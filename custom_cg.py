import torch
from PIL import Image
import numpy as np

# This list must match the order your classifier was trained on.
CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']


def generate_with_custom_cg_on_pipeline(
    pipe, classifier, prompt, run_params, seed,
    cg_params, safe_idx=3
):
    """
    Generate an image using a diffusion pipeline (ESD/UCE/SLD/Baseline) with classifier guidance (CG) applied.
    Args:
        pipe: The loaded diffusion pipeline (e.g., ESD, UCE, SLD, Baseline)
        classifier: The trained safety classifier
        prompt: The text prompt
        run_params: Namespace with pipeline-specific params (e.g., num_inference_steps, guidance_scale, negative_prompt)
        seed: Random seed for reproducibility
        cg_params: Namespace with CG-specific params (e.g., safety_scale, mid_fraction)
        safe_idx: Index of the 'safe' class in the classifier
    Returns:
        image_pil: Generated PIL image
    """
    device = pipe.device
    target_dtype = pipe.unet.dtype
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size),
        generator=generator,
        device=device,
        dtype=target_dtype
    ) * pipe.scheduler.init_noise_sigma

    # Text embeddings
    with torch.no_grad():
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0].to(target_dtype)
        uncond_input = pipe.tokenizer(run_params.negative_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0].to(target_dtype)
        full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    pipe.scheduler.set_timesteps(run_params.num_inference_steps, device=device)
    apply_safety_guidance = (cg_params.safety_scale > 0)
    start_step_idx = int(run_params.num_inference_steps * (1 - cg_params.mid_fraction))
    end_step_idx = run_params.num_inference_steps

    mid_block_features = {}
    hook_handle = pipe.unet.mid_block.register_forward_hook(lambda m, i, o: mid_block_features.update({'output': o[0]}))

    for i, t in enumerate(pipe.scheduler.timesteps):
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input_for_unet = pipe.scheduler.scale_model_input(latent_model_input, t).to(target_dtype)
            noise_pred = pipe.unet(latent_model_input_for_unet, t, encoder_hidden_states=full_text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + run_params.cfg_guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Apply classifier guidance on top of method
        if apply_safety_guidance and start_step_idx <= i < end_step_idx:
            with torch.enable_grad():
                latents_grad = latents.detach().clone().requires_grad_(True)
                latent_grad_input_for_unet = pipe.scheduler.scale_model_input(latents_grad, t).to(target_dtype)
                _ = pipe.unet(latent_grad_input_for_unet, t, encoder_hidden_states=uncond_embeddings)
                features = mid_block_features['output'].to(target_dtype).unsqueeze(0)
                logits = classifier(features)
                probs = torch.softmax(logits, -1).squeeze(0)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    grad = torch.zeros_like(latents_grad)
                else:
                    loss = -torch.log(probs[safe_idx] + 1e-6).sum()
                    grad = torch.autograd.grad(loss, latents_grad)[0]
                latents = latents.detach() - cg_params.safety_scale * grad.to(latents.dtype)
                del grad, latents_grad, features, logits, probs
                if 'output' in mid_block_features: del mid_block_features['output']
                torch.cuda.empty_cache()
        latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

    hook_handle.remove()
    with torch.no_grad():
        latents = 1 / pipe.vae.config.scaling_factor * latents
        image_decoded = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
        image_np = ((image_decoded / 2 + 0.5).clamp(0, 1)).permute(0, 2, 3, 1).cpu().numpy()[0]
        image_pil = Image.fromarray((image_np * 255).round().astype("uint8"))
    return image_pil


def generate_with_custom_cg(pipe, classifier, prompt, run_params, seed):
    """Generates an image using your Custom Classifier Guidance (CG)."""
    device = pipe.device
    
    # --- IMPORTANT FIX 1: Get the correct dtype from the pipe's unet ---
    # This ensures latents and other tensors match the UNet's expected precision
    target_dtype = pipe.unet.dtype 
    
    generator = torch.Generator(device).manual_seed(seed)
    # Initialize latents with the correct dtype
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size),
        generator=generator,
        device=device,
        dtype=target_dtype # Use the dynamically determined dtype
    ) * pipe.scheduler.init_noise_sigma # Apply scheduler's initial noise scaling

    # Pre-process text embeddings
    with torch.no_grad():
        # Text embeddings should also be cast to target_dtype
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0].to(target_dtype) # Cast
        
        uncond_input = pipe.tokenizer(run_params.negative_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0].to(target_dtype) # Cast
        
        full_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    pipe.scheduler.set_timesteps(run_params.num_inference_steps, device=device)
    apply_safety_guidance = (run_params.safety_scale > 0)
    
    # Correct calculation for start_step and end_step
    # mid_fraction should typically be applied to the total steps to define the active region
    start_step_idx = int(run_params.num_inference_steps * (1 - run_params.mid_fraction))
    end_step_idx = run_params.num_inference_steps # Guidance applied until the very end of inference
    # If mid_fraction means the *middle portion*, then:
    # steps_for_guidance = int(run_params.num_inference_steps * run_params.mid_fraction)
    # start_step_idx = (run_params.num_inference_steps - steps_for_guidance) // 2
    # end_step_idx = start_step_idx + steps_for_guidance
    # I'm using the interpretation from your original script where mid_fraction affects the *length* of the applied range.
    # Adjust `start_step_idx` and `end_step_idx` if your interpretation of `mid_fraction` is different.

    mid_block_features = {}
    hook_handle = pipe.unet.mid_block.register_forward_hook(lambda m, i, o: mid_block_features.update({'output': o[0]}))

    for i, t in enumerate(pipe.scheduler.timesteps):
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            # Ensure latent_model_input is cast to target_dtype before passing to UNet
            latent_model_input_for_unet = pipe.scheduler.scale_model_input(latent_model_input, t).to(target_dtype)

            noise_pred = pipe.unet(latent_model_input_for_unet, t, encoder_hidden_states=full_text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + run_params.cfg_guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Apply safety guidance
        if apply_safety_guidance and start_step_idx <= i < end_step_idx: # Changed variable names
            with torch.enable_grad():
                latents_grad = latents.detach().clone().requires_grad_(True)
                # Ensure latents_grad for UNet is cast to target_dtype
                latent_grad_input_for_unet = pipe.scheduler.scale_model_input(latents_grad, t).to(target_dtype)

                # Pass only the unconditional embeddings to the UNet when computing features for the classifier
                # This ensures the features are generated without the positive prompt's influence for safety check
                _ = pipe.unet(latent_grad_input_for_unet, t, encoder_hidden_states=uncond_embeddings)
                
                # --- IMPORTANT FIX 2: Ensure features for classifier match classifier's dtype ---
                # Classifier's dtype is set in main.py, so we must match it here.
                # Assuming classifier's dtype is `target_dtype` from main.py's `classifier.to(dtype=DTYPE)`.
                features = mid_block_features['output'].to(target_dtype).unsqueeze(0) 
                
                logits = classifier(features)
                probs = torch.softmax(logits, -1).squeeze(0)
                
                # Check for NaNs/Infs in probs before log to prevent errors
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    # Handle cases where probabilities become problematic
                    # This can happen in FP16 or with extreme values
                    print(f"Warning: NaNs/Infs in probabilities at timestep {i}. Skipping guidance step.")
                    grad = torch.zeros_like(latents_grad) # Or continue with previous latents
                else:
                    # Sum over potentially multiple dimensions if your loss expects it,
                    # otherwise just select the safe_class_index.
                    # Ensure safety_scale doesn't lead to issues.
                    loss = -torch.log(probs[run_params.safe_class_index] + 1e-6).sum()
                    grad = torch.autograd.grad(loss, latents_grad)[0]
                
            # Apply guidance. Ensure gradient is also on the correct device and dtype if not already.
            # latents = latents.detach() - run_params.safety_scale * grad.to(target_dtype) # Ensure grad is right dtype
            # Better to cast grad to latents' dtype directly
            latents = latents.detach() - run_params.safety_scale * grad.to(latents.dtype) 
            
            # Clear cache to prevent memory issues, especially with repeated grad computations
            del grad, latents_grad, features, logits, probs
            if 'output' in mid_block_features: del mid_block_features['output'] # Clear feature cache
            torch.cuda.empty_cache()
        
        # Scheduler step. latents are already correct dtype.
        latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

    hook_handle.remove() # Always remove hooks to prevent memory leaks

    # --- IMPORTANT FIX 3: Use pipe.vae.decode and proper scaling ---
    # The VAE expects latents to be scaled by its config's scaling factor.
    # The output is then clamped and normalized.
    with torch.no_grad():
        latents = 1 / pipe.vae.config.scaling_factor * latents # Scale for VAE input
        image_decoded = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample # Ensure VAE gets correct dtype input
        
        # Normalize to [0, 1] and convert to numpy for PIL
        image_np = ((image_decoded / 2 + 0.5).clamp(0, 1)).permute(0, 2, 3, 1).cpu().numpy()[0]
    
    image_pil = Image.fromarray((image_np * 255).round().astype("uint8"))

    return image_pil

