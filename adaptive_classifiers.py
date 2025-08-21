#!/usr/bin/env python3
"""
7a_adaptive_classifier.py
Standalone AdaptiveClassifier extracted from 6j_train_research_classifier.py
Follows the project's established naming convention
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import logging

CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']

class AdaptiveClassifier(nn.Module):
    """
    Adaptive classifier - EXACT copy from your working 6j_train_research_classifier.py
    Supports (1280, 8, 8) research-based latents with 86% accuracy
    """
    
    def __init__(self, input_shape, num_classes, architecture='auto', hidden_dims=None):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Determine optimal architecture - same as 6j
        if architecture == 'auto':
            if len(input_shape) >= 3:  # (C, H, W) or higher
                self.architecture = 'cnn'
            elif len(input_shape) == 2:  # (H, W)
                self.architecture = 'cnn_2d'
            else:  # (N,)
                self.architecture = 'mlp'
        
        # Build model - same as 6j
        self.model = self._build_model(hidden_dims)
        
        # Initialize weights - same as 6j
        self.apply(self._init_weights)
    
    def _build_model(self, hidden_dims):
        """Build model based on architecture type - EXACT copy from 6j"""
        if self.architecture == 'cnn':
            return self._build_cnn()
        elif self.architecture == 'cnn_2d':
            return self._build_cnn_2d()
        elif self.architecture == 'mlp':
            return self._build_mlp(hidden_dims)
        elif self.architecture == 'hybrid':
            return self._build_hybrid()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _build_cnn(self):
        """Build CNN for 3D+ inputs like (C, H, W) - EXACT copy from 6j"""
        channels, height, width = self.input_shape[:3]
        
        # Adaptive layer sizing based on input dimensions
        conv_layers = []
        current_channels = channels
        
        # Progressive channel reduction
        target_channels = [min(512, channels), 256, 128, 64]
        
        for i, out_channels in enumerate(target_channels):
            if current_channels <= out_channels and i > 0:
                break
                
            conv_layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2) if min(height, width) >= 4 else nn.Identity(),
                nn.Dropout2d(0.1)
            ])
            
            current_channels = out_channels
            height = max(1, height // 2)
            width = max(1, width // 2)
        
        # Global pooling and classifier
        model = nn.Sequential(
            *conv_layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        return model
    
    def _build_cnn_2d(self):
        """Build CNN for 2D inputs like (H, W) - EXACT copy from 6j"""
        height, width = self.input_shape
        
        class CNN2D(nn.Module):
            def __init__(self, h, w, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2) if min(h, w) >= 4 else nn.Identity(),
                    
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2) if min(h, w) >= 8 else nn.Identity(),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                # Add channel dimension for 2D input
                if len(x.shape) == 3:  # (batch, H, W)
                    x = x.unsqueeze(1)  # (batch, 1, H, W)
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return CNN2D(height, width, self.num_classes)
    
    def _build_mlp(self, hidden_dims):
        """Build MLP for 1D inputs - EXACT copy from 6j"""
        input_size = np.prod(self.input_shape)
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = [nn.Flatten()]
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def _build_hybrid(self):
        """Build hybrid CNN+MLP architecture - EXACT copy from 6j"""
        return self._build_cnn()  # Fallback to CNN for now
    
    def _init_weights(self, m):
        """Initialize weights using best practices - EXACT copy from 6j"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)



# Utility functions for easy loading
def load_trained_classifier(checkpoint_path, device='auto'):
    """
    Easy function to load your trained classifier
    
    Args:
        checkpoint_path: Path to your trained model (e.g., "experiments/full_research_safety_classifier/best_model.pth")
        device: Device to load on ('auto', 'cuda', 'cpu')
    
    Returns:
        Loaded and ready classifier
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model with same config as your 6j training
    model = AdaptiveClassifier(
        input_shape=(1280, 8, 8),
        num_classes=5,
        architecture='cnn'
    ).to(device)
    
    # Load your trained weights
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"✅ Loaded classifier from {checkpoint_path}")
    logging.info(f"   Best epoch: {checkpoint.get('epoch', 'unknown')}")
    logging.info(f"   Best accuracy: {checkpoint.get('balanced_accuracy', 'unknown'):.4f}")
    
    return model

def get_class_names():
    """Get the class names used in training"""
    return ['gore', 'hate', 'medical', 'safe', 'sexual']

def predict_safety(model, features):
    """
    Simple prediction function
    
    Args:
        model: Loaded classifier
        features: Feature tensor (1280, 8, 8) or batch
    
    Returns:
        Dictionary with predictions
    """
    model.eval()
    with torch.no_grad():
        if len(features.shape) == 3:  # Single sample
            features = features.unsqueeze(0)
        
        logits = model(features)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        
        class_names = get_class_names()
        
        results = {
            'predicted_class': [class_names[p] for p in predictions.cpu().numpy()],
            'confidence': confidence.cpu().numpy(),
            'is_safe': (predictions == 3).cpu().numpy(),  # 'safe' is index 3
            'unsafe_probability': 1 - probs[:, 3].cpu().numpy(),  # 1 - P(safe)
            'all_probabilities': {
                name: probs[:, i].cpu().numpy() 
                for i, name in enumerate(class_names)
            }
        }
        
        return results

if __name__ == "__main__":
    print("AdaptiveClassifier module - extracted from 6j_train_research_classifier.py")
    print("Usage:")
    print("  from 7a_adaptive_classifier import load_trained_classifier, predict_safety")
    print("  model = load_trained_classifier('experiments/full_research_safety_classifier/best_model.pth')")


# adaptive_classifiers.py
import torch.nn as nn

def _init_weights(m):
    """Initializes weights for the models."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Classifier_SD_V1_4_V2_1(nn.Module):
    """
    Classifier for Stable Diffusion v1.4 and v2.1 latents.
    Input shape: (1280, 8, 8)
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (512, 4, 4)
            nn.Dropout2d(0.1),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (256, 2, 2)
            nn.Dropout2d(0.1),

            nn.AdaptiveAvgPool2d(1), # -> (256, 1, 1)
            nn.Flatten(),           # -> (256)

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.model(x)

# class Classifier_SD_V1_5(nn.Module):
#     """
#     Classifier for Stable Diffusion v1.5 latents.
#     Input shape: (320, 8, 8)
#     """
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(320, 320, kernel_size=3, padding=1),
#             nn.BatchNorm2d(320),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # -> (320, 4, 4)
#             nn.Dropout2d(0.1),

#             nn.Conv2d(320, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2), # -> (256, 2, 2)
#             nn.Dropout2d(0.1),
            
#             nn.AdaptiveAvgPool2d(1), # -> (256, 1, 1)
#             nn.Flatten(),           # -> (256)
            
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Linear(128, num_classes)
#         )
#         self.apply(_init_weights)

#     def forward(self, x):
#         return self.model(x)

def get_classifier(sd_version, num_classes):
    """
    Factory function to get the correct classifier model for the given SD version.
    """
    if sd_version in ['sd1_4', 'sd2_1']:
        print(f"Instantiating classifier for SD v1.4/v2.1 (1280 channels).")
        return Classifier_SD_V1_4_V2_1(num_classes)
    elif sd_version == 'sd1_5':
        print(f"Instantiating classifier for SD v1.5 (320 channels).")
        return Classifier_SD_V1_5(num_classes)
    else:
        raise ValueError(f"No classifier defined for SD version: {sd_version}")
    
# --- Loading Function (from your 7a script) ---
def preprocess_image_for_vae(image: Image.Image):
    image = image.convert("RGB").resize((512, 512), resample=Image.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np[None].transpose(0, 3, 1, 2))
    return 2.0 * image_tensor - 1.0

def extract_mid_block_features(pipe, latent_tensor):
    mid_block_features = {}
    hook = pipe.unet.mid_block.register_forward_hook(lambda m, i, o: mid_block_features.update({'output': o[0]}))
    with torch.no_grad():
        dummy_text_embeds = torch.zeros((1, pipe.tokenizer.model_max_length, pipe.text_encoder.config.hidden_size), device=pipe.device, dtype=pipe.text_encoder.dtype)
        _ = pipe.unet(latent_tensor, timestep=torch.tensor(0, device=pipe.device), encoder_hidden_states=dummy_text_embeds)
    hook.remove()
    return mid_block_features.get('output')

def post_hoc_classify(image_pil, classifier, ref_pipe, safe_class_index):
    """Uses our custom classifier with robust dtype handling."""
    device = ref_pipe.device
    try:
        image_tensor = preprocess_image_for_vae(image_pil).to(device, dtype=ref_pipe.vae.dtype)
        with torch.no_grad():
            initial_latent = ref_pipe.vae.encode(image_tensor).latent_dist.sample() * ref_pipe.vae.config.scaling_factor
            features = extract_mid_block_features(ref_pipe, initial_latent)
        
        if features is None:
            return "ERROR_POST_HOC", False, [0.0] * len(CLASSIFIER_CLASS_NAMES)
        
        # --- THIS IS THE FIX FOR THE DTYPE ERROR ---
        # Get the expected dtype from the classifier's own parameters (e.g., float32)
        classifier_dtype = next(classifier.parameters()).dtype
        # Cast the input features to the classifier's expected dtype before the forward pass
        logits = classifier(features.unsqueeze(0).to(classifier_dtype))
        # ---------------------------------------------

        probs = torch.softmax(logits, -1).squeeze(0).detach().cpu().numpy()
        pred_idx = np.argmax(probs)
        return CLASSIFIER_CLASS_NAMES[pred_idx], (pred_idx == safe_class_index), probs.tolist()
    except Exception as e:
        logging.error(f"Error during post-hoc classification: {e}")
        return "ERROR_POST_HOC", False, [0.0] * len(CLASSIFIER_CLASS_NAMES)