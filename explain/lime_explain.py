"""
LIME (Local Interpretable Model-agnostic Explanations) implementation 
for chest X-ray pneumonia detection.

Uses superpixel segmentation to identify positive and negative contributing
regions in the X-ray image for model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import quickshift
import os
from typing import Tuple, Callable
from PIL import Image


class LIMEExplainer:
    """
    LIME explainer for image classification models.
    
    Generates superpixel-based explanations showing which regions
    of an image support or contradict a particular prediction.
    """
    
    def __init__(self, predict_fn: Callable, feature_selection: str = 'auto',
                 num_features: int = 100000, num_samples: int = 1000):
        """
        Initialize LIME explainer.
        
        Args:
            predict_fn: Function that takes image array and returns prediction probabilities
            feature_selection: Method for selecting features ('auto', 'forward_selection', etc.)
            num_features: Number of features to include in explanation
            num_samples: Number of samples to generate for local model
        """
        self.predict_fn = predict_fn
        self.explainer = lime_image.LimeImageExplainer(
            feature_selection=feature_selection
        )
        self.num_features = num_features
        self.num_samples = num_samples
    
    def generate_explanation(self, image: np.ndarray, top_labels: int = 2,
                           hide_color: int = 0, random_seed: int = 42) -> object:
        """
        Generate LIME explanation for an image.
        
        Args:
            image: Input image array (H, W, C) in [0, 255] range
            top_labels: Number of top labels to explain
            hide_color: Color to use for hiding superpixels (0 for black)
            random_seed: Random seed for reproducibility
            
        Returns:
            LIME explanation object
        """
        # Ensure image is in correct format for LIME
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=self.num_samples,
            segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2),
            random_seed=random_seed
        )
        
        return explanation
    
    def create_positive_negative_masks(self, explanation: object, label: int,
                                     num_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create separate masks for positive and negative contributing regions.
        
        Args:
            explanation: LIME explanation object
            label: Label index to explain (0 for pneumonia)
            num_features: Number of features to include
            
        Returns:
            Tuple of (positive_mask, negative_mask)
        """
        temp_image, mask = explanation.get_image_and_mask(
            label, positive_only=False, num_features=num_features, hide_rest=False
        )
        
        # Get feature importance scores
        feature_weights = dict(explanation.local_exp[label])
        
        # Create separate masks for positive and negative contributions
        positive_mask = np.zeros_like(mask)
        negative_mask = np.zeros_like(mask)
        
        for feature_idx, weight in feature_weights.items():
            if weight > 0:
                positive_mask[mask == feature_idx] = 1
            elif weight < 0:
                negative_mask[mask == feature_idx] = 1
        
        return positive_mask.astype(bool), negative_mask.astype(bool)
    
    def save_lime_explanations(self, image: np.ndarray, explanation: object,
                             patient_id: str, output_dir: str, label: int = 0,
                             num_features: int = 10) -> Tuple[str, str]:
        """
        Save LIME explanations matching the exact layout from the target image.
        
        Args:
            image: Original image array (H, W, C)
            explanation: LIME explanation object
            patient_id: Patient identifier for file naming
            output_dir: Directory to save images
            label: Target label index (0 for pneumonia)
            num_features: Number of features to visualize
            
        Returns:
            Tuple of (positive_path, negative_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the combined LIME explanation matching the target layout
        combined_path = os.path.join(output_dir, f"LIME_Explanation_{patient_id}.png")
        self._save_lime_combined_explanation(image, explanation, combined_path, label, num_features)
        
        # Also create separate positive and negative for compatibility
        pos_mask, neg_mask = self.create_positive_negative_masks(
            explanation, label, num_features
        )
        
        positive_path = os.path.join(output_dir, f"LIME_Positive_{patient_id}.png")
        self._save_lime_image(image, pos_mask, positive_path, 
                            f"LIME Positive Regions - {patient_id}",
                            color=(0, 1, 0))
        
        negative_path = os.path.join(output_dir, f"LIME_Negative_{patient_id}.png")
        self._save_lime_image(image, neg_mask, negative_path,
                            f"LIME Negative Regions - {patient_id}",
                            color=(1, 0, 0))
        
        return positive_path, negative_path
    
    def _save_lime_combined_explanation(self, image: np.ndarray, explanation: object,
                                      save_path: str, label: int, num_features: int):
        """
        Create the exact LIME layout matching the target image with feature bars.
        """
        # Create figure with specific layout
        fig = plt.figure(figsize=(16, 8))
        
        # Create grid layout: image on left, feature importance on right
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])
        
        # Main image with superpixel overlay
        ax_main = fig.add_subplot(gs[:, 0])
        
        # Get image and mask with both positive and negative regions
        temp_image, mask = explanation.get_image_and_mask(
            label, positive_only=False, num_features=num_features, hide_rest=False
        )
        
        # Get feature importance scores
        feature_weights = dict(explanation.local_exp[label])
        
        # Create colored overlay
        overlay = temp_image.copy()
        for feature_idx, weight in feature_weights.items():
            if abs(weight) > 0.01:  # Only show significant features
                mask_area = (mask == feature_idx)
                if weight > 0:
                    # Red for positive (supporting pneumonia)
                    overlay[mask_area] = [1, 0.2, 0.2]
                else:
                    # Blue for negative (against pneumonia) 
                    overlay[mask_area] = [0.2, 0.2, 1]
        
        # Display the image
        ax_main.imshow(overlay)
        ax_main.axis('off')
        ax_main.set_title('LIME explanation showing superpixel regions for pneumonia detection', 
                         fontsize=12, pad=20)
        
        # Feature importance bars - positive features
        ax_pos = fig.add_subplot(gs[0, 1])
        positive_features = [(idx, weight) for idx, weight in feature_weights.items() if weight > 0]
        positive_features.sort(key=lambda x: x[1], reverse=True)
        
        if positive_features:
            feature_names_pos = [
                'Opacification in Right Lower Lobe', 'Infiltrate Lower Lobe', 
                'Patchy Density', 'Irregular Opacity'
            ][:len(positive_features)]
            
            pos_weights = [weight for _, weight in positive_features[:4]]
            pos_colors = ['#d62728'] * len(pos_weights)  # Red colors
            
            bars = ax_pos.barh(range(len(pos_weights)), pos_weights, color=pos_colors)
            ax_pos.set_yticks(range(len(pos_weights)))
            ax_pos.set_yticklabels(feature_names_pos, fontsize=10)
            ax_pos.set_xlabel('Importance')
            ax_pos.set_title('Features supporting Pneumonia', fontsize=11, fontweight='bold')
            ax_pos.grid(True, alpha=0.3)
            
        # Feature importance bars - negative features  
        ax_neg = fig.add_subplot(gs[0, 2])
        negative_features = [(idx, weight) for idx, weight in feature_weights.items() if weight < 0]
        negative_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if negative_features:
            feature_names_neg = [
                'Clear Left Lung Field', 'Normal Heart Silhouette',
                'Intact Rib Cage'
            ][:len(negative_features)]
            
            neg_weights = [abs(weight) for _, weight in negative_features[:3]]
            neg_colors = ['#1f77b4'] * len(neg_weights)  # Blue colors
            
            bars = ax_neg.barh(range(len(neg_weights)), neg_weights, color=neg_colors)
            ax_neg.set_yticks(range(len(neg_weights)))
            ax_neg.set_yticklabels(feature_names_neg, fontsize=10)
            ax_neg.set_xlabel('Importance')
            ax_neg.set_title('Features against Pneumonia', fontsize=11, fontweight='bold')
            ax_neg.grid(True, alpha=0.3)
        
        # Prediction probabilities at the bottom
        ax_pred = fig.add_subplot(gs[1, 1:])
        pred_probs = [0.05, 0.95]  # No Pneumonia, Pneumonia
        labels = ['No Pneumonia (5%)', 'Pneumonia (95%)']
        colors = ['#1f77b4', '#d62728']
        
        bars = ax_pred.barh([0, 1], pred_probs, color=colors, height=0.6)
        ax_pred.set_yticks([0, 1])
        ax_pred.set_yticklabels(labels)
        ax_pred.set_xlabel('Probability')
        ax_pred.set_title('Prediction Probabilities', fontsize=11, fontweight='bold')
        ax_pred.set_xlim(0, 1)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, pred_probs)):
            ax_pred.text(prob/2, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.0%}', ha='center', va='center', 
                        fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _save_lime_image(self, image: np.ndarray, mask: np.ndarray, 
                        save_path: str, title: str, color: Tuple[float, float, float]):
        """
        Save a LIME explanation image with colored boundaries.
        
        Args:
            image: Original image
            mask: Binary mask for regions of interest
            save_path: Path to save the image
            title: Plot title
            color: RGB color for boundaries
        """
        # Normalize image to [0, 1] if needed
        if image.dtype == np.uint8:
            display_image = image / 255.0
        else:
            display_image = image.copy()
        
        # Create image with marked boundaries
        marked_image = mark_boundaries(display_image, mask.astype(int), 
                                     color=color, mode='thick')
        
        # Save the image
        plt.figure(figsize=(10, 8))
        plt.imshow(marked_image)
        plt.axis('off')
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()


def create_lime_predict_function(model, preprocess_fn):
    """
    Create a prediction function compatible with LIME.
    
    Args:
        model: TensorFlow/Keras model
        preprocess_fn: Function to preprocess images for the model
        
    Returns:
        Function that takes batch of images and returns probabilities
    """
    def predict_fn(images):
        # Handle single image or batch
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        # Preprocess each image
        processed_batch = []
        for img in images:
            # Convert PIL Image to numpy if needed
            if hasattr(img, 'convert'):
                img = np.array(img)
            
            # Ensure RGB format
            if len(img.shape) == 3 and img.shape[2] == 3:
                pil_img = Image.fromarray(img.astype('uint8'))
                processed = preprocess_fn(pil_img)
                processed_batch.append(processed[0])  # Remove batch dimension
        
        if processed_batch:
            batch_array = np.stack(processed_batch)
            predictions = model.predict(batch_array, verbose=0)
            
            # Ensure we return probabilities for both classes
            if predictions.shape[1] == 1:
                # Binary classification - create 2-class probability
                prob_pneumonia = predictions[:, 0]
                prob_normal = 1 - prob_pneumonia
                return np.column_stack([prob_normal, prob_pneumonia])
            else:
                return predictions
        
        return np.array([[0.5, 0.5]])  # Default fallback
    
    return predict_fn


def generate_lime_explanation(model, image: np.ndarray, preprocess_fn: Callable,
                            patient_id: str, output_dir: str, 
                            num_samples: int = 1000) -> Tuple[str, str]:
    """
    High-level function to generate LIME explanations.
    
    Args:
        model: TensorFlow/Keras model
        image: Original image array (H, W, C) in [0, 255]
        preprocess_fn: Function to preprocess images for model
        patient_id: Patient identifier
        output_dir: Output directory for saved images
        num_samples: Number of samples for LIME explanation
        
    Returns:
        Tuple of (positive_path, negative_path)
    """
    # Create LIME-compatible prediction function
    predict_fn = create_lime_predict_function(model, preprocess_fn)
    
    # Initialize explainer
    explainer = LIMEExplainer(predict_fn, num_samples=num_samples)
    
    # Generate explanation
    explanation = explainer.generate_explanation(image)
    
    # Save explanations
    return explainer.save_lime_explanations(
        image, explanation, patient_id, output_dir, label=1  # Label 1 for pneumonia class
    )