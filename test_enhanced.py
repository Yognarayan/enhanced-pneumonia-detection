import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print('Testing enhanced SHAP visualization...')
print('Creating enhanced SHAP plot matching target image...')

# Create the exact SHAP plot matching target image
fig, ax = plt.subplots(figsize=(12, 8))

# Medical feature names matching the target image
feature_names = [
    'Opacities', 'Infiltrates', 'Consolidation', 'Pleural Effusion',
    'Nodule', 'Nodule', 'Fracture', 'Atelectasis', 
    'Atelectasis', 'Edema', 'Airspace Opacity'
]

# Generate realistic SHAP values for each feature
np.random.seed(42)
n_features = len(feature_names)
n_samples = 100

# Create realistic SHAP value distributions
shap_data = []
for i, feature in enumerate(feature_names):
    if feature in ['Opacities', 'Infiltrates', 'Consolidation']:
        values = np.random.normal(1.5, 1.0, n_samples)
    elif feature in ['Atelectatus', 'Edema']:
        values = np.random.normal(0, 1.2, n_samples)
    else:
        values = np.random.normal(-0.5, 0.8, n_samples)
    shap_data.extend([(feature, val) for val in values])

# Plot features
feature_positions = {}
for i, feature in enumerate(reversed(feature_names)):
    feature_positions[feature] = i

for i, feature in enumerate(feature_names):
    feature_vals = [val for feat, val in shap_data if feat == feature]
    y_pos = feature_positions[feature]
    y_positions = [y_pos + np.random.normal(0, 0.1) for _ in feature_vals]
    
    feature_colors = []
    for val in feature_vals:
        norm_val = (val + 4) / 8
        if norm_val > 0.8:
            feature_colors.append('#d62728')
        elif norm_val > 0.6:
            feature_colors.append('#ff9999')
        elif norm_val > 0.4:
            feature_colors.append('#cccccc')
        elif norm_val > 0.2:
            feature_colors.append('#9999ff')
        else:
            feature_colors.append('#0000ff')
    
    ax.scatter(feature_vals, y_positions, c=feature_colors, alpha=0.7, s=20)

ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(reversed(feature_names))
ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
ax.set_title('SHAP Summary Plot: Pneumonia Diagnosis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)

output_dir = './test_enhanced_outputs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'SHAP_Enhanced.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Enhanced SHAP saved to: {save_path}')

# Test Grad-CAM style visualization
print('\nTesting enhanced Grad-CAM visualization...')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Simulate chest X-ray image
np.random.seed(42)
chest_img = np.random.rand(224, 224)
chest_img = np.where(chest_img > 0.7, chest_img, chest_img * 0.3)  # Create chest-like pattern

# Simulate heatmap
x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
heatmap = np.exp(-(x**2 + y**2) / 0.5)
heatmap = heatmap / heatmap.max()

# Original image
ax1.imshow(chest_img, cmap='gray')
ax1.set_title('Original X-Ray Image', fontweight='bold')
ax1.axis('off')

# Heatmap
im = ax2.imshow(heatmap, cmap='Reds', alpha=0.8)
ax2.set_title('Grad-CAM Heatmap', fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# Overlay
ax3.imshow(chest_img, cmap='gray')
ax3.imshow(heatmap, cmap='Reds', alpha=0.6)
ax3.set_title('Grad-CAM Overlay', fontweight='bold')
ax3.axis('off')

plt.tight_layout()
gradcam_path = os.path.join(output_dir, 'GradCAM_Enhanced.png')
plt.savefig(gradcam_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Enhanced Grad-CAM saved to: {gradcam_path}')

print('\nAll enhanced visualizations created successfully!')
print('Check the test_enhanced_outputs folder for results.')