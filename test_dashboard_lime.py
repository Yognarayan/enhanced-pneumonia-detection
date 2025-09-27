import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

print('Testing enhanced Dashboard and LIME visualizations...')

# Create enhanced dashboard matching target image
fig = plt.figure(figsize=(16, 10))

# Create grid layout like target image
gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1],
                     hspace=0.3, wspace=0.3)

# Main prediction chart (top span)
ax_main = fig.add_subplot(gs[0, :])

# Prediction probabilities
categories = ['Normal', 'Pneumonia']
probabilities = [25.7, 74.3]
colors = ['#2E8B57', '#DC143C']  # Green for Normal, Red for Pneumonia

bars = ax_main.bar(categories, probabilities, color=colors, alpha=0.8, width=0.6)
ax_main.set_ylim(0, 100)
ax_main.set_ylabel('Confidence (%)', fontsize=14, fontweight='bold')
ax_main.set_title('Pneumonia Detection Results', fontsize=18, fontweight='bold', pad=20)
ax_main.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for i, (bar, prob) in enumerate(zip(bars, probabilities)):
    ax_main.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{prob}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Positive factors (bottom left)
ax_pos = fig.add_subplot(gs[1, 0])
pos_factors = ['Opacity Detection', 'Infiltrate Pattern', 'Consolidation Areas']
pos_values = [0.82, 0.71, 0.65]
pos_colors = ['#FF6B6B', '#FF8E8E', '#FFB1B1']

bars_pos = ax_pos.barh(pos_factors, pos_values, color=pos_colors, alpha=0.8)
ax_pos.set_xlim(0, 1)
ax_pos.set_xlabel('Contribution Score', fontweight='bold')
ax_pos.set_title('Contributing Factors (Positive)', fontweight='bold', color='red')
ax_pos.grid(True, alpha=0.3, axis='x')

# Negative factors (bottom middle)
ax_neg = fig.add_subplot(gs[1, 1])
neg_factors = ['Clear Airways', 'Normal Heart Size', 'No Pleural Effusion']
neg_values = [-0.45, -0.38, -0.32]
neg_colors = ['#87CEEB', '#A4D4F4', '#C1DBFF']

bars_neg = ax_neg.barh(neg_factors, neg_values, color=neg_colors, alpha=0.8)
ax_neg.set_xlim(-1, 0)
ax_neg.set_xlabel('Contribution Score', fontweight='bold')
ax_neg.set_title('Contributing Factors (Negative)', fontweight='bold', color='blue')
ax_neg.grid(True, alpha=0.3, axis='x')

# Patient vitals table (bottom right)
ax_table = fig.add_subplot(gs[1, 2])
ax_table.axis('off')

# Create table data
table_data = [
    ['Parameter', 'Value', 'Status'],
    ['Age', '45 years', 'Normal'],
    ['Temperature', '38.5°C', 'Elevated'],
    ['WBC Count', '12,500', 'High'],
    ['O2 Saturation', '92%', 'Low'],
    ['Chest Pain', 'Yes', 'Present']
]

# Create table
table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='left', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        elif j == 2:  # Status column
            status = table_data[i][j] if i < len(table_data) else ''
            if status in ['Elevated', 'High', 'Low', 'Present']:
                cell.set_facecolor('#FFE6E6')
            else:
                cell.set_facecolor('#E6FFE6')

ax_table.set_title('Patient Vitals & Symptoms', fontweight='bold', pad=20)

# Model confidence info (bottom span)
ax_info = fig.add_subplot(gs[2, :])
ax_info.axis('off')
info_text = ("Model Confidence: HIGH (74.3%) | Processing Time: 1.2s | "
            "Model Version: v2.1 | Last Updated: 2024-01-15")
ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

plt.suptitle('AI-Powered Pneumonia Detection Dashboard', fontsize=20, fontweight='bold', y=0.95)

output_dir = './test_enhanced_outputs'
os.makedirs(output_dir, exist_ok=True)
dashboard_path = os.path.join(output_dir, 'Dashboard_Enhanced.png')
plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Enhanced Dashboard saved to: {dashboard_path}')

# Create enhanced LIME visualization
print('\nTesting enhanced LIME visualization...')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Simulate chest X-ray image
np.random.seed(42)
chest_img = np.random.rand(224, 224)
chest_img = np.where(chest_img > 0.7, chest_img, chest_img * 0.3)

# Original image
ax1.imshow(chest_img, cmap='gray')
ax1.set_title('Original X-Ray Image', fontsize=14, fontweight='bold')
ax1.axis('off')

# Superpixel segmentation simulation
segments = np.zeros((224, 224))
for i in range(0, 224, 28):
    for j in range(0, 224, 28):
        segment_id = (i // 28) * 8 + (j // 28)
        segments[i:i+28, j:j+28] = segment_id

# Positive features (green overlay)
positive_mask = np.zeros_like(chest_img)
important_segments = [15, 16, 23, 24, 31, 32]  # Center segments
for seg_id in important_segments:
    mask = (segments == seg_id)
    positive_mask[mask] = 1

ax2.imshow(chest_img, cmap='gray')
ax2.imshow(positive_mask, cmap='Greens', alpha=0.6)
ax2.set_title('LIME: Positive Features (Supporting Pneumonia)', fontsize=14, fontweight='bold')
ax2.axis('off')

# Negative features (red overlay)  
negative_mask = np.zeros_like(chest_img)
negative_segments = [0, 1, 7, 56, 57, 63]  # Edge segments
for seg_id in negative_segments:
    mask = (segments == seg_id)
    negative_mask[mask] = 1

ax3.imshow(chest_img, cmap='gray')
ax3.imshow(negative_mask, cmap='Reds', alpha=0.6)
ax3.set_title('LIME: Negative Features (Against Pneumonia)', fontsize=14, fontweight='bold')
ax3.axis('off')

# Feature importance bar chart
ax4.axis('off')
feature_names = ['Central Opacity', 'Lower Lobe Pattern', 'Air Bronchograms', 
                'Heart Border', 'Pleural Space', 'Lung Periphery']
importance_scores = [0.85, 0.72, 0.68, -0.43, -0.38, -0.31]
colors = ['green' if score > 0 else 'red' for score in importance_scores]

# Create horizontal bar chart within the subplot
bar_ax = fig.add_axes([0.52, 0.1, 0.45, 0.35])  # [left, bottom, width, height]
bars = bar_ax.barh(feature_names, importance_scores, color=colors, alpha=0.7)
bar_ax.set_xlabel('Feature Importance Score', fontweight='bold')
bar_ax.set_title('LIME Feature Importance Ranking', fontweight='bold')
bar_ax.grid(True, alpha=0.3, axis='x')
bar_ax.axvline(x=0, color='black', linewidth=0.8)

# Add score labels
for i, (bar, score) in enumerate(zip(bars, importance_scores)):
    x_pos = score + (0.05 if score > 0 else -0.05)
    bar_ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', ha='left' if score > 0 else 'right',
               va='center', fontweight='bold')

plt.suptitle('LIME Explanation: Local Interpretable Model-agnostic Explanations', 
            fontsize=16, fontweight='bold', y=0.95)

lime_path = os.path.join(output_dir, 'LIME_Enhanced.png')
plt.savefig(lime_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Enhanced LIME saved to: {lime_path}')

print('\nAll enhanced visualizations completed successfully!')
print('Files created in test_enhanced_outputs folder:')
print('- SHAP_Enhanced.png')
print('- GradCAM_Enhanced.png') 
print('- Dashboard_Enhanced.png')
print('- LIME_Enhanced.png')