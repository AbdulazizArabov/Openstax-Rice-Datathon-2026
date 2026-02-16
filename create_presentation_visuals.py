"""
Generate all visualizations for Rice Datathon presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Load data
print("Loading data...")
with open('training.json', 'r') as f:
    training = json.load(f)

with open('testing.json', 'r') as f:
    testing = json.load(f)

# Load predictions
predictions = pd.read_csv('predictions_cleaned_urls.csv')

# Extract standards from training
train_standards = []
for book in training['titles']:
    for cluster_group in book.get('items', []):
        for cluster in cluster_group.get('clusters', []):
            for item in cluster.get('items', []):
                stds = item.get('standards', [])
                if stds:
                    train_standards.append(stds[0])

# Extract standards from testing
test_standards = []
test_item_data = []
item_id = 0
for book in testing['titles']:
    for cluster_group in book.get('items', []):
        for cluster in cluster_group.get('clusters', []):
            for item in cluster.get('items', []):
                stds = item.get('standards', [])
                if stds:
                    test_standards.append(stds[0])
                    test_item_data.append({
                        'item_id': item_id,
                        'standard': stds[0]
                    })
                    item_id += 1

test_df = pd.DataFrame(test_item_data)
results_df = test_df.merge(predictions, on='item_id')

print(f"Training standards: {len(set(train_standards))}")
print(f"Testing standards: {len(set(test_standards))}")
print(f"Overlap: {len(set(train_standards) & set(test_standards))}")

# ============================================================================
# VISUALIZATION 1: The Zero-Overlap Problem
# ============================================================================
print("\n1. Creating overlap visualization...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Venn-diagram style visualization
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

circle1 = Circle((0.3, 0.5), 0.25, color='#3498db', alpha=0.4, label='Training')
circle2 = Circle((0.7, 0.5), 0.25, color='#e74c3c', alpha=0.4, label='Testing')

ax.add_patch(circle1)
ax.add_patch(circle2)

ax.text(0.3, 0.5, f'Training\n96 standards', ha='center', va='center', fontsize=14, weight='bold')
ax.text(0.7, 0.5, f'Testing\n15 standards', ha='center', va='center', fontsize=14, weight='bold')
ax.text(0.5, 0.5, 'ZERO\nOVERLAP', ha='center', va='center', fontsize=16, weight='bold', color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('The Zero-Overlap Challenge\nTraining and Test Standards Are Completely Different', 
             fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('viz1_zero_overlap.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: viz1_zero_overlap.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Approach Evolution
# ============================================================================
print("\n2. Creating approach evolution...")

approaches = ['k-NN\n(Training)', 'Definition\nMatching', 'Hierarchical\n+ Semantic', 
              'With URL\nContent', 'Cleaned\nURL Content']
accuracies = [0.0, 45.0, 48.9, 51.1, 57.4]
colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#27ae60']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(approaches, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold')

ax.axhline(y=57.4, color='green', linestyle='--', linewidth=2, label='Final Model')
ax.set_ylabel('Test Accuracy (%)', fontsize=14, weight='bold')
ax.set_title('Model Evolution: From 0% to 57.4%', fontsize=16, weight='bold', pad=20)
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('viz2_approach_evolution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: viz2_approach_evolution.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Per-Standard Performance
# ============================================================================
print("\n3. Creating per-standard performance...")

# Calculate per-standard accuracy
std_performance = []
for std in test_df['standard'].value_counts().index:
    subset = results_df[results_df['standard'] == std]
    correct = (subset['predicted_standard'] == subset['standard']).sum()
    total = len(subset)
    accuracy = correct / total if total > 0 else 0
    
    std_performance.append({
        'standard': std,
        'accuracy': accuracy * 100,
        'correct': correct,
        'total': total,
        'category': 'Perfect (100%)' if accuracy == 1.0 else 
                   'Good (50-99%)' if accuracy >= 0.5 else 
                   'Failed (0-49%)'
    })

perf_df = pd.DataFrame(std_performance).sort_values('accuracy', ascending=True)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 10))

colors_map = {'Perfect (100%)': '#27ae60', 'Good (50-99%)': '#f39c12', 'Failed (0-49%)': '#e74c3c'}
colors = [colors_map[cat] for cat in perf_df['category']]

bars = ax.barh(range(len(perf_df)), perf_df['accuracy'], color=colors, 
               alpha=0.8, edgecolor='black', linewidth=1)

# Add labels
for i, (idx, row) in enumerate(perf_df.iterrows()):
    ax.text(row['accuracy'] + 2, i, f"{row['correct']}/{row['total']}", 
            va='center', fontsize=10, weight='bold')

ax.set_yticks(range(len(perf_df)))
ax.set_yticklabels(perf_df['standard'], fontsize=10)
ax.set_xlabel('Accuracy (%)', fontsize=14, weight='bold')
ax.set_title('Performance by Standard\n8 Standards at 100%, 10 at 50%+', 
             fontsize=16, weight='bold', pad=20)
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', label='Perfect (100%)'),
    Patch(facecolor='#f39c12', label='Good (50-99%)'),
    Patch(facecolor='#e74c3c', label='Failed (0-49%)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('viz3_per_standard_performance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: viz3_per_standard_performance.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Confusion Matrix for Top Standards
# ============================================================================
print("\n4. Creating confusion patterns...")

# Get top 6 most common test standards
top_standards = test_df['standard'].value_counts().head(6).index.tolist()

# Create confusion matrix
confusion_data = []
for true_std in top_standards:
    row = []
    subset = results_df[results_df['standard'] == true_std]
    for pred_std in top_standards:
        count = (subset['predicted_standard'] == pred_std).sum()
        row.append(count)
    confusion_data.append(row)

confusion_matrix = np.array(confusion_data)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')

# Add text annotations
for i in range(len(top_standards)):
    for j in range(len(top_standards)):
        text = ax.text(j, i, confusion_matrix[i, j],
                      ha="center", va="center", color="black" if confusion_matrix[i, j] < 8 else "white",
                      fontsize=12, weight='bold')

ax.set_xticks(range(len(top_standards)))
ax.set_yticks(range(len(top_standards)))
ax.set_xticklabels(top_standards, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(top_standards, fontsize=10)
ax.set_xlabel('Predicted Standard', fontsize=12, weight='bold')
ax.set_ylabel('True Standard', fontsize=12, weight='bold')
ax.set_title('Confusion Matrix: Top 6 Standards\n(Diagonal = Correct Predictions)', 
             fontsize=14, weight='bold', pad=20)

plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig('viz4_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: viz4_confusion_matrix.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Architecture Diagram (Text-based)
# ============================================================================
print("\n5. Creating architecture diagram...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Define components
components = [
    {'pos': (0.5, 0.95), 'text': 'TEST ITEM\n"Multiply Using Scientific Notation"', 
     'color': '#3498db', 'size': 0.12},
    
    {'pos': (0.25, 0.75), 'text': 'Fetch URL\nContent', 'color': '#9b59b6', 'size': 0.1},
    {'pos': (0.75, 0.75), 'text': 'Extract\nMetadata', 'color': '#9b59b6', 'size': 0.1},
    
    {'pos': (0.5, 0.55), 'text': 'Clean Text\n(Remove noise, keep math terms)', 
     'color': '#f39c12', 'size': 0.12},
    
    {'pos': (0.5, 0.35), 'text': 'Sentence Transformer\n(all-MiniLM-L6-v2)\n384-dim embeddings', 
     'color': '#e74c3c', 'size': 0.14},
    
    {'pos': (0.25, 0.15), 'text': 'Category\nFiltering\n(8.EE, HSG.GMD)', 
     'color': '#27ae60', 'size': 0.1},
    {'pos': (0.75, 0.15), 'text': 'Semantic\nMatching\nvs 173 definitions', 
     'color': '#27ae60', 'size': 0.1},
    
    {'pos': (0.5, -0.05), 'text': 'PREDICTION\n8.EE.A.4 (57.4% accuracy)', 
     'color': '#27ae60', 'size': 0.12},
]

# Draw components
for comp in components:
    circle = Circle(comp['pos'], comp['size'], color=comp['color'], alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(comp['pos'][0], comp['pos'][1], comp['text'], 
           ha='center', va='center', fontsize=10, weight='bold', 
           color='white' if comp['color'] == '#e74c3c' else 'black')

# Draw arrows
arrows = [
    ((0.5, 0.83), (0.25, 0.80)),
    ((0.5, 0.83), (0.75, 0.80)),
    ((0.25, 0.65), (0.5, 0.60)),
    ((0.75, 0.65), (0.5, 0.60)),
    ((0.5, 0.43), (0.5, 0.40)),
    ((0.5, 0.21), (0.25, 0.20)),
    ((0.5, 0.21), (0.75, 0.20)),
    ((0.25, 0.05), (0.5, 0.02)),
    ((0.75, 0.05), (0.5, 0.02)),
]

for start, end in arrows:
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_xlim(0, 1)
ax.set_ylim(-0.15, 1.05)
ax.set_title('Model Architecture: Hierarchical Category-Aware Semantic Classification', 
            fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('viz5_architecture.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: viz5_architecture.png")
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n6. Creating summary table...")

summary_data = {
    'Metric': [
        'Total Test Items',
        'Correct Predictions',
        'Overall Accuracy',
        'Standards at 100%',
        'Standards at 50%+',
        'Training/Test Overlap',
        'Embedding Dimension',
        'Text Length (avg)',
        'Processing Time'
    ],
    'Value': [
        '94',
        '54',
        '57.4%',
        '8 / 15 (53%)',
        '10 / 15 (67%)',
        '0 standards',
        '384 dimensions',
        '1253 characters',
        '~3 minutes'
    ]
}

summary_df = pd.DataFrame(summary_data)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                cellLoc='left', loc='center',
                colWidths=[0.5, 0.5])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax.set_title('Model Performance Summary', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('viz6_summary_table.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: viz6_summary_table.png")
plt.close()

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS CREATED!")
print("=" * 80)
print("\nFiles generated:")
print("  1. viz1_zero_overlap.png - The challenge")
print("  2. viz2_approach_evolution.png - Your journey")
print("  3. viz3_per_standard_performance.png - Detailed results")
print("  4. viz4_confusion_matrix.png - Error analysis")
print("  5. viz5_architecture.png - Technical approach")
print("  6. viz6_summary_table.png - Key metrics")
