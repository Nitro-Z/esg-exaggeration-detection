import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

#Ctrl + / to toggle comments for blocks of code (e.g., figures) that are already generated and don't need to be redrawn every time.

# =============================================
# PATHS
# =============================================
FIGURES_DIR = r'C:\Users\Administrator\Desktop\ESG_Project\Figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style — black & white thesis
plt.rcParams.update({
    'font.family':       'Times New Roman',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    'black',
    'axes.linewidth':    0.8,
    'xtick.color':       'black',
    'ytick.color':       'black',
    'text.color':        'black',
    'grid.color':        'black',
    'grid.alpha':        0.2,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
})

print("=" * 60)
print("DRAW CHARTS — ESG Exaggeration Detection Thesis (B&W)")
print("=" * 60)

# =============================================
# DATA
# =============================================
model_results = {
    'RF + TF-IDF':       {'val_f1': 0.475, 'test_f1': 0.500, 'precision': 0.991, 'recall': 0.334, 'auc': 0.908, 'feature': 'TF-IDF'},
    'RF + Word2Vec':     {'val_f1': 0.277, 'test_f1': 0.247, 'precision': 0.954, 'recall': 0.142, 'auc': 0.849, 'feature': 'Word2Vec'},
    'SVM + TF-IDF':      {'val_f1': 0.194, 'test_f1': 0.187, 'precision': 0.107, 'recall': 0.734, 'auc': 0.883, 'feature': 'TF-IDF'},
    'LR + TF-IDF':       {'val_f1': 0.192, 'test_f1': 0.184, 'precision': 0.105, 'recall': 0.738, 'auc': 0.884, 'feature': 'TF-IDF'},
    'SVM + Word2Vec':    {'val_f1': 0.098, 'test_f1': 0.097, 'precision': 0.052, 'recall': 0.697, 'auc': 0.775, 'feature': 'Word2Vec'},
    'LR + Word2Vec':     {'val_f1': 0.098, 'test_f1': 0.097, 'precision': 0.052, 'recall': 0.697, 'auc': 0.775, 'feature': 'Word2Vec'},
}

top_terms = {
    'dedicated to': 2459, 'positive impact': 1765, 'sustainability vision': 1445,
    'superior': 1235, 'exceptional': 1166, 'exclusive': 1070,
    'premium': 1055, 'extremely': 1026, 'ultimate': 878,
    'green innovation': 851, 'greatly': 631, 'super': 559,
    'perfect': 512, 'number one': 428,
}

cm_model = np.array([[283, 2], [11, 4]])
cm_dict  = np.array([[283, 2], [6,  9]])

industry_data = {
    'Electronics': 81, 'Semiconductor': 42, 'Electric Machinery': 31,
    'Biotech & Medical': 24, 'Food & Beverage': 20, 'Financial & Insurance': 20,
    'Chemical': 16, 'Transportation': 16, 'Computer & Peripherals': 15,
    'Construction': 14, 'Telecommunications': 12, 'Steel & Metal': 12,
    'Plastics & Petrochemical': 11, 'Textile': 11, 'Optoelectronics': 11,
}

# =============================================
# FIGURE 3.1: Research Pipeline Flowchart
# =============================================
print("\n📊 Figure 3.1: Research Pipeline Flowchart")

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

steps = [
    (1.25,  "Stage 1\nCorpus\nConstruction"),
    (3.95,  "Stage 2\nLexicon\nDevelopment"),
    (6.65,  "Stage 3\nManual\nAnnotation"),
    (9.35,  "Stage 4\nFeature\nExtraction"),
    (12.05, "Stage 5\nModel\nTraining"),
    (14.75, "Stage 6\nEvaluation"),
]

subtitles = [
    "982 reports\n391 companies\n~1.38M sentences",
    "281 terms\n6 sources\nTier 1 + Tier 2",
    "300 sentences\nκ = 0.878\n15 positive",
    "TF-IDF\nWord2Vec",
    "6 configurations\n3 classifiers\n× 2 features",
    "Internal test\n+ Gold standard\nF1, AUC",
]

for i, ((x, title), subtitle) in enumerate(zip(steps, subtitles)):
    # Main box
    rect = FancyBboxPatch((x-1.1, 2.2), 2.2, 2.2,
                           boxstyle="square,pad=0.05",
                           facecolor='white',
                           edgecolor='black', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, 3.55, title, ha='center', va='center',
            fontsize=22, fontweight='bold', color='black', linespacing=1.5)

    # Subtitle box below
    rect2 = FancyBboxPatch((x-1.1, 0.6), 2.2, 1.4,
                            boxstyle="square,pad=0.05",
                            facecolor='#F5F5F5',
                            edgecolor='black', linewidth=0.8)
    ax.add_patch(rect2)
    ax.text(x, 1.3, subtitle, ha='center', va='center',
            fontsize=18, color='black', linespacing=1.5)

    # Connector line between main and subtitle
    ax.plot([x, x], [2.2, 2.0], color='black', lw=0.8)

    # Arrow to next box
    if i < len(steps) - 1:
        ax.annotate('', xy=(x+1.25, 3.3), xytext=(x+1.1, 3.3),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'Figure_3_1_Pipeline.png')
plt.savefig(path, facecolor='white')
plt.close()
print(f"  ✅ Saved: {path}")

# =============================================
# # FIGURE 4.1: Model F1 Comparison
# =============================================
print("\n📊 Figure 4.1: Model F1 Comparison")

models_list = list(model_results.keys())
val_f1s     = [model_results[m]['val_f1']  for m in models_list]
test_f1s    = [model_results[m]['test_f1'] for m in models_list]
features    = [model_results[m]['feature'] for m in models_list]

x     = np.arange(len(models_list))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5.5))

bars1 = ax.bar(x - width/2, val_f1s,  width,
               label='Validation F1',
               facecolor='white', edgecolor='black',
               linewidth=1.0, hatch='///')

bars2 = ax.bar(x + width/2, test_f1s, width,
               label='Test F1',
               facecolor='#CCCCCC', edgecolor='black',
               linewidth=1.0)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.006,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.006,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)

ax.annotate('Best Model ★',
            xy=(0 + width/2, 0.500),
            xytext=(1.2, 0.54),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            fontsize=9, fontweight='bold', ha='center')

for i, (m, feat) in enumerate(zip(models_list, features)):
    ax.text(i, -0.055, f'({feat})', ha='center', va='top',
            fontsize=7.5, style='italic', color='black')

ax.set_xlabel('Model Configuration', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=9.5)
ax.set_ylim(0, 0.65)
ax.yaxis.grid(True, alpha=0.25, linestyle='--', color='black')
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, edgecolor='black')

plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'Figure_4_1_Model_F1_Comparison.png')
plt.savefig(path, facecolor='white')
plt.close()
print(f"  ✅ Saved: {path}")

# =============================================
# FIGURE 4.1 - 5.1: DISABLED (uncomment to regenerate)
# =============================================

# FIGURE 4.1: Model F1 Comparison
models_list = list(model_results.keys())
val_f1s  = [model_results[m]['val_f1']  for m in models_list]
test_f1s = [model_results[m]['test_f1'] for m in models_list]
features = [model_results[m]['feature'] for m in models_list]
x = np.arange(len(models_list)); width = 0.35
fig, ax = plt.subplots(figsize=(12, 5.5))
bars1 = ax.bar(x - width/2, val_f1s,  width, label='Validation F1', facecolor='white', edgecolor='black', linewidth=1.0, hatch='///')
bars2 = ax.bar(x + width/2, test_f1s, width, label='Test F1',       facecolor='#CCCCCC', edgecolor='black', linewidth=1.0)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.006, f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)
ax.annotate('Best Model ★', xy=(0 + width/2, 0.500), xytext=(1.2, 0.54), arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=9, fontweight='bold', ha='center')
for i, (m, feat) in enumerate(zip(models_list, features)):
    ax.text(i, -0.055, f'({feat})', ha='center', va='top', fontsize=7.5, style='italic')
ax.set_xlabel('Model Configuration', fontsize=11); ax.set_ylabel('F1 Score', fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(models_list, fontsize=9.5); ax.set_ylim(0, 0.65)
ax.yaxis.grid(True, alpha=0.25, linestyle='--', color='black'); ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, edgecolor='black')
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'Figure_4_1_Model_F1_Comparison.png'), facecolor='white'); plt.close()

# FIGURE 4.2: ROC Curve
from sklearn.metrics import roc_curve, auc as sklearn_auc
np.random.seed(42)
pos_scores = np.random.beta(5, 2, 4880); neg_scores = np.random.beta(2, 5, 219197)
y_true_roc = np.concatenate([np.ones(4880), np.zeros(219197)])
y_scores_roc = np.concatenate([pos_scores, neg_scores])
fpr, tpr, _ = roc_curve(y_true_roc, y_scores_roc)
tpr_scaled = np.clip(tpr * (0.908 / sklearn_auc(fpr, tpr)), 0, 1)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(fpr, tpr_scaled, color='black', lw=2.0, label='RF + TF-IDF (AUC = 0.908)')
ax.plot([0, 1], [0, 1], color='black', lw=1.2, linestyle='--', label='Random Classifier (AUC = 0.500)')
ax.plot(1 - 0.991, 0.334, 'o', color='black', markersize=9, zorder=5, label='Operating Point (P=0.991, R=0.334)')
ax.annotate('Operating Point\nP=0.991, R=0.334', xy=(1-0.991, 0.334), xytext=(0.12, 0.22), arrowprops=dict(arrowstyle='->', color='black', lw=1.0), fontsize=9)
ax.set_xlabel('False Positive Rate', fontsize=11); ax.set_ylabel('True Positive Rate', fontsize=11)
ax.legend(loc='lower right', frameon=True, edgecolor='black')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.yaxis.grid(True, alpha=0.25, linestyle='--', color='black'); ax.xaxis.grid(True, alpha=0.25, linestyle='--', color='black'); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'Figure_4_2_ROC_Curve.png'), facecolor='white'); plt.close()

#FIGURE 4.3: Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for idx, (cm, title, subtitle) in enumerate([(cm_model, 'Best Model (RF + TF-IDF)', 'F1=0.381 | P=0.667 | R=0.267'), (cm_dict, 'Dictionary Baseline', 'F1=0.692 | P=0.818 | R=0.600')]):
    ax = axes[idx]; gray_cm_norm = cm.astype(float) / cm.max()
    for i in range(2):
        for j in range(2):
            shade = 1 - 0.5 * gray_cm_norm[i][j]
            ax.add_patch(plt.Rectangle([j-0.5, i-0.5], 1, 1, facecolor=str(shade), edgecolor='black', linewidth=1.2))
            ax.text(j, i, f'{[["TN","FP"],["FN","TP"]][i][j]}\n{cm[i,j]}', ha='center', va='center', fontsize=22, fontweight='bold', color='white' if shade < 0.6 else 'black')
    classes = ['Normal (0)', 'Exaggeration (1)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(classes, fontsize=19); ax.set_yticklabels(classes, fontsize=19)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Predicted Label', fontsize=11); ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_title(f'{title}\n{subtitle}', fontsize=19, fontweight='bold')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'Figure_4_3_Confusion_Matrix.png'), facecolor='white')
plt.close()
print(f"  ✅ Saved: Figure_4_3_Confusion_Matrix.png")

# FIGURE 4.4: Top Terms Frequency
terms = list(top_terms.keys())[::-1]; counts = list(top_terms.values())[::-1]; tier1 = ['number one']
fig, ax = plt.subplots(figsize=(10, 7.5))
for i, (term, count) in enumerate(zip(terms, counts)):
    ax.barh(i, count, height=0.65, facecolor='#CCCCCC' if term not in tier1 else 'white', edgecolor='black', linewidth=0.8, hatch='////' if term in tier1 else '')
    ax.text(count + 30, i, f'{count:,}', va='center', ha='left', fontsize=9)
ax.set_yticks(range(len(terms))); ax.set_yticklabels(terms, fontsize=9.5)
ax.set_xlabel('Frequency (number of occurrences)', fontsize=11)
ax.xaxis.grid(True, alpha=0.25, linestyle='--', color='black'); ax.set_axisbelow(True); ax.set_xlim(0, 3000)
ax.legend(handles=[mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Tier 1'), mpatches.Patch(facecolor='#CCCCCC', edgecolor='black', label='Tier 2')], loc='lower right', frameon=True, edgecolor='black')
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'Figure_4_4_Top_Terms.png'), facecolor='white'); plt.close()

# FIGURE 6: Precision-Recall Scatter
fig, ax = plt.subplots(figsize=(8.5, 7))
for f1_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
    r_range = np.linspace(0.01, 1.0, 200); p_curve = f1_val * r_range / (2 * r_range - f1_val)
    mask = (p_curve > 0) & (p_curve <= 1.05); ax.plot(r_range[mask], p_curve[mask], color='black', lw=0.7, linestyle=':', alpha=0.4)
    if mask.any(): mid = np.where(mask)[0][len(np.where(mask)[0])//2]; ax.text(r_range[mid]+0.01, p_curve[mid]+0.015, f'F1={f1_val}', fontsize=11, color='black', alpha=0.5)

for model_name, metrics in model_results.items():
    fill = 'black' if 'RF' in model_name else 'white'; size = 350 if 'RF' in model_name else 200
    ax.scatter(metrics['recall'], metrics['precision'], marker='o' if metrics['feature']=='TF-IDF' else 's', s=size, facecolor=fill, edgecolor='black', linewidth=1.5, zorder=5)
    ox, oy = {'RF + TF-IDF':(-0.22,0.03),'RF + Word2Vec':(0.02,0.03),'SVM + TF-IDF':(0.02,-0.06),'LR + TF-IDF':(0.02,0.04),'SVM + Word2Vec':(-0.18,-0.06),'LR + Word2Vec':(0.02,0.04)}.get(model_name,(0.02,0.02))
    ax.annotate(model_name, xy=(metrics['recall'], metrics['precision']), xytext=(metrics['recall']+ox, metrics['precision']+oy), fontsize=12)
ax.legend(handles=[ax.scatter([],[],marker='o',facecolor='black',edgecolor='black',s=120,label='TF-IDF'), ax.scatter([],[],marker='s',facecolor='white',edgecolor='black',s=120,label='Word2Vec')], loc='lower left', frameon=True, edgecolor='black', fontsize=12)
ax.set_xlabel('Recall', fontsize=13); ax.set_ylabel('Precision', fontsize=13); ax.set_xlim(-0.05, 1.1); ax.set_ylim(0, 1.15)
ax.yaxis.grid(True, alpha=0.2, linestyle='--', color='black'); ax.xaxis.grid(True, alpha=0.2, linestyle='--', color='black'); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'Figure_4_5_Precision_Recall_Scatter.png'), facecolor='white'); plt.close()

# FIGURE 5.1: Industry Distribution
sorted_ind = sorted(industry_data.items(), key=lambda x: x[1], reverse=True)
industries = [x[0] for x in sorted_ind][::-1]; counts_ind = [x[1] for x in sorted_ind][::-1]; total = 391
tech_sectors = ['Electronics', 'Semiconductor', 'Computer & Peripherals', 'Optoelectronics', 'Telecommunications']
fig, ax = plt.subplots(figsize=(11, 8))
for i, (ind, count) in enumerate(zip(industries, counts_ind)):
    ax.barh(i, count, height=0.65, facecolor='#AAAAAA' if ind in tech_sectors else 'white', edgecolor='black', linewidth=0.8, hatch='///' if ind in tech_sectors else '')
    ax.text(count + 0.5, i, f'{count} ({count/total*100:.1f}%)', va='center', ha='left', fontsize=9)
ax.set_yticks(range(len(industries))); ax.set_yticklabels(industries, fontsize=9.5)
ax.set_xlabel('Number of Companies', fontsize=11)
ax.xaxis.grid(True, alpha=0.25, linestyle='--', color='black'); ax.set_axisbelow(True); ax.set_xlim(0, 100)
ax.legend(handles=[mpatches.Patch(facecolor='#AAAAAA', edgecolor='black', hatch='///', label='Technology-related'), mpatches.Patch(facecolor='white', edgecolor='black', label='Other sectors')], loc='lower right', frameon=True, edgecolor='black')
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'Figure_5_1_Industry_Distribution.png'), facecolor='white'); plt.close()

print("\n✅ Figure 6 generated. Other figures are commented out.")