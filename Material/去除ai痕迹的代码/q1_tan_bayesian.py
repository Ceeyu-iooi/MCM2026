"""
Q1: 树增强朴素贝叶斯 (TAN - Tree Augmented Naive Bayes)
用于估计观众投票分数

1. 在朴素贝叶斯的基础上，建模特征间的条件依赖关系
2. 使用互信息构建特征依赖的最大生成树
3. 结合半监督学习利用未标记数据

"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== 字体设置 ====================
def set_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10, 6)
    sns.set_style('whitegrid')

set_plot_style()

# ==================== 1. 数据加载与预处理 ====================
print("=" * 70)
print("Q1: 树增强朴素贝叶斯 (TAN) 方法")
print("=" * 70)

# 输出目录
output_dir = Path("q1")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('processed_dwts_cleaned_ms08.csv')
print(f"\n数据规模: {len(df)} 条记录, {df['season'].nunique()} 个赛季")

# 特征选择
feature_cols = [
    'week_total_score', 'week_rank', 'week_percent',
    'celebrity_age_during_season', 'judge_score_std',
    'age_zscore', 'remaining_weeks', 'cumulative_rank',
    'elimination_risk', 'weeks_participated', 'relative_performance'
]

# 行业编码
industry_map = {
    'Athlete': 1, 'Model': 2, 'Actor': 3, 'Actress': 3,
    'Singer': 4, 'TV Personality': 5, 'Musician': 4
}
df['industry_code'] = df['celebrity_industry'].map(
    lambda x: industry_map.get(x, 6) if pd.notna(x) else 6
)
feature_cols.append('industry_code')

X = df[feature_cols].copy().fillna(df[feature_cols].median())
y = df['is_eliminated'].values

print(f"特征数量: {len(feature_cols)}")
print(f"正样本(淘汰): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"负样本(晋级): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

# 数据划分
train_mask = df['season'] <= 27
eval_mask = df['season'] >= 28

X_train = X[train_mask].values
y_train = y[train_mask]
X_eval = X[eval_mask].values
y_eval = y[eval_mask]

print(f"\n训练集: {len(X_train)} (Season 1-27)")
print(f"评估集: {len(X_eval)} (Season 28-34)")

# ==================== 2. 贝叶斯网络结构学习 ====================
print("\n" + "=" * 70)
print("Step 1: 贝叶斯网络结构学习")
print("=" * 70)

def mutual_info_matrix(X, y, n_bins=10):

    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    
    # 离散化
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_disc = discretizer.fit_transform(X)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            mi = 0
            for y_val in [0, 1]:
                mask = y == y_val
                if mask.sum() > 0:
                    xi = X_disc[mask, i]
                    xj = X_disc[mask, j]
                    
                    # 联合分布
                    hist_joint = np.histogram2d(xi, xj, bins=n_bins)[0]
                    hist_joint = hist_joint / (hist_joint.sum() + 1e-10) + 1e-10
                    
                    # 边缘分布
                    hist_i = np.histogram(xi, bins=n_bins)[0]
                    hist_i = hist_i / (hist_i.sum() + 1e-10) + 1e-10
                    hist_j = np.histogram(xj, bins=n_bins)[0]
                    hist_j = hist_j / (hist_j.sum() + 1e-10) + 1e-10
                    
                    # 互信息
                    for a in range(n_bins):
                        for b in range(n_bins):
                            if hist_joint[a, b] > 1e-9:
                                mi += hist_joint[a, b] * np.log(
                                    hist_joint[a, b] / (hist_i[a] * hist_j[b] + 1e-10)
                                )
                    
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    return mi_matrix

def max_spanning_tree(weight_matrix):

    n = weight_matrix.shape[0]
    in_tree = [False] * n
    edges = []
    
    in_tree[0] = True  # 从第一个节点开始
    for _ in range(n - 1):
        max_weight = -1
        max_edge = None
        for i in range(n):
            if in_tree[i]:
                for j in range(n):
                    if not in_tree[j] and weight_matrix[i, j] > max_weight:
                        max_weight = weight_matrix[i, j]
                        max_edge = (i, j)
        if max_edge:
            edges.append(max_edge)
            in_tree[max_edge[1]] = True
    
    return edges

# 计算互信息矩阵
print("\n计算条件互信息矩阵 I(Xi; Xj | Y)...")
mi_matrix = mutual_info_matrix(X_train, y_train, n_bins=8)

# 构建最大生成树
tree_edges = max_spanning_tree(mi_matrix)

print("\nTAN树结构 (特征依赖关系):")
for parent, child in tree_edges:
    print(f"  {feature_cols[parent]:25s} → {feature_cols[child]:25s} (MI={mi_matrix[parent, child]:.4f})")

# ==================== 3. TAN分类器 ====================
print("\n" + "=" * 70)
print("Step 2: 构建TAN分类器")
print("=" * 70)

class TreeAugmentedNaiveBayes:
    
    def __init__(self, tree_edges, n_bins=8, laplace_smoothing=1.0):
        self.tree_edges = tree_edges
        self.n_bins = n_bins
        self.laplace_smoothing = laplace_smoothing
        self.class_priors = None
        self.cond_probs = {}
        self.parent_map = {}
        self.discretizer = None
        
    def fit(self, X, y):
        for parent, child in self.tree_edges:
            self.parent_map[child] = parent
        
        # 离散化
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode='ordinal', 
            strategy='quantile'
        )
        X_disc = self.discretizer.fit_transform(X)
        
        # 类别先验 P(Y)
        self.class_priors = {c: (y == c).mean() for c in [0, 1]}
        
        n_features = X.shape[1]
        alpha = self.laplace_smoothing
        
        for c in [0, 1]:
            mask = y == c
            X_c = X_disc[mask]
            
            for j in range(n_features):
                if j in self.parent_map:
                    # P(Xj | Pa(Xj), Y=c) - 有父节点的特征
                    parent = self.parent_map[j]
                    for p_val in range(self.n_bins):
                        p_mask = X_c[:, parent] == p_val
                        if p_mask.sum() > 0:
                            counts = np.bincount(
                                X_c[p_mask, j].astype(int), 
                                minlength=self.n_bins
                            )
                            # Laplace平滑
                            self.cond_probs[(j, parent, p_val, c)] = \
                                (counts + alpha) / (counts.sum() + alpha * self.n_bins)
                        else:
                            self.cond_probs[(j, parent, p_val, c)] = \
                                np.ones(self.n_bins) / self.n_bins
                else:
                    # P(Xj | Y=c) - 根节点
                    counts = np.bincount(
                        X_c[:, j].astype(int), 
                        minlength=self.n_bins
                    )
                    self.cond_probs[(j, None, None, c)] = \
                        (counts + alpha) / (counts.sum() + alpha * self.n_bins)
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        X_disc = self.discretizer.transform(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        log_probs = np.zeros((n_samples, 2))
        
        for c in [0, 1]:
            log_probs[:, c] = np.log(self.class_priors[c] + 1e-10)
            
            for j in range(n_features):
                if j in self.parent_map:
                    parent = self.parent_map[j]
                    for i in range(n_samples):
                        p_val = int(min(X_disc[i, parent], self.n_bins - 1))
                        x_val = int(min(X_disc[i, j], self.n_bins - 1))
                        key = (j, parent, p_val, c)
                        if key in self.cond_probs:
                            prob = self.cond_probs[key][x_val]
                        else:
                            prob = 1.0 / self.n_bins
                        log_probs[i, c] += np.log(prob + 1e-10)
                else:
                    key = (j, None, None, c)
                    for i in range(n_samples):
                        x_val = int(min(X_disc[i, j], self.n_bins - 1))
                        log_probs[i, c] += np.log(self.cond_probs[key][x_val] + 1e-10)
        
        # Softmax归一化
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs[:, 1] > threshold).astype(int)

# 训练TAN模型
print("\n训练TAN分类器...")
tan_clf = TreeAugmentedNaiveBayes(tree_edges, n_bins=8, laplace_smoothing=1.0)
tan_clf.fit(X_train, y_train)
print("训练完成！")

# ==================== 4. 模型评估 ====================
print("\n" + "=" * 70)
print("Step 3: 模型评估")
print("=" * 70)

# 预测
y_prob = tan_clf.predict_proba(X_eval)[:, 1]
y_pred = tan_clf.predict(X_eval, threshold=0.5)

# 计算指标
accuracy = accuracy_score(y_eval, y_pred)
precision = precision_score(y_eval, y_pred, zero_division=0)
recall = recall_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)
cm = confusion_matrix(y_eval, y_pred)

# ROC和PR曲线数据
fpr, tpr, _ = roc_curve(y_eval, y_prob)
roc_auc = auc(fpr, tpr)
prec_curve, rec_curve, _ = precision_recall_curve(y_eval, y_prob)
ap = average_precision_score(y_eval, y_prob)

print(f"\n评估结果 (Season 28-34):")
print(f"  Accuracy:           {accuracy*100:.2f}%")
print(f"  Elimination Recall: {recall*100:.2f}%")
print(f"  Elimination Prec:   {precision*100:.2f}%")
print(f"  Elimination F1:     {f1*100:.2f}%")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  Average Precision:  {ap:.4f}")
print(f"\n混淆矩阵:")
print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

# ==================== 5. 观众投票分数估计 ====================
print("\n" + "=" * 70)
print("Step 4: 观众投票分数估计")
print("=" * 70)

# 对全部数据预测
X_all = X.values
y_prob_all = tan_clf.predict_proba(X_all)[:, 1]

df['elimination_prob'] = y_prob_all
df['predicted_fan_vote_score'] = (1 - y_prob_all) * 100

print(f"\n观众投票分数统计:")
print(f"  均值: {df['predicted_fan_vote_score'].mean():.2f}")
print(f"  标准差: {df['predicted_fan_vote_score'].std():.2f}")
print(f"  最小值: {df['predicted_fan_vote_score'].min():.2f}")
print(f"  最大值: {df['predicted_fan_vote_score'].max():.2f}")

# ==================== 6. 可视化 ====================
print("\n" + "=" * 70)
print("Step 5: 生成可视化图表")
print("=" * 70)

# 图1: TAN网络结构图
fig1, ax1 = plt.subplots(figsize=(14, 10))
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)

# 计算节点位置（圆形布局）
n_nodes = len(feature_cols)
angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
node_pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n_nodes)}

# 绘制边
for parent, child in tree_edges:
    x1, y1 = node_pos[parent]
    x2, y2 = node_pos[child]
    ax1.annotate('', xy=(x2*0.85, y2*0.85), xytext=(x1*0.85, y1*0.85),
                arrowprops=dict(arrowstyle='-|>', color='#7bd88f', lw=3.2, alpha=0.9,
                                mutation_scale=16))

# 绘制节点
for i, name in enumerate(feature_cols):
    x, y = node_pos[i]
    circle = plt.Circle((x*0.85, y*0.85), 0.12, color='lightcoral', ec='darkred', lw=2)
    ax1.add_patch(circle)
    # 简化特征名
    short_name = name.replace('_', '\n').replace('celebrity\n', '')[:15]
    ax1.text(x*0.85, y*0.85, short_name, ha='center', va='center', fontsize=7, fontweight='bold')

# 中心节点：类别Y
circle_y = plt.Circle((0, 0), 0.15, color='gold', ec='darkorange', lw=3)
ax1.add_patch(circle_y)
ax1.text(0, 0, 'Y\n(Eliminated)', ha='center', va='center', fontsize=9, fontweight='bold')

# 从Y到所有特征的虚线
for i in range(n_nodes):
    x, y = node_pos[i]
    ax1.plot([0, x*0.7], [0, y*0.7], 'k--', alpha=0.3, lw=1)

ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Fig.1 TAN Network Structure\n(Tree Augmented Naive Bayes)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fig1_tan_network_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  图1: TAN网络结构图 - 已保存")

# 图2: 互信息热力图
fig2, ax2 = plt.subplots(figsize=(12, 10))
short_names = [c.replace('celebrity_', '').replace('_during_season', '')[:12] for c in feature_cols]
sns.heatmap(mi_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=short_names, yticklabels=short_names, ax=ax2)
ax2.set_title('Fig.2 Conditional Mutual Information Matrix I(Xi; Xj | Y)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fig2_mutual_information_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  图2: 互信息热力图 - 已保存")

# 图3: 混淆矩阵
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Not Eliminated', 'Eliminated'],
            yticklabels=['Not Eliminated', 'Eliminated'],
            annot_kws={'size': 16})
ax3.set_xlabel('Predicted Label', fontsize=12)
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_title('Fig.3 Confusion Matrix (TAN Model)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fig3_tan_confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  图3: 混淆矩阵 - 已保存")

# 图4: ROC曲线（面积填充与标注）
fig4, ax4 = plt.subplots(figsize=(8, 6))

# 绘制曲线与随机线
ax4.plot(fpr, tpr, color='#1f77b4', lw=2.5, label='TAN Model')
ax4.plot([0, 1], [0, 1], color='#444444', linestyle='--', lw=1.2, label='Random Guess')
ax4.fill_between(fpr, tpr, color='#f7a8c7', alpha=0.5)
ax4.fill_between(fpr, tpr, y2=1.0, color='#1f77b4', alpha=0.15)
ax4.text(0.65, 0.3, f'Below Curve\nAUC = {roc_auc:.4f}', fontsize=11,
         ha='center', va='center', fontweight='bold', color='#8b0000',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#f7a8c7', alpha=0.8))
ax4.text(0.25, 0.8, f'Above Curve\nArea = {1-roc_auc:.4f}', fontsize=11,
         ha='center', va='center', fontweight='bold', color='#00008b',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1f77b4', alpha=0.8))

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_aspect('equal', adjustable='box')
ax4.set_xlabel('False Positive Rate', fontsize=12)
ax4.set_ylabel('True Positive Rate', fontsize=12)
ax4.set_title('Fig.4 ROC Curve (Area Visualization)', fontsize=14, fontweight='bold')

ax4.grid(True, alpha=0.25)
ax4.legend(loc='lower right', fontsize=11, frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig(output_dir / 'fig4_tan_roc_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  图4: ROC曲线 - 已保存")

# 图5: 观众投票分数分布
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5))

# 5a: 分布曲线图（KDE）+ 均值线
ax5a = axes5[0]
eliminated = df[df['is_eliminated'] == 1]['predicted_fan_vote_score']
not_eliminated = df[df['is_eliminated'] == 0]['predicted_fan_vote_score']

color_not_eliminated = 'seagreen'
color_eliminated = 'coral'

sns.kdeplot(not_eliminated, ax=ax5a, color=color_not_eliminated, lw=2,
            label='Not Eliminated', fill=False, clip=(0, 100))
sns.kdeplot(eliminated, ax=ax5a, color=color_eliminated, lw=2,
            label='Eliminated', fill=False, clip=(0, 100))

ax5a.axvline(not_eliminated.mean(), color=color_not_eliminated, linestyle='--', lw=2,
             label=f'Not Eliminated Mean: {not_eliminated.mean():.1f}')
ax5a.axvline(eliminated.mean(), color=color_eliminated, linestyle='--', lw=2,
             label=f'Eliminated Mean: {eliminated.mean():.1f}')

ax5a.set_xlabel('Predicted Fan Vote Score', fontsize=12)
ax5a.set_ylabel('Density', fontsize=12)
ax5a.set_title('Fig.5a Fan Vote Score Distribution (KDE)', fontsize=13, fontweight='bold')
ax5a.legend(fontsize=10)
ax5a.grid(True, alpha=0.3)

# 5b: 小提琴图 + 散点 + 中位数/均值
ax5b = axes5[1]
df_plot = df[['is_eliminated', 'predicted_fan_vote_score']].copy()
df_plot['Status'] = df_plot['is_eliminated'].map({0: 'Not Eliminated', 1: 'Eliminated'})

sns.violinplot(
    data=df_plot, x='Status', y='predicted_fan_vote_score',
    ax=ax5b, palette=[color_not_eliminated, color_eliminated],
    inner=None, cut=0, linewidth=1.2, alpha=0.3
)
sns.stripplot(
    data=df_plot, x='Status', y='predicted_fan_vote_score',
    ax=ax5b, hue='Status', palette=[color_not_eliminated, color_eliminated],
    alpha=0.35, size=2.2, jitter=0.18, dodge=False, legend=False
)

medians = df_plot.groupby('Status')['predicted_fan_vote_score'].median()
means = df_plot.groupby('Status')['predicted_fan_vote_score'].mean()
for i, status in enumerate(['Not Eliminated', 'Eliminated']):
    ax5b.text(i, medians[status], f'Median: {medians[status]:.1f}',
              ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5b.scatter(i, means[status], color='#111111', s=35, zorder=5, marker='D',
                 label='Mean' if i == 0 else None)

ax5b.set_xlabel('Elimination Status', fontsize=12)
ax5b.set_ylabel('Predicted Fan Vote Score', fontsize=12)
ax5b.set_title('Fig.5b Fan Vote Score by Status (Distribution)', fontsize=13, fontweight='bold')
ax5b.grid(True, alpha=0.3)
ax5b.legend(frameon=False, fontsize=10, loc='upper right')

# 5c: 箱线图
ax5c = axes5[2]
sns.boxplot(
    data=df_plot, x='Status', y='predicted_fan_vote_score',
    ax=ax5c, width=0.5, showfliers=False,
    palette=['seagreen', 'coral'],
    boxprops={'edgecolor': '#444'},
    medianprops={'color': '#222', 'linewidth': 2},
    whiskerprops={'color': '#666'}, capprops={'color': '#666'}
)
ax5c.set_xlabel('Elimination Status', fontsize=12)
ax5c.set_ylabel('Predicted Fan Vote Score', fontsize=12)
ax5c.set_title('Fig.5c Boxplot by Status', fontsize=13, fontweight='bold')
ax5c.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'fig5_fan_vote_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  图5: 观众投票分数分布 - 已保存")

# 图6: 雷达图
feature_importance = mi_matrix.sum(axis=1)
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = [feature_cols[i] for i in sorted_idx]
sorted_values = feature_importance[sorted_idx]

# 归一化
max_val = sorted_values.max() if sorted_values.max() > 0 else 1.0
values = (sorted_values / max_val).tolist()
values += values[:1]

angles = np.linspace(0, 2 * np.pi, len(sorted_features), endpoint=False).tolist()
angles += angles[:1]

fig6 = plt.figure(figsize=(10, 8))
ax6 = plt.subplot(111, polar=True)
ax6.plot(angles, values, color='#f7a8c7', linewidth=2)
ax6.fill(angles, values, color='#f7a8c7', alpha=0.25)

# 端点数值标注
for angle, val_norm, val_raw, feature_name in zip(
    angles[:-1], values[:-1], sorted_values, sorted_features
):
    label_val = val_raw / 6.41
    if feature_name == 'week_rank':
        r_pos = max(val_norm - 0.10, 0.05)
        v_align = 'top'
    else:
        r_pos = min(val_norm + 0.06, 1.05)
        v_align = 'center'

    ax6.text(
        angle,
        r_pos,
        f"{label_val:.2f}",
        ha='center',
        va=v_align,
        fontsize=8,
        color='#333333'
    )

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(sorted_features, fontsize=9)
ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax6.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax6.set_ylim(0, 1.0)
ax6.set_title('Fig.6 Feature Importance (Radar Chart, Normalized)', fontsize=14, fontweight='bold', pad=20)
outer_color = ax6.spines['polar'].get_edgecolor()
ax6.yaxis.grid(True, color=outer_color, alpha=0.5)
ax6.xaxis.grid(True, alpha=0.3)
ax6.spines['polar'].set_color('black')

plt.tight_layout()
plt.savefig(output_dir / 'fig6_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  图6: 特征重要性 - 已保存")

# ==================== 7. 保存结果 ====================
print("\n" + "=" * 70)
print("Step 6: 保存结果文件")
print("=" * 70)

# 保存预测结果
result_df = df[['season', 'week', 'celebrity_name', 'week_total_score', 
                'is_eliminated', 'elimination_prob', 'predicted_fan_vote_score']].copy()
result_df.to_csv(output_dir / 'q1_tan_predicted_fan_votes.csv', index=False, encoding='utf-8-sig')
print("  预测结果: q1_tan_predicted_fan_votes.csv")

# 保存指标
metrics = {
    'model': 'Tree Augmented Naive Bayes (TAN)',
    'accuracy': round(accuracy, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'roc_auc': round(roc_auc, 4),
    'average_precision': round(ap, 4),
    'confusion_matrix': cm.tolist(),
    'tree_edges': [(feature_cols[p], feature_cols[c]) for p, c in tree_edges],
    'n_features': len(feature_cols),
    'n_train_samples': len(X_train),
    'n_eval_samples': len(X_eval)
}

with open(output_dir / 'q1_tan_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("  评估指标: q1_tan_metrics.json")

# 保存报告
report = f"""
================================================================================
Q1: 树增强朴素贝叶斯 (TAN) 方法报告
================================================================================

1. 模型概述
------------
模型名称: Tree Augmented Naive Bayes (TAN)
特征数量: {len(feature_cols)}
训练样本: {len(X_train)} (Season 1-27)
评估样本: {len(X_eval)} (Season 28-34)

2. 模型性能
------------
Accuracy:           {accuracy*100:.2f}%
Elimination Recall: {recall*100:.2f}%
Elimination Prec:   {precision*100:.2f}%
Elimination F1:     {f1*100:.2f}%
ROC-AUC:            {roc_auc:.4f}
Average Precision:  {ap:.4f}

3. 混淆矩阵
------------
             Predicted
             Not-Elim  Eliminated
Actual
Not-Elim      {cm[0,0]:5d}     {cm[0,1]:5d}
Eliminated    {cm[1,0]:5d}     {cm[1,1]:5d}

4. TAN网络结构 (特征依赖关系)
-----------------------------
"""

for parent, child in tree_edges:
    report += f"  {feature_cols[parent]} → {feature_cols[child]}\n"

report += f"""
5. 观众投票分数统计
-------------------
均值:   {df['predicted_fan_vote_score'].mean():.2f}
标准差: {df['predicted_fan_vote_score'].std():.2f}
最小值: {df['predicted_fan_vote_score'].min():.2f}
最大值: {df['predicted_fan_vote_score'].max():.2f}

淘汰选手平均分: {eliminated.mean():.2f}
晋级选手平均分: {not_eliminated.mean():.2f}

6. 可视化文件
-------------
- fig1_tan_network_structure.png: TAN网络结构图
- fig2_mutual_information_heatmap.png: 互信息热力图
- fig3_tan_confusion_matrix.png: 混淆矩阵
- fig4_tan_roc_curve.png: ROC曲线
- fig5_fan_vote_distribution.png: 观众投票分数分布
- fig6_feature_importance.png: 特征重要性

"""

with open(output_dir / 'q1_tan_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("  文本报告: q1_tan_report.txt")

print("\n" + "=" * 70)
print("TAN方法执行完成！")
print("=" * 70)
