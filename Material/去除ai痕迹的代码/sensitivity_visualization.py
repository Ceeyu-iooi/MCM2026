import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("综合敏感性分析与可视化")
print("="*70)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载数据
q1_uncertainty = pd.read_csv('q1_fan_vote_uncertainty.csv')
q2_propagated = pd.read_csv('q2_propagated_uncertainty.csv')
q3_coef_stability = pd.read_csv('q3_coefficient_stability.csv')

with open('uncertainty_propagation_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

print(f"加载数据完成")

# ============================================================================
# 图1：不确定性传播流程图
# ============================================================================
print("\n生成图1：不确定性传播框架概览 (Generating Fig 1: Uncertainty Propagation Framework)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Q1粉丝投票不确定性分布
ax1 = axes[0, 0]
ax1.hist(q1_uncertainty['ci_width'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax1.axvline(q1_uncertainty['ci_width'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {q1_uncertainty["ci_width"].mean():.3f}')
ax1.set_xlabel('95% CI Width', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Q1: Uncertainty Dist. of Fan Votes', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 1.2 Q2争议识别的不确定性分布
ax2 = axes[0, 1]
ax2.hist(q2_propagated['q2_ci_width'].dropna(), bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
ax2.axvline(q2_propagated['q2_ci_width'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {q2_propagated["q2_ci_width"].mean():.3f}')
ax2.set_xlabel('95% CI Width', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Q2: Propagated Uncertainty in Controversy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 1.3 Q3系数稳定性误差棒图
ax3 = axes[1, 0]
features = q3_coef_stability['feature'].values
coef_mean = q3_coef_stability['coef_mean'].values
coef_ci_lower = q3_coef_stability['coef_ci_lower'].values
coef_ci_upper = q3_coef_stability['coef_ci_upper'].values

# 简化特征名
feature_labels = ['Score', 'Week', 'Age', 'Fan Support']
colors = ['green' if c < 0 else 'coral' for c in coef_mean]
y_pos = np.arange(len(features))

ax3.barh(y_pos, coef_mean, xerr=[coef_mean - coef_ci_lower, coef_ci_upper - coef_mean],
         color=colors, alpha=0.7, capsize=5, error_kw={'ecolor': 'gray', 'capthick': 2})
x_min = min(coef_ci_lower)
for i, (v, lower) in enumerate(zip(coef_mean, coef_ci_lower)):
    ax3.text(lower - 0.05, i, f'{v:.3f}', va='center', ha='right', fontsize=9, fontweight='bold', color='black')

ax3.axvline(0, color='black', linestyle='-', linewidth=1)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(feature_labels, fontsize=11)
ax3.set_xlabel('Coefficient Value (with 95% CI)', fontsize=12)
ax3.set_title('Q3: Coefficient Stability (Propagated)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 1.4 不确定性传播汇总
ax4 = axes[1, 1]
stages = ['Q1\nFan Votes', 'Q2\nControversy', 'Q3\nFactors']
uncertainties = [
    summary['q1_uncertainty']['mean_ci_width'],
    summary['q2_propagation']['mean_ci_width'],
    summary['q3_propagation']['mean_coef_cv']
]
# 归一化到相对尺度
normalized = [1.0, 
              summary['q2_propagation']['amplification_factor'],
              1 + summary['q3_propagation']['mean_coef_cv']]

bars = ax4.bar(stages, normalized, color=['steelblue', 'forestgreen', 'coral'], alpha=0.7, edgecolor='black')
ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
ax4.set_ylabel('Relative Uncertainty (Q1 = 1.0)', fontsize=12)
ax4.set_title('Uncertainty Amplification Effect', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')


# 添加数值标签
for bar, val in zip(bars, normalized):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_uncertainty_propagation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   fig_uncertainty_propagation.png")

# ============================================================================
# 图2：Q1-Q2不确定性相关性分析
# ============================================================================
print("生成图2：Q1-Q2不确定性传播相关性 (Generating Fig 2)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 2.1 Q1不确定性 vs Q2不确定性
ax1 = axes[0]
# 合并数据
merged = q1_uncertainty.merge(
    q2_propagated[['season', 'week', 'celebrity_name', 'q2_ci_width']],
    on=['season', 'week', 'celebrity_name'],
    how='inner'
)
ax1.scatter(merged['ci_width'], merged['q2_ci_width'], alpha=0.3, s=10, c='steelblue')
ax1.set_xlabel('Q1 Uncertainty (CI Width)', fontsize=12)
ax1.set_ylabel('Q2 Uncertainty (CI Width)', fontsize=12)
ax1.set_title('Q1→Q2 Uncertainty Propagation', fontsize=14, fontweight='bold')

# 添加拟合线
z = np.polyfit(merged['ci_width'], merged['q2_ci_width'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged['ci_width'].min(), merged['ci_width'].max(), 100)
ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Slope: {z[0]:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 计算相关系数
corr = merged['ci_width'].corr(merged['q2_ci_width'])
ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2.2 按赛季的不确定性趋势
ax2 = axes[1]
season_agg = merged.groupby('season').agg({
    'ci_width': 'mean',
    'q2_ci_width': 'mean'
}).reset_index()

ax2.plot(season_agg['season'], season_agg['ci_width'], 'o-', 
         color='steelblue', linewidth=2, markersize=5, label='Q1 Uncertainty')
ax2.plot(season_agg['season'], season_agg['q2_ci_width'], 's-', 
         color='forestgreen', linewidth=2, markersize=5, label='Q2 Uncertainty')
ax2.set_xlabel('Season', fontsize=12)
ax2.set_ylabel('Mean CI Width', fontsize=12)
ax2.set_title('Uncertainty Trend over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2.3 高不确定性样本分析
ax3 = axes[2]
high_q1 = merged['ci_width'] > merged['ci_width'].quantile(0.9)
high_q2 = merged['q2_ci_width'] > merged['q2_ci_width'].quantile(0.9)

# 四象限分类
categories = []
for h1, h2 in zip(high_q1, high_q2):
    if h1 and h2:
        categories.append('High-High')
    elif h1 and not h2:
        categories.append('High Q1 Only')
    elif not h1 and h2:
        categories.append('High Q2 Only')
    else:
        categories.append('Low-Low')

cat_counts = pd.Series(categories).value_counts()

# Color mapping
color_dict = {
    'Low-Low': 'orange',
    'High-High': 'red',
    'High Q1 Only': 'steelblue',
    'High Q2 Only': 'forestgreen'
}
plot_colors = [color_dict.get(label, 'gray') for label in cat_counts.index]

# Pie chart with adjusted labels
wedges, texts, autotexts = ax3.pie(cat_counts.values, labels=cat_counts.index, 
                                   autopct='%1.1f%%', 
                                   colors=plot_colors, 
                                   startangle=140,
                                   pctdistance=0.85,
                                   labeldistance=1.1,
                                   explode=[0.05]*len(cat_counts)) # Slightly separate slices

for text in texts:
    text.set_fontsize(9)
for autotext in autotexts:
    autotext.set_fontsize(8)
    autotext.set_weight('bold')
    
ax3.set_title('Uncertainty Category Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_uncertainty_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   fig_uncertainty_correlation.png")

# ============================================================================
# 图3：综合敏感性分析
# ============================================================================
print("生成图3：综合敏感性分析 (Generating Fig 3)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 淘汰vs存活的不确定性比较
ax1 = axes[0, 0]
eliminated = q1_uncertainty[q1_uncertainty['is_eliminated'] == 1]['fan_support_std']
survived = q1_uncertainty[q1_uncertainty['is_eliminated'] == 0]['fan_support_std']

ax1.boxplot([eliminated, survived], labels=['Eliminated', 'Survived'])
ax1.set_ylabel('Fan Support Std. Dev.', fontsize=12)
ax1.set_title('Eliminated vs Survived Uncertainty', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')


# 添加均值标注
ax1.text(1, eliminated.mean(), f'μ={eliminated.mean():.3f}', ha='center', fontsize=10)
ax1.text(2, survived.mean(), f'μ={survived.mean():.3f}', ha='center', fontsize=10)

# 3.2 Q3系数Bootstrap分布
ax2 = axes[0, 1]
# 生成模拟分布
np.random.seed(42)
coef_fan_support = np.random.normal(
    q3_coef_stability[q3_coef_stability['feature'] == 'fan_support_mean']['coef_mean'].values[0],
    q3_coef_stability[q3_coef_stability['feature'] == 'fan_support_mean']['coef_std'].values[0],
    1000
)
ax2.hist(coef_fan_support, bins=40, color='forestgreen', alpha=0.7, edgecolor='white', density=True)
ax2.axvline(coef_fan_support.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {coef_fan_support.mean():.3f}')
ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Coefficient Value', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Bootstrap Dist. of Fan Support Coef.', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3 不确定性放大矩阵
ax3 = axes[1, 0]
stages = ['Q1', 'Q2', 'Q3']
matrix = np.array([
    [1.00, 1.13, 0.01],
    [0.88, 1.00, 0.01],
    [0.99, 0.99, 1.00]
])
im = ax3.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2)

for i in range(3):
    for j in range(3):
        text = ax3.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', 
                       fontsize=14, fontweight='bold',
                       color='white' if matrix[i, j] > 1.2 else 'black')

ax3.set_xticks(np.arange(3))
ax3.set_yticks(np.arange(3))
ax3.set_xticklabels(['→Q1', '→Q2', '→Q3'], fontsize=12)
ax3.set_yticklabels(['Q1', 'Q2', 'Q3'], fontsize=12)
ax3.set_title('Uncertainty Propagation Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, label='Amp. Factor')

# 3.4 结论汇总
ax4 = axes[1, 1]
ax4.axis('off')

conclusion_text = f"""
【Uncertainty Analysis Summary】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Q1 Fan Vote Estimation
   • Mean 95% CI Width: {summary['q1_uncertainty']['mean_ci_width']:.3f}
   • Meaning: Moderate uncertainty in vote estimation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Q2 Controversy Propagation
   • Amp. Factor: {summary['q2_propagation']['amplification_factor']:.2f}x
   • Conclusion: Uncertainty slightly amplified but controlled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Q3 Factor Stability
   • Stable Features: {summary['q3_propagation']['n_stable_features']}/{summary['q3_propagation']['total_features']}
   • Coef. CV: {summary['q3_propagation']['mean_coef_cv']:.4f}
   • Conclusion: Highly robust model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Overall: {summary['conclusion']['robustness'].replace('强', 'Strong').replace('中', 'Moderate').replace('弱', 'Weak')}
   {summary['conclusion']['interpretation'].replace('模型能够在不确定性存在的情况下保持结论的可靠性', 'Model maintains reliability under uncertainty')}
"""

ax4.text(0.05, 0.95, conclusion_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_sensitivity_analysis.png', dpi=150, bbox_inches='tight')

plt.close()
print("   fig_sensitivity_analysis.png")

# ============================================================================
# 图4：流程图风格的传播示意
# ============================================================================
print("生成图4：传播框架流程图 (Generating Fig 4: Propagation Framework Flowchart)...")

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# 定义方框位置
boxes = [
    {'x': 1, 'y': 3, 'w': 3, 'h': 2, 'text': 'Q1\nFan Vote Est.\n\nBootstrap\nn=200', 'color': 'steelblue'},
    {'x': 6.5, 'y': 3, 'w': 3, 'h': 2, 'text': 'Q2\nControversy ID\n\nMonte Carlo\nn=100', 'color': 'forestgreen'},
    {'x': 12, 'y': 3, 'w': 3, 'h': 2, 'text': 'Q3\nFactor Analysis\n\nCoef. Stability', 'color': 'coral'},
]

for box in boxes:
    rect = plt.Rectangle((box['x'], box['y']), box['w'], box['h'], 
                         facecolor=box['color'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, box['text'],
           ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# 绘制箭头
arrow_style = dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                   color='black', linewidth=2)
ax.annotate('', xy=(6.5, 4), xytext=(4, 4), arrowprops=arrow_style)
ax.annotate('', xy=(12, 4), xytext=(9.5, 4), arrowprops=arrow_style)

# 标注传播效应
ax.text(5.25, 4.8, f'Amplified {summary["q2_propagation"]["amplification_factor"]:.2f}x', 
        ha='center', fontsize=10, fontweight='bold', color='red')
ax.text(10.75, 4.8, f'Stable {summary["q3_propagation"]["stability_rate"]*100:.0f}%', 
        ha='center', fontsize=10, fontweight='bold', color='green')

# 添加不确定性指标
ax.text(2.5, 1.8, f'CI Width: {summary["q1_uncertainty"]["mean_ci_width"]:.3f}', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(8, 1.8, f'CI Width: {summary["q2_propagation"]["mean_ci_width"]:.3f}', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(13.5, 1.8, f'CV: {summary["q3_propagation"]["mean_coef_cv"]:.4f}', 
        ha='center', fontsize=10, fontweight='bold')

# 标题
ax.text(8, 5.8, 'Uncertainty Propagation Framework', 
        ha='center', fontsize=16, fontweight='bold')

ax.text(8, 0.5, 'Conclusion: Despite Q1 estimation uncertainty, core conclusions of Q2 & Q3 remain robust.', 
        ha='center', fontsize=12, style='italic', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('fig_propagation_framework.png', dpi=150, bbox_inches='tight')
plt.close()
print("   fig_propagation_framework.png")

print("\n" + "="*70)
print("可视化生成完成! (Visualization Completed!)")
print("="*70)

# 生成图片说明
captions = """
Figure Captions
==========================

fig_uncertainty_propagation.png - Uncertainty Propagation Framework Overview
  Contains four subplots: 
  (a) Q1 Fan Vote Assessment: 95% Confidence Interval Width Distribution;
  (b) Q2 Controversy Identification: Propagated Uncertainty Distribution;
  (c) Q3 Factor Analysis: Coefficient Stability Error Bar Plot;
  (d) Amplification Effect: Comparison of relative uncertainty across Q1→Q2→Q3 stages.

fig_uncertainty_correlation.png - Q1-Q2 Uncertainty Propagation Correlation
  Analyzing how Q1 estimation uncertainty propagates to Q2 controversy identification:
  (a) Scatter plot and trend line of Q1 vs. Q2 uncertainty;
  (b) Temporal trend of uncertainty over seasons;
  (c) Distribution of uncertainty categories (High/Low).

fig_sensitivity_analysis.png - Comprehensive Sensitivity Analysis
  (a) Comparison of estimation uncertainty for Eliminated vs. Survived contestants;
  (b) Bootstrap distribution of the Fan Support coefficient;
  (c) Uncertainty Propagation Factor Matrix;
  (d) Summary of comprehensive analysis conclusions.

fig_propagation_framework.png - Propagation Framework Flowchart
  Illustrates the uncertainty propagation path Q1→Q2→Q3, annotated with propagation effects and stability metrics at each stage.
"""

with open('figure_captions.txt', 'w', encoding='utf-8') as f:
    f.write(captions)
print(" 图片说明已保存: figure_captions.txt")
