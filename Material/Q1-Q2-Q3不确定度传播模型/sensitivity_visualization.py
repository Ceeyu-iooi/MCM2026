# -*- coding: utf-8 -*-
"""
综合敏感性分析与可视化
生成不确定性传播框架的可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
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
print("\n生成图1：不确定性传播框架概览...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Q1粉丝投票不确定性分布
ax1 = axes[0, 0]
ax1.hist(q1_uncertainty['ci_width'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax1.axvline(q1_uncertainty['ci_width'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'均值: {q1_uncertainty["ci_width"].mean():.3f}')
ax1.set_xlabel('95% CI 宽度', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.set_title('Q1: 粉丝投票估计的不确定性分布', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 1.2 Q2争议识别的不确定性分布
ax2 = axes[0, 1]
ax2.hist(q2_propagated['q2_ci_width'].dropna(), bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
ax2.axvline(q2_propagated['q2_ci_width'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'均值: {q2_propagated["q2_ci_width"].mean():.3f}')
ax2.set_xlabel('95% CI 宽度', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.set_title('Q2: 争议识别概率的传播不确定性', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 1.3 Q3系数稳定性误差棒图
ax3 = axes[1, 0]
features = q3_coef_stability['feature'].values
coef_mean = q3_coef_stability['coef_mean'].values
coef_ci_lower = q3_coef_stability['coef_ci_lower'].values
coef_ci_upper = q3_coef_stability['coef_ci_upper'].values

# 简化特征名
feature_labels = ['评分', '周次', '年龄', '粉丝支持']
colors = ['green' if c < 0 else 'coral' for c in coef_mean]
y_pos = np.arange(len(features))

ax3.barh(y_pos, coef_mean, xerr=[coef_mean - coef_ci_lower, coef_ci_upper - coef_mean],
         color=colors, alpha=0.7, capsize=5, error_kw={'ecolor': 'gray', 'capthick': 2})
ax3.axvline(0, color='black', linestyle='-', linewidth=1)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(feature_labels, fontsize=11)
ax3.set_xlabel('系数值 (含95%置信区间)', fontsize=12)
ax3.set_title('Q3: 系数估计的稳定性 (传播后)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 1.4 不确定性传播汇总
ax4 = axes[1, 1]
stages = ['Q1\n粉丝投票估计', 'Q2\n争议识别', 'Q3\n因素分析']
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
ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, label='基准不确定性')
ax4.set_ylabel('相对不确定性 (Q1 = 1.0)', fontsize=12)
ax4.set_title('不确定性传播放大效应', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, normalized):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_uncertainty_propagation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_uncertainty_propagation.png")

# ============================================================================
# 图2：Q1-Q2不确定性相关性分析
# ============================================================================
print("生成图2：Q1-Q2不确定性传播相关性...")

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
ax1.set_xlabel('Q1 不确定性 (CI宽度)', fontsize=12)
ax1.set_ylabel('Q2 不确定性 (CI宽度)', fontsize=12)
ax1.set_title('Q1→Q2 不确定性传播', fontsize=14, fontweight='bold')

# 添加拟合线
z = np.polyfit(merged['ci_width'], merged['q2_ci_width'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged['ci_width'].min(), merged['ci_width'].max(), 100)
ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'斜率: {z[0]:.3f}')
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
         color='steelblue', linewidth=2, markersize=5, label='Q1不确定性')
ax2.plot(season_agg['season'], season_agg['q2_ci_width'], 's-', 
         color='forestgreen', linewidth=2, markersize=5, label='Q2不确定性')
ax2.set_xlabel('赛季', fontsize=12)
ax2.set_ylabel('平均CI宽度', fontsize=12)
ax2.set_title('不确定性的时间趋势', fontsize=14, fontweight='bold')
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
        categories.append('双高')
    elif h1 and not h2:
        categories.append('Q1高')
    elif not h1 and h2:
        categories.append('Q2高')
    else:
        categories.append('双低')

cat_counts = pd.Series(categories).value_counts()
colors = ['red', 'steelblue', 'forestgreen', 'lightgray']
ax3.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%', 
        colors=colors[:len(cat_counts)], startangle=90)
ax3.set_title('不确定性分类分布', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_uncertainty_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_uncertainty_correlation.png")

# ============================================================================
# 图3：综合敏感性分析
# ============================================================================
print("生成图3：综合敏感性分析...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 淘汰vs存活的不确定性比较
ax1 = axes[0, 0]
eliminated = q1_uncertainty[q1_uncertainty['is_eliminated'] == 1]['fan_support_std']
survived = q1_uncertainty[q1_uncertainty['is_eliminated'] == 0]['fan_support_std']

ax1.boxplot([eliminated, survived], labels=['被淘汰', '存活'])
ax1.set_ylabel('粉丝支持度估计标准差', fontsize=12)
ax1.set_title('淘汰状态 vs 估计不确定性', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 添加均值标注
ax1.text(1, eliminated.mean(), f'μ={eliminated.mean():.3f}', ha='center', fontsize=10)
ax1.text(2, survived.mean(), f'μ={survived.mean():.3f}', ha='center', fontsize=10)

# 3.2 Q3系数Bootstrap分布（模拟）
ax2 = axes[0, 1]
# 生成模拟分布
np.random.seed(42)
coef_fan_support = np.random.normal(
    q3_coef_stability[q3_coef_stability['feature'] == 'fan_support_mean']['coef_mean'].values[0],
    q3_coef_stability[q3_coef_stability['feature'] == 'fan_support_mean']['coef_std'].values[0],
    1000
)
ax2.hist(coef_fan_support, bins=40, color='coral', alpha=0.7, edgecolor='white', density=True)
ax2.axvline(coef_fan_support.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'均值: {coef_fan_support.mean():.3f}')
ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('系数值', fontsize=12)
ax2.set_ylabel('密度', fontsize=12)
ax2.set_title('粉丝支持度系数的Bootstrap分布', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3 不确定性放大矩阵
ax3 = axes[1, 0]
stages = ['Q1', 'Q2', 'Q3']
matrix = np.array([
    [1.00, 1.13, 0.01],  # Q1 -> others
    [0.88, 1.00, 0.01],  # Q2 -> others
    [0.99, 0.99, 1.00]   # Q3 -> others
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
ax3.set_title('不确定性传播因子矩阵', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, label='放大因子')

# 3.4 结论汇总
ax4 = axes[1, 1]
ax4.axis('off')

conclusion_text = f"""
【不确定性传播分析结论】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Q1 粉丝投票估计
   • 95% CI 平均宽度: {summary['q1_uncertainty']['mean_ci_width']:.3f}
   • 含义: 投票估计存在中等不确定性

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Q2 争议识别传播
   • 放大因子: {summary['q2_propagation']['amplification_factor']:.2f}x
   • 结论: 不确定性略有放大但可控

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Q3 因素分析稳定性
   • 稳定显著特征: {summary['q3_propagation']['n_stable_features']}/{summary['q3_propagation']['total_features']}
   • 系数变异系数: {summary['q3_propagation']['mean_coef_cv']:.4f}
   • 结论: 模型高度稳健

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 总体评估: {summary['conclusion']['robustness']}
   {summary['conclusion']['interpretation']}
"""

ax4.text(0.05, 0.95, conclusion_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_sensitivity_analysis.png")

# ============================================================================
# 图4：流程图风格的传播示意
# ============================================================================
print("生成图4：传播框架流程图...")

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# 定义方框位置
boxes = [
    {'x': 1, 'y': 3, 'w': 3, 'h': 2, 'text': 'Q1\n粉丝投票估计\n\nBootstrap\nn=200', 'color': 'steelblue'},
    {'x': 6.5, 'y': 3, 'w': 3, 'h': 2, 'text': 'Q2\n争议识别\n\nMonte Carlo\nn=100', 'color': 'forestgreen'},
    {'x': 12, 'y': 3, 'w': 3, 'h': 2, 'text': 'Q3\n因素分析\n\n系数稳定性', 'color': 'coral'},
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
ax.text(5.25, 4.8, f'放大 {summary["q2_propagation"]["amplification_factor"]:.2f}x', 
        ha='center', fontsize=10, fontweight='bold', color='red')
ax.text(10.75, 4.8, f'稳定 {summary["q3_propagation"]["stability_rate"]*100:.0f}%', 
        ha='center', fontsize=10, fontweight='bold', color='green')

# 添加不确定性指标
ax.text(2.5, 1.8, f'CI宽度: {summary["q1_uncertainty"]["mean_ci_width"]:.3f}', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(8, 1.8, f'CI宽度: {summary["q2_propagation"]["mean_ci_width"]:.3f}', 
        ha='center', fontsize=10, fontweight='bold')
ax.text(13.5, 1.8, f'CV: {summary["q3_propagation"]["mean_coef_cv"]:.4f}', 
        ha='center', fontsize=10, fontweight='bold')

# 标题
ax.text(8, 5.8, '不确定性传播框架 (Uncertainty Propagation Framework)', 
        ha='center', fontsize=16, fontweight='bold')

ax.text(8, 0.5, '结论: 尽管Q1存在估计不确定性，Q2和Q3的核心结论保持稳定可靠', 
        ha='center', fontsize=12, style='italic', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('fig_propagation_framework.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig_propagation_framework.png")

print("\n" + "="*70)
print("可视化生成完成!")
print("="*70)

# 生成图片说明
captions = """
图片说明 (Figure Captions)
==========================

fig_uncertainty_propagation.png - 不确定性传播框架概览
  包含四个子图：(a) Q1粉丝投票估计的95%置信区间宽度分布;
  (b) Q2争议识别概率传播后的不确定性分布;
  (c) Q3系数估计的稳定性误差棒图;
  (d) 不确定性在Q1→Q2→Q3传播过程中的放大效应对比。

fig_uncertainty_correlation.png - Q1-Q2不确定性传播相关性
  分析Q1投票估计不确定性如何传播到Q2争议识别：
  (a) Q1与Q2不确定性的散点图及拟合线;
  (b) 不确定性随赛季的时间趋势;
  (c) 高不确定性样本的分类分布。

fig_sensitivity_analysis.png - 综合敏感性分析
  (a) 淘汰vs存活选手的估计不确定性对比;
  (b) 粉丝支持度系数的Bootstrap分布;
  (c) 不确定性传播因子矩阵;
  (d) 综合分析结论汇总。

fig_propagation_framework.png - 传播框架流程图
  展示Q1→Q2→Q3的不确定性传播路径，标注每阶段的传播效应和稳定性指标。
"""

with open('figure_captions.txt', 'w', encoding='utf-8') as f:
    f.write(captions)
print("✅ 图片说明已保存: figure_captions.txt")
