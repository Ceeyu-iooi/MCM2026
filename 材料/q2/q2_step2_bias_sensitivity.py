# -*- coding: utf-8 -*-
"""
Q2 Step 2: 偏倚敏感度分析 (Bias Sensitivity Analysis)
MCM 2026 Problem C

核心任务：
- 分析两种方法中观众投票对最终结果的影响程度
- 计算偏倚敏感度梯度: ∂(placement)/∂(fan_vote)
- 通过方差分解估算观众权重
- 判断哪种方法更偏向观众投票

数学模型：
1. 敏感度梯度: Sensitivity = ΔPlacement / ΔFanVote
2. 方差分解: Fan_Weight = Var(Fan_Component) / Var(Total)
3. 边际效应: 观众投票增加1%时排名变化

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("Q2 Step 2: 偏倚敏感度分析")
print("=" * 70)

# ==================== 1. 加载Step 1数据 ====================
print("\n[1/5] 加载数据")
print("-" * 50)

try:
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'q2_step1_simulated.csv'))
    print(f"  加载仿真数据: {len(df)} 条记录")
except FileNotFoundError:
    print("  ERROR: 请先运行 q2_step1_counterfactual.py")
    raise SystemExit()

# ==================== 2. 偏倚敏感度计算 ====================
print("\n[2/5] 计算偏倚敏感度")
print("-" * 50)

class BiasSensitivityAnalyzer:
    """
    偏倚敏感度分析器
    
    核心思想：
    1. 敏感度梯度：观众投票变化时，最终排名变化多少
    2. 方差分解：综合分数中观众因素贡献的方差占比
    3. 边际效应：观众投票边际增加对结果的影响
    
    如果某方法的观众权重更高，说明该方法更偏向观众投票
    """
    
    def __init__(self):
        self.results = []
    
    def compute_variance_decomposition(self, group, method='rank'):
        """
        方差分解法计算观众权重
        
        原理：
        Var(Combined) ≈ Var(Judge_Component) + Var(Fan_Component)
        Fan_Weight = Var(Fan_Component) / [Var(Judge) + Var(Fan)]
        
        这种方法直接衡量观众因素对结果变异性的贡献
        """
        if len(group) < 3:
            return {'fan_weight': 0.5, 'judge_weight': 0.5}
        
        if method == 'rank':
            fan_var = group['fan_rank'].var()
            judge_var = group['judge_rank'].var()
        else:
            fan_var = group['fan_pct'].var()
            judge_var = group['judge_pct'].var()
        
        total_var = fan_var + judge_var
        if total_var == 0:
            return {'fan_weight': 0.5, 'judge_weight': 0.5}
        
        return {
            'fan_weight': fan_var / total_var,
            'judge_weight': judge_var / total_var
        }
    
    def compute_marginal_effect(self, group, method='rank'):
        """
        边际效应分析
        
        原理：
        对每个选手，假设其观众投票增加10%，计算排名变化
        Marginal_Effect = Δ Placement / Δ FanVote(10%)
        
        高边际效应意味着观众投票对结果影响大
        """
        if len(group) < 2:
            return 0
        
        group = group.copy()
        delta_pct = 0.10  # 10%扰动
        
        marginal_effects = []
        
        for idx in group.index:
            # 原始排名
            if method == 'rank':
                original_placement = group.loc[idx, 'rank_placement']
            else:
                original_placement = group.loc[idx, 'pct_placement']
            
            # 扰动观众投票
            original_vote = group.loc[idx, 'predicted_fan_vote_score']
            perturbed_vote = original_vote * (1 + delta_pct)
            
            # 创建扰动后的数据
            group_perturbed = group.copy()
            group_perturbed.loc[idx, 'predicted_fan_vote_score'] = perturbed_vote
            
            # 重新计算排名
            if method == 'rank':
                group_perturbed['fan_rank_new'] = group_perturbed['predicted_fan_vote_score'].rank(
                    ascending=False, method='min')
                group_perturbed['combined_new'] = group_perturbed['judge_rank'] + group_perturbed['fan_rank_new']
                group_perturbed['placement_new'] = group_perturbed['combined_new'].rank(method='min')
            else:
                total_votes = group_perturbed['predicted_fan_vote_score'].sum()
                group_perturbed['fan_pct_new'] = group_perturbed['predicted_fan_vote_score'] / total_votes
                group_perturbed['combined_new'] = group_perturbed['judge_pct'] + group_perturbed['fan_pct_new']
                group_perturbed['placement_new'] = group_perturbed['combined_new'].rank(ascending=False, method='min')
            
            new_placement = group_perturbed.loc[idx, 'placement_new']
            
            # 计算边际效应（排名变化/投票变化）
            placement_change = original_placement - new_placement  # 正值表示排名上升
            marginal_effect = placement_change / delta_pct
            marginal_effects.append(marginal_effect)
        
        return np.mean(np.abs(marginal_effects))
    
    def compute_sensitivity_gradient(self, group, method='rank'):
        """
        敏感度梯度计算
        
        计算 ∂(placement) / ∂(fan_vote_normalized)
        使用数值微分方法
        """
        if len(group) < 2:
            return 0
        
        group = group.copy()
        
        # 归一化观众投票到[0,1]区间
        vote_min = group['predicted_fan_vote_score'].min()
        vote_max = group['predicted_fan_vote_score'].max()
        vote_range = vote_max - vote_min
        
        if vote_range == 0:
            return 0
        
        # 计算每个选手的敏感度
        sensitivities = []
        delta = 0.01 * vote_range  # 1%的范围作为微扰
        
        for idx in group.index:
            if method == 'rank':
                original = group.loc[idx, 'rank_placement']
            else:
                original = group.loc[idx, 'pct_placement']
            
            # 正向扰动
            group_plus = group.copy()
            group_plus.loc[idx, 'predicted_fan_vote_score'] += delta
            
            if method == 'rank':
                group_plus['fan_rank'] = group_plus['predicted_fan_vote_score'].rank(ascending=False, method='min')
                group_plus['combined'] = group_plus['judge_rank'] + group_plus['fan_rank']
                group_plus['placement'] = group_plus['combined'].rank(method='min')
            else:
                total = group_plus['predicted_fan_vote_score'].sum()
                group_plus['fan_pct'] = group_plus['predicted_fan_vote_score'] / total
                group_plus['combined'] = group_plus['judge_pct'] + group_plus['fan_pct']
                group_plus['placement'] = group_plus['combined'].rank(ascending=False, method='min')
            
            new_placement = group_plus.loc[idx, 'placement']
            
            # 敏感度 = |Δplacement / Δvote|
            sensitivity = abs(original - new_placement) / delta
            sensitivities.append(sensitivity)
        
        return np.mean(sensitivities)
    
    def analyze_week(self, group):
        """分析单个周次的偏倚敏感度"""
        season = group['season'].iloc[0]
        week = group['week'].iloc[0]
        
        # 排名法分析
        rank_var = self.compute_variance_decomposition(group, 'rank')
        rank_marginal = self.compute_marginal_effect(group, 'rank')
        rank_gradient = self.compute_sensitivity_gradient(group, 'rank')
        
        # 百分比法分析
        pct_var = self.compute_variance_decomposition(group, 'percentage')
        pct_marginal = self.compute_marginal_effect(group, 'percentage')
        pct_gradient = self.compute_sensitivity_gradient(group, 'percentage')
        
        return {
            'season': season,
            'week': week,
            'n_contestants': len(group),
            # 排名法
            'rank_fan_weight': rank_var['fan_weight'],
            'rank_judge_weight': rank_var['judge_weight'],
            'rank_marginal_effect': rank_marginal,
            'rank_sensitivity': rank_gradient,
            # 百分比法
            'pct_fan_weight': pct_var['fan_weight'],
            'pct_judge_weight': pct_var['judge_weight'],
            'pct_marginal_effect': pct_marginal,
            'pct_sensitivity': pct_gradient
        }
    
    def analyze_all(self, df):
        """分析所有周次"""
        print("  正在计算偏倚敏感度...")
        
        results = []
        for (season, week), group in df.groupby(['season', 'week']):
            if len(group) < 2:
                continue
            result = self.analyze_week(group)
            results.append(result)
        
        self.results = pd.DataFrame(results)
        print(f"  分析完成: {len(self.results)} 个周次")
        
        return self.results

# 运行分析
analyzer = BiasSensitivityAnalyzer()
sensitivity_df = analyzer.analyze_all(df)

# ==================== 3. 结果统计 ====================
print("\n[3/5] 结果统计")
print("-" * 50)

# 整体统计
mean_rank_fan_weight = sensitivity_df['rank_fan_weight'].mean()
mean_pct_fan_weight = sensitivity_df['pct_fan_weight'].mean()
mean_rank_marginal = sensitivity_df['rank_marginal_effect'].mean()
mean_pct_marginal = sensitivity_df['pct_marginal_effect'].mean()
mean_rank_sensitivity = sensitivity_df['rank_sensitivity'].mean()
mean_pct_sensitivity = sensitivity_df['pct_sensitivity'].mean()

print(f"\n  【方差分解 - 观众权重】")
print(f"    排名法观众权重: {mean_rank_fan_weight:.2%}")
print(f"    百分比法观众权重: {mean_pct_fan_weight:.2%}")
print(f"    差异: {(mean_pct_fan_weight - mean_rank_fan_weight)*100:.2f} 百分点")

print(f"\n  【边际效应】(投票增加10%时的排名变化)")
print(f"    排名法: {mean_rank_marginal:.4f}")
print(f"    百分比法: {mean_pct_marginal:.4f}")

print(f"\n  【敏感度梯度】")
print(f"    排名法: {mean_rank_sensitivity:.6f}")
print(f"    百分比法: {mean_pct_sensitivity:.6f}")

# 判断哪种方法更偏向观众
if mean_pct_fan_weight > mean_rank_fan_weight:
    more_fan_biased = "百分比法"
    weight_diff = mean_pct_fan_weight - mean_rank_fan_weight
else:
    more_fan_biased = "排名法"
    weight_diff = mean_rank_fan_weight - mean_pct_fan_weight

print(f"\n  【结论】")
print(f"    {more_fan_biased}更偏向观众投票")
print(f"    观众权重差异: {weight_diff*100:.2f} 百分点")

# 按赛季分组统计
season_sensitivity = sensitivity_df.groupby('season').agg({
    'rank_fan_weight': 'mean',
    'pct_fan_weight': 'mean',
    'rank_marginal_effect': 'mean',
    'pct_marginal_effect': 'mean'
}).reset_index()

# ==================== 4. 可视化 ====================
print("\n[4/5] 生成可视化")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 观众权重分布对比
ax1 = axes[0, 0]
ax1.hist(sensitivity_df['rank_fan_weight'], bins=25, alpha=0.6, 
         label=f'Rank Method (μ={mean_rank_fan_weight:.2%})', color='#1976D2')
ax1.hist(sensitivity_df['pct_fan_weight'], bins=25, alpha=0.6,
         label=f'Percentage Method (μ={mean_pct_fan_weight:.2%})', color='#F57C00')
ax1.axvline(mean_rank_fan_weight, color='#1976D2', linestyle='--', linewidth=2)
ax1.axvline(mean_pct_fan_weight, color='#F57C00', linestyle='--', linewidth=2)
ax1.set_xlabel('Fan Vote Weight', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Fig 1: Fan Vote Weight Distribution\n(Higher = More Fan-Biased)', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 图2: 按赛季的观众权重趋势
ax2 = axes[0, 1]
ax2.plot(season_sensitivity['season'], season_sensitivity['rank_fan_weight'], 
         'o-', label='Rank Method', color='#1976D2', linewidth=2, markersize=5)
ax2.plot(season_sensitivity['season'], season_sensitivity['pct_fan_weight'],
         's-', label='Percentage Method', color='#F57C00', linewidth=2, markersize=5)
ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Equal Weight (0.5)')
ax2.fill_between(season_sensitivity['season'], 
                  season_sensitivity['rank_fan_weight'],
                  season_sensitivity['pct_fan_weight'],
                  alpha=0.2, color='green')
ax2.set_xlabel('Season', fontsize=11)
ax2.set_ylabel('Fan Vote Weight', fontsize=11)
ax2.set_title('Fig 2: Fan Vote Weight Trend by Season', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 1])

# 图3: 权重差异热力图
ax3 = axes[1, 0]
# 按赛季段分组
season_ranges = [(1, 10), (11, 20), (21, 27), (28, 34)]
range_labels = ['S1-10', 'S11-20', 'S21-27', 'S28-34']
heatmap_data = []

for start, end in season_ranges:
    mask = (sensitivity_df['season'] >= start) & (sensitivity_df['season'] <= end)
    subset = sensitivity_df[mask]
    heatmap_data.append({
        'Range': f'S{start}-{end}',
        'Rank Fan Weight': subset['rank_fan_weight'].mean(),
        'Pct Fan Weight': subset['pct_fan_weight'].mean(),
        'Weight Diff': subset['pct_fan_weight'].mean() - subset['rank_fan_weight'].mean()
    })

heatmap_df = pd.DataFrame(heatmap_data)
heat_matrix = heatmap_df[['Rank Fan Weight', 'Pct Fan Weight']].values
sns.heatmap(heat_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=['Rank Method', 'Pct Method'],
            yticklabels=range_labels, ax=ax3, vmin=0, vmax=1,
            cbar_kws={'label': 'Fan Weight'})
ax3.set_title('Fig 3: Fan Weight by Season Range\n(Green = Higher Fan Influence)', 
              fontsize=12, fontweight='bold')

# 图4: 边际效应对比
ax4 = axes[1, 1]
methods = ['Rank Method', 'Percentage Method']
marginal_values = [mean_rank_marginal, mean_pct_marginal]
colors = ['#1976D2', '#F57C00']
bars = ax4.bar(methods, marginal_values, color=colors, alpha=0.8, width=0.5)
ax4.set_ylabel('Marginal Effect', fontsize=11)
ax4.set_title('Fig 4: Average Marginal Effect of Fan Votes\n(Higher = Fan votes impact ranking more)', 
              fontsize=12, fontweight='bold')
for bar, val in zip(bars, marginal_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q2_fig2_bias_sensitivity.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  已保存: q2_fig2_bias_sensitivity.png")

# ==================== 5. 保存结果 ====================
print("\n[5/5] 保存结果")
print("-" * 50)

# 保存详细数据
sensitivity_df.to_csv(os.path.join(SCRIPT_DIR, 'q2_step2_sensitivity.csv'), index=False)
print("  已保存: q2_step2_sensitivity.csv")

# 保存汇总结果
results_summary = {
    'variance_decomposition': {
        'rank_fan_weight': float(mean_rank_fan_weight),
        'rank_judge_weight': float(1 - mean_rank_fan_weight),
        'pct_fan_weight': float(mean_pct_fan_weight),
        'pct_judge_weight': float(1 - mean_pct_fan_weight),
        'weight_difference': float(mean_pct_fan_weight - mean_rank_fan_weight)
    },
    'marginal_effect': {
        'rank_method': float(mean_rank_marginal),
        'pct_method': float(mean_pct_marginal)
    },
    'sensitivity_gradient': {
        'rank_method': float(mean_rank_sensitivity),
        'pct_method': float(mean_pct_sensitivity)
    },
    'conclusion': {
        'more_fan_biased_method': more_fan_biased,
        'weight_difference_pct': float(weight_diff * 100)
    }
}

with open(os.path.join(SCRIPT_DIR, 'q2_step2_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print("  已保存: q2_step2_results.json")

# ==================== 生成报告 ====================
report = f"""
================================================================================
               Q2 Step 2: 偏倚敏感度分析 - 分析报告
================================================================================

一、研究目的
--------------------------------------------------------------------------------
分析两种投票组合方法中，观众投票对最终结果的影响程度。
判断哪种方法更"偏向"观众投票，即观众投票的权重更大。

二、分析方法
--------------------------------------------------------------------------------
1. 方差分解法 (Variance Decomposition)
   原理: Var(Combined) = Var(Judge) + Var(Fan)
   观众权重 = Var(Fan) / [Var(Judge) + Var(Fan)]
   
2. 边际效应分析 (Marginal Effect)
   原理: 观众投票增加10%时，排名平均变化多少
   高边际效应 = 观众投票对结果影响大

3. 敏感度梯度 (Sensitivity Gradient)
   原理: ∂(placement) / ∂(fan_vote)
   数值微分计算排名对观众投票的导数

三、核心结果
--------------------------------------------------------------------------------
【方差分解 - 观众权重】
  排名法:    观众权重 = {mean_rank_fan_weight:.2%}, 评委权重 = {1-mean_rank_fan_weight:.2%}
  百分比法:  观众权重 = {mean_pct_fan_weight:.2%}, 评委权重 = {1-mean_pct_fan_weight:.2%}
  
  差异: 百分比法观众权重高出 {(mean_pct_fan_weight - mean_rank_fan_weight)*100:.2f} 百分点

【边际效应】(投票增加10%时的排名变化)
  排名法:    {mean_rank_marginal:.4f}
  百分比法:  {mean_pct_marginal:.4f}

【敏感度梯度】
  排名法:    {mean_rank_sensitivity:.6f}
  百分比法:  {mean_pct_sensitivity:.6f}

四、结论
--------------------------------------------------------------------------------
{more_fan_biased}更偏向观众投票，观众权重差异为 {weight_diff*100:.2f} 百分点。

理论解释:
- 百分比法: 高分选手的绝对分数优势被放大（马太效应）
- 排名法: 所有选手权重相等，仅考虑相对排名

对于评委和观众意见分歧大的选手:
- 百分比法: 如果观众投票高，其影响被放大
- 排名法: 影响相对均匀

五、赛季段分析
--------------------------------------------------------------------------------
"""

for item in heatmap_data:
    report += f"  {item['Range']}: 排名法={item['Rank Fan Weight']:.2%}, "
    report += f"百分比法={item['Pct Fan Weight']:.2%}, "
    report += f"差异={item['Weight Diff']*100:.2f}%\n"

report += f"""
================================================================================
                              Step 2 完成
================================================================================
"""

with open(os.path.join(SCRIPT_DIR, 'q2_step2_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print("  已保存: q2_step2_report.txt")

print(report)

print("\n" + "=" * 70)
print("Step 2 偏倚敏感度分析完成！")
print("=" * 70)
print("\n下一步 (Step 3): 争议案例分析")
print("请确认后输入指令继续...")
