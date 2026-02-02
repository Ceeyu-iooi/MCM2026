# -*- coding: utf-8 -*-
"""
Q2 Step 1: 反事实仿真框架 (Counterfactual Simulation Framework)
MCM 2026 Problem C

核心任务：
- 实现排名法(Rank Method)和百分比法(Percentage Method)
- 对所有34赛季的所有周次进行反事实仿真
- 比较两种方法产生的淘汰结果

数学模型：
1. 排名法: Combined_Rank = Judge_Rank + Fan_Rank, 淘汰max
2. 百分比法: Combined_Pct = Judge_Pct + Fan_Pct, 淘汰min

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)

print("=" * 70)
print("Q2 Step 1: 反事实仿真框架")
print("=" * 70)

# ==================== 1. 数据加载 ====================
print("\n[1/5] 数据加载")
print("-" * 50)

try:
    # 加载清洗后的数据
    df = pd.read_csv(os.path.join(DATA_DIR, 'processed_dwts_cleaned_ms08.csv'))
    print(f"  基础数据: {len(df)} 条记录, {df['season'].nunique()} 个赛季")
    
    # 加载Q1估算的观众投票
    fan_votes = pd.read_csv(os.path.join(DATA_DIR, 'q1', 'q1_rf_predicted_fan_votes.csv'))
    print(f"  观众投票估算: {len(fan_votes)} 条记录")
    
except FileNotFoundError as e:
    print(f"  ERROR: 数据文件未找到 - {e}")
    raise SystemExit("请确保数据文件存在")

# 合并数据
df = df.merge(
    fan_votes[['season', 'week', 'celebrity_name', 'predicted_fan_vote_score']], 
    on=['season', 'week', 'celebrity_name'], 
    how='left'
)

# 填充缺失值
df['predicted_fan_vote_score'] = df['predicted_fan_vote_score'].fillna(
    df['predicted_fan_vote_score'].mean()
)
print(f"  合并后数据: {len(df)} 条记录")

# ==================== 2. 定义投票方法类 ====================
print("\n[2/5] 定义投票方法")
print("-" * 50)

class VotingMethodSimulator:
    """
    投票方法仿真器
    
    实现DWTS使用的两种投票组合方法：
    
    1. 排名法 (Rank Method) - Season 1-2, 28-34
       Combined_Rank = Judge_Rank + Fan_Rank
       淘汰者 = argmax(Combined_Rank)  # 综合排名最差者
       
    2. 百分比法 (Percentage Method) - Season 3-27
       Judge_Pct = Score_i / Σ Score_j
       Fan_Pct = FanVote_i / Σ FanVote_j
       Combined_Pct = Judge_Pct + Fan_Pct
       淘汰者 = argmin(Combined_Pct)  # 综合百分比最低者
    """
    
    def __init__(self):
        self.results = []
        self.season_method_map = self._get_season_method_map()
    
    def _get_season_method_map(self):
        """获取各赛季实际使用的方法"""
        method_map = {}
        for s in range(1, 35):
            if s <= 2 or s >= 28:
                method_map[s] = 'rank'
            else:
                method_map[s] = 'percentage'
        return method_map
    
    def compute_rank_method(self, group):
        """
        排名法计算
        
        核心逻辑：
        - 评委排名：分数越高，排名越小（1=最好）
        - 观众排名：投票越高，排名越小（1=最好）
        - 综合排名 = 评委排名 + 观众排名
        - 淘汰者 = 综合排名最大（最差）者
        
        参数:
            group: 某一周的所有选手数据
            
        返回:
            添加了排名法相关字段的group
        """
        n = len(group)
        if n <= 1:
            return group.assign(
                judge_rank=1, fan_rank=1, combined_rank=2,
                rank_eliminated=False, rank_placement=1
            )
        
        group = group.copy()
        
        # 评委排名（分数高→排名小）
        group['judge_rank'] = group['week_total_score'].rank(ascending=False, method='min')
        
        # 观众排名（投票高→排名小）
        group['fan_rank'] = group['predicted_fan_vote_score'].rank(ascending=False, method='min')
        
        # 综合排名 = 评委排名 + 观众排名
        group['combined_rank'] = group['judge_rank'] + group['fan_rank']
        
        # 淘汰者（综合排名最大 = 最差）
        max_rank = group['combined_rank'].max()
        group['rank_eliminated'] = group['combined_rank'] == max_rank
        
        # 处理平局（选择第一个）
        if group['rank_eliminated'].sum() > 1:
            elim_indices = group[group['rank_eliminated']].index.tolist()
            group['rank_eliminated'] = False
            group.loc[elim_indices[0], 'rank_eliminated'] = True
        
        # 最终名次（综合排名小→名次高）
        group['rank_placement'] = group['combined_rank'].rank(method='min')
        
        return group
    
    def compute_percentage_method(self, group):
        """
        百分比法计算
        
        核心逻辑：
        - 评委百分比 = 个人分数 / 所有人分数之和
        - 观众百分比 = 个人投票 / 所有人投票之和
        - 综合百分比 = 评委百分比 + 观众百分比
        - 淘汰者 = 综合百分比最小者
        
        参数:
            group: 某一周的所有选手数据
            
        返回:
            添加了百分比法相关字段的group
        """
        n = len(group)
        if n <= 1:
            return group.assign(
                judge_pct=1.0, fan_pct=1.0, combined_pct=2.0,
                pct_eliminated=False, pct_placement=1
            )
        
        group = group.copy()
        
        # 评委百分比
        total_score = group['week_total_score'].sum()
        if total_score > 0:
            group['judge_pct'] = group['week_total_score'] / total_score
        else:
            group['judge_pct'] = 1 / n
        
        # 观众百分比
        total_votes = group['predicted_fan_vote_score'].sum()
        if total_votes > 0:
            group['fan_pct'] = group['predicted_fan_vote_score'] / total_votes
        else:
            group['fan_pct'] = 1 / n
        
        # 综合百分比
        group['combined_pct'] = group['judge_pct'] + group['fan_pct']
        
        # 淘汰者（综合百分比最小）
        min_pct = group['combined_pct'].min()
        group['pct_eliminated'] = group['combined_pct'] == min_pct
        
        # 处理平局
        if group['pct_eliminated'].sum() > 1:
            elim_indices = group[group['pct_eliminated']].index.tolist()
            group['pct_eliminated'] = False
            group.loc[elim_indices[0], 'pct_eliminated'] = True
        
        # 最终名次（百分比高→名次高）
        group['pct_placement'] = group['combined_pct'].rank(ascending=False, method='min')
        
        return group
    
    def simulate_week(self, group):
        """对单个周次应用两种方法"""
        group = self.compute_rank_method(group)
        group = self.compute_percentage_method(group)
        return group
    
    def simulate_all(self, df):
        """
        对所有赛季-周次进行反事实仿真
        
        反事实的含义：
        - 无论该赛季实际使用哪种方法，都同时计算两种方法的结果
        - 这样可以直接比较如果使用另一种方法，结果会有何不同
        """
        print("  开始反事实仿真...")
        
        results = []
        week_count = 0
        
        for (season, week), group in df.groupby(['season', 'week']):
            # 只分析有淘汰发生的周次
            if group['is_eliminated'].sum() == 0:
                continue
            
            # 至少需要2人才能比较
            if len(group) < 2:
                continue
            
            # 应用两种方法
            simulated = self.simulate_week(group)
            results.append(simulated)
            week_count += 1
        
        if not results:
            raise ValueError("没有有效的仿真数据")
        
        result_df = pd.concat(results, ignore_index=True)
        print(f"  仿真完成: {len(result_df)} 条记录, {week_count} 个淘汰周次")
        
        return result_df

# 创建仿真器并运行
simulator = VotingMethodSimulator()
df_simulated = simulator.simulate_all(df)

# ==================== 3. 结果分析 ====================
print("\n[3/5] 结果分析")
print("-" * 50)

# 3.1 方法一致性分析
df_simulated['methods_agree'] = (
    df_simulated['rank_eliminated'] == df_simulated['pct_eliminated']
)

# 3.2 与真实淘汰对比
df_simulated['rank_correct'] = (
    df_simulated['rank_eliminated'] == df_simulated['is_eliminated']
)
df_simulated['pct_correct'] = (
    df_simulated['pct_eliminated'] == df_simulated['is_eliminated']
)

# 3.3 统计实际淘汰事件
actual_eliminations = df_simulated[df_simulated['is_eliminated'] == 1].copy()
n_eliminations = len(actual_eliminations)

# 计算准确率
rank_accuracy = actual_eliminations['rank_eliminated'].mean()
pct_accuracy = actual_eliminations['pct_eliminated'].mean()
agreement_rate = actual_eliminations['methods_agree'].mean()

print(f"\n  总淘汰事件数: {n_eliminations}")
print(f"  排名法预测正确率: {rank_accuracy:.2%}")
print(f"  百分比法预测正确率: {pct_accuracy:.2%}")
print(f"  两种方法一致率: {agreement_rate:.2%}")

# 3.4 按赛季分析
season_stats = df_simulated.groupby('season').agg({
    'rank_correct': 'mean',
    'pct_correct': 'mean', 
    'methods_agree': 'mean'
}).reset_index()
season_stats.columns = ['Season', 'Rank_Accuracy', 'Pct_Accuracy', 'Agreement_Rate']

# 标记实际使用的方法
season_stats['Actual_Method'] = season_stats['Season'].map(simulator.season_method_map)

print("\n  各赛季统计:")
print(season_stats.to_string(index=False))

# 3.5 按实际使用方法分组统计
rank_seasons = season_stats[season_stats['Actual_Method'] == 'rank']
pct_seasons = season_stats[season_stats['Actual_Method'] == 'percentage']

print(f"\n  排名法赛季 (S1-2, S28-34) 平均准确率:")
print(f"    排名法: {rank_seasons['Rank_Accuracy'].mean():.2%}")
print(f"    百分比法: {rank_seasons['Pct_Accuracy'].mean():.2%}")

print(f"\n  百分比法赛季 (S3-27) 平均准确率:")
print(f"    排名法: {pct_seasons['Rank_Accuracy'].mean():.2%}")
print(f"    百分比法: {pct_seasons['Pct_Accuracy'].mean():.2%}")

# ==================== 4. 可视化 ====================
print("\n[4/5] 生成可视化")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 按赛季的准确率对比
ax1 = axes[0, 0]
x = season_stats['Season']
width = 0.35
bars1 = ax1.bar(x - width/2, season_stats['Rank_Accuracy'], width, 
                label='Rank Method', color='#1976D2', alpha=0.8)
bars2 = ax1.bar(x + width/2, season_stats['Pct_Accuracy'], width,
                label='Percentage Method', color='#F57C00', alpha=0.8)

# 标记实际使用的方法
for i, row in season_stats.iterrows():
    if row['Actual_Method'] == 'rank':
        ax1.axvspan(row['Season'] - 0.5, row['Season'] + 0.5, 
                    alpha=0.1, color='blue')

ax1.axhline(y=rank_accuracy, color='#1976D2', linestyle='--', linewidth=1.5, 
            label=f'Rank Avg ({rank_accuracy:.1%})')
ax1.axhline(y=pct_accuracy, color='#F57C00', linestyle='--', linewidth=1.5,
            label=f'Pct Avg ({pct_accuracy:.1%})')
ax1.set_xlabel('Season', fontsize=11)
ax1.set_ylabel('Prediction Accuracy', fontsize=11)
ax1.set_title('Fig 1: Elimination Prediction Accuracy by Season\n(Blue shading = Rank method seasons)', 
              fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# 图2: 方法一致性饼图
ax2 = axes[0, 1]
agree_count = actual_eliminations['methods_agree'].sum()
disagree_count = n_eliminations - agree_count
colors_pie = ['#4CAF50', '#F44336']
wedges, texts, autotexts = ax2.pie(
    [agree_count, disagree_count], 
    labels=['Methods Agree', 'Methods Disagree'],
    autopct='%1.1f%%', 
    colors=colors_pie, 
    startangle=90,
    explode=[0, 0.05]
)
ax2.set_title(f'Fig 2: Method Agreement Rate\n(Total: {n_eliminations} elimination events)', 
              fontsize=12, fontweight='bold')

# 图3: 各赛季方法一致率
ax3 = axes[1, 0]
colors_bar = ['#1976D2' if m == 'rank' else '#F57C00' 
              for m in season_stats['Actual_Method']]
ax3.bar(season_stats['Season'], season_stats['Agreement_Rate'], 
        color=colors_bar, alpha=0.8)
ax3.axhline(y=agreement_rate, color='green', linestyle='--', linewidth=2,
            label=f'Overall: {agreement_rate:.1%}')
ax3.set_xlabel('Season', fontsize=11)
ax3.set_ylabel('Agreement Rate', fontsize=11)
ax3.set_title('Fig 3: Method Agreement Rate by Season\n(Blue=Rank seasons, Orange=Pct seasons)', 
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.set_ylim([0.5, 1.05])
ax3.grid(axis='y', alpha=0.3)

# 图4: 准确率差异分布
ax4 = axes[1, 1]
accuracy_diff = season_stats['Pct_Accuracy'] - season_stats['Rank_Accuracy']
colors_diff = ['#4CAF50' if d > 0 else '#2196F3' for d in accuracy_diff]
ax4.bar(season_stats['Season'], accuracy_diff, color=colors_diff, alpha=0.8)
ax4.axhline(y=0, color='black', linewidth=1)
ax4.axhline(y=accuracy_diff.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean Diff: {accuracy_diff.mean():.1%}')
ax4.set_xlabel('Season', fontsize=11)
ax4.set_ylabel('Accuracy Difference (Pct - Rank)', fontsize=11)
ax4.set_title('Fig 4: Accuracy Difference by Season\n(Green=Pct better, Blue=Rank better)', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q2_fig1_counterfactual.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  已保存: q2_fig1_counterfactual.png")

# ==================== 5. 保存结果 ====================
print("\n[5/5] 保存结果")
print("-" * 50)

# 保存仿真数据
df_simulated.to_csv(os.path.join(SCRIPT_DIR, 'q2_step1_simulated.csv'), index=False)
print("  已保存: q2_step1_simulated.csv")

# 保存季节统计
season_stats.to_csv(os.path.join(SCRIPT_DIR, 'q2_step1_season_stats.csv'), index=False)
print("  已保存: q2_step1_season_stats.csv")

# 保存模型
joblib.dump(simulator, os.path.join(SCRIPT_DIR, 'q2_step1_simulator.joblib'))
print("  已保存: q2_step1_simulator.joblib")

# 保存汇总结果
results_summary = {
    'total_eliminations': int(n_eliminations),
    'rank_method': {
        'accuracy': float(rank_accuracy),
        'accuracy_in_rank_seasons': float(rank_seasons['Rank_Accuracy'].mean()),
        'accuracy_in_pct_seasons': float(pct_seasons['Rank_Accuracy'].mean())
    },
    'percentage_method': {
        'accuracy': float(pct_accuracy),
        'accuracy_in_rank_seasons': float(rank_seasons['Pct_Accuracy'].mean()),
        'accuracy_in_pct_seasons': float(pct_seasons['Pct_Accuracy'].mean())
    },
    'method_agreement_rate': float(agreement_rate),
    'accuracy_difference_mean': float(accuracy_diff.mean())
}

with open(os.path.join(SCRIPT_DIR, 'q2_step1_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print("  已保存: q2_step1_results.json")

# ==================== 生成报告 ====================
report = f"""
================================================================================
                Q2 Step 1: 反事实仿真框架 - 分析报告
================================================================================

一、研究目的
--------------------------------------------------------------------------------
将两种投票组合方法（排名法和百分比法）应用于所有34个赛季，比较它们产生的
淘汰结果差异。这是一种"反事实"分析：如果某赛季实际使用排名法，我们同时
计算如果使用百分比法会产生什么结果，反之亦然。

二、方法说明
--------------------------------------------------------------------------------
1. 排名法 (Rank Method) - 实际用于Season 1-2, 28-34
   Combined_Rank = Judge_Rank + Fan_Rank
   淘汰者 = 综合排名最大（最差）者

2. 百分比法 (Percentage Method) - 实际用于Season 3-27
   Judge_Pct = 个人分数 / 总分数
   Fan_Pct = 个人投票 / 总投票
   Combined_Pct = Judge_Pct + Fan_Pct
   淘汰者 = 综合百分比最小者

三、核心结果
--------------------------------------------------------------------------------
总淘汰事件数: {n_eliminations}

准确率对比:
  排名法预测准确率: {rank_accuracy:.2%}
  百分比法预测准确率: {pct_accuracy:.2%}
  差异: {(pct_accuracy - rank_accuracy)*100:.2f} 个百分点

方法一致性:
  两种方法一致率: {agreement_rate:.2%}
  一致的淘汰数: {int(agree_count)}
  不一致的淘汰数: {int(disagree_count)}

四、分赛季分析
--------------------------------------------------------------------------------
排名法赛季 (S1-2, S28-34):
  排名法准确率: {rank_seasons['Rank_Accuracy'].mean():.2%}
  百分比法准确率: {rank_seasons['Pct_Accuracy'].mean():.2%}

百分比法赛季 (S3-27):
  排名法准确率: {pct_seasons['Rank_Accuracy'].mean():.2%}
  百分比法准确率: {pct_seasons['Pct_Accuracy'].mean():.2%}

五、初步结论
--------------------------------------------------------------------------------
1. 两种方法在{agreement_rate:.1%}的淘汰决策上产生一致结果
2. {'百分比法' if pct_accuracy > rank_accuracy else '排名法'}整体准确率更高
3. 约{(1-agreement_rate)*100:.1f}%的淘汰决策会因方法选择而改变

================================================================================
                              Step 1 完成
================================================================================
"""

# 保存报告
with open(os.path.join(SCRIPT_DIR, 'q2_step1_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print("  已保存: q2_step1_report.txt")

print(report)

print("\n" + "=" * 70)
print("Step 1 反事实仿真框架完成！")
print("=" * 70)
print("\n下一步 (Step 2): 偏倚敏感度分析")
print("请确认后输入指令继续...")
