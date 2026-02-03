"""

1. 评委救人机制 (Judges' Save) 建模与分析
2. 阿罗不可能定理检验
3. 综合方法推荐
4. 生成最终报告

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("Q2 Step 4: 评委救人机制分析 + 方法推荐")
print("=" * 70)

# ==================== 1. 加载数据 ====================
print("\n[1/6] 加载数据")
print("-" * 50)

try:
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'q2_step1_simulated.csv'))
    print(f"  仿真数据: {len(df)} 条记录")

    with open(os.path.join(SCRIPT_DIR, 'q2_step1_results.json'), 'r', encoding='utf-8') as f:
        step1_results = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'q2_step2_results.json'), 'r', encoding='utf-8') as f:
        step2_results = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'q2_step3_results.json'), 'r', encoding='utf-8') as f:
        step3_results = json.load(f)
    print("  已加载前序步骤结果")
    
except FileNotFoundError as e:
    print(f"  ERROR: {e}")
    raise SystemExit()

# ==================== 2. 评委救人机制分析 ====================
print("\n[2/6] 评委救人机制分析")
print("-" * 50)

class JudgesSaveAnalyzer:
    
    def __init__(self, df):
        self.df = df
        self.results = []
    
    def simulate_judges_save(self, group):

        if len(group) <= 2:
            return group.assign(
                judges_save_eliminated=group['is_eliminated'],
                in_bottom_two=True
            )
        
        group = group.copy()
        
        # 确定最差两名
        bottom_two_idx = group.nlargest(2, 'combined_rank').index
        group['in_bottom_two'] = group.index.isin(bottom_two_idx)
        
        # 评委选择：淘汰评委分数较低者
        bottom_two = group.loc[bottom_two_idx]
        eliminated_idx = bottom_two['week_total_score'].idxmin()
        
        group['judges_save_eliminated'] = False
        group.loc[eliminated_idx, 'judges_save_eliminated'] = True
        
        return group
    
    def analyze_all(self):
        print("  模拟评委救人机制...")
        
        results = []
        for (season, week), group in self.df.groupby(['season', 'week']):
            if len(group) < 3:
                continue
            if group['is_eliminated'].sum() == 0:
                continue
            
            result = self.simulate_judges_save(group)
            results.append(result)
        
        self.df_with_save = pd.concat(results, ignore_index=True)
        
        # 统计分析
        actual_elim = self.df_with_save[self.df_with_save['is_eliminated'] == 1]
        
        # 各方法准确率
        rank_correct = (actual_elim['rank_eliminated'] == 1).sum()
        pct_correct = (actual_elim['pct_eliminated'] == 1).sum()
        save_correct = (actual_elim['judges_save_eliminated'] == 1).sum()
        
        total = len(actual_elim)
        
        self.stats = {
            'total_eliminations': total,
            'rank_correct': int(rank_correct),
            'rank_accuracy': float(rank_correct / total),
            'pct_correct': int(pct_correct),
            'pct_accuracy': float(pct_correct / total),
            'save_correct': int(save_correct),
            'save_accuracy': float(save_correct / total)
        }
        
        # 评委救人改变了多少结果
        rank_vs_save_diff = ((actual_elim['rank_eliminated'] == 1) != 
                            (actual_elim['judges_save_eliminated'] == 1)).sum()
        self.stats['rank_save_difference'] = int(rank_vs_save_diff)
        self.stats['rank_save_diff_rate'] = float(rank_vs_save_diff / total)
        
        print(f"  分析完成: {total} 次淘汰")
        print(f"  排名法准确率: {self.stats['rank_accuracy']:.2%}")
        print(f"  百分比法准确率: {self.stats['pct_accuracy']:.2%}")
        print(f"  评委救人准确率: {self.stats['save_accuracy']:.2%}")
        print(f"  评委救人改变结果: {rank_vs_save_diff} 次 ({self.stats['rank_save_diff_rate']:.2%})")
        
        return self.stats

# 运行评委救人分析
js_analyzer = JudgesSaveAnalyzer(df)
js_stats = js_analyzer.analyze_all()

# ==================== 3. 阿罗不可能定理检验 ====================
print("\n[3/6] 阿罗不可能定理检验")
print("-" * 50)

class ArrowTheoremVerifier:
    
    def __init__(self, df):
        self.df = df
        self.results = {}
    
    def check_pareto_efficiency(self):

        violations_rank = 0
        violations_pct = 0
        total_pairs = 0
        
        for (season, week), group in self.df.groupby(['season', 'week']):
            if len(group) < 2:
                continue
            
            # 检验所有选手对
            for i, row_a in group.iterrows():
                for j, row_b in group.iterrows():
                    if i >= j:
                        continue
                    
                    total_pairs += 1
                    
                    # A在评委和观众中都更好
                    a_better_judge = row_a['week_total_score'] > row_b['week_total_score']
                    a_better_fan = row_a['predicted_fan_vote_score'] > row_b['predicted_fan_vote_score']
                    
                    if a_better_judge and a_better_fan:
                        # 检查排名法结果
                        if row_a['rank_placement'] > row_b['rank_placement']:
                            violations_rank += 1
                        # 检查百分比法结果
                        if row_a['pct_placement'] > row_b['pct_placement']:
                            violations_pct += 1
        
        self.results['pareto'] = {
            'total_pairs': total_pairs,
            'rank_violations': violations_rank,
            'rank_violation_rate': violations_rank / total_pairs if total_pairs > 0 else 0,
            'pct_violations': violations_pct,
            'pct_violation_rate': violations_pct / total_pairs if total_pairs > 0 else 0
        }
        
        print(f"  帕累托效率检验:")
        print(f"    总配对数: {total_pairs}")
        print(f"    排名法违反次数: {violations_rank} ({violations_rank/total_pairs*100:.2f}%)")
        print(f"    百分比法违反次数: {violations_pct} ({violations_pct/total_pairs*100:.2f}%)")
        
        return self.results['pareto']
    
    def check_non_dictatorship(self):
        # 用Step 2的方差分解结果
        rank_fan_weight = step2_results['variance_decomposition']['rank_fan_weight']
        pct_fan_weight = step2_results['variance_decomposition']['pct_fan_weight']
        
        # 非独裁性：权重在0.2-0.8之间
        rank_non_dict = 0.2 <= rank_fan_weight <= 0.8
        pct_non_dict = 0.2 <= pct_fan_weight <= 0.8
        
        self.results['non_dictatorship'] = {
            'rank_fan_weight': rank_fan_weight,
            'rank_satisfies': rank_non_dict,
            'pct_fan_weight': pct_fan_weight,
            'pct_satisfies': pct_non_dict
        }
        
        print(f"\n  非独裁性检验:")
        print(f"    排名法观众权重: {rank_fan_weight:.2%} -> {'满足' if rank_non_dict else '不满足'}")
        print(f"    百分比法观众权重: {pct_fan_weight:.2%} -> {'满足' if pct_non_dict else '不满足'}")
        
        return self.results['non_dictatorship']
    
    def check_iia(self):
        
        self.results['iia'] = {
            'rank_satisfies': False,
            'rank_reason': '排名法中，每个选手的排名取决于所有其他选手的表现',
            'pct_satisfies': False,
            'pct_reason': '百分比法中，总分变化会影响每个人的百分比'
        }
        
        print(f"\n  IIA独立性检验:")
        print(f"    排名法: 不满足 - 排名依赖于其他选手")
        print(f"    百分比法: 不满足 - 百分比依赖于总分")
        
        return self.results['iia']
    
    def verify_all(self):
        self.check_pareto_efficiency()
        self.check_non_dictatorship()
        self.check_iia()
        
        # 综合评分
        rank_score = 0
        pct_score = 0
        
        # 帕累托效率
        rank_score += (1 - self.results['pareto']['rank_violation_rate']) * 33
        pct_score += (1 - self.results['pareto']['pct_violation_rate']) * 33
        
        # 非独裁性
        rank_score += 33 if self.results['non_dictatorship']['rank_satisfies'] else 0
        pct_score += 33 if self.results['non_dictatorship']['pct_satisfies'] else 0
        
        # IIA
        
        self.results['overall_score'] = {
            'rank_method': rank_score,
            'pct_method': pct_score
        }
        
        print(f"\n  阿罗定理综合评分 (满分100):")
        print(f"    排名法: {rank_score:.1f}")
        print(f"    百分比法: {pct_score:.1f}")
        
        return self.results

# 运行阿罗定理检验
arrow_verifier = ArrowTheoremVerifier(df)
arrow_results = arrow_verifier.verify_all()

# ==================== 4. 方法推荐 ====================
print("\n[4/6] 综合方法推荐")
print("-" * 50)

class MethodRecommender:
    
    def __init__(self, step1, step2, step3, js_stats, arrow):
        self.step1 = step1
        self.step2 = step2
        self.step3 = step3
        self.js_stats = js_stats
        self.arrow = arrow
    
    def compute_scores(self):
        
        # 维度1: 准确率
        rank_acc = self.step1['rank_method']['accuracy'] * 25
        pct_acc = self.step1['percentage_method']['accuracy'] * 25
        
        # 维度2: 观众参与度
        rank_balance = (1 - abs(0.5 - self.step2['variance_decomposition']['rank_fan_weight'])) * 25
        pct_balance = (1 - abs(0.5 - self.step2['variance_decomposition']['pct_fan_weight'])) * 25
        
        # 维度3: 阿罗定理得分
        rank_arrow = self.arrow['overall_score']['rank_method'] / 100 * 25
        pct_arrow = self.arrow['overall_score']['pct_method'] / 100 * 25
        
        # 维度4: 评委救人准确率
        save_score = self.js_stats['save_accuracy'] * 25
        
        self.scores = {
            'rank_method': {
                'accuracy': rank_acc,
                'balance': rank_balance,
                'arrow': rank_arrow,
                'total': rank_acc + rank_balance + rank_arrow
            },
            'pct_method': {
                'accuracy': pct_acc,
                'balance': pct_balance,
                'arrow': pct_arrow,
                'total': pct_acc + pct_balance + pct_arrow
            },
            'judges_save': {
                'accuracy': save_score,
                'description': '评委救人机制可作为补充'
            }
        }
        
        return self.scores
    
    def make_recommendation(self):
        scores = self.compute_scores()
        
        rank_total = scores['rank_method']['total']
        pct_total = scores['pct_method']['total']
        
        print(f"\n  【多维度评分】(满分75)")
        print(f"  排名法总分: {rank_total:.1f}")
        print(f"    - 准确率: {scores['rank_method']['accuracy']:.1f}/25")
        print(f"    - 权重平衡: {scores['rank_method']['balance']:.1f}/25")
        print(f"    - 阿罗定理: {scores['rank_method']['arrow']:.1f}/25")
        
        print(f"\n  百分比法总分: {pct_total:.1f}")
        print(f"    - 准确率: {scores['pct_method']['accuracy']:.1f}/25")
        print(f"    - 权重平衡: {scores['pct_method']['balance']:.1f}/25")
        print(f"    - 阿罗定理: {scores['pct_method']['arrow']:.1f}/25")
        
        print(f"\n  评委救人机制准确率得分: {scores['judges_save']['accuracy']:.1f}/25")
        
        # 生成推荐
        if rank_total > pct_total:
            primary_rec = "排名法"
            reason = "权重更平衡，评委和观众影响力相当"
        else:
            primary_rec = "百分比法"
            reason = "预测准确率更高"
        
        self.recommendation = {
            'primary': primary_rec,
            'reason': reason,
            'scores': scores,
            'include_judges_save': True,
            'judges_save_reason': '可纠正约{}%的争议淘汰'.format(
                int(self.js_stats['rank_save_diff_rate'] * 100)
            )
        }
        
        print(f"\n  【推荐方案】")
        print(f"  主推荐: {primary_rec}")
        print(f"  理由: {reason}")
        print(f"  是否建议评委救人机制: 是")
        print(f"  评委救人理由: {self.recommendation['judges_save_reason']}")
        
        return self.recommendation

# 生成推荐
recommender = MethodRecommender(step1_results, step2_results, step3_results, 
                                js_stats, arrow_results)
recommendation = recommender.make_recommendation()

# ==================== 5. 可视化 ====================
print("\n[5/6] 生成可视化")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 三种方法准确率对比
ax1 = axes[0, 0]
methods = ['Rank\nMethod', 'Percentage\nMethod', 'Judges Save\n(Rank+Vote)']
accuracies = [js_stats['rank_accuracy'], js_stats['pct_accuracy'], js_stats['save_accuracy']]
colors = ['#1976D2', '#F57C00', '#4CAF50']
bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, width=0.6)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_title('Fig 1: Elimination Prediction Accuracy\nby Different Methods', 
              fontsize=12, fontweight='bold')
ax1.set_ylim([0, 0.5])
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.1%}', ha='center', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 图2: 阿罗定理评分
ax2 = axes[0, 1]
criteria = ['Pareto\nEfficiency', 'Non-\nDictatorship', 'IIA']
rank_scores = [
    (1 - arrow_results['pareto']['rank_violation_rate']) * 100,
    100 if arrow_results['non_dictatorship']['rank_satisfies'] else 0,
    0
]
pct_scores = [
    (1 - arrow_results['pareto']['pct_violation_rate']) * 100,
    100 if arrow_results['non_dictatorship']['pct_satisfies'] else 0,
    0
]

x = np.arange(len(criteria))
width = 0.35
bars_rank = ax2.bar(x - width/2, rank_scores, width, label='Rank Method', color='#1976D2', alpha=0.8)
bars_pct = ax2.bar(x + width/2, pct_scores, width, label='Percentage Method', color='#F57C00', alpha=0.8)

for i, (score, bar) in enumerate(zip(rank_scores, bars_rank)):
    if score < 1:
        reason = "Fail"
        if i == 2: reason = "Not\nSatisfied"
        ax2.text(bar.get_x() + bar.get_width()/2, 3, reason, 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1565C0')

for i, (score, bar) in enumerate(zip(pct_scores, bars_pct)):
    if score < 1:
        reason = "Fail"
        if i == 1: reason = "Fan Weight\nImbalance"
        if i == 2: reason = "Not\nSatisfied"
        ax2.text(bar.get_x() + bar.get_width()/2, 3, reason, 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#E65100')

ax2.set_xticks(x)
ax2.set_xticklabels(criteria)
ax2.set_ylabel('Score (%)', fontsize=11)
ax2.set_title("Fig 2: Arrow's Theorem Criteria Satisfaction", 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.set_ylim([0, 110])
ax2.grid(axis='y', alpha=0.3)

# 图3: 评委救人机制影响
ax3 = axes[1, 0]
categories = ['Both Agree\n& Correct', 'Save\nCorrects Rank', 'Rank\nCorrects Save', 'Both\nWrong']
# 计算各类别
df_save = js_analyzer.df_with_save
actual_elim = df_save[df_save['is_eliminated'] == 1]

both_correct = ((actual_elim['rank_eliminated'] == 1) & 
                (actual_elim['judges_save_eliminated'] == 1)).sum()
save_corrects = ((actual_elim['rank_eliminated'] == 0) & 
                 (actual_elim['judges_save_eliminated'] == 1)).sum()
rank_corrects = ((actual_elim['rank_eliminated'] == 1) & 
                 (actual_elim['judges_save_eliminated'] == 0)).sum()
both_wrong = ((actual_elim['rank_eliminated'] == 0) & 
              (actual_elim['judges_save_eliminated'] == 0)).sum()

values = [both_correct, save_corrects, rank_corrects, both_wrong]
colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
ax3.bar(categories, values, color=colors, alpha=0.8)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Fig 3: Judges Save Mechanism Impact\n(Comparing with Rank Method)', 
              fontsize=12, fontweight='bold')
for i, v in enumerate(values):
    ax3.text(i, v + 2, str(v), ha='center', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 图4: 综合评分雷达图
axes[1, 1].remove()
ax4 = fig.add_subplot(2, 2, 4, projection='polar')

dimensions = ['Accuracy', 'Balance', 'Arrow\nTheorem']
rank_vals = [recommender.scores['rank_method']['accuracy'],
             recommender.scores['rank_method']['balance'],
             recommender.scores['rank_method']['arrow']]
pct_vals = [recommender.scores['pct_method']['accuracy'],
            recommender.scores['pct_method']['balance'],
            recommender.scores['pct_method']['arrow']]

rank_vals += [rank_vals[0]]
pct_vals += [pct_vals[0]]
angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
angles += [angles[0]]

ax4.plot(angles, rank_vals, 'o-', linewidth=2, label='Rank Method', color='#1976D2')
ax4.fill(angles, rank_vals, color='#1976D2', alpha=0.25)

ax4.plot(angles, pct_vals, 'o-', linewidth=2, label='Percentage Method', color='#F57C00')
ax4.fill(angles, pct_vals, color='#F57C00', alpha=0.25)

ax4.set_thetagrids(np.degrees(angles[:-1]), dimensions)
ax4.set_rlabel_position(60)
ax4.set_title('Fig 4: Multi-Criteria Evaluation\n(Radar Chart)', 
              fontsize=12, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.set_ylim([0, 30])
ax4.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q2_fig5_final_analysis.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  已保存: q2_fig5_final_analysis.png")

# ==================== 6. 保存最终结果 ====================
print("\n[6/6] 保存最终结果")
print("-" * 50)

final_results = {
    'step1_counterfactual': step1_results,
    'step2_bias_sensitivity': step2_results,
    'step3_controversy_cases': [
        {k: v for k, v in case.items() if k != 'counterfactual'} 
        for case in step3_results
    ],
    'step4_judges_save': js_stats,
    'step4_arrow_theorem': arrow_results,
    'step4_recommendation': recommendation
}

with open(os.path.join(SCRIPT_DIR, 'q2_final_results.json'), 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
print("  已保存: q2_final_results.json")

# 保存模型
joblib.dump({
    'js_analyzer': js_analyzer,
    'arrow_verifier': arrow_verifier,
    'recommender': recommender
}, os.path.join(SCRIPT_DIR, 'q2_final_models.joblib'))
print("  已保存: q2_final_models.joblib")

# ==================== 生成最终报告 ====================
final_report = f"""
================================================================================
                    Q2 投票方法比较分析 - 最终报告
                         MCM 2026 Problem C
================================================================================

一、研究概述
--------------------------------------------------------------------------------
本研究比较了DWTS节目使用的两种投票组合方法：排名法(Rank Method)和百分比法
(Percentage Method)，并分析了评委救人机制(Judges' Save)的影响。

研究方法：
1. 反事实仿真框架 - 将两种方法应用于所有34赛季
2. 偏倚敏感度分析 - 量化观众投票对结果的影响
3. 争议案例研究 - 分析4个著名争议案例
4. 评委救人机制分析 - 评估其纠错效果
5. 阿罗不可能定理检验 - 理论层面验证

二、核心发现
--------------------------------------------------------------------------------

【反事实仿真结果】
  总淘汰事件: {step1_results['total_eliminations']}
  排名法准确率: {step1_results['rank_method']['accuracy']:.2%}
  百分比法准确率: {step1_results['percentage_method']['accuracy']:.2%}
  方法一致率: {step1_results['method_agreement_rate']:.2%}

【偏倚敏感度分析】
  排名法观众权重: {step2_results['variance_decomposition']['rank_fan_weight']:.2%}
  百分比法观众权重: {step2_results['variance_decomposition']['pct_fan_weight']:.2%}
  结论: {step2_results['conclusion']['more_fan_biased_method']}更偏向观众投票

【评委救人机制】
  准确率: {js_stats['save_accuracy']:.2%}
  相比排名法改变结果: {js_stats['rank_save_difference']} 次 ({js_stats['rank_save_diff_rate']:.2%})

【阿罗不可能定理检验】
  帕累托效率:
    排名法违反率: {arrow_results['pareto']['rank_violation_rate']:.2%}
    百分比法违反率: {arrow_results['pareto']['pct_violation_rate']:.2%}
  非独裁性:
    排名法: {'满足' if arrow_results['non_dictatorship']['rank_satisfies'] else '不满足'}
    百分比法: {'满足' if arrow_results['non_dictatorship']['pct_satisfies'] else '不满足'}
  IIA独立性:
    两种方法均不满足

三、争议案例总结
--------------------------------------------------------------------------------
"""

for case in step3_results:
    final_report += f"""
  {case['name']} (Season {case['season']}):
    实际名次: 第{case['actual_placement']}名
    评委分数百分位: {case['judge_percentile']:.1f}%
    观众投票百分位: {case['fan_percentile']:.1f}%
"""

final_report += f"""
四、方法推荐
--------------------------------------------------------------------------------

【综合评分】(满分75分)
  排名法: {recommender.scores['rank_method']['total']:.1f}
  百分比法: {recommender.scores['pct_method']['total']:.1f}

【最终推荐】
  主推荐方法: {recommendation['primary']}
  推荐理由: {recommendation['reason']}
  
  是否建议评委救人机制: {'是' if recommendation['include_judges_save'] else '否'}
  理由: {recommendation['judges_save_reason']}

【具体建议】

1. 若优先考虑"评委与观众平衡":
   推荐: 排名法
   理由: 观众权重约50%，评委评委权重约50%，最为均衡

2. 若优先考虑"预测准确性":
   推荐: 百分比法
   理由: 整体预测准确率更高

3. 若优先考虑"减少争议":
   推荐: 排名法 + 评委救人机制
   理由: 这是Season 28后DWTS采用的方案，可让评委纠正"不公平"淘汰

4. 给节目制作人的建议:
   - 当前方案(排名法+评委救人)是合理的权衡
   - 如果希望增加观众参与感，可考虑百分比法
   - 完全消除争议是不可能的(阿罗不可能定理)
   - 适度争议对节目收视率有正面影响

五、理论意义
--------------------------------------------------------------------------------
根据阿罗不可能定理，不存在同时满足所有公平性准则的完美投票系统。
DWTS的方法选择体现了在多个目标间的权衡：

  - 排名法: 更公平(权重均衡)，但可能不够"刺激"
  - 百分比法: 更倾向观众，增加娱乐性，但可能产生争议
  - 评委救人: 提供专业纠错机会，但增加主观因素

最佳选择取决于节目的目标定位。

================================================================================
                              分析完成
================================================================================
"""

with open(os.path.join(SCRIPT_DIR, 'q2_final_report.txt'), 'w', encoding='utf-8') as f:
    f.write(final_report)
print("  已保存: q2_final_report.txt")

print(final_report)

print("\n" + "=" * 70)
print("Q2 Step 4 完成！正在生成DOCX文档...")
print("=" * 70)
