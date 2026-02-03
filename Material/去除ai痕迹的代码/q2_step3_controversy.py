import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
DATA_DIR = os.path.dirname(SCRIPT_DIR)

print("=" * 70)
print("Q2 Step 3: 争议案例分析")
print("=" * 70)

# ==================== 1. 加载数据 ====================
print("\n[1/5] 加载数据")
print("-" * 50)

try:
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'q2_step1_simulated.csv'))
    print(f"  加载仿真数据: {len(df)} 条记录")
    
    # 加载原始数据获取更多信息
    df_original = pd.read_csv(r"D:\pydocument\processed_dwts_cleaned_ms08.csv")
    print(f"  加载原始数据: {len(df_original)} 条记录")
except FileNotFoundError as e:
    print(f"  ERROR: {e}")
    raise SystemExit()

# ==================== 2. 定义争议案例 ====================
print("\n[2/5] 争议案例定义")
print("-" * 50)

controversy_cases = [
    {
        'name': 'Jerry Rice',
        'season': 2,
        'actual_placement': 2,  # 亚军
        'controversy': '尽管5周评委分数最低，仍获得亚军',
        'method_used': 'rank'
    },
    {
        'name': 'Billy Ray Cyrus',
        'season': 4,
        'actual_placement': 5,
        'controversy': '6周评委分数最低，仍获得第5名',
        'method_used': 'percentage'
    },
    {
        'name': 'Bristol Palin',
        'season': 11,
        'actual_placement': 3,
        'controversy': '12次评委分数最低，仍获得第3名',
        'method_used': 'percentage'
    },
    {
        'name': 'Bobby Bones',
        'season': 27,
        'actual_placement': 1,  # 冠军
        'controversy': '评委分数持续较低，却赢得冠军',
        'method_used': 'percentage'
    }
]

# ==================== 3. 案例分析 ====================
print("\n[3/5] 案例详细分析")
print("-" * 50)

class ControversyCaseAnalyzer:

    def __init__(self, df, df_original):
        self.df = df
        self.df_original = df_original
        self.results = []
    
    def find_contestant(self, name, season):
        # 精确匹配
        mask = (self.df['celebrity_name'].str.lower() == name.lower()) & \
               (self.df['season'] == season)
        data = self.df[mask]
        
        # 模糊匹配
        if len(data) == 0:
            first_name = name.split()[0]
            mask = (self.df['celebrity_name'].str.contains(first_name, case=False, na=False)) & \
                   (self.df['season'] == season)
            data = self.df[mask]
        
        return data.sort_values('week')
    
    def analyze_case(self, case):
        name = case['name']
        season = case['season']
        
        print(f"\n  【{name} (Season {season})】")
        print(f"  争议: {case['controversy']}")
        
        # 获取选手数据
        contestant_data = self.find_contestant(name, season)
        
        if len(contestant_data) == 0:
            print(f"     未找到该选手数据")
            return None
        
        # 获取该赛季所有数据用于比较
        season_data = self.df[self.df['season'] == season]
        
        # 基本统计
        n_weeks = len(contestant_data)
        avg_judge_score = contestant_data['week_total_score'].mean()
        avg_fan_vote = contestant_data['predicted_fan_vote_score'].mean()
        
        # 评委排名统计
        avg_judge_rank = contestant_data['judge_rank'].mean()
        times_last_judge = (contestant_data['judge_rank'] == contestant_data['judge_rank'].max()).sum()
        
        # 观众排名统计
        avg_fan_rank = contestant_data['fan_rank'].mean()
        
        # 计算在赛季中的相对位置
        season_avg_score = season_data.groupby('celebrity_name')['week_total_score'].mean()
        season_avg_vote = season_data.groupby('celebrity_name')['predicted_fan_vote_score'].mean()
        
        judge_percentile = (avg_judge_score - season_avg_score.min()) / \
                          (season_avg_score.max() - season_avg_score.min()) * 100
        fan_percentile = (avg_fan_vote - season_avg_vote.min()) / \
                        (season_avg_vote.max() - season_avg_vote.min()) * 100
        
        # 两种方法下的平均名次
        avg_rank_placement = contestant_data['rank_placement'].mean()
        avg_pct_placement = contestant_data['pct_placement'].mean()
        
        # 判断-观众差距
        judge_fan_gap = avg_judge_rank - avg_fan_rank  # 正值=观众更喜欢

        would_be_eliminated_rank = contestant_data['rank_eliminated'].sum()
        would_be_eliminated_pct = contestant_data['pct_eliminated'].sum()
        
        result = {
            'name': name,
            'season': season,
            'actual_placement': case['actual_placement'],
            'method_used': case['method_used'],
            'controversy': case['controversy'],
            'weeks_competed': n_weeks,
            'avg_judge_score': avg_judge_score,
            'avg_fan_vote': avg_fan_vote,
            'avg_judge_rank': avg_judge_rank,
            'avg_fan_rank': avg_fan_rank,
            'judge_fan_gap': judge_fan_gap,
            'judge_percentile': judge_percentile,
            'fan_percentile': fan_percentile,
            'avg_rank_placement': avg_rank_placement,
            'avg_pct_placement': avg_pct_placement,
            'times_eliminated_by_rank': would_be_eliminated_rank,
            'times_eliminated_by_pct': would_be_eliminated_pct
        }
        
        # 打印结果
        print(f"    参赛周数: {n_weeks}")
        print(f"    平均评委分数: {avg_judge_score:.1f} (百分位: {judge_percentile:.1f}%)")
        print(f"    平均观众投票: {avg_fan_vote:.1f} (百分位: {fan_percentile:.1f}%)")
        print(f"    平均评委排名: {avg_judge_rank:.1f}")
        print(f"    平均观众排名: {avg_fan_rank:.1f}")
        print(f"    评委-观众差距: {judge_fan_gap:.2f} {'(观众更喜欢)' if judge_fan_gap > 0 else '(评委更认可)'}")
        print(f"    排名法下平均名次: {avg_rank_placement:.1f}")
        print(f"    百分比法下平均名次: {avg_pct_placement:.1f}")
        print(f"    排名法会淘汰次数: {would_be_eliminated_rank}")
        print(f"    百分比法会淘汰次数: {would_be_eliminated_pct}")
        
        self.results.append(result)
        return result
    
    def counterfactual_analysis(self, case, result):
        name = case['name']
        season = case['season']
        method_used = case['method_used']
        
        contestant_data = self.find_contestant(name, season)
        
        if len(contestant_data) == 0:
            return None

        if method_used == 'rank':
            actual_eliminated = contestant_data['rank_eliminated'].sum()
            alt_eliminated = contestant_data['pct_eliminated'].sum()
            alt_method = 'percentage'
        else:
            actual_eliminated = contestant_data['pct_eliminated'].sum()
            alt_eliminated = contestant_data['rank_eliminated'].sum()
            alt_method = 'rank'
        
        counterfactual = {
            'method_used': method_used,
            'actual_eliminated_times': int(actual_eliminated),
            'alternative_method': alt_method,
            'alt_eliminated_times': int(alt_eliminated),
            'difference': int(alt_eliminated - actual_eliminated)
        }
        
        print(f"\n    【反事实分析】")
        print(f"    实际使用: {method_used}法, 被判淘汰{actual_eliminated}次")
        print(f"    如果使用: {alt_method}法, 会被判淘汰{alt_eliminated}次")
        
        if alt_eliminated > actual_eliminated:
            print(f"    结论: 使用{alt_method}法会更早淘汰 (多{alt_eliminated-actual_eliminated}次)")
        elif alt_eliminated < actual_eliminated:
            print(f"    结论: 使用{alt_method}法会更晚淘汰 (少{actual_eliminated-alt_eliminated}次)")
        else:
            print(f"    结论: 两种方法结果相同")
        
        return counterfactual
    
    def analyze_all(self, cases):
        all_results = []
        
        for case in cases:
            result = self.analyze_case(case)
            if result:
                cf = self.counterfactual_analysis(case, result)
                if cf:
                    result['counterfactual'] = cf
                all_results.append(result)
        
        return all_results

# 运行分析
analyzer = ControversyCaseAnalyzer(df, df_original)
case_results = analyzer.analyze_all(controversy_cases)

# ==================== 4. 可视化 ====================
print("\n\n[4/5] 生成可视化")
print("-" * 50)

if len(case_results) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 图1: 评委vs观众百分位对比
    ax1 = axes[0, 0]
    names = [r['name'] for r in case_results]
    judge_pct = [r['judge_percentile'] for r in case_results]
    fan_pct = [r['fan_percentile'] for r in case_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, judge_pct, width, label='Judge Score Percentile', 
                    color='#1976D2', alpha=0.8)
    bars2 = ax1.bar(x + width/2, fan_pct, width, label='Fan Vote Percentile',
                    color='#F57C00', alpha=0.8)
    
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='Median (50%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r['name']}\n(S{r['season']})" for r in case_results], fontsize=10)
    ax1.set_ylabel('Percentile (%)', fontsize=11)
    ax1.set_title('Fig 1: Controversy Cases - Judge vs Fan Percentile\n(Higher = Better relative performance)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 评委-观众差距
    ax2 = axes[0, 1]
    gaps = [r['judge_fan_gap'] for r in case_results]
    colors = ['#4CAF50' if g > 0 else '#F44336' for g in gaps]
    
    bars = ax2.bar(names, gaps, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel('Judge Rank - Fan Rank', fontsize=11)
    ax2.set_title('Fig 2: Judge-Fan Gap\n(Positive = Fans like more than judges)', 
                  fontsize=12, fontweight='bold')
    
    for bar, gap in zip(bars, gaps):
        y_pos = bar.get_height() / 2
        ax2.text(bar.get_x() + bar.get_width()/2, 
                 y_pos,
                 f'{gap:.2f}', ha='center', va='center', 
                 fontsize=10, fontweight='bold', color='white')
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: 两种方法下的平均名次
    ax3 = axes[1, 0]
    rank_placements = [r['avg_rank_placement'] for r in case_results]
    pct_placements = [r['avg_pct_placement'] for r in case_results]
    
    x = np.arange(len(names))
    bars1 = ax3.bar(x - width/2, rank_placements, width, label='Rank Method', 
                    color='#1976D2', alpha=0.8)
    bars2 = ax3.bar(x + width/2, pct_placements, width, label='Percentage Method',
                    color='#F57C00', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{r['name']}\n(S{r['season']})" for r in case_results], fontsize=10)
    ax3.set_ylabel('Average Placement (Lower = Better)', fontsize=11)
    ax3.set_title('Fig 3: Average Placement by Method\n(Lower value = Better ranking)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.invert_yaxis()
    ax3.grid(axis='y', alpha=0.3)
    
    # 图4: 周次详细趋势
    ax4 = axes[1, 1]
    bristol_data = analyzer.find_contestant('Bristol Palin', 11)

    if len(bristol_data) > 0:
        weeks = range(1, len(bristol_data) + 1)
        ax4.plot(weeks, bristol_data['judge_rank'].values, 'o-', 
                label='Judge Rank', color='#1976D2', linewidth=4, markersize=10, alpha=0.5)
        ax4.plot(weeks, bristol_data['fan_rank'].values, 's--',
                label='Fan Rank', color='#F57C00', linewidth=2, markersize=6)
        ax4.fill_between(weeks, bristol_data['judge_rank'].values, 
                         bristol_data['fan_rank'].values, alpha=0.2, color='green')
        ax4.set_xlabel('Week', fontsize=11)
        ax4.set_ylabel('Rank (1 = Best)', fontsize=11)
        ax4.set_title('Fig 4: Bristol Palin (S11) Weekly Trajectory\n(Gap shows fan preference over judges)', 
                      fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.invert_yaxis()
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Bristol Palin data not available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Fig 4: Bristol Palin Weekly Trajectory')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'q2_fig3_controversy_cases.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  已保存: q2_fig3_controversy_cases.png")

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    controversy_en = {
        'Jerry Rice': "Runner-up despite lowest judge scores for 5 weeks",
        'Billy Ray Cyrus': "5th place despite lowest judge scores for 6 weeks",
        'Bristol Palin': "3rd place despite lowest judge scores 12 times",
        'Bobby Bones': "Won championship despite consistently low judge scores"
    }

    for idx, case in enumerate(controversy_cases):
        ax = axes2[idx // 2, idx % 2]
        contestant_data = analyzer.find_contestant(case['name'], case['season'])
        
        if len(contestant_data) > 0:
            weeks = range(1, len(contestant_data) + 1)
            # Improved visibility: Judge (Blue/Solid/Thick/Transparent), Fan (Orange/Dashed/Thin)
            ax.plot(weeks, contestant_data['judge_rank'].values, 'o-', 
                   label='Judge Rank', color='#1976D2', linewidth=4, markersize=8, alpha=0.5)
            ax.plot(weeks, contestant_data['fan_rank'].values, 's--',
                   label='Fan Rank', color='#F57C00', linewidth=2, markersize=5)
            ax.set_xlabel('Week')
            ax.set_ylabel('Rank (1=Best)')

            title_text = controversy_en.get(case['name'], case['controversy'])
            ax.set_title(f"{case['name']} (Season {case['season']})\n{title_text}")
            
            ax.legend(fontsize=8)
            ax.invert_yaxis()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{case['name']} - Data not found", 
                   ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'q2_fig4_all_trajectories.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  已保存: q2_fig4_all_trajectories.png")

# ==================== 5. 保存结果 ====================
print("\n[5/5] 保存结果")
print("-" * 50)

# 转换为DataFrame保存
if len(case_results) > 0:
    csv_results = []
    for r in case_results:
        row = {k: v for k, v in r.items() if k != 'counterfactual'}
        if 'counterfactual' in r:
            row.update({f"cf_{k}": v for k, v in r['counterfactual'].items()})
        csv_results.append(row)
    
    results_df = pd.DataFrame(csv_results)
    results_df.to_csv(os.path.join(SCRIPT_DIR, 'q2_step3_controversy.csv'), index=False)
    print("  已保存: q2_step3_controversy.csv")

# 保存JSON结果
with open(os.path.join(SCRIPT_DIR, 'q2_step3_results.json'), 'w', encoding='utf-8') as f:
    json.dump(case_results, f, indent=2, ensure_ascii=False, default=str)
print("  已保存: q2_step3_results.json")

# ==================== 生成报告 ====================
report = f"""
================================================================================
               Q2 Step 3: 争议案例分析 - 分析报告
================================================================================

一、研究目的
--------------------------------------------------------------------------------
深入分析DWTS历史上四个著名的争议案例，这些案例中选手的评委分数与最终名次
存在明显不匹配，引发了观众和媒体的讨论。

二、争议案例列表
--------------------------------------------------------------------------------
1. Jerry Rice (Season 2): 5周评委最低分，仍获亚军
2. Billy Ray Cyrus (Season 4): 6周评委最低分，获第5名
3. Bristol Palin (Season 11): 12次评委最低分，获第3名
4. Bobby Bones (Season 27): 持续低分，却赢得冠军

三、详细分析结果
--------------------------------------------------------------------------------
"""

for r in case_results:
    report += f"""
【{r['name']} (Season {r['season']})】
  争议描述: {r['controversy']}
  实际名次: 第{r['actual_placement']}名
  
  表现数据:
    - 参赛周数: {r['weeks_competed']}
    - 评委分数百分位: {r['judge_percentile']:.1f}%
    - 观众投票百分位: {r['fan_percentile']:.1f}%
    - 评委-观众差距: {r['judge_fan_gap']:.2f} {'(观众更喜欢)' if r['judge_fan_gap'] > 0 else '(评委更认可)'}
  
  方法对比:
    - 排名法下平均名次: {r['avg_rank_placement']:.1f}
    - 百分比法下平均名次: {r['avg_pct_placement']:.1f}
"""
    
    if 'counterfactual' in r:
        cf = r['counterfactual']
        report += f"""
  反事实分析:
    - 实际使用{cf['method_used']}法，被判淘汰{cf['actual_eliminated_times']}次
    - 如果使用{cf['alternative_method']}法，会被判淘汰{cf['alt_eliminated_times']}次
"""

report += """
四、综合结论
--------------------------------------------------------------------------------
"""

# 计算平均差距
avg_judge_pct = np.mean([r['judge_percentile'] for r in case_results])
avg_fan_pct = np.mean([r['fan_percentile'] for r in case_results])
avg_gap = np.mean([r['judge_fan_gap'] for r in case_results])

report += f"""
1. 争议案例共性:
   - 平均评委分数百分位: {avg_judge_pct:.1f}%
   - 平均观众投票百分位: {avg_fan_pct:.1f}%
   - 平均评委-观众差距: {avg_gap:.2f}
   
2. 这些选手获得高名次的原因:
   - 观众投票远超其评委分数表现
   - 在百分比法下，高观众投票的权重被放大
   - 体现了"粉丝效应"对结果的显著影响

3. 方法选择的影响:
   - 排名法: 评委和观众权重更均衡
   - 百分比法: 观众高投票的影响被放大
   - 对于争议选手，两种方法可能产生不同结果

4. 公平性讨论:
   - 如果"公平"定义为专业性优先，这些案例显示了问题
   - 如果"公平"定义为观众参与度，这些案例是正常结果
   - 节目需要在两者间权衡

================================================================================
                              Step 3 完成
================================================================================
"""

with open(os.path.join(SCRIPT_DIR, 'q2_step3_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print("  已保存: q2_step3_report.txt")

print(report)

print("\n" + "=" * 70)
print("Step 3 争议案例分析完成！")
print("=" * 70)
print("\n下一步 (Step 4): 评委救人机制分析 + 方法推荐 + 文档生成")