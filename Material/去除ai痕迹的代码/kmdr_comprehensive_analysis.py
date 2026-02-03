import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")

print("=" * 70)
print("KMDR模型综合分析 - 生成定量结果和可视化")
print("=" * 70)

# =====================================================
# 1. 加载数据
# =====================================================
print("\n【1. 加载数据】")
# Use absolute paths
df_judge = pd.read_csv(r'd:\pydocument\processed_dwts_full.csv')
df_fan = pd.read_csv(r'd:\pydocument\q1_predicted_fan_votes.csv')

df_fan['week_num'] = df_fan['week'].apply(lambda x: int(x.replace('week', '')))
df_merged = df_judge.merge(
    df_fan[['season', 'week_num', 'celebrity_name', 'predicted_fan_vote_score', 'actual_elimination']],
    on=['season', 'week_num', 'celebrity_name'],
    how='inner'
)

df_merged['judge_pct'] = df_merged['week_percent']
df_merged['fan_pct'] = df_merged['predicted_fan_vote_score']

print(f"  总数据量: {len(df_merged)} 条")
print(f"  赛季数: {df_merged['season'].nunique()}")
print(f"  选手数: {df_merged['celebrity_name'].nunique()}")

# Q2争议选手
q2_controversial = {
    'Jerry Rice': 2,
    'Billy Ray Cyrus': 4,
    'Bristol Palin': 11,
    'Bobby Bones': 27
}

# =====================================================
# 2. KMDR模型实现
# =====================================================
print("\n【2. 实现KMDR模型】")

class KMDRModel:
    def __init__(self, alpha=0.4, beta=0.6, gamma=0.7, adaptive=True):
        self.name = "KMDR"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.adaptive = adaptive
    
    def reset(self):
        pass
    
    def _borda_score(self, values):
        n = len(values)
        ranks = stats.rankdata(-values, method='average')
        borda = n - ranks
        return borda
    
    def _condorcet_matrix(self, judge_rank, fan_rank):
        n = len(judge_rank)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    judge_wins = judge_rank[i] < judge_rank[j]
                    fan_wins = fan_rank[i] < fan_rank[j]
                    if judge_wins and fan_wins:
                        C[i][j] = 1.0
                    elif (not judge_wins) and (not fan_wins):
                        C[i][j] = 1.0
                    else:
                        C[i][j] = 0.5
        return C
    
    def _metric_distortion(self, judge_scores, fan_scores, judge_rank, fan_rank):
        n = len(judge_scores)
        j_norm = (judge_scores - judge_scores.min()) / max(judge_scores.max() - judge_scores.min(), 0.01)
        f_norm = (fan_scores - fan_scores.min()) / max(fan_scores.max() - fan_scores.min(), 0.01)
        distortion = np.abs(j_norm - f_norm)
        rank_diff = np.abs(judge_rank - fan_rank) / n
        combined_distortion = 0.6 * distortion + 0.4 * rank_diff
        return combined_distortion
    
    def score(self, j, f, week, contestants):
        n = len(j)
        judge_rank = stats.rankdata(-j, method='average')
        fan_rank = stats.rankdata(-f, method='average')
        
        borda_judge = self._borda_score(j)
        borda_fan = self._borda_score(f)
        
        if self.adaptive:
            if week <= 3:
                w_j = 0.75
            elif week <= 6:
                w_j = 0.60
            else:
                w_j = 0.50
        else:
            w_j = self.gamma
        w_f = 1 - w_j
        
        borda_fused = w_j * borda_judge + w_f * borda_fan
        condorcet = self._condorcet_matrix(judge_rank, fan_rank)
        condorcet_score = condorcet.sum(axis=1)
        distortion = self._metric_distortion(j, f, judge_rank, fan_rank)
        
        final_scores = borda_fused + self.beta * condorcet_score - self.alpha * n * distortion
        
        min_score = final_scores.min()
        max_score = final_scores.max()
        if max_score > min_score:
            normalized_scores = 100 * (final_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones(n) * 50
        
        effective_weights = np.ones(n) * w_j
        return normalized_scores, effective_weights, {
            'borda_judge': borda_judge,
            'borda_fan': borda_fan,
            'condorcet_score': condorcet_score,
            'distortion': distortion
        }

class RankMethod:
    def __init__(self):
        self.name = "排名法"
    def reset(self):
        pass
    def score(self, j, f, week, contestants):
        j_ranks = stats.rankdata(-j)
        f_ranks = stats.rankdata(-f)
        return 100 - (j_ranks + f_ranks), np.ones(len(j)) * 0.5, {}

class FixedWeight:
    def __init__(self):
        self.name = "固定权重(50:50)"
    def reset(self):
        pass
    def score(self, j, f, week, contestants):
        return 0.5 * j + 0.5 * f, np.ones(len(j)) * 0.5, {}

class PercentageMethod:
    def __init__(self):
        self.name = "百分比法"
    def reset(self):
        pass
    def score(self, j, f, week, contestants):
        # 归一化到百分比
        j_pct = 100 * j / max(j.sum(), 0.01)
        f_pct = 100 * f / max(f.sum(), 0.01)
        return j_pct + f_pct, np.ones(len(j)) * 0.5, {}

# =====================================================
# 3. 运行仿真并收集详细数据
# =====================================================
print("\n【3. 运行仿真并收集数据】")

methods = [
    FixedWeight(),
    RankMethod(),
    PercentageMethod(),
    KMDRModel(alpha=0.4, beta=0.6, gamma=0.7)
]

all_results = {m.name: [] for m in methods}
kmdr_details = []

for method in methods:
    method.reset()
    
    for season in sorted(df_merged['season'].unique()):
        season_data = df_merged[df_merged['season'] == season].copy()
        
        for week in sorted(season_data['week_num'].unique()):
            week_data = season_data[season_data['week_num'] == week].copy()
            if len(week_data) < 2:
                continue
            
            j = week_data['judge_pct'].values
            f = week_data['fan_pct'].values
            contestants = week_data['celebrity_name'].values
            
            scores, weights, details = method.score(j, f, week, contestants)
            
            week_data = week_data.copy()
            week_data['method_score'] = scores
            week_data['judge_weight'] = weights
            week_data['method_rank'] = stats.rankdata(-scores)
            all_results[method.name].append(week_data)
            
            # 收集KMDR详细数据
            if method.name == "KMDR" and details:
                for i, celeb in enumerate(contestants):
                    kmdr_details.append({
                        'season': season,
                        'week': week,
                        'celebrity': celeb,
                        'borda_judge': details['borda_judge'][i],
                        'borda_fan': details['borda_fan'][i],
                        'condorcet_score': details['condorcet_score'][i],
                        'distortion': details['distortion'][i],
                        'final_score': scores[i],
                        'judge_weight': weights[i]
                    })

for m_name in all_results:
    all_results[m_name] = pd.concat(all_results[m_name])

kmdr_details_df = pd.DataFrame(kmdr_details)

print(f"  仿真完成，收集了 {len(kmdr_details)} 条KMDR详细数据")

# =====================================================
# 4. 定量分析
# =====================================================
print("\n【4. 定量分析】")

# 4.1 整体性能指标
def evaluate_method(results):
    metrics = {}
    
    # 淘汰预测准确率
    correct = 0
    total = 0
    for (season, week), group in results.groupby(['season', 'week_num']):
        if len(group) < 2:
            continue
        eliminated = group[group['actual_elimination'] == 1]
        if len(eliminated) > 0:
            total += 1
            if group['method_score'].idxmin() in eliminated.index:
                correct += 1
    metrics['淘汰准确率'] = correct / max(total, 1)
    
    # 冠军预测准确率
    correct_champ = 0
    total_seasons = 0
    for season in results['season'].unique():
        season_data = results[results['season'] == season]
        final_week = season_data['week_num'].max()
        final_data = season_data[season_data['week_num'] == final_week]
        if len(final_data) >= 2:
            total_seasons += 1
            pred_champ = final_data.loc[final_data['method_score'].idxmax(), 'celebrity_name']
            actual_champ = final_data[final_data['placement'] == 1]
            if len(actual_champ) > 0 and actual_champ.iloc[0]['celebrity_name'] == pred_champ:
                correct_champ += 1
    metrics['冠军准确率'] = correct_champ / max(total_seasons, 1)
    
    # Top3准确率
    correct_top3 = 0
    total_top3 = 0
    for season in results['season'].unique():
        season_data = results[results['season'] == season]
        final_week = season_data['week_num'].max()
        final_data = season_data[season_data['week_num'] == final_week]
        if len(final_data) >= 3:
            pred_top3 = final_data.nlargest(3, 'method_score')['celebrity_name'].values
            actual_top3 = final_data[final_data['placement'] <= 3]['celebrity_name'].values
            for name in pred_top3:
                total_top3 += 1
                if name in actual_top3:
                    correct_top3 += 1
    metrics['Top3准确率'] = correct_top3 / max(total_top3, 1)
    
    # Spearman相关系数
    correlations = []
    for season in results['season'].unique():
        season_data = results[results['season'] == season]
        final_week = season_data['week_num'].max()
        final_data = season_data[season_data['week_num'] == final_week]
        if len(final_data) >= 3:
            pred_ranks = stats.rankdata(-final_data['method_score'])
            actual_ranks = final_data['placement'].values
            corr, _ = stats.spearmanr(pred_ranks, actual_ranks)
            if not np.isnan(corr):
                correlations.append(corr)
    metrics['Spearman相关'] = np.mean(correlations) if correlations else 0
    
    return metrics

all_metrics = {}
for m_name, results in all_results.items():
    all_metrics[m_name] = evaluate_method(results)

metrics_df = pd.DataFrame(all_metrics).T
print("\n整体性能指标:")
print(metrics_df.round(4))

# 4.2 争议选手分析
controversy_analysis = []
for celeb_name, season in q2_controversial.items():
    celeb_data = df_merged[(df_merged['celebrity_name'] == celeb_name) & (df_merged['season'] == season)]
    if len(celeb_data) == 0:
        continue
    
    for m_name in all_results.keys():
        m_data = all_results[m_name][
            (all_results[m_name]['celebrity_name'] == celeb_name) &
            (all_results[m_name]['season'] == season)
        ]
        
        count = 0
        for week in m_data['week_num'].unique():
            week_all = all_results[m_name][
                (all_results[m_name]['season'] == season) &
                (all_results[m_name]['week_num'] == week)
            ]
            celeb_rank = m_data[m_data['week_num'] == week]['method_rank'].values[0]
            max_rank = week_all['method_rank'].max()
            if celeb_rank == max_rank:
                count += 1
        
        controversy_analysis.append({
            '选手': celeb_name,
            '赛季': season,
            '方法': m_name,
            '垫底次数': count,
            '总周数': len(m_data)
        })

controversy_df = pd.DataFrame(controversy_analysis)
print("\n争议选手垫底次数:")
print(controversy_df.pivot(index=['选手', '赛季'], columns='方法', values='垫底次数'))

# 4.3 KMDR组件贡献分析
print("\nKMDR组件统计:")
component_stats = kmdr_details_df[['borda_judge', 'borda_fan', 'condorcet_score', 'distortion']].describe()
print(component_stats)

# =====================================================
# 5. 可视化生成
# =====================================================
print("\n【5. 生成可视化图表】")

# 5.1 Performance Comparison Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KMDR Model Performance Comparison', fontsize=16, fontweight='bold')

ax = axes[0, 0]
methods_names_cn = list(all_metrics.keys())
methods_names_en = ['Fixed Weight (50:50)', 'Rank Method', 'Percentage Method', 'KMDR']
elim_acc = [all_metrics[m]['淘汰准确率'] for m in methods_names_cn]
colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
bars = ax.bar(methods_names_en, elim_acc, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Elimination Prediction Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom', fontsize=11)

ax = axes[0, 1]
champ_acc = [all_metrics[m]['冠军准确率'] for m in methods_names_cn]
bars = ax.bar(methods_names_en, champ_acc, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Champion Prediction Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom', fontsize=11)

ax = axes[1, 0]
top3_acc = [all_metrics[m]['Top3准确率'] for m in methods_names_cn]
bars = ax.bar(methods_names_en, top3_acc, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Top3 Prediction Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom', fontsize=11)

ax = axes[1, 1]
spearman = [all_metrics[m]['Spearman相关'] for m in methods_names_cn]
bars = ax.bar(methods_names_en, spearman, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Correlation', fontsize=12)
ax.set_title('Spearman Rank Correlation', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('q4/fig_kmdr_performance.png', dpi=300, bbox_inches='tight')
print("  保存: q4/fig_kmdr_performance.png")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_data = controversy_df.pivot(index='选手', columns='方法', values='垫底次数')
pivot_data = pivot_data[['固定权重(50:50)', '排名法', '百分比法', 'KMDR']]

pivot_data.columns = ['Fixed Weight (50:50)', 'Rank Method', 'Percentage Method', 'KMDR']

x = np.arange(len(pivot_data.index))
styles = [
    {'color': "#3b9adeb6", 'marker': 'o', 'linestyle': '-', 'label': 'Fixed Weight (50:50)'},
    {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'label': 'Rank Method'},
    {'color': '#9467bd', 'marker': '^', 'linestyle': '-.', 'label': 'Percentage Method'},
    {'color': "#11eeac82", 'marker': 'D', 'linestyle': ':', 'label': 'KMDR'}
]
offsets = [-0.05, -0.02, 0.02, 0.05]

for i, col in enumerate(pivot_data.columns):
    ax.plot(x + offsets[i], pivot_data[col], 
            color=styles[i]['color'], 
            marker=styles[i]['marker'], 
            linestyle=styles[i]['linestyle'],
            linewidth=2.5, 
            markersize=8,
            label=col,
            alpha=0.9)

ax.set_xlabel('Controversial Player', fontsize=12, fontweight='bold')
ax.set_ylabel('Bottom Rank Count', fontsize=12, fontweight='bold')
ax.set_title('Controversial Players Bottom Rank Count by Method', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pivot_data.index, fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('q4/fig_controversy_comparison.png', dpi=300, bbox_inches='tight')
print("  保存: q4/fig_controversy_comparison.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KMDR Model Component Distribution Analysis', fontsize=16, fontweight='bold')

components = ['borda_judge', 'borda_fan', 'condorcet_score', 'distortion']
titles = ['Borda Judge Score', 'Borda Fan Score', 'Condorcet Consensus Score', 'Metric Distortion Penalty']
colors_comp = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, (comp, title, color) in enumerate(zip(components, titles, colors_comp)):
    ax = axes[i//2, i%2]
    data = kmdr_details_df[comp].values
    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Score Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('q4/fig_kmdr_components.png', dpi=300, bbox_inches='tight')
print("  保存: q4/fig_kmdr_components.png")

fig, ax = plt.subplots(figsize=(12, 6))
weeks = sorted(kmdr_details_df['week'].unique())
weights_by_week = []
for week in weeks:
    week_data = kmdr_details_df[kmdr_details_df['week'] == week]
    weights_by_week.append(week_data['judge_weight'].mean())

ax.plot(weeks, weights_by_week, 'o-', linewidth=2.5, markersize=8, color='#3498db', label='Judge Weight')
ax.plot(weeks, [1-w for w in weights_by_week], 's-', linewidth=2.5, markersize=8, color='#e74c3c', label='Fan Weight')
ax.set_xlabel('Competition Week', fontsize=12, fontweight='bold')
ax.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
ax.set_title('KMDR Dynamic Weight Adjustment Strategy', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

ax.axvspan(1, 3, alpha=0.1, color='green', label='Early Stage')
ax.axvspan(4, 6, alpha=0.1, color='yellow')
ax.axvspan(7, max(weeks), alpha=0.1, color='red')
ax.text(2, 0.85, 'Early\n(Judge-led)', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(5, 0.85, 'Middle\n(Transition)', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(8.5, 0.85, 'Late\n(Balanced)', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('q4/fig_kmdr_weight_dynamics.png', dpi=300, bbox_inches='tight')
print("  保存: q4/fig_kmdr_weight_dynamics.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Metric Distortion Effect on Controversial Players', fontsize=16, fontweight='bold')

ax = axes[0]
controversy_names = list(q2_controversial.keys())
controversy_distortion = kmdr_details_df[kmdr_details_df['celebrity'].isin(controversy_names)]['distortion']
normal_distortion = kmdr_details_df[~kmdr_details_df['celebrity'].isin(controversy_names)]['distortion']

ax.hist(normal_distortion, bins=30, alpha=0.6, label='Normal Players', color='#3498db', edgecolor='black')
ax.hist(controversy_distortion, bins=30, alpha=0.6, label='Controversial Players', color='#e74c3c', edgecolor='black')
ax.set_xlabel('Metric Distortion Value', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Metric Distortion Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

ax = axes[1]
controversy_data = []
controversy_labels = []
for name in controversy_names:
    data = kmdr_details_df[kmdr_details_df['celebrity'] == name]['distortion']
    if len(data) > 0:
        controversy_data.append(data)
        controversy_labels.append(name.split()[0])

bp = ax.boxplot(controversy_data, labels=controversy_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#e74c3c')
    patch.set_alpha(0.6)

ax.legend([bp["boxes"][0]], ['Distortion Metrics'], loc='upper right', fontsize=10)

for i, line in enumerate(bp['medians']):
    x, y = line.get_xydata()[1]
    text = f'{y:.3f}'
    ax.annotate(text, xy=(x, y), xytext=(5, 0), textcoords='offset points', 
                verticalalignment='center', fontsize=9, color='black', fontweight='bold')

ax.set_xlabel('Controversial Player', fontsize=11)
ax.set_ylabel('Metric Distortion Value', fontsize=11)
ax.set_title('Distortion Distribution by Controversial Player', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('q4/fig_distortion_analysis.png', dpi=300, bbox_inches='tight')
print("  保存: q4/fig_distortion_analysis.png")

# =====================================================
# 6. 保存数据
# =====================================================
print("\n【6. 保存数据】")

metrics_df.to_csv('q4/kmdr_performance_metrics.csv', encoding='utf-8-sig')
controversy_df.to_csv('q4/kmdr_controversy_analysis.csv', encoding='utf-8-sig', index=False)
component_stats.to_csv('q4/kmdr_component_statistics.csv', encoding='utf-8-sig')
kmdr_details_df.to_csv('q4/kmdr_detailed_components.csv', encoding='utf-8-sig', index=False)

print("  保存: q4/kmdr_performance_metrics.csv")
print("  保存: q4/kmdr_controversy_analysis.csv")
print("  保存: q4/kmdr_component_statistics.csv")
print("  保存: q4/kmdr_detailed_components.csv")

print("\n" + "=" * 70)
print(" KMDR综合分析完成！")
print("=" * 70)
