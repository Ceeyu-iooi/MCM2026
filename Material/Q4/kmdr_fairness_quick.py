# -*- coding: utf-8 -*-
"""Q4: 快速公平性评估"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("KMDR模型公平性评估")
print("="*60)

# 加载数据
df_judge = pd.read_csv('processed_dwts_full.csv')
df_fan = pd.read_csv('q1/q1_predicted_fan_votes.csv')
df_fan['week_num'] = df_fan['week'].apply(lambda x: int(x.replace('week', '')))
df = df_judge.merge(
    df_fan[['season', 'week_num', 'celebrity_name', 'predicted_fan_vote_score']],
    on=['season', 'week_num', 'celebrity_name'], how='inner'
)
df['judge_pct'] = df['week_percent']
df['fan_pct'] = df['predicted_fan_vote_score']
print(f"数据量: {len(df)}")

# 方法定义
def fixed_weight(g): return 0.5*g['judge_pct'] + 0.5*g['fan_pct']
def pct_method(g):
    jn = (g['judge_pct']-g['judge_pct'].min())/(g['judge_pct'].max()-g['judge_pct'].min()+1e-10)
    fn = (g['fan_pct']-g['fan_pct'].min())/(g['fan_pct'].max()-g['fan_pct'].min()+1e-10)
    return jn + fn
def rank_method(g): return -(stats.rankdata(-g['judge_pct']) + stats.rankdata(-g['fan_pct']))

def kmdr_method(g):
    n = len(g)
    if n < 2: return pd.Series([0]*n, index=g.index)
    week = g['week_num'].iloc[0]
    w_j = 0.75 if week<=3 else (0.60 if week<=6 else 0.50)
    w_f = 1-w_j
    borda_j = n - stats.rankdata(-g['judge_pct'].values)
    borda_f = n - stats.rankdata(-g['fan_pct'].values)
    B = w_j*borda_j + w_f*borda_f
    jr = stats.rankdata(-g['judge_pct'].values)
    fr = stats.rankdata(-g['fan_pct'].values)
    C = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i!=j:
                if (jr[i]<jr[j]) and (fr[i]<fr[j]): C[i]+=2
                elif (jr[i]<jr[j]) or (fr[i]<fr[j]): C[i]+=1
    jn = (g['judge_pct'].values-g['judge_pct'].min())/(g['judge_pct'].max()-g['judge_pct'].min()+1e-10)
    fn = (g['fan_pct'].values-g['fan_pct'].min())/(g['fan_pct'].max()-g['fan_pct'].min()+1e-10)
    D = 0.6*np.abs(jn-fn) + 0.4*np.abs(jr-fr)/n
    return pd.Series(B + 0.6*C - 0.4*n*D, index=g.index)

methods = {'固定权重(50:50)': fixed_weight, '百分比法': pct_method, '排名法': rank_method, 'KMDR': kmdr_method}

# ========================================
# 指标1: 满意度平衡
# ========================================
print("\n【指标1: 满意度平衡】")
satisfaction = {}
for name, fn in methods.items():
    jsats, fsats = [], []
    for (s,w), g in df.groupby(['season','week_num']):
        if len(g)<3: continue
        sc = fn(g)
        fr = stats.rankdata(-sc)
        jr = stats.rankdata(-g['judge_pct'].values)
        fanr = stats.rankdata(-g['fan_pct'].values)
        cj,_ = stats.spearmanr(fr, jr)
        cf,_ = stats.spearmanr(fr, fanr)
        if not np.isnan(cj): jsats.append(cj)
        if not np.isnan(cf): fsats.append(cf)
    j_sat = np.mean(jsats)
    f_sat = np.mean(fsats)
    balance = 1 - abs(j_sat - f_sat)
    satisfaction[name] = {'评委满意度': j_sat, '观众满意度': f_sat, '平衡度': balance}
    print(f"  {name}: 评委={j_sat:.4f}, 观众={f_sat:.4f}, 平衡度={balance:.4f}")

# ========================================
# 指标2: 噪声鲁棒性
# ========================================
print("\n【指标2: 噪声鲁棒性】")
groups = [(k, g) for k, g in df.groupby(['season','week_num']) if len(g)>=3]
np.random.seed(42)
sample_idx = np.random.choice(len(groups), min(50, len(groups)), replace=False)
sampled = [groups[i] for i in sample_idx]

robustness = {}
for name, fn in methods.items():
    changes = []
    for _ in range(10):
        for (s,w), g in sampled:
            orig = stats.rankdata(-fn(g))
            gn = g.copy()
            gn['judge_pct'] = g['judge_pct'] + np.random.normal(0, 0.2*g['judge_pct'].std(), len(g))
            gn['fan_pct'] = g['fan_pct'] + np.random.normal(0, 0.2*g['fan_pct'].std(), len(g))
            noisy = stats.rankdata(-fn(gn))
            changes.append(np.mean(orig!=noisy))
    rob = 1 - np.mean(changes)
    robustness[name] = rob
    print(f"  {name}: 鲁棒性={rob:.4f}")

# ========================================
# 指标3: Condorcet效率
# ========================================
print("\n【指标3: Condorcet效率】")
condorcet = {}
for name, fn in methods.items():
    total, correct = 0, 0
    for (s,w), g in df.groupby(['season','week_num']):
        if len(g)<3: continue
        jr = stats.rankdata(-g['judge_pct'].values)
        fr = stats.rankdata(-g['fan_pct'].values)
        n = len(g)
        winner = None
        for i in range(n):
            is_w = True
            for j in range(n):
                if i==j: continue
                votes = (1 if jr[i]<jr[j] else 0) + (1 if fr[i]<fr[j] else 0)
                if votes < 1: is_w=False; break
            if is_w: winner=g.index[i]; break
        if winner:
            total += 1
            sc = fn(g)
            if isinstance(sc, pd.Series):
                method_winner = sc.idxmax()
            else:
                method_winner = g.index[np.argmax(sc)]
            if method_winner == winner: correct += 1
    eff = correct/max(total,1)
    condorcet[name] = {'效率': eff, '正确': correct, '总数': total}
    print(f"  {name}: 效率={eff:.4f} ({correct}/{total})")

# ========================================
# 指标4: 分歧惩罚有效性
# ========================================
print("\n【指标4: 分歧惩罚有效性】")
high_ch, low_ch, all_d, all_c = [], [], [], []
for (s,w), g in df.groupby(['season','week_num']):
    if len(g)<3: continue
    n = len(g)
    jn = (g['judge_pct'].values-g['judge_pct'].min())/(g['judge_pct'].max()-g['judge_pct'].min()+1e-10)
    fn_v = (g['fan_pct'].values-g['fan_pct'].min())/(g['fan_pct'].max()-g['fan_pct'].min()+1e-10)
    jr = stats.rankdata(-g['judge_pct'].values)
    fr = stats.rankdata(-g['fan_pct'].values)
    D = 0.6*np.abs(jn-fn_v) + 0.4*np.abs(jr-fr)/n
    kmdr_r = stats.rankdata(-kmdr_method(g))
    base_r = stats.rankdata(-fixed_weight(g))
    thresh = np.percentile(D, 75)
    for d, kr, br in zip(D, kmdr_r, base_r):
        all_d.append(d)
        all_c.append(kr-br)
        if d >= thresh: high_ch.append(kr-br)
        else: low_ch.append(kr-br)

corr, pval = stats.spearmanr(all_d, all_c)
print(f"  高分歧选手平均排名变化: {np.mean(high_ch):+.4f}")
print(f"  低分歧选手平均排名变化: {np.mean(low_ch):+.4f}")
print(f"  惩罚有效性(差值): {np.mean(high_ch)-np.mean(low_ch):+.4f}")
print(f"  分歧-排名变化相关系数: {corr:.4f} (p={pval:.2e})")

# ========================================
# 保存结果
# ========================================
print("\n【保存结果】")
results_df = pd.DataFrame({
    '方法': list(methods.keys()),
    '满意度平衡': [satisfaction[m]['平衡度'] for m in methods.keys()],
    '噪声鲁棒性': [robustness[m] for m in methods.keys()],
    'Condorcet效率': [condorcet[m]['效率'] for m in methods.keys()]
})
results_df.to_csv('q4/kmdr_fairness_metrics.csv', index=False, encoding='utf-8-sig')
print("  ✓ q4/kmdr_fairness_metrics.csv")

penalty_df = pd.DataFrame([{
    '高分歧选手数': len(high_ch),
    '低分歧选手数': len(low_ch),
    '高分歧平均排名变化': np.mean(high_ch),
    '低分歧平均排名变化': np.mean(low_ch),
    '惩罚有效性差值': np.mean(high_ch)-np.mean(low_ch),
    '分歧排名相关系数': corr,
    'p值': pval
}])
penalty_df.to_csv('q4/kmdr_penalty_effectiveness.csv', index=False, encoding='utf-8-sig')
print("  ✓ q4/kmdr_penalty_effectiveness.csv")

# ========================================
# 可视化
# ========================================
print("\n【生成图表】")

# Fig1: Fairness Metrics Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
method_names = list(methods.keys())

# Left: Three metrics bar comparison
ax1 = axes[0]
x = np.arange(3)
width = 0.20
for i, name in enumerate(method_names):
    vals = [satisfaction[name]['平衡度'], robustness[name], condorcet[name]['效率']]
    ax1.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
ax1.set_xlabel('Evaluation Metrics', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Fairness Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + 1.5*width)
ax1.set_xticklabels(['Satisfaction Balance', 'Noise Robustness', 'Condorcet Efficiency'])
ax1.legend(loc='lower right')
ax1.set_ylim(0, 1.1)

# Right: Satisfaction Balance Details
ax2 = axes[1]
x2 = np.arange(len(methods))
width2 = 0.35
j_sats = [satisfaction[m]['评委满意度'] for m in method_names]
f_sats = [satisfaction[m]['观众满意度'] for m in method_names]
ax2.bar(x2 - width2/2, j_sats, width2, label='Judge Satisfaction', color='#3498db', alpha=0.85)
ax2.bar(x2 + width2/2, f_sats, width2, label='Fan Satisfaction', color='#e74c3c', alpha=0.85)
ax2.set_xlabel('Method', fontsize=12)
ax2.set_ylabel('Satisfaction', fontsize=12)
ax2.set_title('Judge vs Fan Satisfaction', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels([n.replace('(50:50)','').replace('固定权重','Fixed').replace('排名法','Rank').replace('百分比法','Percent') for n in method_names], rotation=15)
ax2.legend()
ax2.set_ylim(0.9, 1.0)

plt.tight_layout()
plt.savefig('q4/fig_fairness_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig_fairness_metrics.png")

# Fig2: Distortion Penalty Effect
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_d, all_c, alpha=0.3, s=20, c='#3498db')
z = np.polyfit(all_d, all_c, 1)
p = np.poly1d(z)
x_line = np.linspace(min(all_d), max(all_d), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend Line (slope={z[0]:.2f})')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Metric Distortion (Judge-Fan Disagreement)', fontsize=12)
ax.set_ylabel('Rank Change (KMDR Rank - Fixed Weight Rank)', fontsize=12)
ax.set_title('Disagreement Penalty Effectiveness', fontsize=14, fontweight='bold')
ax.legend()
ax.text(0.02, 0.98, f'Spearman corr: {corr:.4f}\nPositive=Rank drops (penalized)', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
plt.savefig('q4/fig_penalty_effectiveness.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig_penalty_effectiveness.png")

print("\n" + "="*60)
print("公平性评估完成！")
print("="*60)
print(f"\n核心发现:")
print(f"  1. 满意度平衡: KMDR={satisfaction['KMDR']['平衡度']:.4f} (百分比法={satisfaction['百分比法']['平衡度']:.4f})")
print(f"  2. 噪声鲁棒性: KMDR={robustness['KMDR']:.4f} (百分比法={robustness['百分比法']:.4f})")
print(f"  3. Condorcet效率: KMDR={condorcet['KMDR']['效率']:.4f} (百分比法={condorcet['百分比法']['效率']:.4f})")
print(f"  4. 分歧惩罚有效性: 相关系数={corr:.4f}, 差值={np.mean(high_ch)-np.mean(low_ch):+.4f}")
