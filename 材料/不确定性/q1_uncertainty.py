# -*- coding: utf-8 -*-
"""
Q1 不确定性估计模块
为粉丝投票估计生成Bootstrap置信区间
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Q1 不确定性估计模块 - Bootstrap置信区间")
print("="*70)

# 切换到q1目录加载模型
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载Q1的预测结果
q1_predictions = pd.read_csv('../q1/q1_tan_predicted_fan_votes.csv')
print(f"\n加载Q1预测结果: {len(q1_predictions)} 条记录")

# 加载处理后的数据
df = pd.read_csv('../processed_dwts_full.csv')

# 统一列名 results -> result
if 'results' in df.columns and 'result' not in df.columns:
    df['result'] = df['results']

# 创建二分类标签：是否被淘汰（Eliminated开头的都是被淘汰）
df['is_eliminated'] = df['result'].str.contains('Eliminated', case=False, na=False).astype(int)
print(f"淘汰比例: {df['is_eliminated'].mean()*100:.1f}%")

# 统一舞者列名
if 'ballroom_partner' in df.columns and 'pro_dancer_name' not in df.columns:
    df['pro_dancer_name'] = df['ballroom_partner']

# 提取周次数字
if 'week' in df.columns:
    df['week_num'] = df['week'].astype(str).str.extract(r'(\d+)').astype(float).fillna(1)

print(f"数据列: {df.columns.tolist()[:15]}...")

# 定义特征列（与Q1模型一致）
feature_cols = [
    'week_total_score', 'avg_score_before', 'score_std',
    'weeks_participated', 'elimination_risk', 'relative_performance',
    'cumulative_rank', 'rank_improvement', 'is_bottom_3'
]

# 确保特征存在
available_features = [f for f in feature_cols if f in df.columns]
print(f"可用特征: {len(available_features)}/{len(feature_cols)}")

# 补充缺失特征
if 'avg_score_before' not in df.columns:
    df['avg_score_before'] = df.groupby(['season', 'celebrity_name'])['week_total_score'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0])
    )

if 'score_std' not in df.columns:
    df['score_std'] = df.groupby(['season', 'celebrity_name'])['week_total_score'].transform(
        lambda x: x.expanding().std().shift(1).fillna(0)
    )

if 'weeks_participated' not in df.columns:
    df['weeks_participated'] = df.groupby(['season', 'celebrity_name']).cumcount() + 1

if 'elimination_risk' not in df.columns:
    df['elimination_risk'] = df.groupby(['season', 'week'])['week_total_score'].transform(
        lambda x: (x.max() - x) / (x.max() - x.min() + 1e-6)
    )

if 'relative_performance' not in df.columns:
    df['relative_performance'] = df.groupby(['season', 'week'])['week_total_score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

if 'cumulative_rank' not in df.columns:
    df['cumulative_rank'] = df.groupby(['season', 'week'])['week_total_score'].rank(ascending=False)

if 'rank_improvement' not in df.columns:
    df['rank_improvement'] = df.groupby(['season', 'celebrity_name'])['cumulative_rank'].transform(
        lambda x: x.shift(1) - x
    ).fillna(0)

if 'is_bottom_3' not in df.columns:
    df['is_bottom_3'] = (df['cumulative_rank'] >= df.groupby(['season', 'week'])['cumulative_rank'].transform('max') - 2).astype(int)

# 更新可用特征列表
available_features = [f for f in feature_cols if f in df.columns]

# 准备Bootstrap分析的数据
X = df[available_features].fillna(0)
y = df['is_eliminated']

print(f"\n特征矩阵: {X.shape}")
print(f"淘汰样本: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")

# ============================================================================
# Bootstrap置信区间估计
# ============================================================================
print("\n" + "-"*70)
print("Bootstrap不确定性估计 (n=200)")
print("-"*70)

n_bootstrap = 200
n_samples = len(X)

# 存储每次bootstrap的预测概率
bootstrap_probs = np.zeros((n_samples, n_bootstrap))

for i in range(n_bootstrap):
    if (i+1) % 50 == 0:
        print(f"  Bootstrap迭代: {i+1}/{n_bootstrap}")
    
    # 有放回抽样
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=i,
        n_jobs=-1
    )
    model.fit(X_boot, y_boot)
    
    # 预测所有样本的概率
    probs = model.predict_proba(X)[:, 1]  # 淘汰概率
    bootstrap_probs[:, i] = probs

# 计算统计量
prob_mean = bootstrap_probs.mean(axis=1)
prob_std = bootstrap_probs.std(axis=1)
prob_ci_lower = np.percentile(bootstrap_probs, 2.5, axis=1)
prob_ci_upper = np.percentile(bootstrap_probs, 97.5, axis=1)

# 将粉丝投票估计转换为置信区间
# 粉丝支持度 = 1 - 淘汰概率（高支持度=低淘汰风险）
fan_support_mean = 1 - prob_mean
fan_support_std = prob_std
fan_support_ci_lower = 1 - prob_ci_upper  # 注意反转
fan_support_ci_upper = 1 - prob_ci_lower

# 创建结果DataFrame
uncertainty_df = df[['season', 'week', 'celebrity_name', 'pro_dancer_name', 'result', 'is_eliminated']].copy()
uncertainty_df['fan_support_mean'] = fan_support_mean
uncertainty_df['fan_support_std'] = fan_support_std
uncertainty_df['fan_support_ci_lower'] = fan_support_ci_lower
uncertainty_df['fan_support_ci_upper'] = fan_support_ci_upper
uncertainty_df['ci_width'] = fan_support_ci_upper - fan_support_ci_lower
uncertainty_df['elimination_prob_mean'] = prob_mean
uncertainty_df['elimination_prob_std'] = prob_std

# ============================================================================
# 不确定性分析
# ============================================================================
print("\n" + "-"*70)
print("不确定性分析结果")
print("-"*70)

print(f"\n粉丝支持度估计:")
print(f"  均值范围: [{fan_support_mean.min():.3f}, {fan_support_mean.max():.3f}]")
print(f"  平均标准差: {fan_support_std.mean():.4f}")
print(f"  平均95% CI宽度: {(fan_support_ci_upper - fan_support_ci_lower).mean():.4f}")

# 按淘汰状态分组
eliminated = uncertainty_df[uncertainty_df['is_eliminated'] == 1]
survived = uncertainty_df[uncertainty_df['is_eliminated'] == 0]

print(f"\n淘汰选手 vs 存活选手:")
print(f"  淘汰者平均支持度: {eliminated['fan_support_mean'].mean():.3f} ± {eliminated['fan_support_std'].mean():.4f}")
print(f"  存活者平均支持度: {survived['fan_support_mean'].mean():.3f} ± {survived['fan_support_std'].mean():.4f}")
print(f"  差异: {survived['fan_support_mean'].mean() - eliminated['fan_support_mean'].mean():.3f}")

# 高不确定性样本分析
high_uncertainty = uncertainty_df[uncertainty_df['ci_width'] > uncertainty_df['ci_width'].quantile(0.9)]
print(f"\n高不确定性样本 (CI宽度 > 90%分位):")
print(f"  数量: {len(high_uncertainty)} ({len(high_uncertainty)/len(uncertainty_df)*100:.1f}%)")
print(f"  平均CI宽度: {high_uncertainty['ci_width'].mean():.4f}")

# 按赛季分析不确定性趋势
season_uncertainty = uncertainty_df.groupby('season').agg({
    'fan_support_std': 'mean',
    'ci_width': 'mean'
}).reset_index()

print(f"\n赛季不确定性趋势:")
early_seasons = season_uncertainty[season_uncertainty['season'] <= 10]['ci_width'].mean()
late_seasons = season_uncertainty[season_uncertainty['season'] > 25]['ci_width'].mean()
print(f"  早期赛季(1-10) 平均CI宽度: {early_seasons:.4f}")
print(f"  近期赛季(26+) 平均CI宽度: {late_seasons:.4f}")

# 保存结果
output_path = 'q1_fan_vote_uncertainty.csv'
uncertainty_df.to_csv(output_path, index=False)
print(f"\n✅ 不确定性估计已保存: {output_path}")

# 保存摘要统计
summary = {
    'n_samples': len(uncertainty_df),
    'n_bootstrap': n_bootstrap,
    'mean_fan_support': float(fan_support_mean.mean()),
    'mean_uncertainty_std': float(fan_support_std.mean()),
    'mean_ci_width': float((fan_support_ci_upper - fan_support_ci_lower).mean()),
    'eliminated_mean_support': float(eliminated['fan_support_mean'].mean()),
    'survived_mean_support': float(survived['fan_support_mean'].mean()),
    'high_uncertainty_count': int(len(high_uncertainty)),
    'high_uncertainty_pct': float(len(high_uncertainty)/len(uncertainty_df)*100)
}

import json
with open('q1_uncertainty_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ 摘要统计已保存: q1_uncertainty_summary.json")
print("\n" + "="*70)
print("Q1不确定性估计完成!")
print("="*70)
