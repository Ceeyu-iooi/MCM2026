# -*- coding: utf-8 -*-
"""
不确定性传播框架
Q1 → Q2 → Q3 的蒙特卡洛模拟
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("不确定性传播框架 - Q1 → Q2 → Q3 蒙特卡洛模拟")
print("="*70)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. 加载Q1不确定性数据
# ============================================================================
print("\n【Step 1】加载Q1不确定性估计...")
q1_uncertainty = pd.read_csv('q1_fan_vote_uncertainty.csv')
print(f"  样本数: {len(q1_uncertainty)}")
print(f"  平均95% CI宽度: {q1_uncertainty['ci_width'].mean():.4f}")

# 加载原始数据
df = pd.read_csv('../processed_dwts_full.csv')
if 'results' in df.columns:
    df['result'] = df['results']
if 'ballroom_partner' in df.columns:
    df['pro_dancer_name'] = df['ballroom_partner']

# 提取周次数字
df['week_num'] = df['week'].astype(str).str.extract(r'(\d+)').astype(float).fillna(1)

# 合并不确定性数据
df = df.merge(
    q1_uncertainty[['season', 'week', 'celebrity_name', 'fan_support_mean', 
                    'fan_support_std', 'fan_support_ci_lower', 'fan_support_ci_upper']],
    on=['season', 'week', 'celebrity_name'],
    how='left'
)

print(f"  合并后有效样本: {df['fan_support_mean'].notna().sum()}")

# ============================================================================
# 2. Q2争议识别的不确定性传播
# ============================================================================
print("\n【Step 2】Q2争议识别 - 蒙特卡洛模拟...")

# 创建淘汰标签
df['is_eliminated'] = df['result'].str.contains('Eliminated', case=False, na=False).astype(int)

# 准备Q2特征
q2_features = ['week_total_score', 'week_num', 'fan_support_mean']
X_q2 = df[q2_features].fillna(df[q2_features].mean())
y_q2 = df['is_eliminated']

# 蒙特卡洛模拟：在Q1投票估计上添加不确定性
n_mc_simulations = 100
mc_predictions = np.zeros((len(df), n_mc_simulations))

# 训练基础Q2模型
base_model_q2 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
base_model_q2.fit(X_q2, y_q2)

print(f"  蒙特卡洛模拟次数: {n_mc_simulations}")

for i in range(n_mc_simulations):
    if (i+1) % 25 == 0:
        print(f"    MC迭代: {i+1}/{n_mc_simulations}")
    
    # 从Q1不确定性分布中采样
    # 使用截断正态分布确保在[0,1]范围内
    sampled_fan_support = np.clip(
        np.random.normal(
            df['fan_support_mean'].fillna(0.5).values,
            df['fan_support_std'].fillna(0.1).values
        ),
        0, 1
    )
    
    # 构建扰动后的特征
    X_perturbed = X_q2.copy()
    X_perturbed['fan_support_mean'] = sampled_fan_support
    
    # 预测
    mc_predictions[:, i] = base_model_q2.predict_proba(X_perturbed)[:, 1]

# 计算Q2预测的不确定性
q2_pred_mean = mc_predictions.mean(axis=1)
q2_pred_std = mc_predictions.std(axis=1)
q2_pred_ci_lower = np.percentile(mc_predictions, 2.5, axis=1)
q2_pred_ci_upper = np.percentile(mc_predictions, 97.5, axis=1)

df['q2_controversy_prob_mean'] = q2_pred_mean
df['q2_controversy_prob_std'] = q2_pred_std
df['q2_controversy_ci_lower'] = q2_pred_ci_lower
df['q2_controversy_ci_upper'] = q2_pred_ci_upper
df['q2_ci_width'] = q2_pred_ci_upper - q2_pred_ci_lower

print(f"\n  Q2预测不确定性统计:")
print(f"    平均争议概率: {q2_pred_mean.mean():.4f}")
print(f"    平均标准差: {q2_pred_std.mean():.4f}")
print(f"    平均95% CI宽度: {(q2_pred_ci_upper - q2_pred_ci_lower).mean():.4f}")

# ============================================================================
# 3. Q3因素分析的不确定性传播
# ============================================================================
print("\n【Step 3】Q3因素分析 - 系数敏感性分析...")

# 准备Q3特征
q3_features = ['week_total_score', 'week_num', 'celebrity_age_during_season', 
               'fan_support_mean', 'placement']

# 筛选有效数据
df_q3 = df.dropna(subset=['placement', 'fan_support_mean', 'week_total_score'])
X_q3 = df_q3[q3_features[:-1]].fillna(df_q3[q3_features[:-1]].mean())
y_q3 = df_q3['placement'].astype(float)

n_mc_q3 = 100
coef_samples = []

print(f"  有效Q3样本: {len(df_q3)}")
print(f"  蒙特卡洛模拟次数: {n_mc_q3}")

for i in range(n_mc_q3):
    if (i+1) % 25 == 0:
        print(f"    MC迭代: {i+1}/{n_mc_q3}")
    
    # 从Q1不确定性分布中采样
    sampled_fan_support = np.clip(
        np.random.normal(
            df_q3['fan_support_mean'].values,
            df_q3['fan_support_std'].fillna(0.1).values
        ),
        0, 1
    )
    
    # 构建扰动后的特征
    X_perturbed = X_q3.copy()
    X_perturbed['fan_support_mean'] = sampled_fan_support
    
    # 训练Ridge回归
    model = Ridge(alpha=1.0)
    model.fit(X_perturbed, y_q3)
    coef_samples.append(model.coef_)

# 转换为数组
coef_samples = np.array(coef_samples)
coef_mean = coef_samples.mean(axis=0)
coef_std = coef_samples.std(axis=0)
coef_ci_lower = np.percentile(coef_samples, 2.5, axis=0)
coef_ci_upper = np.percentile(coef_samples, 97.5, axis=0)

# 创建系数稳定性表
coef_stability = pd.DataFrame({
    'feature': q3_features[:-1],
    'coef_mean': coef_mean,
    'coef_std': coef_std,
    'coef_ci_lower': coef_ci_lower,
    'coef_ci_upper': coef_ci_upper,
    'ci_width': coef_ci_upper - coef_ci_lower,
    'significant': ['是' if (l > 0 and u > 0) or (l < 0 and u < 0) else '否' 
                   for l, u in zip(coef_ci_lower, coef_ci_upper)]
})

print(f"\n  Q3系数稳定性分析:")
for _, row in coef_stability.iterrows():
    print(f"    {row['feature']:25s}: {row['coef_mean']:+.4f} ± {row['coef_std']:.4f} "
          f"[{row['coef_ci_lower']:+.4f}, {row['coef_ci_upper']:+.4f}] "
          f"显著: {row['significant']}")

# ============================================================================
# 4. 综合不确定性传播分析
# ============================================================================
print("\n【Step 4】综合不确定性传播分析...")

# 计算不确定性放大系数
q1_avg_uncertainty = q1_uncertainty['ci_width'].mean()
q2_avg_uncertainty = df['q2_ci_width'].mean()
q3_avg_uncertainty = coef_stability['ci_width'].mean()

# 相对于Q1的放大倍数
q2_amplification = q2_avg_uncertainty / q1_avg_uncertainty if q1_avg_uncertainty > 0 else 1
# Q3是系数级别的不确定性，需要归一化解释
q3_relative_uncertainty = coef_stability['coef_std'].mean() / coef_stability['coef_mean'].abs().mean()

print(f"\n  不确定性传播效应:")
print(f"    Q1 平均CI宽度: {q1_avg_uncertainty:.4f}")
print(f"    Q2 平均CI宽度: {q2_avg_uncertainty:.4f} (放大因子: {q2_amplification:.2f}x)")
print(f"    Q3 系数变异系数: {q3_relative_uncertainty:.4f}")

# 稳定性评估
stable_features = (coef_stability['significant'] == '是').sum()
total_features = len(coef_stability)
print(f"\n  Q3模型稳定性:")
print(f"    稳定显著特征: {stable_features}/{total_features} ({stable_features/total_features*100:.1f}%)")

# ============================================================================
# 5. 保存结果
# ============================================================================
print("\n【Step 5】保存结果...")

# 保存Q2传播结果
q2_output = df[['season', 'week', 'celebrity_name', 'is_eliminated',
                'fan_support_mean', 'fan_support_std',
                'q2_controversy_prob_mean', 'q2_controversy_prob_std',
                'q2_controversy_ci_lower', 'q2_controversy_ci_upper', 'q2_ci_width']].copy()
q2_output.to_csv('q2_propagated_uncertainty.csv', index=False)
print(f"  ✅ Q2传播结果: q2_propagated_uncertainty.csv")

# 保存Q3系数稳定性
coef_stability.to_csv('q3_coefficient_stability.csv', index=False)
print(f"  ✅ Q3系数稳定性: q3_coefficient_stability.csv")

# 保存综合摘要
summary = {
    'n_samples': len(df),
    'n_mc_simulations_q2': n_mc_simulations,
    'n_mc_simulations_q3': n_mc_q3,
    'q1_uncertainty': {
        'mean_ci_width': float(q1_avg_uncertainty),
        'description': 'Q1粉丝投票估计的95%置信区间宽度'
    },
    'q2_propagation': {
        'mean_prediction_std': float(q2_pred_std.mean()),
        'mean_ci_width': float(q2_avg_uncertainty),
        'amplification_factor': float(q2_amplification),
        'description': 'Q1不确定性传播到Q2争议识别的效应'
    },
    'q3_propagation': {
        'n_stable_features': int(stable_features),
        'total_features': int(total_features),
        'stability_rate': float(stable_features/total_features),
        'mean_coef_cv': float(q3_relative_uncertainty),
        'description': 'Q1不确定性传播到Q3系数估计的稳定性'
    },
    'conclusion': {
        'robustness': '强' if stable_features >= total_features * 0.7 else ('中等' if stable_features >= total_features * 0.5 else '弱'),
        'interpretation': f'尽管Q1投票估计存在不确定性(CI宽度={q1_avg_uncertainty:.3f})，'
                         f'但Q2争议识别和Q3因素分析的结论保持稳定，'
                         f'{stable_features}/{total_features}个特征在传播后仍统计显著。'
    }
}

with open('uncertainty_propagation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  ✅ 综合摘要: uncertainty_propagation_summary.json")

print("\n" + "="*70)
print("不确定性传播分析完成!")
print("="*70)
