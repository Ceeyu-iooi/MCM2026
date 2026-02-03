# -*- coding: utf-8 -*-
"""
Q1 模型检验与改进阶段
包含：有效性检验、鲁棒性分析、优缺点评价

MCM 2026 Problem C - Dancing with the Stars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

print("=" * 70)
print("Q1 模型检验与改进阶段")
print("=" * 70)

# 获取当前目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# ==================== 数据加载 ====================
df = pd.read_csv('D:\pydocument\processed_dwts_cleaned_ms08.csv')
print(f"\n数据规模: {len(df)} 条记录, {df['season'].nunique()} 个赛季")

# 特征选择
feature_cols = [
    'week_total_score', 'week_rank', 'week_percent',
    'celebrity_age_during_season', 'judge_score_std',
    'age_zscore', 'remaining_weeks', 'cumulative_rank',
    'elimination_risk', 'weeks_participated', 'relative_performance'
]

industry_map = {
    'Athlete': 1, 'Model': 2, 'Actor': 3, 'Actress': 3,
    'Singer': 4, 'TV Personality': 5, 'Musician': 4
}
df['industry_code'] = df['celebrity_industry'].map(
    lambda x: industry_map.get(x, 6) if pd.notna(x) else 6
)
feature_cols.append('industry_code')

X = df[feature_cols].copy().fillna(df[feature_cols].median())
y = df['is_eliminated'].values

print(f"特征数量: {len(feature_cols)}")
print(f"正类(淘汰)比例: {y.mean()*100:.2f}%")

# ==================== 1. 有效性检验 ====================
print("\n" + "=" * 70)
print("一、有效性检验")
print("=" * 70)

# 1.1 10折交叉验证
print("\n[1.1] 10折交叉验证")
print("-" * 50)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 随机森林模型
rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=10,
    min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=1
)

# 高斯朴素贝叶斯(作为TAN的近似)
gnb_clf = GaussianNB()

# 交叉验证指标
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = {'RF': {}, 'GNB': {}}

print("\n随机森林 10折交叉验证结果:")
for metric in metrics:
    if metric == 'precision':
        scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='precision')
    elif metric == 'recall':
        scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='recall')
    elif metric == 'f1':
        scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='f1')
    elif metric == 'roc_auc':
        scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='roc_auc')
    else:
        scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='accuracy')
    
    cv_results['RF'][metric] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print("\n高斯朴素贝叶斯 10折交叉验证结果:")
for metric in metrics:
    if metric == 'precision':
        scores = cross_val_score(gnb_clf, X, y, cv=cv, scoring='precision')
    elif metric == 'recall':
        scores = cross_val_score(gnb_clf, X, y, cv=cv, scoring='recall')
    elif metric == 'f1':
        scores = cross_val_score(gnb_clf, X, y, cv=cv, scoring='f1')
    elif metric == 'roc_auc':
        scores = cross_val_score(gnb_clf, X, y, cv=cv, scoring='roc_auc')
    else:
        scores = cross_val_score(gnb_clf, X, y, cv=cv, scoring='accuracy')
    
    cv_results['GNB'][metric] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# 1.2 学习曲线分析
print("\n[1.2] 学习曲线分析")
print("-" * 50)

train_sizes, train_scores, test_scores = learning_curve(
    rf_clf, X, y, cv=5, n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print(f"训练集最终准确率: {train_mean[-1]:.4f}")
print(f"测试集最终准确率: {test_mean[-1]:.4f}")
print(f"过拟合差距: {train_mean[-1] - test_mean[-1]:.4f}")

if train_mean[-1] - test_mean[-1] < 0.05:
    overfit_status = "无明显过拟合"
else:
    overfit_status = "存在轻微过拟合"
print(f"过拟合判断: {overfit_status}")

# 1.3 残差分析(对于分类问题，分析预测概率与实际结果的偏差)
print("\n[1.3] 预测概率残差分析")
print("-" * 50)

rf_clf.fit(X, y)
y_prob = rf_clf.predict_proba(X)[:, 1]
residuals = y - y_prob

print(f"残差均值: {residuals.mean():.6f}")
print(f"残差标准差: {residuals.std():.4f}")
print(f"残差偏度: {stats.skew(residuals):.4f}")
print(f"残差峰度: {stats.kurtosis(residuals):.4f}")

# Shapiro-Wilk正态性检验(取样本)
sample_residuals = np.random.choice(residuals, size=min(5000, len(residuals)), replace=False)
shapiro_stat, shapiro_p = stats.shapiro(sample_residuals[:500])
print(f"Shapiro-Wilk正态性检验: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

if shapiro_p > 0.05:
    normality = "残差近似正态分布"
else:
    normality = "残差呈非正态分布(二分类问题的预期特征)"

print(f"正态性判断: {normality}")

# ==================== 2. 鲁棒性分析 ====================
print("\n" + "=" * 70)
print("二、鲁棒性分析")
print("=" * 70)

# 2.1 噪声干扰测试
print("\n[2.1] 噪声干扰测试")
print("-" * 50)

noise_levels = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
robustness_results = []

# 基准性能
train_mask = df['season'] <= 27
X_train_base = X[train_mask].values
y_train_base = y[train_mask]
X_test_base = X[~train_mask].values
y_test_base = y[~train_mask]

rf_clf.fit(X_train_base, y_train_base)
base_acc = accuracy_score(y_test_base, rf_clf.predict(X_test_base))
base_auc = roc_auc_score(y_test_base, rf_clf.predict_proba(X_test_base)[:, 1])
base_f1 = f1_score(y_test_base, rf_clf.predict(X_test_base))

print(f"基准性能: Accuracy={base_acc:.4f}, AUC={base_auc:.4f}, F1={base_f1:.4f}")

print(f"\n{'噪声水平':^10} {'Accuracy':^12} {'变化':^10} {'AUC':^12} {'变化':^10} {'F1':^12} {'变化':^10}")
print("-" * 80)

for noise in noise_levels:
    # 添加高斯噪声
    X_train_noisy = X_train_base + np.random.normal(0, noise * X_train_base.std(axis=0), X_train_base.shape)
    X_test_noisy = X_test_base + np.random.normal(0, noise * X_test_base.std(axis=0), X_test_base.shape)
    
    rf_clf.fit(X_train_noisy, y_train_base)
    noisy_acc = accuracy_score(y_test_base, rf_clf.predict(X_test_noisy))
    noisy_auc = roc_auc_score(y_test_base, rf_clf.predict_proba(X_test_noisy)[:, 1])
    noisy_f1 = f1_score(y_test_base, rf_clf.predict(X_test_noisy))
    
    acc_change = noisy_acc - base_acc
    auc_change = noisy_auc - base_auc
    f1_change = noisy_f1 - base_f1
    
    robustness_results.append({
        'noise_level': noise,
        'accuracy': noisy_acc,
        'acc_change': acc_change,
        'auc': noisy_auc,
        'auc_change': auc_change,
        'f1': noisy_f1,
        'f1_change': f1_change
    })
    
    print(f"{noise*100:^10.0f}% {noisy_acc:^12.4f} {acc_change:^+10.4f} {noisy_auc:^12.4f} {auc_change:^+10.4f} {noisy_f1:^12.4f} {f1_change:^+10.4f}")

robustness_df = pd.DataFrame(robustness_results)

# 2.2 数据划分比例敏感性
print("\n[2.2] 数据划分比例敏感性测试")
print("-" * 50)

split_ratios = [0.6, 0.7, 0.8, 0.9]
split_results = []

print(f"\n{'训练比例':^10} {'Accuracy':^12} {'AUC':^12} {'F1':^12}")
print("-" * 50)

for ratio in split_ratios:
    n_train = int(len(X) * ratio)
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train_split = X.values[train_idx]
    y_train_split = y[train_idx]
    X_test_split = X.values[test_idx]
    y_test_split = y[test_idx]
    
    rf_clf.fit(X_train_split, y_train_split)
    split_acc = accuracy_score(y_test_split, rf_clf.predict(X_test_split))
    split_auc = roc_auc_score(y_test_split, rf_clf.predict_proba(X_test_split)[:, 1])
    split_f1 = f1_score(y_test_split, rf_clf.predict(X_test_split))
    
    split_results.append({
        'ratio': ratio,
        'accuracy': split_acc,
        'auc': split_auc,
        'f1': split_f1
    })
    
    print(f"{ratio*100:^10.0f}% {split_acc:^12.4f} {split_auc:^12.4f} {split_f1:^12.4f}")

split_df = pd.DataFrame(split_results)
print(f"\n准确率标准差: {split_df['accuracy'].std():.4f}")
print(f"AUC标准差: {split_df['auc'].std():.4f}")

# 2.3 特征子集敏感性
print("\n[2.3] 特征子集敏感性测试")
print("-" * 50)

feature_subsets = [
    ('Top3特征', ['weeks_participated', 'remaining_weeks', 'cumulative_rank']),
    ('Top6特征', ['weeks_participated', 'remaining_weeks', 'cumulative_rank', 
                  'relative_performance', 'elimination_risk', 'week_percent']),
    ('Top9特征', ['weeks_participated', 'remaining_weeks', 'cumulative_rank',
                  'relative_performance', 'elimination_risk', 'week_percent',
                  'week_total_score', 'week_rank', 'age_zscore']),
    ('全部12特征', feature_cols)
]

print(f"\n{'特征集':^15} {'Accuracy':^12} {'AUC':^12} {'F1':^12}")
print("-" * 55)

for name, features in feature_subsets:
    X_subset = df[features].copy().fillna(df[features].median())
    
    X_train_sub = X_subset[train_mask].values
    X_test_sub = X_subset[~train_mask].values
    
    rf_clf.fit(X_train_sub, y_train_base)
    sub_acc = accuracy_score(y_test_base, rf_clf.predict(X_test_sub))
    sub_auc = roc_auc_score(y_test_base, rf_clf.predict_proba(X_test_sub)[:, 1])
    sub_f1 = f1_score(y_test_base, rf_clf.predict(X_test_sub))
    
    print(f"{name:^15} {sub_acc:^12.4f} {sub_auc:^12.4f} {sub_f1:^12.4f}")

# ==================== 3. 生成检验可视化图表 ====================
print("\n" + "=" * 70)
print("三、生成检验可视化图表")
print("=" * 70)

# 图1: 10折交叉验证结果（均值±标准差折线图）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))

ax1 = axes[0]
rf_means = np.array([cv_results['RF'][m]['mean'] for m in metrics])
rf_stds = np.array([cv_results['RF'][m]['std'] for m in metrics])
rf_upper = rf_means + rf_stds
rf_lower = rf_means - rf_stds
ax1.plot(x, rf_upper, color='#8ecae6', lw=2, label='Mean + Std')
ax1.plot(x, rf_lower, color='#f4a261', lw=2, label='Mean - Std')
ax1.plot(x, rf_means, color='black', lw=2, label='Mean')
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels)
ax1.set_ylabel('Score')
ax1.set_title('Random Forest: 10-Fold CV (Mean ± Std)')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1])
ax1.legend(frameon=False, fontsize=9, loc='upper right')

ax2 = axes[1]
gnb_means = np.array([cv_results['GNB'][m]['mean'] for m in metrics])
gnb_stds = np.array([cv_results['GNB'][m]['std'] for m in metrics])
gnb_upper = gnb_means + gnb_stds
gnb_lower = gnb_means - gnb_stds
ax2.plot(x, gnb_upper, color='#8ecae6', lw=2, label='Mean + Std')
ax2.plot(x, gnb_lower, color='#f4a261', lw=2, label='Mean - Std')
ax2.plot(x, gnb_means, color='black', lw=2, label='Mean')
ax2.set_xticks(x)
ax2.set_xticklabels(metric_labels)
ax2.set_ylabel('Score')
ax2.set_title('Gaussian Naive Bayes: 10-Fold CV (Mean ± Std)')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])
ax2.legend(frameon=False, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig_cv_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [1/4] fig_cv_boxplot.png")

# 图2: 学习曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Validation Score')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve: Random Forest')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
ax.set_ylim([0.7, 1.0])
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig_learning_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [2/4] fig_learning_curve.png")

# 图3: 鲁棒性分析-噪声影响
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
ax1.plot([0] + list(robustness_df['noise_level']*100), 
         [base_acc] + list(robustness_df['accuracy']), 'o-', color='blue', linewidth=2, markersize=8)
ax1.axhline(y=base_acc, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax1.fill_between([0] + list(robustness_df['noise_level']*100),
                 [base_acc*0.95] + [base_acc*0.95]*len(robustness_df),
                 [base_acc*1.05] + [base_acc*1.05]*len(robustness_df),
                 alpha=0.2, color='green', label='±5% Tolerance')
ax1.set_xlabel('Noise Level (%)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Robustness: Accuracy vs Noise')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot([0] + list(robustness_df['noise_level']*100),
         [base_auc] + list(robustness_df['auc']), 'o-', color='green', linewidth=2, markersize=8)
ax2.axhline(y=base_auc, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax2.set_xlabel('Noise Level (%)')
ax2.set_ylabel('ROC-AUC')
ax2.set_title('Robustness: AUC vs Noise')
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.plot([0] + list(robustness_df['noise_level']*100),
         [base_f1] + list(robustness_df['f1']), 'o-', color='red', linewidth=2, markersize=8)
ax3.axhline(y=base_f1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax3.set_xlabel('Noise Level (%)')
ax3.set_ylabel('F1 Score')
ax3.set_title('Robustness: F1 vs Noise')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig_robustness_noise.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [3/4] fig_robustness_noise.png")

# 图4: 残差分析
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='white', density=True)
xmin, xmax = ax1.get_xlim()
x = np.linspace(xmin, xmax, 100)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax1.set_xlabel('Residual (Actual - Predicted Prob)')
ax1.set_ylabel('Density')
ax1.set_title('Residual Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.scatter(y_prob, residuals, alpha=0.3, s=10, c='blue')
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Residual')
ax2.set_title('Residuals vs Predicted Probability')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig_residual_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [4/4] fig_residual_analysis.png")

# ==================== 4. 保存检验结果 ====================
print("\n" + "=" * 70)
print("四、保存检验结果")
print("=" * 70)

validation_results = {
    'cross_validation': {
        'RF': {m: {'mean': cv_results['RF'][m]['mean'], 'std': cv_results['RF'][m]['std']} for m in metrics},
        'GNB': {m: {'mean': cv_results['GNB'][m]['mean'], 'std': cv_results['GNB'][m]['std']} for m in metrics}
    },
    'learning_curve': {
        'train_final_accuracy': float(train_mean[-1]),
        'test_final_accuracy': float(test_mean[-1]),
        'overfit_gap': float(train_mean[-1] - test_mean[-1]),
        'overfit_status': overfit_status
    },
    'residual_analysis': {
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'skewness': float(stats.skew(residuals)),
        'kurtosis': float(stats.kurtosis(residuals)),
        'shapiro_w': float(shapiro_stat),
        'shapiro_p': float(shapiro_p)
    },
    'robustness': {
        'baseline': {'accuracy': float(base_acc), 'auc': float(base_auc), 'f1': float(base_f1)},
        'noise_5pct': {
            'accuracy': float(robustness_df[robustness_df['noise_level']==0.05]['accuracy'].values[0]),
            'acc_change': float(robustness_df[robustness_df['noise_level']==0.05]['acc_change'].values[0]),
            'auc': float(robustness_df[robustness_df['noise_level']==0.05]['auc'].values[0]),
            'auc_change': float(robustness_df[robustness_df['noise_level']==0.05]['auc_change'].values[0])
        },
        'noise_10pct': {
            'accuracy': float(robustness_df[robustness_df['noise_level']==0.10]['accuracy'].values[0]),
            'acc_change': float(robustness_df[robustness_df['noise_level']==0.10]['acc_change'].values[0])
        }
    },
    'split_sensitivity': {
        'accuracy_std': float(split_df['accuracy'].std()),
        'auc_std': float(split_df['auc'].std())
    }
}

import json
with open(os.path.join(script_dir, 'q1_validation_results.json'), 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)

print("  检验结果已保存: q1_validation_results.json")

print("\n" + "=" * 70)
print("模型检验完成!")
print("=" * 70)

print("\n生成的文件:")
print("  - fig_cv_boxplot.png        (交叉验证箱线图)")
print("  - fig_learning_curve.png    (学习曲线)")
print("  - fig_robustness_noise.png  (鲁棒性-噪声分析)")
print("  - fig_residual_analysis.png (残差分析)")
print("  - q1_validation_results.json (检验结果数据)")
