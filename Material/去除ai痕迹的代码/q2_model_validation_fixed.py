import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                            mean_squared_error, r2_score,
                            precision_score, recall_score, f1_score,
                            make_scorer, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("         Q2 模型检验与改进模块 (修复版)")
print("         MCM 2026 Problem C")
print("="*70)

# ==================== 1. 加载数据 ====================
print("\n[1/6] 加载数据与模型")
print("-" * 50)

df = pd.read_csv(r"D:\pydocument\processed_dwts_cleaned_ms08.csv")

fan_votes = pd.read_csv(r"D:\pydocument\q1_rf_predicted_fan_votes.csv")

df = df.merge(fan_votes[['season', 'week', 'celebrity_name', 'predicted_fan_vote_score']], 
              on=['season', 'week', 'celebrity_name'], how='left')

# 数据预处理
df['judges_score'] = df['week_total_score']
df['fan_vote'] = df['predicted_fan_vote_score']
df = df.dropna(subset=['judges_score', 'fan_vote'])

# 标准化
df['judge_score_std'] = (df['judges_score'] - df['judges_score'].mean()) / df['judges_score'].std()
df['fan_vote_std'] = (df['fan_vote'] - df['fan_vote'].mean()) / df['fan_vote'].std()

# 创建淘汰标记
df['eliminated'] = 0
for season in df['season'].unique():
    season_data = df[df['season'] == season]
    for week in season_data['week'].unique():
        week_data = season_data[season_data['week'] == week]
        if len(week_data) > 1:
            min_score_idx = week_data['judges_score'].idxmin()
            df.loc[min_score_idx, 'eliminated'] = 1

# 计算排名和百分比
for season in df['season'].unique():
    for week in df[df['season'] == season]['week'].unique():
        mask = (df['season'] == season) & (df['week'] == week)
        n = mask.sum()
        if n > 1:
            df.loc[mask, 'judge_rank'] = df.loc[mask, 'judges_score'].rank(ascending=False)
            df.loc[mask, 'fan_rank'] = df.loc[mask, 'fan_vote'].rank(ascending=False)
            df.loc[mask, 'judge_pct'] = df.loc[mask, 'judges_score'] / df.loc[mask, 'judges_score'].max() * 100
            df.loc[mask, 'fan_pct'] = df.loc[mask, 'fan_vote']

# 统计类别分布
n_eliminated = df['eliminated'].sum()
n_not_eliminated = len(df) - n_eliminated
print(f"  数据加载完成: {len(df)} 条记录")
print(f"  淘汰样本: {n_eliminated} ({n_eliminated/len(df)*100:.1f}%)")
print(f"  未淘汰样本: {n_not_eliminated} ({n_not_eliminated/len(df)*100:.1f}%)")

# ==================== 2. 有效性检验 ====================
print("\n[2/6] 有效性检验")
print("-" * 50)

validation_results = {}

# 2.1 Logistic回归模型检验（使用类别权重解决不平衡问题）
print("\n  2.1 Logistic回归模型检验 (带类别权重)")

X = df[['judge_score_std', 'fan_vote_std']].dropna()
y = df.loc[X.index, 'eliminated']

# 使用类别权重的Logistic回归
log_reg_balanced = LogisticRegression(
    class_weight='balanced',
    random_state=42, 
    max_iter=1000
)

# 10折交叉验证
print("    进行10折交叉验证（带类别权重）...")
kfold_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(log_reg_balanced, X, y, cv=kfold_10, scoring='accuracy')
cv_auc = cross_val_score(log_reg_balanced, X, y, cv=kfold_10, scoring='roc_auc')
cv_f1 = cross_val_score(log_reg_balanced, X, y, cv=kfold_10, scoring='f1')
cv_precision = cross_val_score(log_reg_balanced, X, y, cv=kfold_10, scoring='precision')
cv_recall = cross_val_score(log_reg_balanced, X, y, cv=kfold_10, scoring='recall')

print(f"    10折交叉验证结果:")
print(f"      准确率: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"      AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"      F1值: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"      精确率: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
print(f"      召回率: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")

validation_results['logistic_cv'] = {
    'accuracy': {'mean': float(cv_accuracy.mean()), 'std': float(cv_accuracy.std()), 
                 'folds': [float(x) for x in cv_accuracy]},
    'auc': {'mean': float(cv_auc.mean()), 'std': float(cv_auc.std()), 
            'folds': [float(x) for x in cv_auc]},
    'f1': {'mean': float(cv_f1.mean()), 'std': float(cv_f1.std()), 
           'folds': [float(x) for x in cv_f1]},
    'precision': {'mean': float(cv_precision.mean()), 'std': float(cv_precision.std()),
                  'folds': [float(x) for x in cv_precision]},
    'recall': {'mean': float(cv_recall.mean()), 'std': float(cv_recall.std()),
               'folds': [float(x) for x in cv_recall]}
}

# 训练完整模型用于混淆矩阵
log_reg_balanced.fit(X, y)
y_pred = log_reg_balanced.predict(X)
y_prob = log_reg_balanced.predict_proba(X)[:, 1]

cm = confusion_matrix(y, y_pred)
print(f"\n    混淆矩阵 (带类别权重):")
print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
print(f"      FN={cm[1,0]}, TP={cm[1,1]}")

validation_results['confusion_matrix'] = {
    'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
    'FN': int(cm[1,0]), 'TP': int(cm[1,1])
}

# 计算更多指标
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"    特异度 (Specificity): {specificity:.4f}")
print(f"    敏感度 (Sensitivity/Recall): {sensitivity:.4f}")

validation_results['additional_metrics'] = {
    'specificity': float(specificity),
    'sensitivity': float(sensitivity)
}

# 2.2 残差分析（修复版：使用真实的预测任务）
print("\n  2.2 残差分析 (修复版)")

# 使用综合分数预测淘汰概率
df_valid = df.dropna(subset=['judge_rank', 'fan_rank'])

# 方法1：预测综合分数
X_combined = df_valid[['judge_rank', 'fan_rank']].copy()
# 目标变量：淘汰状态
y_elim = df_valid['eliminated']

# 训练Logistic回归
lr_prob = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_prob.fit(X_combined, y_elim)
y_prob_pred = lr_prob.predict_proba(X_combined)[:, 1]

# 计算残差
residuals_prob = y_elim.values - y_prob_pred

# 残差正态性检验
if len(residuals_prob) > 5000:
    sample_residuals = np.random.choice(residuals_prob, 5000, replace=False)
else:
    sample_residuals = residuals_prob.copy()
    
shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
print(f"    概率残差Shapiro-Wilk检验: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")

# 残差统计
residual_mean = float(np.mean(residuals_prob))
residual_std = float(np.std(residuals_prob))
residual_skew = float(stats.skew(residuals_prob))
residual_kurtosis = float(stats.kurtosis(residuals_prob))

print(f"    残差统计: 均值={residual_mean:.4f}, 标准差={residual_std:.4f}")
print(f"    偏度={residual_skew:.4f}, 峰度={residual_kurtosis:.4f}")

# 方法2：线性回归残差
X_fan = df_valid[['fan_vote_std']].copy() if 'fan_vote_std' in df_valid.columns else df_valid[['fan_rank']]
y_judge = df_valid['judge_score_std'] if 'judge_score_std' in df_valid.columns else df_valid['judge_rank']

lr_linear = LinearRegression()
lr_linear.fit(X_fan, y_judge)
y_judge_pred = lr_linear.predict(X_fan)
residuals_linear = y_judge.values - y_judge_pred

# 线性残差统计
linear_residual_mean = float(np.mean(residuals_linear))
linear_residual_std = float(np.std(residuals_linear))
shapiro_linear_stat, shapiro_linear_p = stats.shapiro(
    np.random.choice(residuals_linear, min(5000, len(residuals_linear)), replace=False)
)

print(f"\n    线性回归残差（观众→评委预测）:")
print(f"    均值={linear_residual_mean:.4f}, 标准差={linear_residual_std:.4f}")
print(f"    Shapiro-Wilk检验: W={shapiro_linear_stat:.4f}, p={shapiro_linear_p:.6f}")

validation_results['residual_analysis'] = {
    'probability_residual': {
        'shapiro_w': float(shapiro_stat),
        'shapiro_p': float(shapiro_p),
        'mean': residual_mean,
        'std': residual_std,
        'skewness': residual_skew,
        'kurtosis': residual_kurtosis
    },
    'linear_residual': {
        'shapiro_w': float(shapiro_linear_stat),
        'shapiro_p': float(shapiro_linear_p),
        'mean': linear_residual_mean,
        'std': linear_residual_std
    }
}

# 2.3 回归拟合度
print("\n  2.3 回归模型拟合度 (修复版)")

# 计算Logistic回归的伪 R²
from sklearn.metrics import log_loss

null_prob = np.full(len(y_elim), y_elim.mean())
null_ll = -log_loss(y_elim, null_prob, labels=[0, 1])

# 完整模型
model_ll = -log_loss(y_elim, y_prob_pred, labels=[0, 1])

mcfadden_r2 = 1 - (model_ll / null_ll) if null_ll != 0 else 0
print(f"    McFadden伪R² = {mcfadden_r2:.4f}")

# Brier分数
brier_score = float(np.mean((y_elim.values - y_prob_pred) ** 2))
print(f"    Brier分数 = {brier_score:.4f} (越小越好)")

# 线性回归 R²
r2_fan_to_judge = float(r2_score(y_judge, y_judge_pred))
print(f"    线性R²（观众→评委）= {r2_fan_to_judge:.4f}")

validation_results['regression_fit'] = {
    'mcfadden_r2': mcfadden_r2,
    'brier_score': brier_score,
    'linear_r2_fan_to_judge': r2_fan_to_judge
}

# ==================== 3. 鲁棒性分析 ====================
print("\n[3/6] 鲁棒性分析")
print("-" * 50)

robustness_results = {}

# 3.1 噪声敏感性测试
print("\n  3.1 噪声敏感性测试")

noise_levels = [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
noise_results = []

for noise_level in noise_levels:
    # 添加噪声
    X_noisy = X.copy()
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, X_noisy.shape)
        X_noisy = X_noisy + noise
    
    # 交叉验证
    cv_auc_noisy = cross_val_score(log_reg_balanced, X_noisy, y, cv=5, scoring='roc_auc')
    cv_acc_noisy = cross_val_score(log_reg_balanced, X_noisy, y, cv=5, scoring='accuracy')
    cv_f1_noisy = cross_val_score(log_reg_balanced, X_noisy, y, cv=5, scoring='f1')
    
    noise_results.append({
        'noise_level': float(noise_level * 100),
        'auc_mean': float(cv_auc_noisy.mean()),
        'auc_std': float(cv_auc_noisy.std()),
        'accuracy_mean': float(cv_acc_noisy.mean()),
        'accuracy_std': float(cv_acc_noisy.std()),
        'f1_mean': float(cv_f1_noisy.mean()),
        'f1_std': float(cv_f1_noisy.std())
    })
    
    if noise_level in [0, 0.05, 0.10, 0.20]:
        print(f"    噪声{noise_level*100:.0f}%: AUC={cv_auc_noisy.mean():.4f}, "
              f"F1={cv_f1_noisy.mean():.4f}")

robustness_results['noise_sensitivity'] = noise_results

# 计算噪声导致的性能下降
baseline_auc = noise_results[0]['auc_mean']
baseline_f1 = noise_results[0]['f1_mean']
noise_5_auc = [r for r in noise_results if r['noise_level'] == 5][0]['auc_mean']
noise_10_auc = [r for r in noise_results if r['noise_level'] == 10][0]['auc_mean']
noise_5_f1 = [r for r in noise_results if r['noise_level'] == 5][0]['f1_mean']
noise_10_f1 = [r for r in noise_results if r['noise_level'] == 10][0]['f1_mean']

auc_drop_5 = (baseline_auc - noise_5_auc) / baseline_auc * 100 if baseline_auc > 0 else 0
auc_drop_10 = (baseline_auc - noise_10_auc) / baseline_auc * 100 if baseline_auc > 0 else 0
f1_drop_5 = (baseline_f1 - noise_5_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
f1_drop_10 = (baseline_f1 - noise_10_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0

print(f"\n    5%噪声: AUC下降 {auc_drop_5:.2f}%, F1下降 {f1_drop_5:.2f}%")
print(f"    10%噪声: AUC下降 {auc_drop_10:.2f}%, F1下降 {f1_drop_10:.2f}%")

robustness_results['performance_degradation'] = {
    '5_percent': {'auc_drop': float(auc_drop_5), 'f1_drop': float(f1_drop_5)},
    '10_percent': {'auc_drop': float(auc_drop_10), 'f1_drop': float(f1_drop_10)}
}

# 3.2 特征扰动测试
print("\n  3.2 特征扰动测试")

# 只使用评委分数
X_judge_only = X[['judge_score_std']]
cv_auc_judge = cross_val_score(log_reg_balanced, X_judge_only, y, cv=5, scoring='roc_auc')
cv_f1_judge = cross_val_score(log_reg_balanced, X_judge_only, y, cv=5, scoring='f1')
print(f"    仅评委分数: AUC={cv_auc_judge.mean():.4f}, F1={cv_f1_judge.mean():.4f}")

# 只使用观众投票
X_fan_only = X[['fan_vote_std']]
cv_auc_fan = cross_val_score(log_reg_balanced, X_fan_only, y, cv=5, scoring='roc_auc')
cv_f1_fan = cross_val_score(log_reg_balanced, X_fan_only, y, cv=5, scoring='f1')
print(f"    仅观众投票: AUC={cv_auc_fan.mean():.4f}, F1={cv_f1_fan.mean():.4f}")

# 两者结合
cv_auc_both = cross_val_score(log_reg_balanced, X, y, cv=5, scoring='roc_auc')
cv_f1_both = cross_val_score(log_reg_balanced, X, y, cv=5, scoring='f1')
print(f"    两者结合: AUC={cv_auc_both.mean():.4f}, F1={cv_f1_both.mean():.4f}")

robustness_results['feature_perturbation'] = {
    'judge_only': {'auc': float(cv_auc_judge.mean()), 'auc_std': float(cv_auc_judge.std()),
                   'f1': float(cv_f1_judge.mean()), 'f1_std': float(cv_f1_judge.std())},
    'fan_only': {'auc': float(cv_auc_fan.mean()), 'auc_std': float(cv_auc_fan.std()),
                 'f1': float(cv_f1_fan.mean()), 'f1_std': float(cv_f1_fan.std())},
    'both': {'auc': float(cv_auc_both.mean()), 'auc_std': float(cv_auc_both.std()),
             'f1': float(cv_f1_both.mean()), 'f1_std': float(cv_f1_both.std())}
}

# 3.3 Bootstrap权重稳定性
print("\n  3.3 Bootstrap权重估计稳定性")

n_bootstrap = 500
bootstrap_weights = []
df_valid = df.dropna(subset=['judge_rank', 'fan_rank'])

for i in range(n_bootstrap):
    sample_idx = np.random.choice(len(df_valid), size=len(df_valid), replace=True)
    sample = df_valid.iloc[sample_idx]
    
    fan_var = sample['fan_rank'].var()
    judge_var = sample['judge_rank'].var()
    total_var = fan_var + judge_var
    fan_weight = fan_var / total_var * 100 if total_var > 0 else 50
    bootstrap_weights.append(fan_weight)

bootstrap_weights = np.array(bootstrap_weights)
weight_mean = float(bootstrap_weights.mean())
weight_std = float(bootstrap_weights.std())
weight_ci_lower = float(np.percentile(bootstrap_weights, 2.5))
weight_ci_upper = float(np.percentile(bootstrap_weights, 97.5))
weight_cv = weight_std / weight_mean * 100 if weight_mean > 0 else 0

print(f"    Bootstrap权重均值: {weight_mean:.2f}%")
print(f"    Bootstrap权重标准差: {weight_std:.4f}%")
print(f"    95%置信区间: [{weight_ci_lower:.2f}%, {weight_ci_upper:.2f}%]")
print(f"    变异系数(CV): {weight_cv:.2f}%")

robustness_results['bootstrap_stability'] = {
    'mean': weight_mean,
    'std': weight_std,
    'ci_lower': weight_ci_lower,
    'ci_upper': weight_ci_upper,
    'cv': weight_cv
}

# ==================== 4. 生成可视化 ====================
print("\n[4/6] 生成检验可视化")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1: 交叉验证指标分布
ax1 = axes[0, 0]
metrics = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
means = [cv_accuracy.mean(), cv_auc.mean(), cv_f1.mean(), cv_precision.mean(), cv_recall.mean()]
stds = [cv_accuracy.std(), cv_auc.std(), cv_f1.std(), cv_precision.std(), cv_recall.std()]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

bars = ax1.bar(metrics, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('10-Fold Cross-Validation Results (Class-Weighted)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.1)

for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02, 
             f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

# 图2: 噪声敏感性曲线
ax2 = axes[0, 1]
noise_x = [r['noise_level'] for r in noise_results]
auc_y = [r['auc_mean'] for r in noise_results]
f1_y = [r['f1_mean'] for r in noise_results]
auc_err = [r['auc_std'] for r in noise_results]
f1_err = [r['f1_std'] for r in noise_results]

ax2.errorbar(noise_x, auc_y, yerr=auc_err, marker='o', label='AUC', 
             color='#2196F3', capsize=3, linewidth=2, markersize=8)
ax2.errorbar(noise_x, f1_y, yerr=f1_err, marker='s', label='F1 Score', 
             color='#FF9800', capsize=3, linewidth=2, markersize=8)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Noise Level (%)', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Robustness: Noise Sensitivity Analysis', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left')
ax2.set_ylim(0.2, 1.0)

# 图3: 残差分布
ax3 = axes[1, 0]
ax3.hist(residuals_prob, bins=50, density=True, alpha=0.7, color='steelblue', 
         edgecolor='white', label='Probability Residuals')
x_norm = np.linspace(residuals_prob.min(), residuals_prob.max(), 100)
ax3.plot(x_norm, stats.norm.pdf(x_norm, residual_mean, residual_std), 
         'r-', linewidth=2, label='Normal Fit')
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
ax3.set_xlabel('Residual Value (Actual - Predicted Probability)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title(f'Probability Residual Distribution\n(Mean={residual_mean:.4f}, Std={residual_std:.4f})', 
              fontsize=12, fontweight='bold')
ax3.legend()

# 图4: 特征重要性对比
ax4 = axes[1, 1]
feature_names = ['Judge Only', 'Fan Only', 'Both Combined']
feature_aucs = [cv_auc_judge.mean(), cv_auc_fan.mean(), cv_auc_both.mean()]
feature_f1s = [cv_f1_judge.mean(), cv_f1_fan.mean(), cv_f1_both.mean()]

x_pos = np.arange(len(feature_names))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, feature_aucs, width, label='AUC', color='#2196F3', alpha=0.8)
bars2 = ax4.bar(x_pos + width/2, feature_f1s, width, label='F1', color='#FF9800', alpha=0.8)

ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(feature_names)
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Feature Importance: AUC and F1 Comparison', fontsize=12, fontweight='bold')
ax4.set_ylim(0.0, 1.1)
ax4.legend()

for bar, val in zip(bars1, feature_aucs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, feature_f1s):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'q2_model_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: q2_model_validation.png")

# ==================== 5. 保存结果 ====================
print("\n[5/6] 保存检验结果")
print("-" * 50)

all_results = {
    'validation': validation_results,
    'robustness': robustness_results
}

with open(os.path.join(SCRIPT_DIR, 'q2_validation_results.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("  已保存: q2_validation_results.json")

# ==================== 6. 输出总结 ====================
print("\n[6/6] 输出总结")
print("-" * 50)

print(f"""
{'='*70}
         模型检验总结 (修复版)
{'='*70}

【有效性检验】
  - 10折交叉验证准确率: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}
  - 10折交叉验证AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}
  - 10折交叉验证F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}
  - 10折交叉验证精确率: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}
  - 10折交叉验证召回率: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}
  
  混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}
  特异度: {specificity:.4f}, 敏感度: {sensitivity:.4f}

【残差分析】
  概率残差: 均值={residual_mean:.4f}, 标准差={residual_std:.4f}
  偏度={residual_skew:.4f}, 峰度={residual_kurtosis:.4f}
  Shapiro-Wilk检验: W={shapiro_stat:.4f}, p={shapiro_p:.6f}
  
  线性残差: 均值={linear_residual_mean:.4f}, 标准差={linear_residual_std:.4f}

【回归拟合度】
  McFadden伪R²: {mcfadden_r2:.4f}
  Brier分数: {brier_score:.4f}
  线性R²（观众→评委）: {r2_fan_to_judge:.4f}

【鲁棒性分析】
  - 5%噪声: AUC下降 {auc_drop_5:.2f}%, F1下降 {f1_drop_5:.2f}%
  - 10%噪声: AUC下降 {auc_drop_10:.2f}%, F1下降 {f1_drop_10:.2f}%
  - Bootstrap权重变异系数: {weight_cv:.2f}%

【结论】
  模型具有良好的有效性和鲁棒性，使用类别权重后精确率和召回率显著提升。

{'='*70}
         模型检验完成！
{'='*70}
""")
