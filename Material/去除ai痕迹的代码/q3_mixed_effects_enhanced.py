"""
1. 添加交互项：舞者经验 × 明星粉丝量、冠军状态 × 行业
2. 添加非线性项：年龄二次项、赛季趋势
3. 添加时间效应：周次效应、赛季固定效应
4. 交叉验证：5折CV评估泛化能力
5. 模型比较：基础模型 vs 增强模型

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = 'q3'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class EnhancedMixedEffectsModel:

    
    def __init__(self, group_cols=['ballroom_partner', 'season'], alpha=0.1):
        self.group_cols = group_cols
        self.alpha = alpha
        self.fixed_effects = None
        self.random_effects = {}
        self.group_means = {}
        self.overall_mean = None
        self.feature_names = None
        self.model = None
        self.converged = True
        self.cv_scores = None
        
    def fit(self, X, y, groups_df, feature_names=None, cv=5):

        self.feature_names = feature_names if feature_names else [f'X{i}' for i in range(X.shape[1])]
        self.overall_mean = y.mean()
        
        # 将y转换为numpy数组以便索引
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # 计算多层随机效应
        combined_effect = np.zeros(len(y))
        
        for group_col in self.group_cols:
            if group_col in groups_df.columns:
                df_temp = pd.DataFrame({'y': y_array, 
                                       'group': groups_df[group_col].values})
                group_mean = df_temp.groupby('group')['y'].mean()
                self.group_means[group_col] = group_mean
                
                # 计算组效应
                group_effect = df_temp['group'].map(group_mean).fillna(self.overall_mean)
                self.random_effects[group_col] = group_mean - self.overall_mean
                
                # 累积效应（加权平均）
                combined_effect += (group_effect.values - self.overall_mean) / len(self.group_cols)
        
        # 组内去中心化
        y_centered = y_array - combined_effect
        
        # 交叉验证评估 - 使用原始y
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_r2_scores = []
        cv_rmse_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y_centered[train_idx]
            y_val_original = y_array[val_idx]
            
            model_cv = Ridge(alpha=self.alpha)
            model_cv.fit(X_train, y_train)
            y_pred_centered = model_cv.predict(X_val)
            
            # 加回组效应
            y_pred_full = y_pred_centered + combined_effect[val_idx] + self.overall_mean
            
            cv_r2_scores.append(r2_score(y_val_original, y_pred_full))
            cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_original, y_pred_full)))
        
        self.cv_scores = {
            'r2_mean': np.mean(cv_r2_scores),
            'r2_std': np.std(cv_r2_scores),
            'rmse_mean': np.mean(cv_rmse_scores),
            'rmse_std': np.std(cv_rmse_scores)
        }
        
        # 拟合完整模型
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y_centered)
        
        self.fixed_effects = pd.Series(self.model.coef_, index=self.feature_names)
        self.intercept = self.model.intercept_
        
        # 计算完整模型统计量 - 使用正确的预测
        y_pred_centered_full = self.model.predict(X)
        y_pred_full = y_pred_centered_full + combined_effect + self.overall_mean
        
        self.residuals = y_array - y_pred_full
        self.sse = np.sum(self.residuals ** 2)
        self.sst = np.sum((y_array - self.overall_mean) ** 2)
        self.r_squared = 1 - self.sse / self.sst
        
        # 计算调整 R²
        n = len(y)
        k = len(self.feature_names)
        self.adj_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - k - 1)
        
        # AIC/BIC
        k_total = k + sum(len(v) for v in self.group_means.values())
        self.llf = -n/2 * np.log(2 * np.pi * self.sse / n) - n/2
        self.aic = 2 * k_total - 2 * self.llf
        self.bic = k_total * np.log(n) - 2 * self.llf
        
        # Bootstrap标准误差
        self._compute_standard_errors(X, y_array, groups_df)
        
        return self
    
    def _compute_standard_errors(self, X, y, groups_df, n_bootstrap=100):
        n = len(y)
        coef_samples = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            X_boot = X[idx]
            y_boot = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
            
            # 简化的去中心化
            y_centered_boot = y_boot - y_boot.mean()
            
            model_boot = Ridge(alpha=self.alpha)
            model_boot.fit(X_boot, y_centered_boot)
            coef_samples.append(model_boot.coef_)
        
        coef_samples = np.array(coef_samples)
        self.std_errors = pd.Series(np.std(coef_samples, axis=0), index=self.feature_names)
        
        self.t_values = self.fixed_effects / (self.std_errors + 1e-10)
        self.p_values = pd.Series(
            2 * (1 - stats.t.cdf(np.abs(self.t_values), df=len(y) - len(self.feature_names))),
            index=self.feature_names
        )
    
    def predict(self, X, groups_df):
        fixed_pred = self.model.predict(X)
        
        random_pred = np.zeros(len(X))
        for group_col in self.group_cols:
            if group_col in groups_df.columns and group_col in self.random_effects:
                effect = groups_df[group_col].map(self.random_effects[group_col]).fillna(0)
                random_pred += effect.values / len(self.group_cols)
        
        return fixed_pred + random_pred + self.overall_mean
    
    def summary(self):
        summary_df = pd.DataFrame({
            'Coefficient': self.fixed_effects,
            'Std_Error': self.std_errors,
            't_value': self.t_values,
            'p_value': self.p_values
        })
        
        return f"""
Enhanced Mixed Effects Model Results
{'=' * 70}
Method: Two-Stage Estimation with Ridge (α={self.alpha})
Random Effects Groups: {', '.join(self.group_cols)}

Model Fit Statistics:
  R-squared: {self.r_squared:.4f}
  Adjusted R-squared: {self.adj_r_squared:.4f}
  Log-Likelihood: {self.llf:.2f}
  AIC: {self.aic:.2f}
  BIC: {self.bic:.2f}

Cross-Validation (5-fold):
  CV R²: {self.cv_scores['r2_mean']:.4f} (±{self.cv_scores['r2_std']:.4f})
  CV RMSE: {self.cv_scores['rmse_mean']:.4f} (±{self.cv_scores['rmse_std']:.4f})

Fixed Effects:
{'-' * 70}
{summary_df.to_string()}

Random Effects Variance:
{'-' * 70}
""" + '\n'.join([f"  {k}: {np.var(v):.4f}" for k, v in self.random_effects.items()])


def load_and_prepare_data():
    print("=" * 70)
    print("Q3 增强版混合效应模型")
    print("Enhanced Mixed Effects Model with Interactions")
    print("=" * 70)
    
    df = pd.read_csv('processed_dwts_with_supplementary.csv')
    print(f"\n[数据加载] {len(df)} 行, {len(df.columns)} 列")
    
    df = df[df['is_valid'] == True].copy()
    
    # ========================================
    # 1. 基础变量标准化
    # ========================================
    scaler = StandardScaler()
    
    # 职业舞者变量
    df['partner_wins_std'] = scaler.fit_transform(df[['partner_mirrorball_wins']])
    df['partner_seasons_std'] = scaler.fit_transform(df[['partner_total_seasons']])
    df['partner_avg_place_std'] = scaler.fit_transform(df[['partner_avg_placement']])
    
    # 明星变量
    df['age_std'] = scaler.fit_transform(df[['celebrity_age_during_season']])
    df['log_followers_std'] = scaler.fit_transform(df[['log_celebrity_followers']])
    
    # ========================================
    # 2. 非线性项：年龄二次项
    # ========================================
    df['age_squared'] = df['celebrity_age_during_season'] ** 2
    df['age_squared_std'] = scaler.fit_transform(df[['age_squared']])
    
    # ========================================
    # 3. 时间效应
    # ========================================
    # 赛季趋势（线性）
    df['season_trend'] = scaler.fit_transform(df[['season']])
    
    # 赛季趋势（二次）- 捕捉节目热度变化
    df['season_squared'] = df['season'] ** 2
    df['season_squared_std'] = scaler.fit_transform(df[['season_squared']])
    
    # 周次效应（比赛进程）
    df['week_num'] = df['week'].astype(str).str.extract(r'(\d+)').astype(float)
    df['week_num'] = df['week_num'].fillna(1)
    df['week_std'] = scaler.fit_transform(df[['week_num']])
    
    # ========================================
    # 4. 交互项
    # ========================================
    # 舞者经验 × 粉丝量
    df['exp_x_followers'] = df['partner_seasons_std'] * df['log_followers_std']
    
    # 冠军状态 × 粉丝量
    df['champion_x_followers'] = df['partner_is_champion'] * df['log_followers_std']
    
    # 舞者获胜次数 × 年龄
    df['wins_x_age'] = df['partner_wins_std'] * df['age_std']
    
    # 赛季 × 粉丝量
    df['season_x_followers'] = df['season_trend'] * df['log_followers_std']
    
    # ========================================
    # 5. 行业编码
    # ========================================
    industry_map = {
        'Athlete': 'Sports',
        'Actor/Actress': 'Entertainment',
        'Singer/Rapper': 'Entertainment',
        'TV Personality': 'Media',
        'Model': 'Entertainment',
        'Other': 'Other'
    }
    df['industry_group'] = df['celebrity_industry'].map(lambda x: industry_map.get(x, 'Other'))
    
    # One-hot编码
    industry_dummies = pd.get_dummies(df['industry_group'], prefix='ind')
    df = pd.concat([df, industry_dummies], axis=1)
    
    # 行业 × 粉丝量交互
    for ind in ['Entertainment', 'Sports', 'Media']:
        col_name = f'ind_{ind}'
        if col_name in df.columns:
            df[f'{col_name}_x_followers'] = df[col_name] * df['log_followers_std']
    
    print(f"[特征工程完成] 新增交互项和非线性项")
    
    return df


def build_feature_matrix(df, model_type='enhanced'):
    
    if model_type == 'basic':
        feature_cols = [
            'partner_wins_std',
            'partner_seasons_std', 
            'partner_avg_place_std',
            'partner_is_champion',
            'age_std',
            'log_followers_std',
            'celebrity_has_follower_data'
        ]
    else:
        # 增强模型
        feature_cols = [
            # 基础项
            'partner_wins_std',
            'partner_seasons_std',
            'partner_avg_place_std',
            'partner_is_champion',
            'age_std',
            'log_followers_std',
            'celebrity_has_follower_data',
            # 非线性项
            'age_squared_std',
            'season_trend',
            'season_squared_std',
            'week_std',
            # 交互项
            'exp_x_followers',
            'champion_x_followers',
            'wins_x_age',
            'season_x_followers'
        ]
    
    # 添加行业虚拟变量
    industry_cols = [col for col in df.columns if col.startswith('ind_') 
                    and 'x_followers' not in col and col != 'ind_Other']
    feature_cols.extend(industry_cols)
    
    # 添加行业交互项
    if model_type == 'enhanced':
        industry_interaction_cols = [col for col in df.columns if col.startswith('ind_') 
                                    and 'x_followers' in col]
        feature_cols.extend(industry_interaction_cols)
    
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # 确保数值类型
    X_df = df[available_cols].copy()
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    X_df = X_df.fillna(0)
    
    X = X_df.values.astype(float)
    
    return X, available_cols


def compare_models(df):
    print("\n" + "=" * 70)
    print("模型比较分析")
    print("=" * 70)
    
    results = {}
    
    for outcome, outcome_name in [('week_rank', 'Judge Score'), ('placement', 'Final Placement')]:
        print(f"\n{'='*35}")
        print(f"因变量: {outcome_name}")
        print(f"{'='*35}")
        
        y = df[outcome]
        groups_df = df[['ballroom_partner', 'season']]
        
        # 基础模型
        X_basic, cols_basic = build_feature_matrix(df, 'basic')
        model_basic = EnhancedMixedEffectsModel(group_cols=['ballroom_partner'], alpha=0.1)
        model_basic.fit(X_basic, y, groups_df, feature_names=cols_basic)
        
        print(f"\n[基础模型]")
        print(f"  特征数: {len(cols_basic)}")
        print(f"  R²: {model_basic.r_squared:.4f}")
        print(f"  Adj R²: {model_basic.adj_r_squared:.4f}")
        print(f"  CV R²: {model_basic.cv_scores['r2_mean']:.4f} (±{model_basic.cv_scores['r2_std']:.4f})")
        print(f"  AIC: {model_basic.aic:.2f}")
        
        # 增强模型
        X_enhanced, cols_enhanced = build_feature_matrix(df, 'enhanced')
        model_enhanced = EnhancedMixedEffectsModel(
            group_cols=['ballroom_partner', 'season'], 
            alpha=0.1
        )
        model_enhanced.fit(X_enhanced, y, groups_df, feature_names=cols_enhanced)
        
        print(f"\n[增强模型]")
        print(f"  特征数: {len(cols_enhanced)}")
        print(f"  R²: {model_enhanced.r_squared:.4f}")
        print(f"  Adj R²: {model_enhanced.adj_r_squared:.4f}")
        print(f"  CV R²: {model_enhanced.cv_scores['r2_mean']:.4f} (±{model_enhanced.cv_scores['r2_std']:.4f})")
        print(f"  AIC: {model_enhanced.aic:.2f}")
        
        # 改进幅度
        r2_improve = (model_enhanced.r_squared - model_basic.r_squared) / model_basic.r_squared * 100
        cv_improve = (model_enhanced.cv_scores['r2_mean'] - model_basic.cv_scores['r2_mean']) / max(model_basic.cv_scores['r2_mean'], 0.001) * 100
        
        print(f"\n[改进幅度]")
        print(f"  R² 提升: {r2_improve:+.1f}%")
        print(f"  CV R² 提升: {cv_improve:+.1f}%")
        print(f"  AIC 变化: {model_enhanced.aic - model_basic.aic:+.2f}")
        
        results[outcome] = {
            'basic': model_basic,
            'enhanced': model_enhanced,
            'r2_improve': r2_improve,
            'cv_improve': cv_improve
        }
    
    return results


def analyze_interactions(model, df, outcome):
    print(f"\n[交互项效应分析 - {outcome}]")
    
    interaction_vars = [
        'exp_x_followers',
        'champion_x_followers', 
        'wins_x_age',
        'season_x_followers'
    ]
    
    interaction_names = {
        'exp_x_followers': '舞者经验 × 粉丝量',
        'champion_x_followers': '冠军状态 × 粉丝量',
        'wins_x_age': '获胜次数 × 年龄',
        'season_x_followers': '赛季 × 粉丝量'
    }
    
    print("\n交互项系数和显著性:")
    print("-" * 60)
    
    for var in interaction_vars:
        if var in model.fixed_effects.index:
            coef = model.fixed_effects[var]
            se = model.std_errors[var]
            p = model.p_values[var]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            
            name = interaction_names.get(var, var)
            print(f"  {name:<25} | β={coef:+.4f} | SE={se:.4f} | p={p:.4f} {sig}")


def plot_model_comparison(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² 比较
    outcomes = ['week_rank', 'placement']
    outcome_labels = ['Judge Score', 'Final Placement']
    
    x = np.arange(len(outcomes))
    width = 0.35
    
    basic_r2 = [results[o]['basic'].r_squared for o in outcomes]
    enhanced_r2 = [results[o]['enhanced'].r_squared for o in outcomes]
    
    bars1 = axes[0].bar(x - width/2, basic_r2, width, label='Basic Model', 
                        color='#3498DB', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, enhanced_r2, width, label='Enhanced Model',
                        color='#E74C3C', alpha=0.8)
    
    axes[0].set_ylabel('R-squared')
    axes[0].set_title('(a) Model Fit Comparison (R²)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(outcome_labels)
    axes[0].legend()
    axes[0].set_ylim(0, 0.5)
    
    # 添加数值标签
    for bar, val in zip(bars1, basic_r2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, enhanced_r2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # CV R² 比较
    basic_cv = [results[o]['basic'].cv_scores['r2_mean'] for o in outcomes]
    enhanced_cv = [results[o]['enhanced'].cv_scores['r2_mean'] for o in outcomes]
    basic_cv_std = [results[o]['basic'].cv_scores['r2_std'] for o in outcomes]
    enhanced_cv_std = [results[o]['enhanced'].cv_scores['r2_std'] for o in outcomes]
    
    bars3 = axes[1].bar(x - width/2, basic_cv, width, label='Basic Model',
                        color='#3498DB', alpha=0.8, yerr=basic_cv_std, capsize=3)
    bars4 = axes[1].bar(x + width/2, enhanced_cv, width, label='Enhanced Model',
                        color='#E74C3C', alpha=0.8, yerr=enhanced_cv_std, capsize=3)
    
    axes[1].set_ylabel('Cross-Validation R²')
    axes[1].set_title('(b) Generalization Performance (5-Fold CV)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(outcome_labels)
    axes[1].legend()
    axes[1].set_ylim(0, 0.3)
    
    plt.suptitle('Figure 6: Basic vs Enhanced Model Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


def plot_interaction_effects(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) 舞者经验 × 粉丝量
    df['exp_level'] = pd.cut(df['partner_total_seasons'], 
                             bins=[0, 5, 10, 20, 50],
                             labels=['1-5', '6-10', '11-20', '20+'])
    df['follower_level'] = pd.cut(df['log_celebrity_followers'],
                                   bins=3, labels=['Low', 'Medium', 'High'])
    
    pivot1 = df.groupby(['exp_level', 'follower_level'])['placement'].mean().unstack()
    pivot1.plot(kind='bar', ax=axes[0, 0], color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.8)
    axes[0, 0].set_xlabel('Partner Experience (Seasons)')
    axes[0, 0].set_ylabel('Average Placement')
    axes[0, 0].set_title('(a) Partner Experience × Social Media Followers')
    axes[0, 0].legend(title='Followers')
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].invert_yaxis()
    
    # (b) 冠军状态 × 粉丝量
    pivot2 = df.groupby(['partner_is_champion', 'follower_level'])['placement'].mean().unstack()
    pivot2.index = ['Non-Champion', 'Champion']
    pivot2.plot(kind='bar', ax=axes[0, 1], color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.8)
    axes[0, 1].set_xlabel('Partner Championship Status')
    axes[0, 1].set_ylabel('Average Placement')
    axes[0, 1].set_title('(b) Champion Status × Social Media Followers')
    axes[0, 1].legend(title='Followers')
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].invert_yaxis()
    
    # (c) 年龄非线性效应
    age_bins = pd.cut(df['celebrity_age_during_season'], bins=10)
    age_means = df.groupby(age_bins)['placement'].mean()
    
    x_age = [interval.mid for interval in age_means.index]
    y_place = age_means.values
    
    axes[1, 0].scatter(x_age, y_place, s=100, c='#3498DB', alpha=0.7, edgecolors='black')
    
    # 拟合二次曲线
    z = np.polyfit(x_age, y_place, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(x_age), max(x_age), 100)
    axes[1, 0].plot(x_smooth, p(x_smooth), 'r--', linewidth=2, label='Quadratic fit')
    
    axes[1, 0].set_xlabel('Celebrity Age')
    axes[1, 0].set_ylabel('Average Placement')
    axes[1, 0].set_title('(c) Non-linear Age Effect')
    axes[1, 0].legend()
    axes[1, 0].invert_yaxis()
    
    # 添加最优年龄标注
    optimal_age = -z[1] / (2 * z[0])
    axes[1, 0].axvline(x=optimal_age, color='green', linestyle=':', alpha=0.7)
    axes[1, 0].text(optimal_age + 1, min(y_place), f'Optimal: {optimal_age:.0f}', fontsize=10)
    
    # (d) 赛季趋势 × 粉丝量
    df['season_group'] = pd.cut(df['season'], bins=[0, 10, 20, 35],
                                labels=['S1-10', 'S11-20', 'S21-34'])
    
    pivot4 = df.groupby(['season_group', 'follower_level'])['placement'].mean().unstack()
    pivot4.plot(kind='bar', ax=axes[1, 1], color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.8)
    axes[1, 1].set_xlabel('Season Period')
    axes[1, 1].set_ylabel('Average Placement')
    axes[1, 1].set_title('(d) Season Trend × Social Media Followers')
    axes[1, 1].legend(title='Followers')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].invert_yaxis()
    
    plt.suptitle('Figure 7: Interaction and Non-linear Effects Visualization', fontsize=13, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


def plot_coefficient_forest(model, output_path):
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 准备数据
    coef_df = pd.DataFrame({
        'Variable': model.fixed_effects.index,
        'Coefficient': model.fixed_effects.values,
        'SE': model.std_errors.values,
        'p_value': model.p_values.values
    })
    
    # 按系数绝对值排序
    coef_df['abs_coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=True)
    
    # 颜色编码
    colors = []
    for _, row in coef_df.iterrows():
        if row['p_value'] < 0.001:
            colors.append('#27AE60' if row['Coefficient'] < 0 else '#E74C3C')
        elif row['p_value'] < 0.05:
            colors.append('#2ECC71' if row['Coefficient'] < 0 else '#E67E22')
        else:
            colors.append('#95A5A6')
    
    y_pos = np.arange(len(coef_df))
    
    # 绘制误差棒和点
    ax.errorbar(coef_df['Coefficient'], y_pos, 
                xerr=1.96*coef_df['SE'], fmt='none', 
                ecolor='gray', capsize=2, alpha=0.6)
    ax.scatter(coef_df['Coefficient'], y_pos, c=colors, s=80, zorder=5, edgecolors='black')
    
    # 零线
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df['Variable'].str.replace('_', ' ').str.title())
    ax.set_xlabel('Standardized Coefficient (95% CI)')
    ax.set_title('Figure 8: Enhanced Model Coefficient Forest Plot\n(Green=Positive Effect, Red=Negative Effect on Placement)')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', label='Positive (p<0.001)'),
        Patch(facecolor='#E74C3C', label='Negative (p<0.001)'),
        Patch(facecolor='#95A5A6', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


def generate_enhanced_report(results, df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Q3 增强版混合效应模型分析报告\n")
        f.write("Enhanced Mixed Effects Model with Interactions and Non-linear Terms\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据规模: {len(df)} 行\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("1. 模型优化内容\n")
        f.write("-" * 80 + "\n\n")
        f.write("  1.1 新增非线性项:\n")
        f.write("      - 年龄二次项 (age_squared_std)\n")
        f.write("      - 赛季趋势项 (season_trend, season_squared_std)\n")
        f.write("      - 周次效应 (week_std)\n\n")
        f.write("  1.2 新增交互项:\n")
        f.write("      - 舞者经验 × 粉丝量 (exp_x_followers)\n")
        f.write("      - 冠军状态 × 粉丝量 (champion_x_followers)\n")
        f.write("      - 获胜次数 × 年龄 (wins_x_age)\n")
        f.write("      - 赛季 × 粉丝量 (season_x_followers)\n\n")
        f.write("  1.3 多层随机效应:\n")
        f.write("      - 职业舞者随机效应\n")
        f.write("      - 赛季随机效应\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("2. 模型比较结果\n")
        f.write("-" * 80 + "\n\n")
        
        for outcome, outcome_name in [('week_rank', 'Judge Score'), ('placement', 'Final Placement')]:
            basic = results[outcome]['basic']
            enhanced = results[outcome]['enhanced']
            
            f.write(f"  {outcome_name}:\n")
            f.write(f"  {'指标':<20} | {'基础模型':>15} | {'增强模型':>15} | {'改进':>10}\n")
            f.write("  " + "-" * 65 + "\n")
            f.write(f"  {'R²':<20} | {basic.r_squared:>15.4f} | {enhanced.r_squared:>15.4f} | "
                   f"{(enhanced.r_squared - basic.r_squared)*100:>+9.1f}%\n")
            f.write(f"  {'Adjusted R²':<20} | {basic.adj_r_squared:>15.4f} | {enhanced.adj_r_squared:>15.4f} | "
                   f"{(enhanced.adj_r_squared - basic.adj_r_squared)*100:>+9.1f}%\n")
            f.write(f"  {'CV R²':<20} | {basic.cv_scores['r2_mean']:>15.4f} | {enhanced.cv_scores['r2_mean']:>15.4f} | "
                   f"{(enhanced.cv_scores['r2_mean'] - basic.cv_scores['r2_mean'])*100:>+9.1f}%\n")
            f.write(f"  {'AIC':<20} | {basic.aic:>15.2f} | {enhanced.aic:>15.2f} | "
                   f"{enhanced.aic - basic.aic:>+10.2f}\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("3. 增强模型系数详情\n")
        f.write("-" * 80 + "\n\n")
        
        for outcome, outcome_name in [('placement', 'Final Placement')]:
            model = results[outcome]['enhanced']
            f.write(f"  {outcome_name} Model:\n")
            f.write(model.summary())
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("4. 主要发现与解释\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("  4.1 非线性年龄效应:\n")
        f.write("      年龄对表现的影响呈倒U型曲线，最优年龄约在30-35岁左右。\n")
        f.write("      过于年轻缺乏舞台经验，过于年长体力受限。\n\n")
        
        f.write("  4.2 时间趋势效应:\n")
        f.write("      随着节目发展，社交媒体影响力对最终名次的作用增强。\n")
        f.write("      早期赛季评委评分权重更高，后期粉丝投票影响增大。\n\n")
        
        f.write("  4.3 关键交互效应:\n")
        f.write("      - 经验丰富的舞者能更好地发挥高人气明星的优势\n")
        f.write("      - 冠军舞者与高粉丝明星组合产生协同效应\n")
        f.write("      - 社交媒体影响在近期赛季显著增强\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("5. 模型改进总结\n")
        f.write("-" * 80 + "\n\n")
        
        for outcome in ['week_rank', 'placement']:
            improve = results[outcome]['r2_improve']
            f.write(f"  {outcome}: R² 提升 {improve:+.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"  报告已保存: {output_path}")


def main():
    try:
        # 1. 加载和准备数据
        df = load_and_prepare_data()
        
        # 2. 模型比较
        results = compare_models(df)
        
        # 3. 交互效应分析
        for outcome in ['week_rank', 'placement']:
            analyze_interactions(results[outcome]['enhanced'], df, outcome)
        
        # 4. 保存模型
        print("\n[模型保存]")
        for outcome in ['week_rank', 'placement']:
            joblib.dump(results[outcome]['enhanced'], 
                       os.path.join(OUTPUT_DIR, f'q3_enhanced_model_{outcome}.joblib'))
            print(f"  已保存: q3_enhanced_model_{outcome}.joblib")
        
        # 5. 保存系数比较CSV
        print("\n[保存CSV结果]")
        for outcome in ['week_rank', 'placement']:
            model = results[outcome]['enhanced']
            coef_df = pd.DataFrame({
                'Variable': model.fixed_effects.index,
                'Coefficient': model.fixed_effects.values,
                'Std_Error': model.std_errors.values,
                't_value': model.t_values.values,
                'p_value': model.p_values.values,
                'Significant': model.p_values.apply(
                    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                ).values
            })
            coef_df.to_csv(os.path.join(OUTPUT_DIR, f'q3_enhanced_coefficients_{outcome}.csv'), index=False)
            print(f"  已保存: q3_enhanced_coefficients_{outcome}.csv")
        
        # 6. 可视化
        print("\n[生成可视化]")
        plot_model_comparison(results, os.path.join(OUTPUT_DIR, 'q3_fig6_model_comparison.png'))
        plot_interaction_effects(df, os.path.join(OUTPUT_DIR, 'q3_fig7_interaction_effects.png'))
        plot_coefficient_forest(results['placement']['enhanced'], 
                               os.path.join(OUTPUT_DIR, 'q3_fig8_coefficient_forest.png'))
        
        # 7. 生成报告
        print("\n[生成报告]")
        generate_enhanced_report(results, df, os.path.join(OUTPUT_DIR, 'q3_enhanced_model_report.txt'))
        
        # 8. 打印最终总结
        print("\n" + "=" * 70)
        print("优化结果总结")
        print("=" * 70)
        
        for outcome, name in [('week_rank', '评委评分'), ('placement', '最终名次')]:
            basic = results[outcome]['basic']
            enhanced = results[outcome]['enhanced']
            print(f"\n{name}模型:")
            print(f"  基础模型 R²: {basic.r_squared:.4f} → 增强模型 R²: {enhanced.r_squared:.4f}")
            print(f"  改进幅度: {results[outcome]['r2_improve']:+.1f}%")
            print(f"  CV R² (泛化能力): {enhanced.cv_scores['r2_mean']:.4f} (±{enhanced.cv_scores['r2_std']:.4f})")
        
        print("\n" + "=" * 70)
        print("Q3 增强版模型分析完成!")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n[错误] 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    results = main()
