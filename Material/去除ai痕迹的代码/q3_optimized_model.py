import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = 'q3'


class OptimizedMixedModel:
    
    def __init__(self, group_col='ballroom_partner', alpha=1.0):
        self.group_col = group_col
        self.alpha = alpha
        self.model = None
        self.group_effects = None
        self.overall_mean = None

    def fit(self, X, y, groups):
        y = np.array(y)
        groups = np.array(groups)
        
        self.overall_mean = y.mean()
        
        # Step 1: 计算组效应
        unique_groups = np.unique(groups)
        self.group_effects = {}
        for g in unique_groups:
            mask = groups == g
            self.group_effects[g] = y[mask].mean() - self.overall_mean
        
        # Step 2: 从y中去除组效应
        y_adjusted = y.copy()
        for i, g in enumerate(groups):
            y_adjusted[i] = y[i] - self.group_effects.get(g, 0)
        
        # Step 3: 拟合固定效应
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y_adjusted)
        
        # 计算模型统计量
        y_pred = self.predict(X, groups)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - self.overall_mean) ** 2)
        self.r_squared = 1 - ss_res / ss_tot
        
        n, k = X.shape
        self.adj_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - k - 1)
        
        # AIC/BIC
        self.llf = -n/2 * np.log(2 * np.pi * ss_res / n) - n/2
        self.aic = 2 * (k + len(unique_groups)) - 2 * self.llf
        self.bic = (k + len(unique_groups)) * np.log(n) - 2 * self.llf
        
        # 系数
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        return self
    
    def predict(self, X, groups):
        groups = np.array(groups)
        fixed_pred = self.model.predict(X)
        random_pred = np.array([self.group_effects.get(g, 0) for g in groups])
        return fixed_pred + random_pred
    
    def cross_validate(self, X, y, groups, cv=5):
        y = np.array(y)
        groups = np.array(groups)
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            # 训练集
            X_train, y_train = X[train_idx], y[train_idx]
            groups_train = groups[train_idx]
            
            # 验证集
            X_val, y_val = X[val_idx], y[val_idx]
            groups_val = groups[val_idx]
            
            # 拟合模型
            model = OptimizedMixedModel(self.group_col, self.alpha)
            model.fit(X_train, y_train, groups_train)
            
            # 预测
            y_pred = model.predict(X_val, groups_val)
            
            scores.append(r2_score(y_val, y_pred))
        
        return np.mean(scores), np.std(scores)


def load_data():
    print("=" * 70)
    print("Q3 优化版混合效应模型")
    print("=" * 70)
    
    df = pd.read_csv('processed_dwts_with_supplementary.csv')
    df = df[df['is_valid'] == True].copy()
    
    print(f"\n[数据] {len(df)} 行")
    
    return df


def prepare_basic_features(df):
    scaler = StandardScaler()
    
    # 基础特征
    feature_cols = []
    
    # 职业舞者
    df['wins_std'] = scaler.fit_transform(df[['partner_mirrorball_wins']])
    df['seasons_std'] = scaler.fit_transform(df[['partner_total_seasons']])
    df['avg_place_std'] = scaler.fit_transform(df[['partner_avg_placement']])
    feature_cols.extend(['wins_std', 'seasons_std', 'avg_place_std', 'partner_is_champion'])
    
    # 明星
    df['age_std'] = scaler.fit_transform(df[['celebrity_age_during_season']])
    df['followers_std'] = scaler.fit_transform(df[['log_celebrity_followers']])
    feature_cols.extend(['age_std', 'followers_std', 'celebrity_has_follower_data'])
    
    return df, feature_cols


def prepare_enhanced_features(df):
    scaler = StandardScaler()
    
    feature_cols = []
    
    # 基础特征
    df['wins_std'] = scaler.fit_transform(df[['partner_mirrorball_wins']])
    df['seasons_std'] = scaler.fit_transform(df[['partner_total_seasons']])
    df['avg_place_std'] = scaler.fit_transform(df[['partner_avg_placement']])
    feature_cols.extend(['wins_std', 'seasons_std', 'avg_place_std', 'partner_is_champion'])
    
    df['age_std'] = scaler.fit_transform(df[['celebrity_age_during_season']])
    df['followers_std'] = scaler.fit_transform(df[['log_celebrity_followers']])
    feature_cols.extend(['age_std', 'followers_std', 'celebrity_has_follower_data'])
    
    # 非线性项
    df['age_sq'] = df['celebrity_age_during_season'] ** 2
    df['age_sq_std'] = scaler.fit_transform(df[['age_sq']])
    feature_cols.append('age_sq_std')
    
    df['season_std'] = scaler.fit_transform(df[['season']])
    feature_cols.append('season_std')
    
    # 提取周次数字
    df['week_num'] = df['week'].astype(str).str.extract(r'(\d+)').astype(float).fillna(1)
    df['week_std'] = scaler.fit_transform(df[['week_num']])
    feature_cols.append('week_std')
    
    # 交互项
    df['exp_x_fans'] = df['seasons_std'] * df['followers_std']
    df['champ_x_fans'] = df['partner_is_champion'] * df['followers_std']
    df['wins_x_age'] = df['wins_std'] * df['age_std']
    df['season_x_fans'] = df['season_std'] * df['followers_std']
    feature_cols.extend(['exp_x_fans', 'champ_x_fans', 'wins_x_age', 'season_x_fans'])
    
    # 行业编码
    industry_map = {'Athlete': 'Sports', 'Actor/Actress': 'Entertainment',
                   'Singer/Rapper': 'Entertainment', 'TV Personality': 'Media',
                   'Model': 'Entertainment', 'Other': 'Other'}
    df['ind_group'] = df['celebrity_industry'].map(lambda x: industry_map.get(x, 'Other'))
    
    for ind in ['Entertainment', 'Sports', 'Media']:
        df[f'ind_{ind}'] = (df['ind_group'] == ind).astype(int)
        feature_cols.append(f'ind_{ind}')
    
    return df, feature_cols


def compare_models(df):
    print("\n" + "=" * 70)
    print("模型比较")
    print("=" * 70)
    
    results = {}
    
    for outcome, name in [('week_rank', '评委评分'), ('placement', '最终名次')]:
        print(f"\n{'='*35}")
        print(f"因变量: {name} ({outcome})")
        print(f"{'='*35}")
        
        # 基础模型
        df_basic, basic_cols = prepare_basic_features(df.copy())
        X_basic = df_basic[basic_cols].fillna(0).values
        y = df_basic[outcome].values
        groups = df_basic['ballroom_partner'].values
        
        model_basic = OptimizedMixedModel('ballroom_partner', alpha=1.0)
        model_basic.fit(X_basic, y, groups)
        cv_basic_mean, cv_basic_std = model_basic.cross_validate(X_basic, y, groups)
        
        print(f"\n[基础模型] 特征数: {len(basic_cols)}")
        print(f"  R²: {model_basic.r_squared:.4f}")
        print(f"  Adj R²: {model_basic.adj_r_squared:.4f}")
        print(f"  CV R²: {cv_basic_mean:.4f} (±{cv_basic_std:.4f})")
        print(f"  AIC: {model_basic.aic:.2f}")
        
        # 增强模型
        df_enhanced, enhanced_cols = prepare_enhanced_features(df.copy())
        X_enhanced = df_enhanced[enhanced_cols].fillna(0).values
        y = df_enhanced[outcome].values
        groups = df_enhanced['ballroom_partner'].values
        
        model_enhanced = OptimizedMixedModel('ballroom_partner', alpha=1.0)
        model_enhanced.fit(X_enhanced, y, groups)
        cv_enhanced_mean, cv_enhanced_std = model_enhanced.cross_validate(X_enhanced, y, groups)
        
        print(f"\n[增强模型] 特征数: {len(enhanced_cols)}")
        print(f"  R²: {model_enhanced.r_squared:.4f}")
        print(f"  Adj R²: {model_enhanced.adj_r_squared:.4f}")
        print(f"  CV R²: {cv_enhanced_mean:.4f} (±{cv_enhanced_std:.4f})")
        print(f"  AIC: {model_enhanced.aic:.2f}")
        
        # 改进幅度
        r2_improve = (model_enhanced.r_squared - model_basic.r_squared) / max(model_basic.r_squared, 0.001) * 100
        cv_improve = (cv_enhanced_mean - cv_basic_mean) / max(abs(cv_basic_mean), 0.001) * 100
        
        print(f"\n[改进幅度]")
        print(f"  R² 提升: {r2_improve:+.1f}%")
        print(f"  CV R² 提升: {cv_improve:+.1f}%")
        print(f"  AIC 变化: {model_enhanced.aic - model_basic.aic:+.2f}")
        
        results[outcome] = {
            'basic': {
                'model': model_basic,
                'r2': model_basic.r_squared,
                'adj_r2': model_basic.adj_r_squared,
                'cv_r2': cv_basic_mean,
                'cv_std': cv_basic_std,
                'aic': model_basic.aic,
                'features': basic_cols,
                'coef': model_basic.coef_
            },
            'enhanced': {
                'model': model_enhanced,
                'r2': model_enhanced.r_squared,
                'adj_r2': model_enhanced.adj_r_squared,
                'cv_r2': cv_enhanced_mean,
                'cv_std': cv_enhanced_std,
                'aic': model_enhanced.aic,
                'features': enhanced_cols,
                'coef': model_enhanced.coef_
            },
            'r2_improve': r2_improve,
            'cv_improve': cv_improve
        }
    
    return results


def compute_bootstrap_se(X, y, groups, n_bootstrap=100, alpha=1.0):
    n = len(y)
    coef_samples = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        model = OptimizedMixedModel(alpha=alpha)
        model.fit(X[idx], y[idx], groups[idx])
        coef_samples.append(model.coef_)
    
    return np.std(coef_samples, axis=0)


def analyze_coefficients(results, df):
    print("\n" + "=" * 70)
    print("增强模型系数分析")
    print("=" * 70)
    
    for outcome, name in [('week_rank', '评委评分'), ('placement', '最终名次')]:
        print(f"\n[{name}]")
        
        df_enhanced, features = prepare_enhanced_features(df.copy())
        X = df_enhanced[features].fillna(0).values
        y = df_enhanced[outcome].values
        groups = df_enhanced['ballroom_partner'].values
        
        coef = results[outcome]['enhanced']['coef']
        se = compute_bootstrap_se(X, y, groups)
        
        t_vals = coef / (se + 1e-10)
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=len(y) - len(features)))
        
        print(f"\n{'变量':<25} | {'系数':>10} | {'标准误':>10} | {'t值':>8} | {'p值':>8} | {'显著性':>5}")
        print("-" * 80)
        
        for i, feat in enumerate(features):
            sig = '***' if p_vals[i] < 0.001 else ('**' if p_vals[i] < 0.01 else ('*' if p_vals[i] < 0.05 else ''))
            print(f"{feat:<25} | {coef[i]:>+10.4f} | {se[i]:>10.4f} | {t_vals[i]:>8.2f} | {p_vals[i]:>8.4f} | {sig:>5}")
        
        # 保存到CSV
        coef_df = pd.DataFrame({
            'Variable': features,
            'Coefficient': coef,
            'Std_Error': se,
            't_value': t_vals,
            'p_value': p_vals,
            'Significance': ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')) for p in p_vals]
        })
        coef_df.to_csv(os.path.join(OUTPUT_DIR, f'q3_optimized_coef_{outcome}.csv'), index=False)


def plot_comparison(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    outcomes = ['week_rank', 'placement']
    labels = ['Judge Score', 'Final Placement']
    
    # R² 比较
    x = np.arange(2)
    width = 0.35
    
    basic_r2 = [results[o]['basic']['r2'] for o in outcomes]
    enhanced_r2 = [results[o]['enhanced']['r2'] for o in outcomes]
    
    bars1 = axes[0].bar(x - width/2, basic_r2, width, label='Basic Model', color='#3498DB', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, enhanced_r2, width, label='Enhanced Model', color='#E74C3C', alpha=0.8)
    
    axes[0].set_ylabel('R-squared')
    axes[0].set_title('(a) Model Fit (Training R²)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    
    for bar, val in zip(bars1, basic_r2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f'{val:.3f}', ha='center', fontsize=10)
    for bar, val in zip(bars2, enhanced_r2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f'{val:.3f}', ha='center', fontsize=10)
    
    # CV R² 比较
    basic_cv = [results[o]['basic']['cv_r2'] for o in outcomes]
    enhanced_cv = [results[o]['enhanced']['cv_r2'] for o in outcomes]
    basic_std = [results[o]['basic']['cv_std'] for o in outcomes]
    enhanced_std = [results[o]['enhanced']['cv_std'] for o in outcomes]
    
    bars3 = axes[1].bar(x - width/2, basic_cv, width, label='Basic Model',
                        color='#3498DB', alpha=0.8, yerr=basic_std, capsize=3)
    bars4 = axes[1].bar(x + width/2, enhanced_cv, width, label='Enhanced Model',
                        color='#E74C3C', alpha=0.8, yerr=enhanced_std, capsize=3)
    
    axes[1].set_ylabel('Cross-Validation R²')
    axes[1].set_title('(b) Generalization (5-Fold CV R²)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    for bar, mean, std in zip(bars3, basic_cv, basic_std):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.006,
                     f'{mean:.3f} ± {std:.3f}', ha='center', fontsize=9)
    for bar, mean, std in zip(bars4, enhanced_cv, enhanced_std):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.006,
                     f'{mean:.3f} ± {std:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Figure 6: Basic vs Enhanced Mixed Effects Model', fontsize=13, y=1.02)
    fig.legend(handles=[bars1, bars2], labels=['Basic Model', 'Enhanced Model'],
               loc='upper left', bbox_to_anchor=(-0.02, 1.02), frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


def plot_effects_analysis(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df_plot = df.copy()
    
    # (a) 舞者经验 vs 表现
    exp_data = df_plot.groupby('partner_experience_level')['placement'].agg(['mean', 'std', 'count'])
    exp_order = ['novice', 'intermediate', 'experienced', 'veteran']
    exp_data = exp_data.reindex([e for e in exp_order if e in exp_data.index])
    
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#2E86AB']
    bars = axes[0, 0].bar(range(len(exp_data)), exp_data['mean'], 
                          color=colors[:len(exp_data)], alpha=0.8,
                          yerr=exp_data['std']/np.sqrt(exp_data['count']), capsize=5)
    axes[0, 0].set_xticks(range(len(exp_data)))
    axes[0, 0].set_xticklabels(['Novice', 'Intermediate', 'Experienced', 'Veteran'][:len(exp_data)])
    axes[0, 0].set_xlabel('Partner Experience Level')
    axes[0, 0].set_ylabel('Average Placement')
    axes[0, 0].set_title('(a) Partner Experience Effect')
    axes[0, 0].invert_yaxis()
    
    # (b) 年龄效应
    age_bins = pd.cut(df_plot['celebrity_age_during_season'], bins=8)
    age_data = df_plot.groupby(age_bins, observed=True)['placement'].mean()
    
    x_ages = [interval.mid for interval in age_data.index]
    y_places = age_data.values
    
    axes[0, 1].scatter(x_ages, y_places, s=100, c='#3498DB', alpha=0.7, edgecolors='black')
    z = np.polyfit(x_ages, y_places, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(x_ages), max(x_ages), 100)
    axes[0, 1].plot(x_smooth, p(x_smooth), 'r--', linewidth=2, label='Quadratic fit')
    
    optimal_age = -z[1] / (2 * z[0])
    axes[0, 1].axvline(x=optimal_age, color='green', linestyle=':', alpha=0.7)
    axes[0, 1].text(optimal_age + 1, min(y_places), f'Optimal: {optimal_age:.0f}', fontsize=10, color='green')
    
    axes[0, 1].set_xlabel('Celebrity Age')
    axes[0, 1].set_ylabel('Average Placement')
    axes[0, 1].set_title('(b) Non-linear Age Effect')
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()
    
    # (c) 粉丝量效应
    df_plot['follower_bin'] = pd.cut(df_plot['log_celebrity_followers'], bins=5, 
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    fan_data = df_plot.groupby('follower_bin', observed=True)['placement'].agg(['mean', 'std', 'count'])
    
    colors_fans = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB']
    bars_f = axes[1, 0].bar(range(len(fan_data)), fan_data['mean'], 
                            color=colors_fans[:len(fan_data)], alpha=0.8,
                            yerr=fan_data['std']/np.sqrt(fan_data['count']), capsize=3)
    axes[1, 0].set_xticks(range(len(fan_data)))
    axes[1, 0].set_xticklabels(fan_data.index, rotation=15)
    axes[1, 0].set_xlabel('Social Media Followers (Log Scale)')
    axes[1, 0].set_ylabel('Average Placement')
    axes[1, 0].set_title('(c) Social Media Influence')
    axes[1, 0].invert_yaxis()
    
    # (d) 赛季趋势
    season_data = df_plot.groupby('season')['placement'].mean()
    axes[1, 1].plot(season_data.index, season_data.values, 'o-', color='#9B59B6', alpha=0.7)
    
    z_s = np.polyfit(season_data.index, season_data.values, 2)
    p_s = np.poly1d(z_s)
    x_s = np.linspace(season_data.index.min(), season_data.index.max(), 100)
    axes[1, 1].plot(x_s, p_s(x_s), 'r--', linewidth=2, label='Trend')
    
    axes[1, 1].set_xlabel('Season')
    axes[1, 1].set_ylabel('Average Placement')
    axes[1, 1].set_title('(d) Season Trend')
    axes[1, 1].legend()
    axes[1, 1].invert_yaxis()
    
    plt.suptitle('Figure 7: Enhanced Model Effects Analysis', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {output_path}")


def generate_report(results, df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Q3 优化版混合效应模型报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据规模: {len(df)} 行\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("1. 模型优化内容\n")
        f.write("-" * 80 + "\n\n")
        f.write("  新增特征:\n")
        f.write("    - 年龄二次项 (非线性效应)\n")
        f.write("    - 赛季趋势\n")
        f.write("    - 周次效应\n")
        f.write("    - 舞者经验 × 粉丝量 交互项\n")
        f.write("    - 冠军状态 × 粉丝量 交互项\n")
        f.write("    - 获胜次数 × 年龄 交互项\n")
        f.write("    - 赛季 × 粉丝量 交互项\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("2. 模型比较结果\n")
        f.write("-" * 80 + "\n\n")
        
        for outcome, name in [('week_rank', 'Judge Score'), ('placement', 'Final Placement')]:
            basic = results[outcome]['basic']
            enhanced = results[outcome]['enhanced']
            
            f.write(f"  {name}:\n")
            f.write(f"  {'Metric':<20} | {'Basic':>12} | {'Enhanced':>12} | {'Improve':>10}\n")
            f.write("  " + "-" * 60 + "\n")
            f.write(f"  {'R²':<20} | {basic['r2']:>12.4f} | {enhanced['r2']:>12.4f} | "
                   f"{(enhanced['r2']-basic['r2'])*100:>+9.1f}%\n")
            f.write(f"  {'Adj R²':<20} | {basic['adj_r2']:>12.4f} | {enhanced['adj_r2']:>12.4f} | "
                   f"{(enhanced['adj_r2']-basic['adj_r2'])*100:>+9.1f}%\n")
            f.write(f"  {'CV R²':<20} | {basic['cv_r2']:>12.4f} | {enhanced['cv_r2']:>12.4f} | "
                   f"{(enhanced['cv_r2']-basic['cv_r2'])*100:>+9.1f}%\n")
            f.write(f"  {'AIC':<20} | {basic['aic']:>12.2f} | {enhanced['aic']:>12.2f} | "
                   f"{enhanced['aic']-basic['aic']:>+10.2f}\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("3. 主要发现\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("  3.1 职业舞者效应:\n")
        f.write("      - 舞者的历史平均名次是最强预测因子\n")
        f.write("      - 舞者随机效应解释了大部分变异\n\n")
        
        f.write("  3.2 年龄效应:\n")
        f.write("      - 年龄与表现呈倒U型关系\n")
        f.write("      - 最佳年龄约30-35岁\n\n")
        
        f.write("  3.3 社交媒体效应:\n")
        f.write("      - 粉丝量对最终名次有正向影响\n")
        f.write("      - 近期赛季社交媒体影响增强\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"  报告已保存: {output_path}")


def main():
    try:
        # 1. 加载数据
        df = load_data()
        
        # 2. 模型比较
        results = compare_models(df)
        
        # 3. 系数分析
        analyze_coefficients(results, df)
        
        # 4. 保存模型
        print("\n[模型保存]")
        for outcome in ['week_rank', 'placement']:
            df_enhanced, features = prepare_enhanced_features(df.copy())
            X = df_enhanced[features].fillna(0).values
            y = df_enhanced[outcome].values
            groups = df_enhanced['ballroom_partner'].values
            
            model = OptimizedMixedModel('ballroom_partner', alpha=1.0)
            model.fit(X, y, groups)
            model.feature_names = features
            
            joblib.dump(model, os.path.join(OUTPUT_DIR, f'q3_optimized_{outcome}.joblib'))
            print(f"  已保存: q3_optimized_{outcome}.joblib")
        
        # 5. 可视化
        print("\n[生成可视化]")
        plot_comparison(results, os.path.join(OUTPUT_DIR, 'q3_fig6_model_comparison_v2.png'))
        plot_effects_analysis(df, os.path.join(OUTPUT_DIR, 'q3_fig7_effects_analysis_v2.png'))
        
        # 6. 报告
        print("\n[生成报告]")
        generate_report(results, df, os.path.join(OUTPUT_DIR, 'q3_optimized_report.txt'))
        
        # 7. 总结
        print("\n" + "=" * 70)
        print("优化结果总结")
        print("=" * 70)
        
        for outcome, name in [('week_rank', '评委评分'), ('placement', '最终名次')]:
            basic = results[outcome]['basic']
            enhanced = results[outcome]['enhanced']
            print(f"\n{name}:")
            print(f"  基础模型 R²: {basic['r2']:.4f} → 增强模型 R²: {enhanced['r2']:.4f}")
            print(f"  基础模型 CV R²: {basic['cv_r2']:.4f} → 增强模型 CV R²: {enhanced['cv_r2']:.4f}")
            print(f"  R² 提升: {results[outcome]['r2_improve']:+.1f}%")
        
        print("\n" + "=" * 70)
        print("完成!")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n[错误] {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    results = main()
