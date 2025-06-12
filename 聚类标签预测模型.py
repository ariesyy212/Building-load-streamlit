
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
import joblib
# 参数
BUILDING_TYPES = []  # 自动识别
SEED = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 加载标签数据
def load_label_data(filepath):
    df = pd.read_excel(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['building_type', 'building', 'date']).reset_index(drop=True)
    return df

# 构建标签编码（cold_1 ~ transition_4 -> 1 ~ 12）
def encode_cluster_labels(df):
    season_order = ['cold', 'transition', 'hot']
    label_map = {}
    label_counter = 1
    for season in season_order:
        for i in range(1, 5):
            label_map[f'{season}_{i}'] = label_counter
            label_counter += 1
    df['cluster_key'] = df['season'] + '_' + df['cluster_label'].astype(str)
    df['cluster_id'] = df['cluster_key'].map(label_map)
    return df, label_map

def decode_pred(pred):
    """
    把 pred (1~12) 解码为 season, cluster_id(1~4)
    """
    season_order = ['cold', 'transition', 'hot']
    idx = pred - 1
    season = season_order[idx // 4]
    cluster_id = (idx % 4) + 1
    return season, cluster_id

# 提取用户行为特征（全年统计特征）
def extract_behavior_features(df, weather_df):
    feature_list = []
    for (btype, user), group in df.groupby(['building_type', 'building']):
        power_cols = [f'P{i}' for i in range(1, 97)]
        user_power = group[power_cols].dropna()

        if user_power.empty:
            continue

        all_vals = user_power.values.flatten()
        peak_vals = user_power.max(axis=1)
        valley_vals = user_power.min(axis=1)
        peak_valley_ratios = (peak_vals - valley_vals) / (peak_vals + 1e-5)

        start_hours = []
        end_hours = []
        slopes = []

        for row in user_power.values:
            peak = np.max(row)
            if peak == 0:
                continue
            above_30 = np.where(row >= 0.3 * peak)[0]
            if len(above_30) > 0:
                start_hours.append(above_30[0] * 0.25)
                end_hours.append(above_30[-1] * 0.25)

            summer_segment = row[24:40]  # 6:00-10:00
            if np.max(summer_segment) > 0:
                try:
                    start = np.where(summer_segment > 0.3 * np.max(summer_segment))[0][0]
                    end = np.where(summer_segment > 0.9 * np.max(summer_segment))[0][0]
                    if end > start:
                        slope = (summer_segment[end] - summer_segment[start]) / ((end - start) * 0.25)
                        slopes.append(slope)
                except:
                    continue

        user_feats = {
            'building_type': btype,
            'building': user,
            'annual_max': all_vals.max(),
            'annual_cv': all_vals.std() / all_vals.mean() if all_vals.mean() else 0,
            'avg_daily_peak_valley_ratio': peak_valley_ratios.mean(),
            'median_startup_hour': np.median(start_hours) if start_hours else np.nan,
            'median_shutdown_hour': np.median(end_hours) if end_hours else np.nan,
            'median_rise_slope_summer': np.median(slopes) if slopes else np.nan
        }

        feature_list.append(user_feats)

    return pd.DataFrame(feature_list)

def generate_weather_time_features(weather_df, date_range):
    """
    从逐时气象数据中提取每日天气特征 + 时间特征
    :param weather_df: 包含 Timestamp, DryBulbT(C), Humidity(%), SolarRadiation(W/m2) 等列
    :param date_range: 一个包含所有日期的 pd.date_range，如 pd.date_range("2023-03-01", "2024-03-31")
    :return: 按日期提取的特征表 DataFrame（以 date 为索引）
    """

    # 确保时间格式正确
    weather_df = weather_df.copy()
    weather_df['Timestamp'] = pd.to_datetime(weather_df['Timestamp'])
    weather_df['date'] = weather_df['Timestamp'].dt.date
    weather_df['hour'] = weather_df['Timestamp'].dt.hour
    
    # 聚合为每日特征
    daily_weather = weather_df.groupby('date').agg({
        'DryBulbT(C)': ['mean', 'max', 'min', lambda x: x.max() - x.min()],
        'Humidity(%)': ['mean', 'max', 'min'],
        'SolarRadiation(W/m2)': ['sum', 'max'],
         'Precipitation': 'sum', 
        'Enthalpy(kJ/kg)': ['mean', lambda x: x.max() - x.min()]
    })
    daily_weather.columns = ['_'.join(col).strip() for col in daily_weather.columns.values]
    daily_weather = daily_weather.rename(columns={
        'DryBulbT(C)_<lambda_0>': 'T_range',
        'Enthalpy(kJ/kg)_<lambda_0>': 'Enthalpy_range'
    })
    daily_weather['is_rainy'] = weather_df.groupby('date')['Precipitation'].sum() > 0
    daily_weather['is_rainy'] = daily_weather['is_rainy'].astype(int)
    daily_weather = daily_weather.reset_index()
    # 时间特征（构造完整日期表）
    date_df = pd.DataFrame({'date': date_range})
    date_df['month'] = date_df['date'].dt.month
    date_df['weekday'] = date_df['date'].dt.weekday  # 0 = Monday
    date_df['dayofyear'] = date_df['date'].dt.dayofyear
    date_df['is_weekend'] = date_df['weekday'].isin([5, 6]).astype(int)

    # sin/cos 编码
    date_df['month_sin'] = np.sin(2 * np.pi * date_df['month'] / 12)
    date_df['month_cos'] = np.cos(2 * np.pi * date_df['month'] / 12)
    date_df['weekday_sin'] = np.sin(2 * np.pi * date_df['weekday'] / 7)
    date_df['weekday_cos'] = np.cos(2 * np.pi * date_df['weekday'] / 7)
    date_df['dayofyear_sin'] = np.sin(2 * np.pi * date_df['dayofyear'] / 365)
    date_df['dayofyear_cos'] = np.cos(2 * np.pi * date_df['dayofyear'] / 365)

    # 合并天气与时间特征
    date_df['date'] = pd.to_datetime(date_df['date'])
    daily_weather['date'] = pd.to_datetime(daily_weather['date'])
    daily_features = pd.merge(date_df, daily_weather, on='date', how='left')

    return daily_features


def compute_temperature_sensitivity(label_df, raw_weather_df):
    raw_weather_df = raw_weather_df.copy()
    raw_weather_df['Timestamp'] = pd.to_datetime(raw_weather_df['Timestamp'])
    raw_weather_df['date'] = raw_weather_df['Timestamp'].dt.date
    raw_weather_df['Enthalpy(kJ/kg)'] = raw_weather_df['Enthalpy(kJ/kg)'].astype(float)

    daily_weather = raw_weather_df.groupby('date').agg({
        'DryBulbT(C)': 'mean',
        'Enthalpy(kJ/kg)': 'mean'
    }).reset_index()
    daily_weather.columns = ['date', 'DryBulbT(C)_mean', 'Enthalpy(kJ/kg)_mean']

    result = []
    for (btype, building), group in label_df.groupby(['building_type', 'building']):
        temp = group.copy()
        temp['date'] = pd.to_datetime(temp['date']).dt.date
        merged = temp.merge(daily_weather, on='date', how='left')
        merged['daily_load'] = merged[[f'P{i}' for i in range(1, 97)]].sum(axis=1)

        try:
            temp_corr = np.corrcoef(merged['daily_load'], merged['DryBulbT(C)_mean'])[0, 1]
        except:
            temp_corr = None
        try:
            enthalpy_corr = np.corrcoef(merged['daily_load'], merged['Enthalpy(kJ/kg)_mean'])[0, 1]
        except:
            enthalpy_corr = None

        result.append({
            'building_type': btype,
            'building': building,
            'temperature_sensitivity': temp_corr,
            'enthalpy_sensitivity': enthalpy_corr
        })
    return pd.DataFrame(result)

# 构建训练数据集
def build_training_data(label_df, user_feature_df, weather_df, sensitivity_df):
    df = label_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    df = df.merge(weather_df, on='date', how='left')
    df = df.merge(user_feature_df, on=['building_type', 'building'], how='left')
    df = df.merge(sensitivity_df, on=['building_type', 'building'], how='left')
    df = df.dropna(subset=['cluster_id'])

    features = [ 'is_weekend', 'month_sin', 'month_cos', 'weekday_sin', 
                'weekday_cos', 'dayofyear_sin', 'dayofyear_cos',
        'DryBulbT(C)_mean', 'DryBulbT(C)_max', 'DryBulbT(C)_min', 'T_range',
        'Humidity(%)_mean', 'Humidity(%)_max', 'Humidity(%)_min',
        'SolarRadiation(W/m2)_sum', 'SolarRadiation(W/m2)_max',
        'Enthalpy(kJ/kg)_mean', 'Enthalpy_range', 'is_rainy',
        'annual_max', 'annual_cv', 'avg_daily_peak_valley_ratio',
        'median_startup_hour', 'median_shutdown_hour', 'median_rise_slope_summer',
        'temperature_sensitivity', 'enthalpy_sensitivity'
    ]

    X = df[features].values
    y = df['cluster_id'].astype(int).values - 1
    return X, y, df

# 可视化混淆矩阵
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

#%%  ============主函数============

# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
#         'random_state': SEED,
#         'eval_metric': 'mlogloss',
#         'use_label_encoder': False
#     }

#     model = XGBClassifier(**params)
#     score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
#     return score

if __name__ == '__main__':
    
    
    LABEL_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/cluster_labels_by_day.xlsx'
    WEATHER_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/杭州天气数据集.xlsx'
    CENTER_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/cluster_centers_by_type_season.xlsx'
    
    
    center_df = pd.read_excel(CENTER_FILE)
    weather_df = pd.read_excel(WEATHER_FILE)
    label_df = load_label_data(LABEL_FILE)
    label_df, label_map = encode_cluster_labels(label_df)
    building_types = label_df['building_type'].unique()
    date_range = pd.date_range(start="2023-03-01", end="2024-03-31", freq='D')
    
    user_feature_df = extract_behavior_features(label_df, weather_df)
    sensitivity_df = compute_temperature_sensitivity(label_df, weather_df)
    daily_weather_time_df = generate_weather_time_features(weather_df, date_range)
    
#     _, _, feature_dataset = build_training_data(label_df, user_feature_df, daily_weather_time_df, sensitivity_df)
# with pd.ExcelWriter("D:/Doccument/Papers/00 Journal/Electricity load prediction/behaviour_feature_dataset.xlsx", engine='openpyxl') as writer:
#     feature_dataset.to_excel(writer)

    building_type_best_params = []
    building_type_shap_values = []
    building_type_report = []
    for btype in building_types[6:]:
        print(f"=== 建筑类型: {btype} ===")
        btype_df = label_df[label_df['building_type'] == btype]
        btype_user_feature_df = user_feature_df[user_feature_df['building_type'] == btype]
        X, y, merged = build_training_data(btype_df, btype_user_feature_df, daily_weather_time_df, sensitivity_df)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/X_standard_scaler_{btype}.pkl")
        
        param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 9, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        model = XGBClassifier(random_state=SEED, eval_metric='mlogloss')
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            verbose=0,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        building_type_best_params.append(best_params)

        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        plot_confusion(y_test, y_pred, title=f"{btype} - Confusion Matrix")
        
        explainer = shap.TreeExplainer(best_model)
    
        feature_names = [ 'is_weekend', 'month_sin', 'month_cos', 'weekday_sin', 
                        'weekday_cos', 'dayofyear_sin', 'dayofyear_cos',
                'DryBulbT(C)_mean', 'DryBulbT(C)_max', 'DryBulbT(C)_min', 'T_range',
                'Humidity(%)_mean', 'Humidity(%)_max', 'Humidity(%)_min',
                'SolarRadiation(W/m2)_sum', 'SolarRadiation(W/m2)_max',
                'Enthalpy(kJ/kg)_mean', 'Enthalpy_range', 'is_rainy',
                'annual_max', 'annual_cv', 'avg_daily_peak_valley_ratio',
                'median_startup_hour', 'median_shutdown_hour', 'median_rise_slope_summer',
                'temperature_sensitivity', 'enthalpy_sensitivity'
                ]
        
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        shap_values = explainer.shap_values(X_test_df)
        explanation = explainer(X_test_df)
        # shap.initjs()
        # shap.plots.force(explainer.expected_value[0], shap_values[0][0], matplotlib=True)
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, feature_names=feature_names)
        # 可视化所有类别（多分类 shap 会返回 list）
        # for i in range(len(shap_values)):
        #     # shap.summary_plot(shap_values[i], X_test_df, plot_type='bar', show=False)
        #     shap.summary_plot(shap_values[i], X_test_df, plot_type='dot')
        #     plt.title(f'SHAP Feature Importance - Class {i+1}')
        #     plt.tight_layout()
        #     plt.show()
        building_type_shap_values.append(shap_values)

        best_model.save_model(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/xgb_model_{btype}.json")
        print("模型已保存")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        building_type_report.append(report_dict)

