# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 20:46:13 2025

@author: wqxlm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 13:41:24 2025

@author: wqxlm
"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
# 加载数据
SEED = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def residual_matrix(center_df, label_df): 
    # 创建字典用于快速查找中心曲线
    center_dict = {}
    for _, row in center_df.iterrows():
        key = (row['building_type'], row['season'], row['cluster_id'])
        center_curve = [row[f'P{i}'] for i in range(1, 97)]
        center_dict[key] = center_curve
    
    # 初始化列表存储残差记录
    residual_records = []
    
    # 遍历 label_df 计算残差
    for idx, row in label_df.iterrows():
        btype = row['building_type']
        building = row['building']
        season = row['season']
        label = row['cluster_label']
        date = row['date']
    
        key = (btype, season, label)
        if key not in center_dict:
            continue  # 跳过找不到中心曲线的情况
        
        center_curve = center_dict[key]
        actual_curve = np.array([row[f'P{i}'] for i in range(1, 97)], dtype=np.float32)
        safe_center = np.clip(center_curve, 1e-4, None)
        
        residual = actual_curve / safe_center
    
        record = {
            'building_type': btype,
            'building': building,
            'season': season,
            'date': date
        }
        for i in range(96):
            record[f'residual_{i+1}'] = residual[i]
        
        residual_records.append(record)
    
    # 构造 residual_df
    residual_df = pd.DataFrame(residual_records)
    return residual_df


def compute_pca_basis_by_building_type(residual_df, n_components=6):
    
    """
    对每类建筑的残差曲线进行 PCA，提取主成分形状基
    :param residual_df: 包含 building_type, residual_1 ~ residual_96 的 DataFrame
    :param variance_threshold: PCA 保留的累计方差比例（如 0.95）
    :param plot: 是否可视化每类建筑的前几个主成分
    :return: pca_basis_dict: dict[building_type] = {'pca': PCA对象, 'components': ndarray, 'explained_ratio': list}
    """
    
    pca_basis_dict = {}

    for btype, group in residual_df.groupby('building_type'):
        print(f"处理建筑类型：{btype}，样本数：{len(group)}")

        residual_matrix = group[[f'residual_{i+1}' for i in range(96)]].values

        pca = PCA(n_components=n_components, svd_solver='full')
        basis_weights = pca.fit_transform(residual_matrix)

        residual_basis = pca.components_  # shape: [n_components, 96]

        pca_basis_dict[btype] = {
            'pca': pca,
            'basis_weights': basis_weights,
            'residual_basis': residual_basis,
        }

    rows = []

    for btype, content in pca_basis_dict.items():
        basis = content['residual_basis']  # shape: [n_components, 96]
        for i, row in enumerate(basis):
            row_dict = {
                'building_type': btype,
                'PC': f'PC{i+1}'
            }
            row_dict.update({f't{j+1}': val for j, val in enumerate(row)})
            rows.append(row_dict)

    flat_df = pd.DataFrame(rows)
    flat_df.to_excel('D:/Doccument/Papers/00 Journal/Electricity load prediction/residual_shape_base.xlsx', index=False)
    
    return pca_basis_dict
    

# 加载标签数据
def load_label_data(filepath):
    df = pd.read_excel(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['building_type', 'building', 'date']).reset_index(drop=True)
    return df


def build_training_data(user_feature_df):
    
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

    feature = user_feature_df[features].values
    return feature

#%% ==========主函数============

LABEL_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/cluster_labels_by_day.xlsx'
CENTERS_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/cluster_centers_by_type_season.xlsx'
FEATURE_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/behaviour_feature_dataset.xlsx'

center_df = pd.read_excel(CENTERS_FILE)
label_df = load_label_data(LABEL_FILE)
feature_df = pd.read_excel(FEATURE_FILE)

building_types = label_df['building_type'].unique()

residual_df = residual_matrix(center_df, label_df)
pca_basis_dict = compute_pca_basis_by_building_type(residual_df)


building_type_best_params = []
building_type_score_values = []

for btype in building_types[1:]:
    print(f"=== 建筑类型: {btype} ===")
    
    btype_user_feature_df = feature_df[feature_df['building_type'] == btype]
    X_features = build_training_data(btype_user_feature_df)
    pca_dataset = pca_basis_dict.get(btype)
    y = pca_dataset['basis_weights']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.1, shuffle=True, random_state=SEED)
    # 加载 scaler
    loaded_X_scaler = joblib.load(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/X_standard_scaler_{btype}.pkl")
    X_train = loaded_X_scaler.transform(X_train)
    X_test = loaded_X_scaler.transform(X_test)
    
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    joblib.dump(y_scaler, f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/y_weight_standard_scaler_{btype}.pkl")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 9, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        
    cv = KFold(n_splits=10, shuffle=True)
        
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=cv,
            verbose=0,
            n_jobs=-1
        )
        
    grid_search.fit(X_train, y_train)
        
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    building_type_score_values.append(cv_score)
    building_type_best_params.append(best_params)

    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    r2 = []
    for i in range(len(y_test)):
        r2.append(r2_score(y_test[i], y_pred[i])) 
    r2_mean = np.array(r2).mean()
    print(f'R² (Coefficient of Determination): {r2_mean}')
    best_model.save_model(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/residual_xgb_model_{btype}.json")
    
#%% TEST for label prediction
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
  
for btype in building_types:
    print(f"=== 建筑类型: {btype} ===")

    btype_user_feature_df = feature_df[feature_df['building_type'] == btype]
    all_building_features = build_training_data(btype_user_feature_df)
    loaded_X_scaler = joblib.load(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/X_standard_scaler_{btype}.pkl")
    # 按 building 分组后取第2个用户
    user = (
    btype_user_feature_df.groupby('building')
    .head(1)  # 每个用户只取一条（如果重复）
    .reset_index(drop=True)
    .iloc[-5:]  # 取第2个用户（第1个是索引0）
    )

    # 如果你想要这个用户的所有数据：
    user_id = user['building']
    user_id_value = user_id.iloc[2]
    user_data = btype_user_feature_df[btype_user_feature_df['building'] == user_id_value]
    
    X_features = build_training_data(user_data)
    X_features = loaded_X_scaler.transform(X_features)
    y_real =  np.array(user_data['cluster_id']) - 1
    
    model = xgb.XGBClassifier()
    model.load_model(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/xgb_model_{btype}.json")
    model.n_classes_ = 12
    model.classes_ = np.arange(12)  # 即 [0, 1, ..., 11]
    y_pred = model.predict(X_features)
    
    plot_confusion(y_real, y_pred, title=f"{btype} - Confusion Matrix")
    print(classification_report(y_real, y_pred))