# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:20:38 2025

@author: wqxlm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score
import joblib
#%% 加载中心曲线和残差基
CENTER_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/cluster_centers_by_type_season.xlsx'
center_df = pd.read_excel(CENTER_FILE) # P1~P96、building_type、cluster_id
SHAPE_BASE_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/residual_shape_base.xlsx'
shape_base_df = pd.read_excel(SHAPE_BASE_FILE) # P1~P96、building_type、cluster_id

#%% 加载标签预测模型
def load_label_model(building_type):
    label_model = xgb.XGBClassifier()
    label_model.load_model(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/xgb_model_{building_type}.json")
    label_model.n_classes_ = 12
    label_model.classes_ = np.arange(12)
    return label_model

#%% 加载基于残差的曲线修正模型
def load_residual_model(building_type):
    residual_model = xgb.XGBRegressor()
    residual_model.load_model(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/residual_xgb_model_{building_type}.json")
    return residual_model
    

#%% 加载or输入行为特征及天气时间特征
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

WEATHER_FILE = 'D:/Doccument/Papers/00 Journal/Electricity load prediction/杭州天气数据集.xlsx'
weather_df = pd.read_excel(WEATHER_FILE)

date_range = pd.date_range(start="2023-03-01", end="2024-03-31", freq='D')

daily_weather_time_df = generate_weather_time_features(weather_df, date_range)
#%% 合并特征

def build_training_data(user_feature_df, weather_df, sensitivity_df):
    df = weather_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 将一行广播成与 df 行数相同
    user_features = pd.concat([user_feature_df]*len(df), ignore_index=True)
    sensitivity_features = pd.concat([sensitivity_df]*len(df), ignore_index=True)
    
    # 合并：直接按行位置拼接
    df = pd.concat([df.reset_index(drop=True), user_features, sensitivity_features], axis=1)

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

    building_features = df[features].values
    
    return building_features

#%% 簇编码解码

def decode_pred(pred):
    """
    把 pred (1~12) 解码为 season, cluster_id(1~4)
    """
    season_order = ['cold', 'transition', 'hot']
    idx = pred - 1
    season = season_order[idx // 4]
    cluster_id = (idx % 4) + 1
    return season, cluster_id

#%% 预测标签和曲线修正值
def building_load_generation(building_type, user_features, center_df, shape_base_df):
    label_model = load_label_model(building_type)
    residual_model = load_residual_model(building_type)
    
    loaded_X_scaler = joblib.load(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/X_standard_scaler_{building_type}.pkl")
    user_features = loaded_X_scaler.transform(user_features)
    
    label_pred = label_model.predict(user_features)
    residual_pred = residual_model.predict(user_features)
    center_originals = []
    for index, pred in enumerate(label_pred):
        season, cluster_id = decode_pred(pred)
        ct = center_df[
            (center_df['building_type'] == building_type) &
            (center_df['season'] == season) &
            (center_df['cluster_id'] == cluster_id)
        ]
        if ct.empty:
            raise ValueError(f"找不到中心线: {building_type}, {season}, 簇{cluster_id}")
        center_original = ct[[f'P{i}' for i in range(1,97)]].values.flatten()
        center_originals.append(center_original)
    center_curves = np.vstack(center_originals).flatten()
    loaded_y_resid_scaler = joblib.load(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/y_weight_standard_scaler_{building_type}.pkl")
    
    retransfered_resid_pred = loaded_y_resid_scaler.inverse_transform(residual_pred)
    
    shape_base_curves = shape_base_df[shape_base_df['building_type'] == building_type].iloc[:, 2:]
    resid_factor = np.dot(retransfered_resid_pred, shape_base_curves).flatten()
    modified_curve = center_curves * (resid_factor+1)
        
    return modified_curve
    
#%% 调用


"""
用户输入项
behaviour_features =  'annual_max', 'annual_cv', 'avg_daily_peak_valley_ratio',
        'median_startup_hour', 'median_shutdown_hour', 'median_rise_slope_summer',
behaviour_features =  'temperature_sensitivity', 'enthalpy_sensitivity
"""

annual_max = 4.18
annual_cv = 0.78
peak_valley_ratio = 0.74
startup_hour = 5.5
shutdown_hour = 22.75
rise_slope_summer = 0.357

temperature_sensitivity = 0.0648
enthalpy_sensitivity = 0.182

building_types = ['Office', 'Mall', 'Hotel', 'School', 'Gym', 'Hospital', 'Industries', 'Hi_tech_industries']

gui_user_feature = pd.DataFrame(data=[[annual_max,annual_cv,peak_valley_ratio, startup_hour,shutdown_hour,rise_slope_summer]], 
                  columns=['annual_max', 'annual_cv', 'avg_daily_peak_valley_ratio',
        'median_startup_hour', 'median_shutdown_hour', 'median_rise_slope_summer'])

gui_sensitivity_df = pd.DataFrame(data=[[temperature_sensitivity,enthalpy_sensitivity]], 
                  columns=['temperature_sensitivity', 'enthalpy_sensitivity'])

user_features = build_training_data(gui_user_feature, daily_weather_time_df, gui_sensitivity_df)


generated_curve = building_load_generation(building_types[0], user_features, center_df, shape_base_df)

#%% 调试

annual_max = 4.18
annual_cv = 0.78
peak_valley_ratio = 0.74
startup_hour = 5.5
shutdown_hour = 22.75
rise_slope_summer = 0.357

temperature_sensitivity = 0.0648
enthalpy_sensitivity = 0.182

gui_user_feature = pd.DataFrame(data=[[annual_max,annual_cv,peak_valley_ratio, startup_hour,shutdown_hour,rise_slope_summer]], 
                  columns=['annual_max', 'annual_cv', 'avg_daily_peak_valley_ratio',
        'median_startup_hour', 'median_shutdown_hour', 'median_rise_slope_summer'])

gui_sensitivity_df = pd.DataFrame(data=[[temperature_sensitivity,enthalpy_sensitivity]], 
                  columns=['temperature_sensitivity', 'enthalpy_sensitivity'])

user_features = build_training_data(gui_user_feature, daily_weather_time_df, gui_sensitivity_df)


building_type = 'Office'
label_model = load_label_model('Office')
residual_model = load_residual_model(building_type)

loaded_X_scaler = joblib.load(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/X_standard_scaler_{building_type}.pkl")
user_features = loaded_X_scaler.transform(user_features)

label_pred = label_model.predict(user_features)
residual_pred = residual_model.predict(user_features)
center_originals = []
for index, pred in enumerate(label_pred):
    season, cluster_id = decode_pred(pred)
    ct = center_df[
        (center_df['building_type'] == building_type) &
        (center_df['season'] == season) &
        (center_df['cluster_id'] == cluster_id)
    ]
    if ct.empty:
        raise ValueError(f"找不到中心线: {building_type}, {season}, 簇{cluster_id}")
    center_original = ct[[f'P{i}' for i in range(1,97)]].values.flatten()
    center_originals.append(center_original)
center_curves = np.vstack(center_originals).flatten()
loaded_y_resid_scaler = joblib.load(f"D:/Doccument/Papers/00 Journal/Electricity load prediction/XGB model/y_weight_standard_scaler_{building_type}.pkl")

retransfered_resid_pred = loaded_y_resid_scaler.inverse_transform(residual_pred)

shape_base_curves = shape_base_df[shape_base_df['building_type'] == building_type].iloc[:, 2:]
resid_factor = np.dot(retransfered_resid_pred, shape_base_curves).flatten()
modified_curve = center_curves * (resid_factor+1)
print(shape_base_curves)