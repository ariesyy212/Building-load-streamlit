# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:39:35 2025

@author: wqxlm
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from execute_model import (
    generate_weather_time_features,
    build_training_data,
    building_load_generation
)

# 文件上传区
st.sidebar.subheader("📤 上传模型数据文件")
uploaded_center = st.sidebar.file_uploader("中心曲线 (cluster_centers_by_type_season.xlsx)", type="xlsx")
uploaded_resid = st.sidebar.file_uploader("残差基曲线 (residual_shape_base.xlsx)", type="xlsx")
uploaded_weather = st.sidebar.file_uploader("天气数据 ", type="xlsx")

# 文件加载状态检查
ready = uploaded_center and uploaded_resid and uploaded_weather
if ready:
    try:
        center_df = pd.read_excel(uploaded_center)
        shape_base_df = pd.read_excel(uploaded_resid)
        weather_df = pd.read_excel(uploaded_weather)
        date_range = pd.date_range(start="2023-03-01", end="2024-03-31", freq='D')  # 可调整
        daily_weather_time_df = generate_weather_time_features(weather_df, date_range)
        st.sidebar.success("✅ 文件加载成功")
    except Exception as e:
        st.sidebar.error(f"❌ 文件读取失败：{e}")
        ready = False
else:
    st.sidebar.info("📂 请上传全部三个 Excel 文件")
    
    
st.set_page_config(layout="wide")
st.title("🔋 建筑日负荷曲线生成工具")

# 建筑类型列表
building_types = ['Office', 'Mall', 'Hotel', 'School', 'Gym', 'Hospital', 'Industries', 'Hi_tech_industries']
daily_weather_time_df = generate_weather_time_features(weather_df, date_range)

# 表单输入区
with st.sidebar.form("input_form"):
    st.subheader("📋 输入参数")

    annual_max = st.number_input("年最大负荷 (annual_max)", value=4.18)
    annual_cv = st.number_input("年变异系数 (annual_cv)", value=0.78)
    peak_valley_ratio = st.number_input("平均峰谷比", value=0.74)
    startup_hour = st.number_input("平均启动时间 (小时)", value=5.5)
    shutdown_hour = st.number_input("平均关闭时间 (小时)", value=22.75)
    rise_slope_summer = st.number_input("夏季早上爬坡斜率", value=0.357)

    temperature_sensitivity = st.number_input("温度敏感性", value=0.0648)
    enthalpy_sensitivity = st.number_input("焓敏感性", value=0.182)

    building_type = st.selectbox("建筑类型", building_types)

    submit = st.form_submit_button("生成负荷曲线")

# 用户点击生成后
if submit:
    if not ready:
        st.error("❌ 请先上传完整的 Excel 文件再生成曲线")
    
    else:
        try:
            user_feature_df = pd.DataFrame([[annual_max, annual_cv, peak_valley_ratio, startup_hour,
                                             shutdown_hour, rise_slope_summer]],
                                           columns=['annual_max', 'annual_cv', 'avg_daily_peak_valley_ratio',
                                                    'median_startup_hour', 'median_shutdown_hour', 'median_rise_slope_summer'])
    
            sensitivity_df = pd.DataFrame([[temperature_sensitivity, enthalpy_sensitivity]],
                                          columns=['temperature_sensitivity', 'enthalpy_sensitivity'])
    
            user_features = build_training_data(user_feature_df, daily_weather_time_df, sensitivity_df)
    
            curve = building_load_generation(building_type, user_features, center_df, shape_base_df)
    
            st.success("✅ 成功生成负荷曲线！")
    
            # 展示图像
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(curve, label="负荷曲线")
            ax.set_title(f"{building_type} 建筑的96点日负荷曲线")
            ax.set_xlabel("15分钟间隔")
            ax.set_ylabel("负荷")
            ax.grid(True)
            st.pyplot(fig)
    
            # 可选导出
            download_df = pd.DataFrame(curve.reshape(-1, 96), columns=[f"P{i+1}" for i in range(96)])
            csv = download_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载负荷曲线 CSV", csv, "load_curve.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ 出错了: {str(e)}")
