import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from collections import defaultdict
# === 参数 ===
N_CLUSTERS = 4
MIN_CLUSTER_RATIO = 0.05
NORMALIZE_METHOD = 0  # 1 = 每日归一化，0 = 全年归一化

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


# === 季节分类函数 ===
def classify_season(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return 'cold'
    elif m in [6, 7, 8, 9]:
        return 'hot'
    else:
        return 'transition'

# === 归一化函数 ===
def normalize_daily(pivot_df, method=0):
    pivot_base = pivot_df.drop(columns='season')
    if method == 1:
        pivot_norm = pivot_base.div(pivot_base.mean(axis=1), axis=0)
    elif method == 0:
        pivot_norm = pivot_base / pivot_base.values.mean()
    else:
        raise ValueError("method 参数只能为 0 或 1")
    pivot_norm['season'] = pivot_df['season']
    return pivot_norm

# === 特征提取函数 ===
def extract_curve_features(curve, threshold_ratio=0.8):
    peak_val = np.max(curve)
    valley_val = np.min(curve)
    peak_idx = np.argmax(curve)
    valley_idx = np.argmin(curve)
    mean_val = np.mean(curve)
    std_val = np.std(curve)

    feats = {
        'peak_val': peak_val,
        'valley_val': valley_val,
        'range': peak_val - valley_val,
        'avg': mean_val,
        'std': std_val,
        'cv': std_val / mean_val if mean_val else 0,
        'peak_hour': round(peak_idx * 0.25, 2),
        'valley_hour': round(valley_idx * 0.25, 2),
        'concentration_index': peak_val / np.sum(curve),
        'daytime_ratio': np.sum(curve[32:76]) / np.sum(curve),
        'sharpness': (peak_val - mean_val) / std_val if std_val else 0,
    }

    peaks, _ = find_peaks(curve, prominence=0.1, distance=8)
    feats['multi_peak_count'] = len(peaks)

    high_threshold = threshold_ratio * peak_val
    feats['flat_peak_duration'] = np.sum(curve >= high_threshold) * 0.25

    # 爬坡
    start, end = None, None
    rise_window = curve[24:40]
    rise_peak = np.max(rise_window)
    for i, val in enumerate(rise_window):
        if start is None and val > 0.3 * rise_peak:
            start = i
        if end is None and val > 0.9 * rise_peak:
            end = i
            break
    if start is not None and end is not None and end > start:
        feats['slope'] = (rise_window[end] - rise_window[start]) / ((end - start) * 0.25)
        feats['rise_start_hour'] = 6 + start * 0.25
        feats['fall_end_hour'] = 6 + end * 0.25
    else:
        feats['slope'] = 0
        feats['rise_start_hour'] = None
        feats['fall_end_hour'] = None

    # 启停行为
    above = np.where(curve >= 0.3 * peak_val)[0]
    if len(above):
        feats['start_hour'] = above[0] * 0.25
        feats['end_hour'] = above[-1] * 0.25
        feats['active_duration'] = (above[-1] - above[0] + 1) * 0.25
    else:
        feats['start_hour'] = feats['end_hour'] = feats['active_duration'] = None

    # FFT 高频能量
    f = fft(curve)
    energy = np.abs(f) ** 2
    feats['fft_high_energy_ratio'] = np.sum(energy[10:30]) / (np.sum(energy[1:48]) + 1e-8)

    # 熵
    p = curve / (np.sum(curve) + 1e-8)
    feats['load_entropy'] = entropy(p, base=2)

    return feats


def plot_cluster_heatmap(daily_label_df):
    # 获取所有建筑类型
    for btype, group in daily_label_df.groupby('building_type'):
        pivot = group.pivot(index='building', columns='date', values='cluster_label')
        pivot = pivot.sort_index().sort_index(axis=1)

        # 构造一致的颜色映射：簇1~簇4 映射为 tab20(0, 2, 4, 6)
        cluster_palette = {
            1: mcolors.to_hex(plt.cm.tab20(0)),
            2: mcolors.to_hex(plt.cm.tab20(2)),
            3: mcolors.to_hex(plt.cm.tab20(4)),
            4: mcolors.to_hex(plt.cm.tab20(6))
        }

        cmap = mcolors.ListedColormap([cluster_palette[i] for i in sorted(cluster_palette.keys())])

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap=cmap, cbar_kws={'label': 'Cluster Label'}, linewidths=0.1, linecolor='gray', square=False)
        plt.title(f'{btype} 不同用户的日簇分布热力图')
        plt.xlabel('日期')
        plt.ylabel('用户编号')
        plt.tight_layout()
        plt.show()
        
      
def analyze_cluster_distribution(daily_label_df):


    results_type_level = []
    results_user_level = []

    for (btype, season), group_season in daily_label_df.groupby(['building_type', 'season']):
        group_season['weekday'] = pd.to_datetime(group_season['date']).dt.weekday

        # === 类型级别 ===
        total_days = len(group_season)
        cluster_to_weekday = group_season.groupby(['cluster_label', 'weekday']).size().unstack(fill_value=0)
        cluster_to_weekday_ratio = cluster_to_weekday.div(cluster_to_weekday.sum(axis=1), axis=0)

        weekday_to_cluster = group_season.groupby(['weekday', 'cluster_label']).size().unstack(fill_value=0)
        weekday_to_cluster_ratio = weekday_to_cluster.div(weekday_to_cluster.sum(axis=1), axis=0)

        cluster_to_weekday_ratio['building_type'] = btype
        cluster_to_weekday_ratio['season'] = season
        weekday_to_cluster_ratio['building_type'] = btype
        weekday_to_cluster_ratio['season'] = season

        results_type_level.append(('cluster_to_weekday', cluster_to_weekday_ratio))
        results_type_level.append(('weekday_to_cluster', weekday_to_cluster_ratio))

        # === 用户级别 ===
        for building, group_user in group_season.groupby('building'):
            group_user['weekday'] = pd.to_datetime(group_user['date']).dt.weekday

            cl_to_wd = group_user.groupby(['cluster_label', 'weekday']).size().unstack(fill_value=0)
            wd_to_cl = group_user.groupby(['weekday', 'cluster_label']).size().unstack(fill_value=0)

            cl_to_wd_ratio = cl_to_wd.div(cl_to_wd.sum(axis=1), axis=0)
            wd_to_cl_ratio = wd_to_cl.div(wd_to_cl.sum(axis=1), axis=0)

            cl_to_wd_ratio['building'] = building
            cl_to_wd_ratio['building_type'] = btype
            cl_to_wd_ratio['season'] = season

            wd_to_cl_ratio['building'] = building
            wd_to_cl_ratio['building_type'] = btype
            wd_to_cl_ratio['season'] = season

            results_user_level.append(('cluster_to_weekday', cl_to_wd_ratio))
            results_user_level.append(('weekday_to_cluster', wd_to_cl_ratio))

    return results_type_level, results_user_level

       
# === 主函数 ===
def run_clustering_by_building_type(filepath):
    xls = pd.ExcelFile(filepath)
    all_feature_records = []
    center_records = []
    daily_label_records = []
    
    for sheet in xls.sheet_names:
        print(f"处理建筑类型: {sheet}")
        df = xls.parse(sheet)
        df['时间'] = pd.to_datetime(df['时间'])
        df['date'] = df['时间'].dt.date
        df['time'] = df['时间'].dt.strftime('%H:%M')
        df['season'] = df['时间'].apply(classify_season)
        season_series = df.groupby('date')['season'].first()

        buildings = [col for col in df.columns if col not in ['时间', 'date', 'time', 'season']]

        season_data_dict = {'hot': [], 'cold': [], 'transition': []}

        for b in buildings:
            pivot = df.pivot_table(index='date', columns='time', values=b)
            pivot['season'] = pivot.index.map(season_series)
            pivot_norm = normalize_daily(pivot, method=NORMALIZE_METHOD)
            for season in ['hot', 'cold', 'transition']:
                part = pivot_norm[pivot_norm['season'] == season].drop(columns='season').dropna()
                if not part.empty:
                    part['building'] = b
                    season_data_dict[season].append(part)

        for season, season_parts in season_data_dict.items():
            if not season_parts:
                continue
            season_df = pd.concat(season_parts)
            season_df_noid = season_df.drop(columns='building')

            km = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(season_df_noid)
            labels = km.labels_
            centers = km.cluster_centers_

            # 簇重排序
            max_vals = [np.max(c) for c in centers]
            sorted_idx = np.argsort(-np.array(max_vals))
            centers = centers[sorted_idx]
            label_map = {old: new for new, old in enumerate(sorted_idx)}
            labels = np.array([label_map[l] for l in labels])

            season_df['label'] = labels
            total = len(season_df)
            for i in range(N_CLUSTERS):
                subset = season_df[season_df['label'] == i].drop(columns=['label', 'building'])
                ratio = len(subset) / total
                # if ratio < MIN_CLUSTER_RATIO:
                #     continue
                feats = extract_curve_features(centers[i])
                feats.update({
                    'building_type': sheet,
                    'season': season,
                    'cluster': i+1,
                    'cluster_ratio': round(ratio, 4),
                    'days': len(subset)
                })
                all_feature_records.append(feats)
            
                # === 添加中心线保存数据 ===
                center_row = dict(zip([f'P{i+1}' for i in range(96)], centers[i]))
                center_row.update({
                    'building_type': sheet,
                    'season': season,
                    'cluster_id': i+1
                })
                center_records.append(center_row)
            # === 添加每日标签记录 ===
            for idx, row in season_df.iterrows():
                record = {
                    'building': row['building'],
                    'building_type': sheet,
                    'season': season,
                    'date': idx,
                    'cluster_label': row['label'] + 1
                }
                for i in range(96):
                    record[f'P{i+1}'] = row.iloc[i]
                daily_label_records.append(record)

            # 可视化
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(N_CLUSTERS):
                members = season_df[season_df['label'] == i].drop(columns=['label', 'building'])
                # if len(members) / total < MIN_CLUSTER_RATIO:
                #     continue
                for j in range(min(20, len(members))):
                    ax.plot(members.iloc[j].values, color='gray', alpha=0.3)
                
                cluster_color_map = {
                                    0: plt.cm.tab20(0),
                                    1: plt.cm.tab20(2),
                                    2: plt.cm.tab20(4),
                                    3: plt.cm.tab20(6)
                                }
                ax.plot(centers[i], color=cluster_color_map[i], label=f'簇{i+1} ({len(members)/total*100:.1f}%)', linewidth=2)
            ax.set_title(f'{sheet} - {season.capitalize()} Season ({total} days)')
            ax.set_xlabel('Hour of Day')
            ax.set_xticks(np.arange(0, 96, 4))
            ax.set_xticklabels([str(int(i * 0.25)) for i in np.arange(0, 96, 4)])
            ax.set_ylabel('Normalized Load')
            ax.legend()
            plt.tight_layout()
            plt.show()
    # 保存中心线特征
    feature_df = pd.DataFrame(all_feature_records)
    feature_df.to_excel("典型日特征提取结果_建筑类型级别.xlsx", index=False)
    
    # 保存中心线
    center_df = pd.DataFrame(center_records)
    center_df.to_excel("cluster_centers_by_type_season.xlsx", index=False)
    
    # 保存每条数据对应的簇标签
    label_df = pd.DataFrame(daily_label_records)
    label_df.to_excel("cluster_labels_by_day.xlsx", index=False)
    # plot_cluster_heatmap(label_df)
    # type_level_label_distribution, user_level_distribution = analyze_cluster_distribution(label_df)
    # with pd.ExcelWriter("簇分布统计结果.xlsx") as writer:
    #     for name, df in type_level_label_distribution:
    #         df.to_excel(writer, sheet_name=f"type_{name}")
    #     for name, df in user_level_distribution:
    #         df.to_excel(writer, sheet_name=f"user_{name}", index=True)


    print("聚类中心线与标签数据已保存。")


# === 脚本入口 ===
if __name__ == '__main__':
    run_clustering_by_building_type('D:/Doccument/Papers/00 Journal/Electricity load prediction/Building_dataset.xlsx')



