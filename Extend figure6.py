import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# ================================================================
# 1. 文件路径设置
# ================================================================
Hadisdh = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/HadISDH_2024_cleaned.nc'
gsod_file = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/均一化_均值_Global.nc'
obs_file = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/观测_均值_Global.nc'
era5_file = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/再分析数据所有陆地的/ERA5_rh_land.nc'
jra3q_file = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/再分析数据所有陆地的/JRA3q_rh_land.nc'
# —— 全局字体统一 18 ——
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.size': 24,           # 全局字体
    'axes.titlesize': 24,      # 坐标轴标题
    'axes.labelsize': 24,      # 坐标轴标签
    'xtick.labelsize': 24,     # x 刻度标签
    'ytick.labelsize': 24,     # y 刻度标签
    'legend.fontsize': 24      # 图例文字
})
# 输出目录
output_dir = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/CMIP6/'
os.makedirs(output_dir, exist_ok=True)

# 读取CMIP6数据
data_cmip6 = pd.read_excel(
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/CMIP6/CMIP6_Land_RH_Anomalies_Averaged_20250807_1433.xlsx')
print("CMIP6数据预览:")
print(data_cmip6.head())

print("=" * 60)
print("多数据源相对湿度对比分析（含CMIP6模式范围）")
print("=" * 60)

# 设置字体为Times New Roman，并设置字体大小为16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24


# ================================================================
# 2. 定义通用函数
# ================================================================
def calculate_area_weights(lats):
    """计算面积权重"""
    weights = np.cos(np.deg2rad(lats))
    return weights / weights.sum()


def calculate_global_mean_with_weights(data, lats):
    """计算面积加权的全球平均"""
    weights = np.cos(np.deg2rad(lats))

    if data.ndim == 3:  # (time, lat, lon)
        weights_3d = weights[np.newaxis, :, np.newaxis]
    elif data.ndim == 2:  # (lat, lon)
        weights_3d = weights[:, np.newaxis]
    else:
        raise ValueError(f"数据维度不支持: {data.ndim}")

    weighted_data = data * weights_3d
    sum_weights = np.sum(weights_3d * ~np.isnan(data), axis=(-2, -1), keepdims=True)
    global_mean = np.sum(weighted_data, axis=(-2, -1), keepdims=True) / sum_weights

    return global_mean.squeeze()


def calculate_anomaly_spatial(data, lat, time_df, baseline_start=1980, baseline_end=2010):
    """计算空间网格的距平"""
    print(f"  计算距平，基准期: {baseline_start}-{baseline_end}")

    baseline_mask = (time_df['year'] >= baseline_start) & (time_df['year'] <= baseline_end)

    if np.sum(baseline_mask) == 0:
        print(f"  警告：基准期无数据，使用前30年作为基准期")
        baseline_mask = time_df['time_index'] < min(360, len(time_df))

    print(f"  基准期月数: {np.sum(baseline_mask)}")

    monthly_clim = np.full((12, data.shape[1], data.shape[2]), np.nan)

    for month in range(1, 13):
        month_baseline_mask = baseline_mask & (time_df['month'] == month)
        if np.sum(month_baseline_mask) > 0:
            monthly_clim[month - 1] = np.nanmean(data[month_baseline_mask], axis=0)

    anomalies = np.full_like(data, np.nan)
    for i, month in enumerate(time_df['month']):
        anomalies[i] = data[i] - monthly_clim[month - 1]

    annual_anomalies = []
    annual_years = []

    for year in sorted(set(time_df['year'])):
        year_mask = time_df['year'] == year
        year_data = anomalies[year_mask]

        if np.sum(year_mask) >= 6:
            annual_mean = np.nanmean(year_data, axis=0)
            annual_anomalies.append(annual_mean)
            annual_years.append(year)

    weights = np.cos(np.deg2rad(lat))
    weights_grid = np.broadcast_to(weights[:, np.newaxis], (len(lat), data.shape[2]))

    global_annual_anomalies = []
    for annual_anom in annual_anomalies:
        valid_mask = ~np.isnan(annual_anom)
        if np.sum(valid_mask) > 0:
            weighted_sum = np.nansum(annual_anom * weights_grid)
            weight_sum = np.nansum(weights_grid * valid_mask)
            global_mean = weighted_sum / weight_sum if weight_sum > 0 else np.nan
            global_annual_anomalies.append(global_mean)
        else:
            global_annual_anomalies.append(np.nan)

    return annual_years, global_annual_anomalies


def spatial_downsample(data, target_lat_size=72, target_lon_size=144):
    """空间降采样"""
    original_shape = data.shape
    lat_factor = original_shape[1] // target_lat_size
    lon_factor = original_shape[2] // target_lon_size

    downsampled = np.full((original_shape[0], target_lat_size, target_lon_size), np.nan)

    for i in range(target_lat_size):
        for j in range(target_lon_size):
            lat_start = i * lat_factor
            lat_end = (i + 1) * lat_factor
            lon_start = j * lon_factor
            lon_end = (j + 1) * lon_factor

            subset = data[:, lat_start:lat_end, lon_start:lon_end]
            downsampled[:, i, j] = np.nanmean(subset, axis=(1, 2))

    return downsampled


def align_data_to_years(data_years, data_values, target_years):
    """将数据对齐到目标年份"""
    aligned_values = []
    for year in target_years:
        if year in data_years:
            idx = data_years.index(year)
            aligned_values.append(data_values[idx])
        else:
            aligned_values.append(np.nan)
    return np.array(aligned_values)


# ================================================================
# 3. 处理CMIP6数据（从Excel读取）
# ================================================================
print("\n处理CMIP6数据...")

# 从Excel数据中提取年份和模式数据
cmip6_years = data_cmip6['Year'].values
print(f"CMIP6年份范围: {cmip6_years.min()}-{cmip6_years.max()}")

# 获取所有模式列（除了Year列）
model_columns = [col for col in data_cmip6.columns if col != 'Year']
print(f"CMIP6模式数量: {len(model_columns)}")
print(f"CMIP6模式: {model_columns}")

# 计算集合统计
cmip6_ensemble_mean = []
cmip6_ensemble_std = []
cmip6_ensemble_min = []
cmip6_ensemble_max = []

for i, year in enumerate(cmip6_years):
    year_data = []
    for model in model_columns:
        value = data_cmip6.loc[i, model]
        if not pd.isna(value):
            year_data.append(value)

    if len(year_data) >= 3:  # 至少3个模式有数据
        cmip6_ensemble_mean.append(np.mean(year_data))
        cmip6_ensemble_std.append(np.std(year_data))
        cmip6_ensemble_min.append(np.min(year_data))
        cmip6_ensemble_max.append(np.max(year_data))
    else:
        cmip6_ensemble_mean.append(np.nan)
        cmip6_ensemble_std.append(np.nan)
        cmip6_ensemble_min.append(np.nan)
        cmip6_ensemble_max.append(np.nan)

cmip6_ensemble_mean = np.array(cmip6_ensemble_mean)
cmip6_ensemble_std = np.array(cmip6_ensemble_std)
cmip6_ensemble_min = np.array(cmip6_ensemble_min)
cmip6_ensemble_max = np.array(cmip6_ensemble_max)

print(f"CMIP6集合统计计算完成")

# ================================================================
# 4. 处理HadISDH数据
# ================================================================
print("\n处理HadISDH数据...")
hadisdh_ds = xr.open_dataset(Hadisdh)
print(f"HadISDH数据维度: {hadisdh_ds.dims}")

hadisdh_rh = hadisdh_ds['rh_abs']
hadisdh_lat = hadisdh_ds['latitude']
hadisdh_lon = hadisdh_ds['longitude']
hadisdh_time = hadisdh_ds['time']

start_year = 1973
hadisdh_time_length = len(hadisdh_time)
hadisdh_years = []
hadisdh_months = []
for i in range(hadisdh_time_length):
    year = start_year + i // 12
    month = (i % 12) + 1
    hadisdh_years.append(year)
    hadisdh_months.append(month)

hadisdh_df = pd.DataFrame({
    'time_index': range(hadisdh_time_length),
    'year': hadisdh_years,
    'month': hadisdh_months
})

print(f"HadISDH时间范围: {min(hadisdh_years)}-{max(hadisdh_years)}")

if hadisdh_rh.shape[1] <= 72 and hadisdh_rh.shape[2] <= 144:
    hadisdh_rh_processed = hadisdh_rh.values
    hadisdh_lat_processed = hadisdh_lat.values
else:
    hadisdh_rh_processed = spatial_downsample(hadisdh_rh.values)
    hadisdh_lat_processed = np.linspace(hadisdh_lat.min(), hadisdh_lat.max(), 72)

hadisdh_years_proc, hadisdh_anomalies = calculate_anomaly_spatial(
    hadisdh_rh_processed, hadisdh_lat_processed, hadisdh_df)

print(f"HadISDH处理完成，年份范围: {min(hadisdh_years_proc)}-{max(hadisdh_years_proc)}")

# ================================================================
# 5. 处理GSOD数据
# ================================================================
print("\n处理GSOD数据...")
gsod_ds = xr.open_dataset(gsod_file)
print(f"GSOD数据维度: {gsod_ds.dims}")

gsod_rhum = gsod_ds['rhum']
gsod_lat = gsod_ds['lat']
gsod_time = gsod_ds['time']

time_length = len(gsod_time)
years = []
months = []
for i in range(time_length):
    year = start_year + i // 12
    month = (i % 12) + 1
    years.append(year)
    months.append(month)

gsod_df = pd.DataFrame({
    'time_index': range(time_length),
    'year': years,
    'month': months
})

print(f"GSOD时间范围: {min(years)}-{max(years)}")

gsod_years, gsod_anomalies = calculate_anomaly_spatial(
    gsod_rhum.values, gsod_lat.values, gsod_df)

print(f"GSOD处理完成，年份范围: {min(gsod_years)}-{max(gsod_years)}")

# ================================================================
# 6. 处理观测数据
# ================================================================
print("\n处理观测数据...")
obs_ds = xr.open_dataset(obs_file)
print(f"观测数据维度: {obs_ds.dims}")

obs_rhum = obs_ds['rhum']
obs_lat = obs_ds['lat']
obs_time = obs_ds['time']

obs_time_length = len(obs_time)
obs_years = []
obs_months = []
for i in range(obs_time_length):
    year = start_year + i // 12
    month = (i % 12) + 1
    obs_years.append(year)
    obs_months.append(month)

obs_df = pd.DataFrame({
    'time_index': range(obs_time_length),
    'year': obs_years,
    'month': obs_months
})

print(f"观测时间范围: {min(obs_years)}-{max(obs_years)}")

obs_years_proc, obs_anomalies = calculate_anomaly_spatial(
    obs_rhum.values, obs_lat.values, obs_df)

print(f"观测处理完成，年份范围: {min(obs_years_proc)}-{max(obs_years_proc)}")

# ================================================================
# 7. 处理ERA5数据
# ================================================================
print("\n处理ERA5数据...")
era5_ds = xr.open_dataset(era5_file)
print(f"ERA5数据维度: {era5_ds.dims}")

era5_rh = era5_ds['rh']
era5_lat = era5_ds['latitude']
era5_time = era5_ds['time']

era5_time_length = len(era5_time)
era5_years = []
era5_months = []
for i in range(era5_time_length):
    year = start_year + i // 12
    month = (i % 12) + 1
    era5_years.append(year)
    era5_months.append(month)

era5_df = pd.DataFrame({
    'time_index': range(era5_time_length),
    'year': era5_years,
    'month': era5_months
})

print(f"ERA5时间范围: {min(era5_years)}-{max(era5_years)}")

era5_rh_downsampled = spatial_downsample(era5_rh.values)
era5_lat_downsampled = np.linspace(era5_lat.min(), era5_lat.max(), 72)

era5_years_proc, era5_anomalies = calculate_anomaly_spatial(
    era5_rh_downsampled, era5_lat_downsampled, era5_df)

print(f"ERA5处理完成，年份范围: {min(era5_years_proc)}-{max(era5_years_proc)}")

# ================================================================
# 8. 处理JRA3q数据
# ================================================================
print("\n处理JRA3q数据...")
jra3q_ds = xr.open_dataset(jra3q_file)
print(f"JRA3q数据维度: {jra3q_ds.dims}")

jra3q_rh = jra3q_ds['rh']
jra3q_lat = jra3q_ds['lat']
jra3q_time = jra3q_ds['time']

jra3q_time_length = len(jra3q_time)
jra3q_years = []
jra3q_months = []
for i in range(jra3q_time_length):
    year = start_year + i // 12
    month = (i % 12) + 1
    jra3q_years.append(year)
    jra3q_months.append(month)

jra3q_df = pd.DataFrame({
    'time_index': range(jra3q_time_length),
    'year': jra3q_years,
    'month': jra3q_months
})

print(f"JRA3q时间范围: {min(jra3q_years)}-{max(jra3q_years)}")

jra3q_years_proc, jra3q_anomalies = calculate_anomaly_spatial(
    jra3q_rh.values, jra3q_lat.values, jra3q_df)

print(f"JRA3q处理完成，年份范围: {min(jra3q_years_proc)}-{max(jra3q_years_proc)}")

# ================================================================
# ================================================================
# 9. 处理并对齐数据到各自年份范围
# ================================================================
print("\n数据对齐...")

# CMIP6 只画到 2014
cmip6_years_aligned = cmip6_years  # 1973-2014
cmip6_mean_aligned = cmip6_ensemble_mean
cmip6_min_aligned  = cmip6_ensemble_min
cmip6_max_aligned  = cmip6_ensemble_max

# 其它数据画到 2024
other_years = list(range(1973, 2025))  # 1973-2024

# 对齐各观测／再分析到 1973-2024
hadisdh_aligned = align_data_to_years(hadisdh_years_proc, hadisdh_anomalies, other_years)
gsod_aligned   = align_data_to_years(gsod_years, gsod_anomalies, other_years)
obs_aligned    = align_data_to_years(obs_years_proc, obs_anomalies, other_years)
era5_aligned   = align_data_to_years(era5_years_proc, era5_anomalies, other_years)
jra3q_aligned  = align_data_to_years(jra3q_years_proc, jra3q_anomalies, other_years)

print("数据对齐完成：")
print(f"  CMIP6: {cmip6_years_aligned[0]}–{cmip6_years_aligned[-1]}")
print(f"  其它:  {other_years[0]}–{other_years[-1]}")

# ================================================================
# 10. 绘制综合对比图 - 第一张：±2σ范围
# ================================================================
# ================================================================
# 10. 绘图：CMIP6 范围画到 2014，其它数据画到 2024
# ================================================================
print("\n绘制综合对比图（±2σ范围）...")

fig1, ax1 = plt.subplots(figsize=(16, 10))

# —— CMIP6 范围（1973–2014）——
ax1.fill_between(cmip6_years_aligned,
                 cmip6_min_aligned,
                 cmip6_max_aligned,
                 color='lightgray', alpha=0.4,
                 label=f'CMIP6 Models Range (n={len(model_columns)})')
ax1.plot(cmip6_years_aligned, cmip6_mean_aligned,
         color='green', linewidth=3, label='CMIP6 Ensemble Mean')

# —— 其它观测／再分析（1973–2024）——
ax1.plot(other_years, hadisdh_aligned, color='#8B4513',  linewidth=2.5, label='HadISDH')
ax1.plot(other_years, gsod_aligned,   color='#55b7e6',  linewidth=2.5, label='Homogenization (this study)')
ax1.plot(other_years, obs_aligned,    color='#193e8f',  linewidth=2.5, label='HadISD')
ax1.plot(other_years, era5_aligned,   color='#E53528',  linewidth=2.5, label='ERA5')
ax1.plot(other_years, jra3q_aligned,  color='#F09739',  linewidth=2.5, label='JRA-3Q')

# x 轴范围扩展到 2024
ax1.set_xlim(1973, 2024)
ax1.set_xticks(range(1975, 2025, 5))

# 其余格式同原代码
ax1.set_xlabel('Year', fontsize=24)
ax1.set_ylabel('RH Anomaly (%)', fontsize=24)
ax1.set_title('Global Land Annual RH Anomaly(1973–2024)', fontsize=24)
ax1.axhline(0, color='black', linestyle='--', alpha=0.7)
ax1.legend(loc='lower left', fontsize=20, ncol=2)
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'global_RH_comparison_CMIP6_2sigma_1973-2024.svg'), dpi=800)
plt.show()