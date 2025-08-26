import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

# ================================================================
# 1. 文件路径
# ================================================================
raw_vpd_file  = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/VPD_dat/Raw_vpd_Global.nc'
homo_vpd_file = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/VPD_dat/Homo_vpd_Global.nc'

# —— 全局字体统一 20 ——
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24
})

output_dir = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/VPD_dat/'
os.makedirs(output_dir, exist_ok=True)

# ================================================================
# 2. 公共函数
# ================================================================
def calculate_anomaly_spatial(data, lat, time_df, baseline_start=1980, baseline_end=2010):
    base_mask = (time_df['year'] >= baseline_start) & (time_df['year'] <= baseline_end)
    if np.sum(base_mask) == 0:
        base_mask = time_df['time_index'] < min(360, len(time_df))

    monthly_clim = np.full((12, data.shape[1], data.shape[2]), np.nan)
    for m in range(1, 13):
        mm = base_mask & (time_df['month'] == m)
        if np.sum(mm) > 0:
            monthly_clim[m-1] = np.nanmean(data[mm], axis=0)

    anomalies = np.full_like(data, np.nan)
    for i, m in enumerate(time_df['month']):
        anomalies[i] = data[i] - monthly_clim[m-1]

    years = sorted(set(time_df['year']))
    lat_weights = np.cos(np.deg2rad(lat))
    W = np.broadcast_to(lat_weights[:, None], (len(lat), data.shape[2]))

    ann_years, ann_global = [], []
    for yr in years:
        idx = (time_df['year'] == yr)
        if np.sum(idx) < 6:
            ann_years.append(yr)
            ann_global.append(np.nan)
            continue
        grid_ann = np.nanmean(anomalies[idx], axis=0)
        valid = ~np.isnan(grid_ann)
        if np.any(valid):
            num = np.nansum(grid_ann[valid] * W[valid])
            den = np.nansum(W[valid])
            ann_global.append(num/den if den > 0 else np.nan)
        else:
            ann_global.append(np.nan)
        ann_years.append(yr)
    return ann_years, np.array(ann_global, dtype=float)

def align_data_to_years(data_years, data_values, target_years):
    mp = dict(zip(data_years, data_values))
    return np.array([mp.get(y, np.nan) for y in target_years], dtype=float)

def slope_per_decade(years, series):
    years = np.asarray(years, dtype=float)
    y = np.asarray(series, dtype=float)
    m = np.isfinite(years) & np.isfinite(y)
    if m.sum() < 2: return np.nan
    slope_year = np.polyfit(years[m], y[m], 1)[0]
    return slope_year * 10.0

# ================================================================
# 3. 读取数据 + 计算 anomaly
# ================================================================
def load_vpd(file, var='rhum', start_year=1973):
    ds = xr.open_dataset(file)
    lat = ds['lat'].values
    var_data = ds[var].astype('float64').values
    fv = ds[var].attrs.get('_FillValue', None)
    if fv is not None:
        var_data = np.where(var_data == fv, np.nan, var_data)
    years_seq = [start_year + i//12 for i in range(var_data.shape[0])]
    months_seq = [(i % 12) + 1 for i in range(var_data.shape[0])]
    df = pd.DataFrame({'time_index': range(var_data.shape[0]), 'year': years_seq, 'month': months_seq})
    years_proc, ann = calculate_anomaly_spatial(var_data, lat, df)
    ds.close()
    return years_proc, ann

raw_years_proc, raw_ann = load_vpd(raw_vpd_file)
homo_years_proc, homo_ann = load_vpd(homo_vpd_file)

TARGET_YEARS = list(range(1973, 2025))
raw_aligned  = align_data_to_years(raw_years_proc,  raw_ann,  TARGET_YEARS)
homo_aligned = align_data_to_years(homo_years_proc, homo_ann, TARGET_YEARS)

raw_slope  = slope_per_decade(TARGET_YEARS, raw_aligned)
homo_slope = slope_per_decade(TARGET_YEARS, homo_aligned)

# ================================================================
# 4. 绘图
# ================================================================
# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(16, 10))

# 画曲线
ax.plot(TARGET_YEARS, raw_aligned,  color='#193e8f', linewidth=3, label='VPD is calculated from HadISD RH')
ax.plot(TARGET_YEARS, homo_aligned, color='#E53528', linewidth=3, label='VPD is calculated from Homogenization RH')

ax.set_xlim(1973, 2024)
ax.set_xticks(range(1973, 2025, 5))
ax.set_xlabel('Year', fontsize=20)
ax.set_ylabel('VPD Anomaly (kPa)', fontsize=24)
ax.set_title('Global Land Annual VPD Anomaly Comparison (1973–2024, baseline 1980–2010)', fontsize=24)
ax.axhline(0, color='black', linestyle='--', alpha=0.7)


# ====== 在图内添加 slope 文字 ======
# Raw (蓝线) → 放在最后几年附近
ax.text(2009.5, raw_aligned[-1], f'Trend: {raw_slope:+.3f} kPa/decade',
        color='#193e8f', fontsize=24, va='center')

# Homo (红线) → 稍微低一点
ax.text(2011, homo_aligned[-14], f'Trend: {homo_slope:+.3f} kPa/decade',
        color='#E53528', fontsize=24, va='center')

# ====== legend 简洁版（只标数据来源） ======
ax.legend(loc='upper left', fontsize=24, frameon=False)

plt.tight_layout()
out_fig = os.path.join(output_dir, 'global_VPD_comparison_with_slopes_inside.svg')
plt.savefig(out_fig, dpi=800, bbox_inches='tight')
plt.show()