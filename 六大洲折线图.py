import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ——— 全局字体设置 ———
plt.rcParams['font.size']          = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family']        = 'sans-serif'

# ——— 配置区 ———
root_nc_dir = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/六大洲'
fig_dir     = os.path.join(root_nc_dir, '图')
os.makedirs(fig_dir, exist_ok=True)

continents = [
    'Global',
    'Africa', 'Asia', 'Europe',
    'North_America', 'South_America', 'Oceania'
]

# 变量读取函数（跟之前一样）
def annual_anomaly_series(nc_path):
    ds = nc.Dataset(nc_path)
    if 'rhum' in ds.variables:
        arr = ds.variables['rhum'][:].astype(float)
        lat = ds.variables['lat'][:]
        fill = getattr(ds.variables['rhum'], '_FillValue', None)
    else:
        arr = ds.variables['rh_abs'][:].astype(float)
        lat = ds.variables.get('lat', ds.variables.get('latitude'))[:]
        fill = getattr(ds.variables['rh_abs'], '_FillValue', None)
    ds.close()

    if fill is not None:
        arr[arr == fill] = np.nan
    arr[(arr < 0) | (arr > 100)] = np.nan

    years = np.arange(1973, 2024 + 1)
    n_years = len(years)
    arr = arr[:n_years*12].reshape(n_years, 12, *arr.shape[1:])

    clim = np.nanmean(arr[1980-1973:2010-1973+1], axis=(0,1))
    anom = np.nanmean(arr - clim[None,None,...], axis=1)

    w = np.cos(np.deg2rad(lat))
    w2 = np.broadcast_to(w[:,None], anom.shape[1:])
    ts = []
    for y in range(n_years):
        fld = anom[y]
        m = ~np.isnan(fld)
        ts.append(np.nansum(fld[m]*w2[m]) / np.nansum(w2[m]) if m.any() else np.nan)
    return pd.Series(ts, index=years)

# 构造路径字典
obs_tpl = {'Global': os.path.join(root_nc_dir, '观测_均值_Global.nc')}
hom_tpl = {'Global': os.path.join(root_nc_dir, '均一化_均值_Global.nc')}
had_tpl = {'Global': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/HadISDH_2024_cleaned.nc'}
for cont in continents:
    if cont!='Global':
        sub = os.path.join(root_nc_dir, cont)
        obs_tpl[cont] = os.path.join(sub, '观测_均值_Global.nc')
        hom_tpl[cont] = os.path.join(sub, '均一化_均值_Global.nc')
        had_tpl[cont] = os.path.join(sub, f'HadISDH_{cont}.nc')

# 读取所有序列
series = {}
for cont in continents:
    series[f'{cont}_obs'] = annual_anomaly_series(obs_tpl[cont])
    series[f'{cont}_hom'] = annual_anomaly_series(hom_tpl[cont])
    if os.path.exists(had_tpl[cont]):
        series[f'{cont}_had'] = annual_anomaly_series(had_tpl[cont])

years = series['Global_obs'].index

# ——— 绘图 ———
fig = plt.figure(figsize=(22,20))
gs  = GridSpec(4, 2, figure=fig,
               height_ratios=[1,1,1,1],  # 第一行高度是下面的两倍
               hspace=0.2, wspace=0.1)


# —— (0) Global 跨两列 ——
ax0 = fig.add_subplot(gs[0, :])
# 可选：再微调它的框高宽比
ax0.set_box_aspect(0.4)
ax0.plot(years, series['Global_obs'], color='#f16c23', lw=2, label='Obs')
ax0.plot(years, series['Global_hom'], color='#2b6a99', lw=2, label='Hom')
if 'Global_had' in series:
    ax0.plot(years, series['Global_had'], color='#1b7c3d', lw=2, label='HadISDH')
ax0.axhline(0, color='gray', ls='--', lw=1)
# 标题恢复为默认粗细
ax0.set_title('(a) Global', fontsize=18)
ax0.set_ylabel('RH Anomaly (%)', fontsize=16)
ax0.tick_params(labelsize=14)
# 不在这里显示 legend

# —— (1–6) 六大洲 3 行 × 2 列 ——
cont6 = ['Africa','Asia','Europe','North_America','South_America','Oceania']
labels = ['b','c','d','e','f','g']
for idx, cont in enumerate(cont6):
    row = 1 + idx//2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])
    ax.plot(years, series[f'{cont}_obs'], color='#f16c23', lw=1.5)
    ax.plot(years, series[f'{cont}_hom'], color='#2b6a99', lw=1.5)
    if f'{cont}_had' in series:
        ax.plot(years, series[f'{cont}_had'], color='#1b7c3d', lw=1.5)
    ax.axhline(0, color='gray', ls='--', lw=0.8)

    # —— (1–6) 六大洲 3 行 × 2 列 ——
    cont6 = ['Africa', 'Asia', 'Europe', 'North_America', 'South_America', 'Oceania']
    # 对应的子图标号 b-g
    labels = ['b', 'c', 'd', 'e', 'f', 'g']

    ax.set_title(f'({labels[idx]}) {cont.replace("_", " ")}', fontsize=16)
    ax.tick_params(labelsize=12)
    if row == 4 - 1:  # 最底排
        ax.set_xlabel('Year', fontsize=14)
    if col == 0:
        ax.set_ylabel('RH Anomaly (%)', fontsize=14)

# —— 底部图例（可选） ——
lines = [
    Line2D([0], [0], color='#f16c23', lw=2),
    Line2D([0], [0], color='#2b6a99', lw=2)
]
# —— 底部图例（可选） ——

labels = ['HadISD', 'Homogenization (this study)']  # 修改这里，加入 (this study)
if 'Global_had' in series:
    lines.append(Line2D([0], [0], color='#1b7c3d', lw=2))
    labels.append('HadISDH')
fig.legend(lines, labels,
           ncol=len(labels), fontsize=14,
           bbox_to_anchor=(0.52, 0.05), loc='lower center')


# —— 保存矢量图 ——
out_svg = os.path.join(root_nc_dir, 'Global_plus_6continents.svg')
plt.savefig(out_svg, format='svg', bbox_inches='tight')
plt.show()

print("Saved SVG to:", out_svg)