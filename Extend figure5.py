import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Set default font (to display English correctly)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# —— 全局字体统一 18 ——
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.size': 20,           # 全局字体
    'axes.titlesize': 20,      # 坐标轴标题
    'axes.labelsize': 20,      # 坐标轴标签
    'xtick.labelsize': 20,     # x 刻度标签
    'ytick.labelsize': 20,     # y 刻度标签
    'legend.fontsize': 20      # 图例文字
})
# Year range
start_year = 1973
end_year = 2024
years = np.arange(start_year, end_year + 1)

# Specify each country’s instrument change start/end years
# end_year = None means only a single line
vertical_lines = {
    'BRAZIL': (2000,None),
    'JAPAN': (1996,None),
    'CHINA': (2003,None),
    'SPAIN': (2010,None),
    'RUSSIA': ( 2014,None),
    'ITALY': (2004,None),
}

# Function to compute global area-weighted annual series (cos(lat) weights)
def get_global_annual_series(nc_file):
    ds = nc.Dataset(nc_file)
    rhum = ds.variables['rhum'][:]  # [time, lat, lon]
    lat = ds.variables['lat'][:]   # [lat]
    ds.close()

    if np.ma.is_masked(rhum):
        rhum = rhum.filled(np.nan)

    n_years = end_year - start_year + 1
    rhum_yearly = np.nanmean(
        rhum.reshape(n_years, 12, *rhum.shape[1:]), axis=1
    )

    weights = np.cos(np.deg2rad(lat))
    w2d = np.broadcast_to(weights[:, None], rhum_yearly.shape[1:])

    global_annual = []
    for yr in range(n_years):
        data = rhum_yearly[yr]
        mask = ~np.isnan(data)
        if not np.any(mask):
            global_annual.append(np.nan)
        else:
            d_valid = data[mask]
            w_valid = w2d[mask]
            global_annual.append(np.nansum(d_valid * w_valid) /
                                np.nansum(w_valid))
    return pd.Series(global_annual, index=years)

# Root directory containing continent folders
root_rhtest = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/RHtest国家/'

# New output folder
output_folder = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/国家和更换仪器信息图片'
os.makedirs(output_folder, exist_ok=True)

# ……前面的代码保持不变……
# ================= 画图部分 =================
from matplotlib.lines import Line2D

# 2 行 3 列
nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), dpi=800, sharex=True, sharey=True)
axes = axes.flatten()

legend_handles = [
    Line2D([0], [0], color='#2564af', marker='o', lw=1.5, label='Raw'),
    Line2D([0], [0], color='#e13937', marker='s', lw=1.5, label='Homogenization (this study)'),
    Line2D([0], [0], color='#B7B2D0', ls='--', lw=2, label='Observation instrument change time'),

]

for idx, (country, (v_start, v_end)) in enumerate(vertical_lines.items()):
    if idx >= nrows * ncols:
        break

    # 找国家文件夹
    country_path = None
    for continent in os.listdir(root_rhtest):
        cpath = os.path.join(root_rhtest, continent, country)
        if os.path.isdir(cpath):
            country_path = cpath
            break
    if country_path is None:
        print(f"{country}: folder not found, skipping")
        continue

    # 找观测和均一化文件
    obs_file = next((os.path.join(country_path, f)
                     for f in os.listdir(country_path)
                     if f.startswith('观测_') and f.endswith('.nc')), None)
    hom_file = next((os.path.join(country_path, f)
                     for f in os.listdir(country_path)
                     if f.startswith('均一化_') and f.endswith('.nc')), None)
    if obs_file is None or hom_file is None:
        print(f"{country}: missing observed or homogenized file, skipping")
        continue

    series_obs = get_global_annual_series(obs_file)
    series_hom = get_global_annual_series(hom_file)

    ax = axes[idx]
    sns.lineplot(x=series_obs.index, y=series_obs.values,
                 marker='o', linewidth=1.5, ax=ax, color='#2564af', label=None)
    sns.lineplot(x=series_hom.index, y=series_hom.values,
                 marker='s', linewidth=1.5, ax=ax, color='#e13937', label=None)

    if v_start is not None:
        ax.axvline(x=v_start, color='#B7B2D0', linestyle='--', linewidth=2)


    ax.set_title(country.title())
    ax.set_xlabel('Year')
    ax.set_ylabel('RH Anomaly(%)')

    ax.set_xticks(np.arange(start_year, end_year + 1, 10))

    # —— 强制所有子图都显示 x/y 轴刻度及标签 ——
    ax.tick_params(axis='x', which='both', labelbottom=True, rotation=45)
    ax.tick_params(axis='y', which='both', labelleft=True)

# 删除多余子图
last_used = idx
for j in range(last_used + 1, nrows * ncols):
    fig.delaxes(axes[j])

# 布局和统一图例
plt.tight_layout(rect=[0, 0.12, 1, 1])
fig.legend(handles=legend_handles,
           loc='lower center',
           ncol=4,
           frameon=False,
           bbox_to_anchor=(0.52, 0.1))

out_fig = os.path.join(output_folder, "Global_RH_6_Countries.svg")
plt.savefig(out_fig, dpi=800)
plt.show()
plt.close()

print(f"Combined plot saved to: {out_fig}")