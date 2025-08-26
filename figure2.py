

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import cartopy.mpl.ticker as cticker

# ----------------------------
# 1. 读取 NetCDF 文件
# ----------------------------
data = nc.Dataset(
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/'
    '重新计算的相对湿度到2024.nc'
)
data2 = nc.Dataset(
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/'
    '原始相对湿度.nc'
)

# ----------------------------
# 2. 提取数据
# ----------------------------
time      = data.variables['time'][:]
lat       = data.variables['lat'][:]
lon       = data.variables['lon'][:]
rhum_rec  = data.variables['rhum'][:]
fill_rec  = getattr(data.variables['rhum'], '_FillValue', np.nan)

time2     = data2.variables['time'][:]
rhum_ori  = data2.variables['rhum'][:]
fill_ori  = getattr(data2.variables['rhum'], '_FillValue', np.nan)

# ----------------------------
# 3. 时间转换
# ----------------------------
dates_rec = nc.num2date(
    time,
    units=data.variables['time'].units,
    calendar=getattr(data.variables['time'], 'calendar', 'standard')
)
dates_ori = nc.num2date(
    time2,
    units=data2.variables['time'].units,
    calendar=getattr(data2.variables['time'], 'calendar', 'standard')
)

# ----------------------------
# 4. 截取 1990–1999
# ----------------------------
start = datetime(1990, 1, 1)
end   = datetime(1999,12,31)
mask1 = (dates_rec >= start) & (dates_rec <= end)
mask2 = (dates_ori >= start) & (dates_ori <= end)

rhum_rec = rhum_rec[mask1]
rhum_ori = rhum_ori[mask2]

# ----------------------------
# 5. 填充值替换为 NaN
# ----------------------------
rhum_rec = np.where(rhum_rec == fill_rec, np.nan, rhum_rec)
rhum_ori = np.where(rhum_ori == fill_ori, np.nan, rhum_ori)

# ----------------------------
# 6. 计算空间平均差 (重构–原始)
# ----------------------------
mean_rec = np.nanmean(rhum_rec, axis=0)
mean_ori = np.nanmean(rhum_ori, axis=0)
diff     = mean_rec - mean_ori

# ----------------------------
# 7. 计算全球面积加权平均并打印
# ----------------------------
weights     = np.cos(np.deg2rad(lat))
w2d         = np.broadcast_to(weights[:, None], diff.shape)
valid       = ~np.isnan(diff)
global_mean = np.nansum(diff[valid] * w2d[valid]) / np.nansum(w2d[valid])
print(f"Global area-weighted mean RH difference (recon−orig) 1990–1999: {global_mean:.3f}%")

# ----------------------------
# 8. 绘制地图（带经纬度标签但无格网线，所有文字 fontsize=16）
# ----------------------------
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'font.size': 16,
})

fig = plt.figure(figsize=(12, 6))
ax  = plt.axes(projection=ccrs.PlateCarree())

# 主图：差值
mesh = ax.pcolormesh(
    lon, lat, diff,
    cmap='RdBu',
    transform=ccrs.PlateCarree(),
    shading='auto',
    vmin=-3, vmax=3
)

# 海岸线 & 国界
ax.coastlines(linewidth=0.6)
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

# 只画刻度标签，不画格网线
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0,       # 不画线
    xlocs=np.arange(-180, 181, 60),
    ylocs=np.arange(-90,  91, 30),
    crs=ccrs.PlateCarree()
)
gl.top_labels   = False
gl.right_labels = False
gl.xformatter   = cticker.LongitudeFormatter()
gl.yformatter   = cticker.LatitudeFormatter()

# 色条：更长更细
cbar = plt.colorbar(
    mesh,
    ax=ax,
    orientation='horizontal',
    fraction=0.05,  # 色条厚度 ≈ 总图高度的 5%
    shrink=0.9,     # 色条长度 90%
    aspect=80,      # 色条的细长程度
    pad=0.08,
    extend='both'
)
cbar.set_label('RH difference (%)', fontsize=16)
cbar.ax.tick_params(labelsize=16)

# 标题
ax.set_title(
    'Spatial Mean RH Difference (Reconstructed – Original)\n'
    'Period: 1990–1999'
)

# 保存 & 显示
out_png = (
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_'
    '根据干湿球系数/RH_diff_map_1990-1999.svg'
)
plt.savefig(out_png, dpi=800, bbox_inches='tight')
plt.show()

# 关闭文件
data.close()
data2.close()