# import netCDF4 as nc
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from datetime import datetime
#
# # 读取NetCDF文件
# data = nc.Dataset('/Users/yincj/Desktop/GSOD均一化/HadISD/重新计算得到相对湿度2/monthly/重新计算相对湿度到2024.nc')  # 重构数据
# data2 = nc.Dataset('/Users/yincj/Desktop/GSOD均一化/HadISD/重新计算得到相对湿度2/monthly/相对湿度2.nc')  # 原始观测数据 (1973-2024)
# print(data)   # 查看重构数据结构
# print(data2)  # 查看原始观测数据结构
#
# # 提取维度和变量
# # 重构数据
# time = data.variables['time'][:]
# lat = data.variables['lat'][:]
# lon = data.variables['lon'][:]
# var_reconstructed = data.variables['rhum'][:]  # 重构相对湿度
#
# # 原始观测数据
# time2 = data2.variables['time'][:]
# lat2 = data2.variables['lat'][:]  # 假设纬度和经度维度一致
# lon2 = data2.variables['lon'][:]
# var_original = data2.variables['rhum'][:]  # 原始相对湿度
#
# # 检查时间单位并转换为日期
# time_units = data.variables['time'].units  # 例如 'days since 1970-01-01'
# time_calendar = data.variables['time'].calendar if 'calendar' in data.variables['time'].ncattrs() else 'standard'
# dates = nc.num2date(time, units=time_units, calendar=time_calendar)
#
# time_units2 = data2.variables['time'].units
# time_calendar2 = data2.variables['time'].calendar if 'calendar' in data2.variables['time'].ncattrs() else 'standard'
# dates2 = nc.num2date(time2, units=time_units2, calendar=time_calendar2)
#
# # 筛选1998-1999年的数据
# start_date = datetime(1973, 1, 1)
# end_date = datetime(2003, 12, 31)
#
# # 重构数据
# mask1 = (dates >= start_date) & (dates <= end_date)
# var_reconstructed_1998_1999 = var_reconstructed[mask1, :, :]
#
# # 原始观测数据
# mask2 = (dates2 >= start_date) & (dates2 <= end_date)
# var_original_1998_1999 = var_original[mask2, :, :]
#
# # 计算1998-1999年的平均值（沿时间轴求平均），结果为二维数组 (lat, lon)
# var_reconstructed_mean = np.mean(var_reconstructed_1998_1999, axis=0)
# var_original_mean = np.mean(var_original_1998_1999, axis=0)
#
# # 计算差值：重构数据减去原始数据
# var_diff = var_reconstructed_mean - var_original_mean
# # 设置阈值，提取绝对值超过阈值的“异常”格点，比如 ±3%
# high_threshold = 5.0
# low_threshold = -5.0
#
# # 找出异常高值和异常低值的索引（lat_idx, lon_idx）
# high_mask = var_diff > high_threshold
# low_mask = var_diff < low_threshold
#
# # 使用 np.where 获取经纬度索引
# high_indices = np.where(high_mask)
# low_indices = np.where(low_mask)
#
# # 打印或保存异常高值和低值的经纬度及差值
# print("\n🌡️ 异常高值区域 (RH diff > +3%)：")
# for lat_idx, lon_idx in zip(*high_indices):
#     print(f"Lat: {lat[lat_idx]:.2f}, Lon: {lon[lon_idx]:.2f}, Diff: {var_diff[lat_idx, lon_idx]:.2f}%")
#
# print("\n❄️ 异常低值区域 (RH diff < -3%)：")
# for lat_idx, lon_idx in zip(*low_indices):
#     print(f"Lat: {lat[lat_idx]:.2f}, Lon: {lon[lon_idx]:.2f}, Diff: {var_diff[lat_idx, lon_idx]:.2f}%")
# # ---------- 核心绘图部分开始 ----------
# plt.figure(figsize=(12, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
#
# # 使用 pcolormesh 绘制格点图
# # 如果 lon、lat 为 1D 数组，并且 var_diff 的形状为 (lat, lon),
# # 则新版 Matplotlib 建议使用 shading='auto' 来自动处理插值关系。
# mesh = ax.pcolormesh(lon, lat, var_diff,
#                      cmap='RdBu_r',
#                      transform=ccrs.PlateCarree(),
#                      shading='auto',
#                      vmin=-3, vmax=3)
#
# # 添加 colorbar，并设置 extend='both' 提示超出范围的值
# plt.colorbar(mesh, ax=ax, label='Relative Humidity Difference (%)', extend='both')
#
# # 添加地理特征
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
#
# # 添加网格线，并只显示左下角的经纬度标签
# gl = ax.gridlines(draw_labels=True)
# gl.top_labels = False
# gl.right_labels = False
#
# plt.title('Reconstructed - Original RH Mean (2005-2022)', fontsize=14)
#
# # 保存图像
# plt.savefig('/Users/yincj/Desktop/GSOD均一化/HadISD/重新计算得到相对湿度2/monthly/7400个站.png', dpi=300, bbox_inches='tight')
# plt.show()
# # ---------- 核心绘图部分结束 ----------
#
# data.close()
# data2.close()






# import netCDF4 as nc
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import matplotlib.dates as mdates  # 用于处理日期
#
# # 读取NetCDF文件
# data = nc.Dataset('/Users/yincj/Desktop/GSOD均一化/HadISD/重新计算相对湿度/monthly/重构相对湿度到2022.nc')  # 重构数据
# data2 = nc.Dataset('/Users/yincj/Desktop/GSOD均一化/HadISD/重新计算相对湿度/monthly/RH2022.nc')  # 原始观测数据
#
# # 提取变量
# # 重构数据
# time = data.variables['time'][:]
# var_reconstructed = data.variables['rhum'][:]  # 重构相对湿度
#
# # 原始观测数据
# time2 = data2.variables['time'][:-24]
# var_original = data2.variables['rhum'][:-24]  # 原始相对湿度
#
# # 检查时间单位并转换为日期
# time_units = data.variables['time'].units
# time_calendar = data.variables['time'].calendar if 'calendar' in data.variables['time'].ncattrs() else 'standard'
# dates = nc.num2date(time, units=time_units, calendar=time_calendar)
#
# time_units2 = data2.variables['time'].units
# time_calendar2 = data2.variables['time'].calendar if 'calendar' in data2.variables['time'].ncattrs() else 'standard'
# dates2 = nc.num2date(time2, units=time_units2, calendar=time_calendar2)
#
# # 将 cftime 对象转换为 Python 的 datetime 对象
# dates = [datetime(d.year, d.month, d.day) for d in dates]
# dates2 = [datetime(d.year, d.month, d.day) for d in dates2]
#
# # 检查时间长度是否一致
# if len(dates) != len(dates2):
#     raise ValueError("时间维度长度不一致，请检查数据对齐情况！")
#
# # 计算所有格点的差值（重构数据 - 原始数据）
# var_diff = var_reconstructed - var_original  # 形状仍为 (time, lat, lon)
#
# # 计算差值的全球平均值（沿纬度和经度轴平均）
# diff_mean = np.mean(var_diff, axis=(1, 2))  # 平均纬度和经度，得到 (time,)
#
# # 创建折线图
# plt.figure(figsize=(12, 6))
#
# # 绘制差值的折线
# plt.plot(dates, diff_mean, label='Reconstructed - Original RH', color='purple', linewidth=1.5)
#
# # 添加零线作为参考
# plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
#
# # 添加图表元素
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('RH Difference (%)', fontsize=12)
# plt.title('Global Mean RH Difference (Reconstructed - Original, 1973-2022)', fontsize=14)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # 设置x轴刻度显示年份
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))  # 每5年显示一个刻度
# plt.xticks(rotation=45)
#
# # 调整布局以防止标签被截断
# plt.tight_layout()
#
# # 保存图像
# plt.savefig('/Users/yincj/Desktop/GSOD均一化/HadISD/重新计算相对湿度/monthly/RH_diff_1973_2022_lineplot.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # 关闭数据集
# data.close()
# data2.close()
#
# import netCDF4 as nc
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from datetime import datetime
#
# # 读取NetCDF文件
# data = nc.Dataset('/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/重新计算的相对湿度到2024.nc')  # 重构数据
# data2 = nc.Dataset('/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/原始相对湿度.nc')  # 原始观测数据 (1973-2024)
# print(data)   # 查看重构数据结构
# print(data2)  # 查看原始观测数据结构
#
# # 提取维度和变量
# # 重构数据
# time = data.variables['time'][:]
# lat = data.variables['lat'][:]
# lon = data.variables['lon'][:]
# var_reconstructed = data.variables['rhum'][:]  # 重构相对湿度
# fill_value1 = data.variables['rhum']._FillValue if hasattr(data.variables['rhum'], '_FillValue') else np.nan
#
# # 原始观测数据
# time2 = data2.variables['time'][:]
# lat2 = data2.variables['lat'][:]  # 假设纬度和经度维度一致
# lon2 = data2.variables['lon'][:]
# var_original = data2.variables['rhum'][:]  # 原始相对湿度
# fill_value2 = data2.variables['rhum']._FillValue if hasattr(data2.variables['rhum'], '_FillValue') else np.nan
#
# # 检查时间单位并转换为日期
# time_units = data.variables['time'].units  # e.g. 'days since 1970-01-01'
# time_calendar = data.variables['time'].calendar if 'calendar' in data.variables['time'].ncattrs() else 'standard'
# dates = nc.num2date(time, units=time_units, calendar=time_calendar)
#
# time_units2 = data2.variables['time'].units
# time_calendar2 = data2.variables['time'].calendar if 'calendar' in data2.variables['time'].ncattrs() else 'standard'
# dates2 = nc.num2date(time2, units=time_units2, calendar=time_calendar2)
#
# # 筛选1973-2003年的数据
# start_date = datetime(1990, 1, 1)
# end_date = datetime(1999, 12, 31)
#
# # 重构数据
# mask1 = (dates >= start_date) & (dates <= end_date)
# time_selected = time[mask1]
# dates_selected = dates[mask1]
# var_reconstructed_selected = var_reconstructed[mask1, :, :]
#
# # 原始观测数据
# mask2 = (dates2 >= start_date) & (dates2 <= end_date)
# time2_selected = time2[mask2]
# dates2_selected = dates2[mask2]
# var_original_selected = var_original[mask2, :, :]
#
# # 确保时间轴一致
# if not np.array_equal(dates_selected, dates2_selected):
#     print("时间轴不一致，请检查数据！")
#     exit()
#
# # 识别有效格点（两组数据均非缺测值）
# # 将缺测值替换为 NaN 以便处理
# var_reconstructed_selected = np.where(var_reconstructed_selected == fill_value1, np.nan, var_reconstructed_selected)
# var_original_selected = np.where(var_original_selected == fill_value2, np.nan, var_original_selected)
#
# # 计算每个格点在整个时间段内是否始终有有效数据
# valid_mask = (~np.isnan(var_reconstructed_selected).any(axis=0)) & (~np.isnan(var_original_selected).any(axis=0))
#
# # 检查是否有有效格点
# if not valid_mask.any():
#     print("没有共同的有效格点！")
#     exit()
#
# # 计算有效格点的月均值（沿纬度和经度轴平均）
# # 注意：np.nanmean 对空序列会给出警告，如果没有有效点会导致空数组
# reconstructed_mean = np.nanmean(var_reconstructed_selected[:, valid_mask], axis=1)
# original_mean = np.nanmean(var_original_selected[:, valid_mask], axis=1)
#
# # 计算整个时段内的空间均值图像（用于地图绘制）
# var_reconstructed_mean = np.nanmean(var_reconstructed_selected, axis=0)
# var_original_mean = np.nanmean(var_original_selected, axis=0)
#
# # 计算差值：重构数据减去原始数据
# var_diff = var_reconstructed_mean - var_original_mean
# # 设置阈值，提取绝对值超过阈值的“异常”格点，比如 ±5%
# high_threshold = 3.0
# low_threshold = -3.0
#
# # 找出异常高值和异常低值的索引（lat_idx, lon_idx）
# high_mask = var_diff > high_threshold
# low_mask = var_diff < low_threshold
#
# # 使用 np.where 获取经纬度索引
# high_indices = np.where(high_mask)
# low_indices = np.where(low_mask)
#
# # 打印异常区域的经纬度及差值
# print("\n🌡️ 异常高值区域 (RH diff > +5%)：")
# for lat_idx, lon_idx in zip(*high_indices):
#     print(f"Lat: {lat[lat_idx]:.2f}, Lon: {lon[lon_idx]:.2f}, Diff: {var_diff[lat_idx, lon_idx]:.2f}%")
#
# print("\n❄️ 异常低值区域 (RH diff < -5%)：")
# for lat_idx, lon_idx in zip(*low_indices):
#     print(f"Lat: {lat[lat_idx]:.2f}, Lon: {lon[lon_idx]:.2f}, Diff: {var_diff[lat_idx, lon_idx]:.2f}%")
#
# # ---------- 绘图部分开始 ----------
# fig = plt.figure(figsize=(12, 10))
#
# # 地图子图
# ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
# mesh = ax1.pcolormesh(lon, lat, var_diff,
#                       cmap='RdBu',
#                       transform=ccrs.PlateCarree(),
#                       shading='auto',
#                       vmin=-3, vmax=3)
# plt.colorbar(mesh, ax=ax1, label='Relative Humidity Difference (%)', extend='both')
# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS, linestyle=':')
# gl = ax1.gridlines(draw_labels=True)
# gl.top_labels = False
# gl.right_labels = False
# ax1.set_title('Reconstructed - Original RH Mean (1990-1999)', fontsize=14)
#
# # 时间序列子图
# ax2 = fig.add_subplot(2, 1, 2)
#
# # 将 dates_selected 转换为 Python 原生 datetime 对象，matplotlib 不能直接处理 cftime 对象
# dates_selected_py = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates_selected])
#
# # 绘制两条折线
# ax2.plot(dates_selected_py, reconstructed_mean, label='Reconstructed RH', color='blue')
# ax2.plot(dates_selected_py, original_mean, label='Original RH', color='red')
#
# # 设置标签和标题
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Relative Humidity (%)')
# ax2.set_title('Monthly Mean RH (1973-2003, Valid Grid Points)')
# ax2.legend()
# ax2.grid(True)
# plt.setp(ax2.get_xticklabels(), rotation=45)
#
# fig.tight_layout()
# plt.savefig('/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/6000个站_with_timeseries1990-1999.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # 关闭数据文件
# data.close()
# data2.close()


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