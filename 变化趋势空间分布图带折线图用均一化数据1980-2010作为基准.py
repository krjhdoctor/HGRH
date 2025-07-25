# import netCDF4
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import linregress
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.mpl.ticker as cticker
# import cartopy.mpl.ticker as cticker
# # ——— Parameters ———
# MIN_COUNT = 60  # minimum months required (>=360)
# order = ['ERA5','MERRA2','HadISDH','Homogenization']
# # set MERRA2 start to 1980, others to 1973
# start_years = {'ERA5':1973, 'MERRA2':1980, 'HadISDH':1973, 'Homogenization':1973}
# paths = {
#     'ERA5':       '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/ERA5_on_HadISDH_masked2024.nc',
#     'MERRA2':     '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/MERRA2_on_HadISDH_2024.nc',
#     'HadISDH':    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/HadISDH_2024_cleaned.nc',
#     'Homogenization': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/均一化_均值_on_HadISDH_masked.nc'
# }
# # ——— 在定义 order, colors 之后，增加一个“显示名称”映射 ———
# display_names = {
#     'ERA5':          'ERA5',
#     'MERRA2':        'MERRA2',
#     'HadISDH':       'HadISDH',
#     'Homogenization':'Homogenization (this study)'
# }
# # ——— Global matplotlib settings ———
# plt.rcParams['font.size'] = 16
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'sans-serif'
#
# def reorder_data_with_lon(lon, data):
#     lon2 = np.where(lon>180, lon-360, lon)
#     idx  = np.argsort(lon2)
#     return lon2[idx], data[..., idx]
#
# def calculate_slope_p(data, min_count=360):
#     nt, nlat, nlon = data.shape
#     t = np.arange(nt)
#     valid_count = np.sum(~np.isnan(data), axis=0)
#     slope = np.full((nlat, nlon), np.nan)
#     pval  = np.full((nlat, nlon), np.nan)
#     for i in range(nlat):
#         for j in range(nlon):
#             if valid_count[i,j] >= min_count:
#                 y = data[:,i,j]
#                 ok = ~np.isnan(y)
#                 res = linregress(t[ok], y[ok])
#                 slope[i,j] = res.slope * 120.0  # trend per decade
#                 pval[i,j]  = res.pvalue
#     return slope, pval, valid_count
#
# def calculate_anomalies(data, start_year, base_start=1980, base_end=2010):
#     """
#     Compute monthly anomalies relative to the 1980–2010 climatology.
#     data shape = (ntime, nlat, nlon)
#     start_year = year of first month in data
#     """
#     nt = data.shape[0]
#     years = start_year + (np.arange(nt) // 12)
#     mask = (years >= base_start) & (years <= base_end)
#     climatology = np.nanmean(data[mask, ...], axis=0)
#     return data - climatology[None, ...]
#
# def calculate_annual_mean(data):
#     nyears = data.shape[0] // 12
#     return np.nanmean(
#         data[:nyears*12].reshape(nyears,12,*data.shape[1:]),
#         axis=1
#     )
#
# # ——— Load and process each dataset ———
# datasets = {}
# for name in order:
#     ds = netCDF4.Dataset(paths[name])
#     rh = ds.variables['rh_abs'][:].astype(float)
#     fill = getattr(ds.variables['rh_abs'], '_FillValue', None)
#     if fill is not None:
#         rh[rh == fill] = np.nan
#     rh[(rh < 0) | (rh > 100)] = np.nan
#
#     lat = ds.variables.get('latitude', ds.variables.get('lat'))[:]
#     lon = ds.variables.get('longitude', ds.variables.get('lon'))[:]
#     lon2, rh = reorder_data_with_lon(lon, rh)
#     if lat[0] > lat[-1]:
#         lat, rh = lat[::-1], rh[:, ::-1, :]
#
#     # store coords once
#     if 'lats' not in globals():
#         lats, lons = lat, lon2
#
#     # 1) Monthly anomalies using 1980–2010 baseline
#     anom = calculate_anomalies(rh, start_years[name], base_start=1980, base_end=2010)
#
#     # 2) Trend computation
#     slope, pval, valid_count = calculate_slope_p(anom, min_count=MIN_COUNT)
#     # 3) Annual mean anomaly
#     annual = calculate_annual_mean(anom)
#
#     ds.close()
#     datasets[name] = {
#         'slope': slope,
#         'pval': pval,
#         'valid_count': valid_count,
#         'annual': annual
#     }
#
# # ——— Compute global area-weighted annual series ———
# time_series = {}
# weights = np.cos(np.deg2rad(lats))
# weights /= np.nansum(weights)
# for name in order:
#     arr = datasets[name]['annual']
#     mask_short = datasets[name]['valid_count'] < MIN_COUNT
#     arr = np.where(mask_short[None,...], np.nan, arr)
#
#     ts = []
#     for year_slice in arr:
#         w2d = np.broadcast_to(weights[:,None], year_slice.shape)
#         num = np.nansum(year_slice * w2d)
#         den = np.nansum(w2d[~np.isnan(year_slice)])
#         ts.append(num/den)
#     years = np.arange(start_years[name], start_years[name] + len(ts))
#     time_series[name] = (years, np.array(ts))
#
# # ——— Plot in a 3×2 grid ———
# fig = plt.figure(figsize=(16,18))
# gs = fig.add_gridspec(3,2, height_ratios=[1,1,0.9], hspace=-0.5, wspace=0.16)
# labels = ['(a)','(b)','(c)','(d)']
# pcm_list = []
# colors = {'ERA5':'#7fb2d3','MERRA2':'#8dd3c9','HadISDH':'#ffb55f','Homogenization':'#fc7f71'}
#
# for idx, name in enumerate(order):
#     ax = fig.add_subplot(gs[idx//2, idx%2], projection=ccrs.PlateCarree())
#     ax.set_extent([-180, 180, -90, 90])
#
#     # 5.4 经纬度标签（无格网线）
#     gl = ax.gridlines(
#         draw_labels=True,
#         linewidth=0,                 # 线宽为 0，即不画网格线
#         crs=ccrs.PlateCarree()
#     )
#     gl.top_labels   = False
#     gl.right_labels = False
#     gl.xformatter   = cticker.LongitudeFormatter()
#     gl.yformatter   = cticker.LatitudeFormatter()
#     gl.xlabel_style = {'size': 14}
#     gl.ylabel_style = {'size': 14}
#     # 保留原来的刻度间隔（根据需要调整）
#     gl.xlocator    = plt.MultipleLocator(60)
#     gl.ylocator    = plt.MultipleLocator(30)
#
#
#     ax.coastlines()
#
#     lon2d, lat2d = np.meshgrid(lons, lats)
#     slope_map = np.where(datasets[name]['valid_count'] < MIN_COUNT,
#                          np.nan, datasets[name]['slope'])
#     pval_map  = np.where(datasets[name]['valid_count'] < MIN_COUNT,
#                          np.nan, datasets[name]['pval'])
#
#     pcm = ax.pcolormesh(
#         lon2d, lat2d, slope_map,
#         transform=ccrs.PlateCarree(),
#         cmap='RdBu', vmin=-1.5, vmax=1.5
#     )
#     sig = (pval_map < 0.05)
#     ax.scatter(lon2d[sig], lat2d[sig], s=5, color='black',
#                transform=ccrs.PlateCarree())
#
#     # area-weighted global trend
#     w2d = np.broadcast_to(weights[:,None], slope_map.shape)
#     global_trend = np.nansum(slope_map * w2d) / np.nansum(w2d[~np.isnan(slope_map)])
#
#     ax.set_title(f"{labels[idx]} {display_names[name]}", loc='left')
#     ax.text(-170, -80,
#             f"Global mean: {global_trend:.2f} %/decade",
#             transform=ccrs.PlateCarree(),
#             bbox=dict(facecolor='white', alpha=0.7),
#             fontsize=14)
#     pcm_list.append(pcm)
#
# # shared colorbar
# cax = fig.add_axes([0.13,0.356,0.76,0.01])
# cbar = fig.colorbar(pcm_list[0], cax=cax, orientation='horizontal', extend='both')
# cbar.set_label('Trend (% per decade)', fontsize=16)
# cbar.set_ticks(np.linspace(-1.5, 1.5, 7))
#
# # (e) time series panel
# axf = fig.add_subplot(gs[2,0])
# axf.set_position([0.124,0.135,0.775,0.17])
# for name in order:
#     yrs, ts = time_series[name]
#     axf.plot(yrs, ts, label=display_names[name], color=colors[name])
# axf.set_xticks(np.arange(1973, 2028, 5))
# axf.set_xlabel('Year')
# axf.set_ylabel('RH anomaly (%)')
# axf.set_title('(e) Global RH anomaly', loc='left')
# axf.legend(frameon=False, fontsize=16)
#
# plt.savefig(
#     '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/四种数据的空间分布图和时间分布图.svg',
#     format='svg',
#     bbox_inches='tight'
# )
#
# plt.show()
#
# # ——— Print time-series trends ———
# print("Linear trend of global area-weighted series (% per decade):")
# for name in order:
#     yrs, ts = time_series[name]
#     ok = ~np.isnan(ts)
#     res = linregress(yrs[ok], ts[ok])
#     print(f"  {name}: slope = {res.slope:.3f} %/decade, p-value = {res.pvalue:.4f}")



import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.mpl.ticker as cticker
# ——— Parameters ———
MIN_COUNT = 0  # minimum months required (>=360)
order = ['ERA5','MERRA2','HadISD','Homogenization']
# set MERRA2 start to 1980, others to 1973
start_years = {'ERA5':1973, 'MERRA2':1980, 'HadISD':1973, 'Homogenization':1973}
paths = {
    'ERA5':       '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/ERA5_on_HadISD_2024.nc',
    'MERRA2':     '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/MERRA2_on_HadISD_2024.nc',
    'HadISD':    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/观测_均值_Global.nc',
    'Homogenization': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/均一化_均值_Global.nc'
}
# ——— 在定义 order, colors 之后，增加一个“显示名称”映射 ———
display_names = {
    'ERA5':          'ERA5',
    'MERRA2':        'MERRA2',
    'HadISD':       'HadISD',
    'Homogenization':'Homogenization (this study)'
}
# ——— Global matplotlib settings ———
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def reorder_data_with_lon(lon, data):
    lon2 = np.where(lon>180, lon-360, lon)
    idx  = np.argsort(lon2)
    return lon2[idx], data[..., idx]

def calculate_slope_p(data, min_count=360):
    nt, nlat, nlon = data.shape
    t = np.arange(nt)
    valid_count = np.sum(~np.isnan(data), axis=0)
    slope = np.full((nlat, nlon), np.nan)
    pval  = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            if valid_count[i,j] >= min_count:
                y = data[:,i,j]
                ok = ~np.isnan(y)
                res = linregress(t[ok], y[ok])
                slope[i,j] = res.slope * 120.0  # trend per decade
                pval[i,j]  = res.pvalue
    return slope, pval, valid_count

def calculate_anomalies(data, start_year, base_start=1980, base_end=2010):
    """
    Compute monthly anomalies relative to the 1980–2010 climatology.
    data shape = (ntime, nlat, nlon)
    start_year = year of first month in data
    """
    nt = data.shape[0]
    years = start_year + (np.arange(nt) // 12)
    mask = (years >= base_start) & (years <= base_end)
    climatology = np.nanmean(data[mask, ...], axis=0)
    return data - climatology[None, ...]

def calculate_annual_mean(data):
    nyears = data.shape[0] // 12
    return np.nanmean(
        data[:nyears*12].reshape(nyears,12,*data.shape[1:]),
        axis=1
    )

# ——— Load and process each dataset ———
datasets = {}
# ——— Load and process each dataset ———
datasets = {}
for name in order:
    ds = netCDF4.Dataset(paths[name])

    # —— 根据文件名选择变量 ——
    if name in ['ERA5', 'MERRA2']:
        varname = 'rh_abs'
    else:
        varname = 'rhum'

    rh = ds.variables[varname][:].astype(float)
    fill = getattr(ds.variables[varname], '_FillValue', None)
    if fill is not None:
        rh[rh == fill] = np.nan
    rh[(rh < 0) | (rh > 100)] = np.nan

    lat = ds.variables.get('latitude', ds.variables.get('lat'))[:]
    lon = ds.variables.get('longitude', ds.variables.get('lon'))[:]
    lon2, rh = reorder_data_with_lon(lon, rh)
    if lat[0] > lat[-1]:
        lat, rh = lat[::-1], rh[:, ::-1, :]

    # store coords once
    if 'lats' not in globals():
        lats, lons = lat, lon2

    # 1) Monthly anomalies using 1980–2010 baseline
    anom = calculate_anomalies(rh, start_years[name], base_start=1980, base_end=2010)

    # 2) Trend computation
    slope, pval, valid_count = calculate_slope_p(anom, min_count=MIN_COUNT)
    # 3) Annual mean anomaly
    annual = calculate_annual_mean(anom)

    ds.close()
    datasets[name] = {
        'slope': slope,
        'pval': pval,
        'valid_count': valid_count,
        'annual': annual
    }

# ——— Compute global area-weighted annual series ———
time_series = {}
weights = np.cos(np.deg2rad(lats))
weights /= np.nansum(weights)
for name in order:
    arr = datasets[name]['annual']
    mask_short = datasets[name]['valid_count'] < MIN_COUNT
    arr = np.where(mask_short[None,...], np.nan, arr)

    ts = []
    for year_slice in arr:
        w2d = np.broadcast_to(weights[:,None], year_slice.shape)
        num = np.nansum(year_slice * w2d)
        den = np.nansum(w2d[~np.isnan(year_slice)])
        ts.append(num/den)
    years = np.arange(start_years[name], start_years[name] + len(ts))
    time_series[name] = (years, np.array(ts))

# ——— Plot in a 3×2 grid ———
fig = plt.figure(figsize=(16,18))
gs = fig.add_gridspec(3,2, height_ratios=[1,1,0.9], hspace=-0.5, wspace=0.16)
labels = ['(a)','(b)','(c)','(d)']
pcm_list = []
colors = {'ERA5':'#fb3e38','MERRA2':'#79add6','HadISD':'#ee8227','Homogenization':'#5a5a5a'}

for idx, name in enumerate(order):
    ax = fig.add_subplot(gs[idx//2, idx%2], projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90])

    # 5.4 经纬度标签（无格网线）
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0,                 # 线宽为 0，即不画网格线
        crs=ccrs.PlateCarree()
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = cticker.LongitudeFormatter()
    gl.yformatter   = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    # 保留原来的刻度间隔（根据需要调整）
    gl.xlocator    = plt.MultipleLocator(60)
    gl.ylocator    = plt.MultipleLocator(30)


    ax.coastlines()

    lon2d, lat2d = np.meshgrid(lons, lats)
    slope_map = np.where(datasets[name]['valid_count'] < MIN_COUNT,
                         np.nan, datasets[name]['slope'])
    pval_map  = np.where(datasets[name]['valid_count'] < MIN_COUNT,
                         np.nan, datasets[name]['pval'])

    pcm = ax.pcolormesh(
        lon2d, lat2d, slope_map,
        transform=ccrs.PlateCarree(),
        cmap='RdBu', vmin=-1.5, vmax=1.5
    )
    sig = (pval_map < 0.05)
    ax.scatter(lon2d[sig], lat2d[sig], s=0.5, color='black',alpha=0.5,
               transform=ccrs.PlateCarree())

    # area-weighted global trend
    w2d = np.broadcast_to(weights[:,None], slope_map.shape)
    global_trend = np.nansum(slope_map * w2d) / np.nansum(w2d[~np.isnan(slope_map)])

    ax.set_title(f"{labels[idx]} {display_names[name]}", loc='left')
    ax.text(-170, -80,
            f"Global mean: {global_trend:.2f} %/decade",
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=14)
    pcm_list.append(pcm)

# shared colorbar
cax = fig.add_axes([0.13,0.356,0.76,0.01])
cbar = fig.colorbar(pcm_list[0], cax=cax, orientation='horizontal', extend='both')
cbar.set_label('Trend (% per decade)', fontsize=16)
cbar.set_ticks(np.linspace(-1.5, 1.5, 7))

# (e) time series panel
axf = fig.add_subplot(gs[2,0])
axf.set_position([0.124,0.135,0.775,0.17])
for name in order:
    yrs, ts = time_series[name]
    axf.plot(yrs, ts, label=display_names[name], color=colors[name])
axf.set_xticks(np.arange(1973, 2028, 5))
axf.set_xlabel('Year')
axf.set_ylabel('RH anomaly (%)')
axf.set_title('(e) Global RH anomaly', loc='left')
# 方法5: 将图例放在下方，分多列显示
axf.legend(frameon=False, fontsize=16,
          bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)

plt.savefig(
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/四种数据的空间分布图和时间分布图.svg',
    format='svg',
    bbox_inches='tight'
)

plt.show()

# ——— Print time-series trends ———
print("Linear trend of global area-weighted series (% per decade):")
for name in order:
    yrs, ts = time_series[name]
    ok = ~np.isnan(ts)
    res = linregress(yrs[ok], ts[ok])
    print(f"  {name}: slope = {res.slope:.3f} %/decade, p-value = {res.pvalue:.4f}")
