
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
    'Homogenization': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/均一化_均值_Global.nc',
    'HadISDH':'/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/HadISDH_2024_cleaned.nc',
    "JRA3Q":'/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/再分析数据所有陆地的/JRA3q_rh_land.nc'

}
data1=netCDF4.Dataset(paths['HadISDH'],'r')
print(data1.variables['latitude'][:])
print(data1.variables['longitude'][:])
data2=netCDF4.Dataset(paths['ERA5'],'r')
print(data2.variables['latitude'][:])
print(data2.variables['longitude'][:])
print(data2)
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

# ======================= 新增：差值“趋势地图” + “差值折线图” =======================
# 1) 准备公共网格
lon2d, lat2d = np.meshgrid(lons, lats)
weights = np.cos(np.deg2rad(lats))
weights2d = np.broadcast_to(weights[:, None], lon2d.shape)

# 2) 构建有效掩膜（两两同时有效且满足 MIN_COUNT）
valid_homo_had = (datasets['Homogenization']['valid_count'] >= MIN_COUNT) & \
                 (datasets['HadISD']['valid_count']        >= MIN_COUNT)
valid_homo_era = (datasets['Homogenization']['valid_count'] >= MIN_COUNT) & \
                 (datasets['ERA5']['valid_count']          >= MIN_COUNT)

# 3) 趋势差（单位：%/10yr）
slope_homo = datasets['Homogenization']['slope']
slope_had  = datasets['HadISD']['slope']
slope_era  = datasets['ERA5']['slope']

diff_slope_homo_minus_had = np.where(valid_homo_had, slope_had-slope_homo, np.nan)
diff_slope_homo_minus_era = np.where(valid_homo_era, slope_era-slope_homo, np.nan)

# 供配色对称范围（根据数据自动取最大绝对值，保证居中0）
def sym_vlim(a):
    m = np.nanmax(np.abs(a))
    if not np.isfinite(m) or m == 0:
        m = 1.0
    # 稍微留白
    return -m, m

vmin1, vmax1 = sym_vlim(diff_slope_homo_minus_had)
vmin2, vmax2 = sym_vlim(diff_slope_homo_minus_era)
# 两张图统一色标范围，便于比较
vmax = max(abs(vmin1), abs(vmax1), abs(vmin2), abs(vmax2))
vmin = -vmax

# 4) 计算全球加权平均“趋势差”（单值，用于角标说明）
def area_weighted_mean(field):
    w = weights2d.copy()
    # 只对有效格点加权
    mask = ~np.isnan(field)
    if not np.any(mask):
        return np.nan
    return np.nansum(field * w) / np.nansum(w[mask])

gm_diff_had = area_weighted_mean(diff_slope_homo_minus_had)
gm_diff_era = area_weighted_mean(diff_slope_homo_minus_era)


# 6) 折线图：年平均距平差（Homogenization − HadISD 与 Homogenization − ERA5）
#    先把年距平做同一掩膜，再面积加权平均到单个时间序列
annual_homo = datasets['Homogenization']['annual']  # (nyear, lat, lon)
annual_had  = datasets['HadISD']['annual']
annual_era  = datasets['ERA5']['annual']

# 对齐长度（一般相同起点：1973；ERA5 也是 1973 这里 OK）
ny = min(annual_homo.shape[0], annual_had.shape[0], annual_era.shape[0])
annual_homo = annual_homo[:ny]
annual_had  = annual_had[:ny]
annual_era  = annual_era[:ny]

# 年度掩膜：两两都必须为有效格点（沿时间每年逐年平均时各自已处理 NaN，这里只按 valid_count 过滤）
mask_hh = np.broadcast_to(valid_homo_had, annual_homo.shape[1:])
mask_he = np.broadcast_to(valid_homo_era, annual_homo.shape[1:])

def area_weighted_series(diff_annual, mask):
    ts = []
    for y in range(diff_annual.shape[0]):
        A = np.where(mask, diff_annual[y], np.nan)
        m = ~np.isnan(A)
        if not np.any(m):
            ts.append(np.nan)
        else:
            ts.append(np.nansum(A * weights2d) / np.nansum(weights2d[m]))
    return np.array(ts)

diff_annual_hh = annual_had-annual_homo
diff_annual_he = annual_era-annual_homo

ts_hh = area_weighted_series(diff_annual_hh, mask_hh)
ts_he = area_weighted_series(diff_annual_he, mask_he)

years_common = np.arange(start_years['Homogenization'], start_years['Homogenization'] + ny)
# =============== 合并成三联图：上两张地图 + 下方一张折线图（色标固定 -3~3） ===============
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 0.6], wspace=0.12, hspace=-0.05)

# 固定色标范围
vmin, vmax = -1.5, 1.5

# (a) Homogenization − HadISD
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.set_extent([-180, 180, -90, 90])
gl1 = ax1.gridlines(draw_labels=True, linewidth=0, crs=ccrs.PlateCarree())
gl1.top_labels   = False
gl1.right_labels = False
gl1.xformatter   = cticker.LongitudeFormatter()
gl1.yformatter   = cticker.LatitudeFormatter()
gl1.xlabel_style = {'size': 16}
gl1.ylabel_style = {'size': 16}
gl1.xlocator     = plt.MultipleLocator(60)
gl1.ylocator     = plt.MultipleLocator(30)
ax1.coastlines()
pcm1 = ax1.pcolormesh(lon2d, lat2d, diff_slope_homo_minus_had,
                      transform=ccrs.PlateCarree(), cmap='RdBu', vmin=vmin, vmax=vmax)
ax1.set_title('(a) Trend difference: HadISD - Homogenization', loc='left')
ax1.text(-170, -80, f"Global mean difference: {gm_diff_had:.2f} %/decade",
         transform=ccrs.PlateCarree(), bbox=dict(facecolor='white', alpha=0.8))

# (b) Homogenization − ERA5
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.set_extent([-180, 180, -90, 90])
gl2 = ax2.gridlines(draw_labels=True, linewidth=0, crs=ccrs.PlateCarree())
gl2.top_labels   = False
gl2.right_labels = False
gl2.xformatter   = cticker.LongitudeFormatter()
gl2.yformatter   = cticker.LatitudeFormatter()
gl2.xlabel_style = {'size': 16}
gl2.ylabel_style = {'size': 16}
gl2.xlocator     = plt.MultipleLocator(60)
gl2.ylocator     = plt.MultipleLocator(30)
ax2.coastlines()
pcm2 = ax2.pcolormesh(lon2d, lat2d, diff_slope_homo_minus_era,
                      transform=ccrs.PlateCarree(), cmap='RdBu', vmin=vmin, vmax=vmax)
ax2.set_title('(b) Trend difference: ERA5 - Homogenization', loc='left')
ax2.text(-170, -80, f"Global mean difference: {gm_diff_era:.2f} %/decade",
         transform=ccrs.PlateCarree(), bbox=dict(facecolor='white', alpha=0.8))

# 手动指定位置 [左, 下, 宽度, 高度]，宽度调大，高度很小
cax = fig.add_axes([0.1, 0.43, 0.8, 0.02])  # 0.8 表示占整个画布的80%宽度
cbar = fig.colorbar(
    pcm1, cax=cax,
    orientation='horizontal',
    extend='both'
)
cbar.set_label('Trend difference (% / decade)')
cbar.set_ticks(np.linspace(vmin, vmax, 7))


# (c) 底部折线图
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(years_common, ts_hh, label='HadISD - Homogenization')
ax3.plot(years_common, ts_he, label='ERA5 - Homogenization')
ax3.axhline(0, linewidth=1, linestyle='--')
ax3.set_xlabel('Year')
ax3.set_ylabel('RH anomaly difference (%)')
ax3.set_title('(c) Global annual RH anomaly differences', loc='left')
ax3.legend(frameon=False)

plt.savefig('/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/差值_趋势地图+年序列_TNR.svg',
            format='svg', bbox_inches='tight', dpi=500)
plt.show()