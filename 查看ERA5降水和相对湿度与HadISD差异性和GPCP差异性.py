import netCDF4 as nc
import numpy as np
import pandas as pd
import calendar
from scipy import stats
import matplotlib.pyplot as plt
import os
from netCDF4 import num2date
import matplotlib as mpl
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False
mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "lines.linewidth": 1.7
})

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not available, fall back to simple maps.")

# ================== 配置 ==================
DATA_ERA5_RH   = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/ERA5_on_HadISD_2024.nc'
DATA_HADISD_RH = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/观测_均值_Global.nc'
DATA_ERA5_PR   = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/ERA5_precip_monthly_2.5deg_had.nc'
DATA_MSWEP_PR  = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/mswep_precip_monthly_2.5deg_HadISD.nc'
DATA_GPCP_PR   = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask/GPCP_monthly_2.5deg_HadISD_land_only.nc'

OUT_DIR = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/陆地再分析相对湿度/用HadISD做mask'
os.makedirs(OUT_DIR, exist_ok=True)

START_YEAR = 1980
END_YEAR   = 2024
BASE_START = 1980   # 基准期开始
BASE_END   = 2010   # 基准期结束
MIN_MONTHS = 6
APPLY_MASK_TO_PRECIP = True

# ================== 函数 ==================
def read_time_variable_standard(ds, name='time'):
    tv = ds.variables[name]; arr = tv[:]
    units = getattr(tv, 'units', None)
    if units is None:
        raise AttributeError("time variable has no 'units' attribute.")
    cal = getattr(tv, 'calendar', 'standard')
    dts = num2date(arr, units, calendar=cal)
    return pd.to_datetime([pd.Timestamp(str(x)) for x in dts])

def read_monthly_var(path, varname):
    ds = nc.Dataset(path)
    times = read_time_variable_standard(ds, 'time')
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    data = ds.variables[varname][:].astype(np.float32)
    ds.close()
    return times, lat, lon, data

def read_and_convert(path, varname, to_mm=True):
    ds = nc.Dataset(path)
    tv = ds.variables['time']
    times = num2date(tv[:], tv.units, calendar=getattr(tv,'calendar','standard'))
    lat = ds.variables.get('lat', ds.variables.get('latitude'))[:]
    lon = ds.variables.get('lon', ds.variables.get('longitude'))[:]
    data = ds.variables[varname][:].astype(np.float32)
    ds.close()
    if to_mm == 'gpcp':
        for t, dt in enumerate(times):
            data[t] *= calendar.monthrange(dt.year, dt.month)[1]
    elif to_mm == 'era5':
        for t, dt in enumerate(times):
            data[t] = data[t]*1000.0*calendar.monthrange(dt.year, dt.month)[1]
    times_pd = pd.to_datetime([pd.Timestamp(str(x)) for x in times])
    return times_pd, lat, lon, data

def subset_year(times, data, start, end):
    sel = (times.year >= start) & (times.year <= end)
    return times[sel], data[sel]

def monthly_to_annual_rh(times, data, min_months=6):
    years = times.year.values
    target_years = np.arange(START_YEAR, END_YEAR+1)
    ny, nx = data.shape[1:]
    out = np.full((len(target_years), ny, nx), np.nan, dtype=np.float32)
    for i, yr in enumerate(target_years):
        m = (years == yr)
        if not m.any(): continue
        block = data[m]
        cnt = np.sum(~np.isnan(block), axis=0)
        enough = cnt >= min_months
        vals = np.nanmean(block, axis=0)
        vals[~enough] = np.nan
        out[i] = vals
    return target_years, out

def monthly_to_annual_precip(times, data, min_months=6):
    years = times.year.values
    target_years = np.arange(START_YEAR, END_YEAR+1)
    ny, nx = data.shape[1:]
    out = np.full((len(target_years), ny, nx), np.nan, dtype=np.float32)
    for i, yr in enumerate(target_years):
        m = (years == yr)
        if not m.any(): continue
        block = data[m]
        cnt = np.sum(~np.isnan(block), axis=0)
        enough = cnt >= min_months
        vals = np.nansum(block, axis=0)
        vals[~enough] = np.nan
        out[i] = vals
    return target_years, out

def area_weighted_mean(cube, lat):
    w = np.cos(np.deg2rad(lat))[:, None]
    series = np.full(cube.shape[0], np.nan)
    for t in range(cube.shape[0]):
        fld = cube[t]; valid = ~np.isnan(fld)
        if valid.any():
            ww = w * valid
            series[t] = np.nansum(fld*ww)/np.nansum(ww)
    return series

def spatial_corr(a, b, min_valid=10):
    nt, ny, nx = a.shape
    rmat = np.full((ny, nx), np.nan, dtype=np.float32)
    pmat = np.full((ny, nx), np.nan, dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            s1 = a[:,i,j]; s2 = b[:,i,j]
            good = ~np.isnan(s1) & ~np.isnan(s2)
            if good.sum() >= min_valid:
                try:
                    r,p = stats.pearsonr(s1[good], s2[good])
                    rmat[i,j] = r; pmat[i,j] = p
                except:
                    pass
    return rmat, pmat

def year_index(src_years, ref_years):
    return np.array([np.where(src_years==y)[0][0] for y in ref_years])

# ========== 数据读取与预处理 ==========
print("读取 RH 与趋势")
had_times_all, lat, lon, had_rh_all = read_monthly_var(DATA_HADISD_RH, 'rhum')
n_time, n_lat, n_lon = had_rh_all.shape
trend = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
t_idx = np.arange(n_time, dtype=np.float32)
for i in range(n_lat):
    for j in range(n_lon):
        s = had_rh_all[:,i,j]; good = ~np.isnan(s)
        if good.sum()>10:
            try:
                sl,_,_,_,_ = stats.linregress(t_idx[good], s[good])
                trend[i,j] = sl
            except:
                pass
mask = ~np.isnan(trend)

print("ERA5 RH")
ds_era5_rh = nc.Dataset(DATA_ERA5_RH)
era5_raw_time = ds_era5_rh.variables['time'][:]
era5_rh_all = ds_era5_rh.variables['rh_abs'][:].astype(np.float32)
lat_rh = ds_era5_rh.variables['latitude'][:]
lon_rh = ds_era5_rh.variables['longitude'][:]
ds_era5_rh.close()
assert np.allclose(lat, lat_rh) and np.allclose(lon, lon_rh)
era5_rh_times_all = pd.to_datetime(era5_raw_time, unit='s', utc=True).tz_convert(None)

print("裁剪 RH 年份")
had_times, had_rh = subset_year(had_times_all, had_rh_all, START_YEAR, END_YEAR)
era5_rh_times, era5_rh = subset_year(era5_rh_times_all, era5_rh_all, START_YEAR, END_YEAR)

print("读取降水并转换")
gpcp_times_all, plat_g, plon_g, gpcp_mon  = read_and_convert(DATA_GPCP_PR, 'tp', to_mm='gpcp')
mswep_times_all, plat_m, plon_m, mswep_mon = read_and_convert(DATA_MSWEP_PR, 'tp', to_mm=False)
era5p_times_all, plat_e, plon_e, era5p_mon = read_and_convert(DATA_ERA5_PR, 'tp', to_mm='era5')

assert np.allclose(lat, plat_g) and np.allclose(lon, plon_g)
assert np.allclose(lat, plat_m) and np.allclose(lon, plon_m)
assert np.allclose(lat, plat_e) and np.allclose(lon, plon_e)

gpcp_times, gpcp_mon = subset_year(gpcp_times_all, gpcp_mon, START_YEAR, END_YEAR)
mswep_times, mswep_mon = subset_year(mswep_times_all, mswep_mon, START_YEAR, END_YEAR)
era5p_times, era5p_mon = subset_year(era5p_times_all, era5p_mon, START_YEAR, END_YEAR)

print("应用 Mask")
had_rh_masked  = had_rh.copy();  had_rh_masked[:, ~mask]  = np.nan
era5_rh_masked = era5_rh.copy(); era5_rh_masked[:, ~mask] = np.nan
if APPLY_MASK_TO_PRECIP:
    gpcp_mon_masked  = gpcp_mon.copy();  gpcp_mon_masked[:, ~mask]  = np.nan
    mswep_mon_masked = mswep_mon.copy(); mswep_mon_masked[:, ~mask] = np.nan
    era5p_mon_masked = era5p_mon.copy(); era5p_mon_masked[:, ~mask] = np.nan
else:
    gpcp_mon_masked, mswep_mon_masked, era5p_mon_masked = gpcp_mon, mswep_mon, era5p_mon

print("年聚合 (RH 年平均, 降水年总)")
had_years,   had_rh_annual   = monthly_to_annual_rh(had_times,     had_rh_masked,   MIN_MONTHS)
era5r_years, era5_rh_annual  = monthly_to_annual_rh(era5_rh_times, era5_rh_masked,  MIN_MONTHS)
gpcp_years,  gpcp_annual     = monthly_to_annual_precip(gpcp_times,  gpcp_mon_masked,  MIN_MONTHS)
mswep_years, mswep_annual    = monthly_to_annual_precip(mswep_times, mswep_mon_masked, MIN_MONTHS)
era5p_years, era5p_annual    = monthly_to_annual_precip(era5p_times, era5p_mon_masked, MIN_MONTHS)

print("计算 RH 基准期并生成距平")
# 基准期掩码
base_mask_had  = (had_years  >= BASE_START) & (had_years  <= BASE_END)
base_mask_era5 = (era5r_years>= BASE_START) & (era5r_years<= BASE_END)
had_rh_clim  = np.nanmean(had_rh_annual[base_mask_had], axis=0)
era5_rh_clim = np.nanmean(era5_rh_annual[base_mask_era5], axis=0)
had_rh_anom  = had_rh_annual  - had_rh_clim
era5_rh_anom = era5_rh_annual - era5_rh_clim

print("公共年份 & 差值 (距平差)")
common_years = np.intersect1d(
    np.intersect1d(gpcp_years, mswep_years),
    np.intersect1d(era5p_years, np.intersect1d(had_years, era5r_years))
)
if common_years.size == 0:
    raise RuntimeError("没有公共年份。")

i_had   = year_index(had_years,   common_years)
i_era5r = year_index(era5r_years, common_years)
i_gpcp  = year_index(gpcp_years,  common_years)
i_mswep = year_index(mswep_years, common_years)
i_era5p = year_index(era5p_years, common_years)

# RH 距平差
rh_anom_diff     = era5_rh_anom[i_era5r] - had_rh_anom[i_had]
era5_gpcp_diff   = era5p_annual[i_era5p] - gpcp_annual[i_gpcp]
era5_mswep_diff  = era5p_annual[i_era5p] - mswep_annual[i_mswep]

print("面积加权时间序列")
rh_anom_diff_mean    = area_weighted_mean(rh_anom_diff, lat)
era5_gpcp_diff_mean  = area_weighted_mean(era5_gpcp_diff, lat)
era5_mswep_diff_mean = area_weighted_mean(era5_mswep_diff, lat)

print("空间相关 (基于 RH 距平差 与 降水差)")
corr_rh_mswep, p_rh_mswep = spatial_corr(rh_anom_diff, era5_mswep_diff)
corr_rh_gpcp,  p_rh_gpcp  = spatial_corr(rh_anom_diff, era5_gpcp_diff)

# ========== 绘图 (相关系数 -1~1) ==========
import matplotlib.gridspec as gridspec
height_ratio_ts   = 1.0
height_ratio_maps = 1.3
hspace             = 0.02
wspace             = 0.2
cbar_pad_rel    = 0.030
cbar_height_rel = 0.012

fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(
    2, 2,
    height_ratios=[height_ratio_ts, height_ratio_maps],
    width_ratios=[1, 1],
    hspace=hspace,
    wspace=wspace,
    figure=fig
)

# 时间序列
ax_a = fig.add_subplot(gs[0,0])
ax_b = fig.add_subplot(gs[0,1])

ax_a.plot(common_years, rh_anom_diff_mean, '-o', ms=4)
ax_a.axhline(0, color='k', lw=0.8, ls='--')
ax_a.set_xlabel('Year')
ax_a.set_ylabel('RH Anomaly Difference (%)')
ax_a.set_title('(a) (ERA5 - HadISD) RH Anomaly Difference')

ax_b.plot(common_years, era5_gpcp_diff_mean, '-o', ms=4, label='ERA5 - GPCP')
ax_b.plot(common_years, era5_mswep_diff_mean,'-s', ms=4, label='ERA5 - MSWEP')
ax_b.axhline(0, color='k', lw=0.8, ls='--')
ax_b.set_xlabel('Year')
ax_b.set_ylabel('Precip Difference (mm/yr)')
ax_b.set_title('(b) Precipitation Differences')
ax_b.legend(frameon=False)

# 地图
if HAS_CARTOPY:
    ax_c = fig.add_subplot(gs[1,0], projection=ccrs.PlateCarree())
    ax_d = fig.add_subplot(gs[1,1], projection=ccrs.PlateCarree())
else:
    ax_c = fig.add_subplot(gs[1,0])
    ax_d = fig.add_subplot(gs[1,1])

vmin, vmax = -1, 1
if HAS_CARTOPY:
    im_c = ax_c.pcolormesh(lon, lat, corr_rh_mswep, vmin=vmin, vmax=vmax, cmap='RdBu_r')
    ax_c.coastlines()
else:
    im_c = ax_c.imshow(corr_rh_mswep, origin='lower', vmin=vmin, vmax=vmax,
                       cmap='RdBu_r', extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect='auto')
ax_c.set_title('(c) Corr: RH anomaly vs (ERA5 - MSWEP) precipitation')

if HAS_CARTOPY:
    im_d = ax_d.pcolormesh(lon, lat, corr_rh_gpcp, vmin=vmin, vmax=vmax, cmap='RdBu_r')
    ax_d.coastlines()
else:
    im_d = ax_d.imshow(corr_rh_gpcp, origin='lower', vmin=vmin, vmax=vmax,
                       cmap='RdBu_r', extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect='auto')
ax_d.set_title('(d) Corr: RH anomaly vs (ERA5 - GPCP) precipitation')

# Colorbar
fig.canvas.draw()
bb_c = ax_c.get_position(); bb_d = ax_d.get_position()
left_maps  = bb_c.x0
right_maps = bb_d.x1
width_maps = right_maps - left_maps
maps_bottom = min(bb_c.y0, bb_d.y0)
cbar_height = cbar_height_rel
cbar_bottom = maps_bottom - cbar_pad_rel - cbar_height
if cbar_bottom < 0.02:
    cbar_bottom = 0.02

cax = fig.add_axes([left_maps, cbar_bottom, width_maps, cbar_height])
cbar = fig.colorbar(im_d, cax=cax, orientation='horizontal', extend='both')
cbar.set_label('Correlation')
ticks = np.linspace(-1, 1, 9)
cbar.set_ticks(ticks)
cbar.ax.set_xticklabels([f'{t:.2f}' for t in ticks])

out_path = os.path.join(OUT_DIR, 'combined_2x2_RHAnom_Precip_corr.svg')
plt.savefig(out_path, format='svg', dpi=600)
plt.show()
plt.close()
print("Saved:", out_path)
