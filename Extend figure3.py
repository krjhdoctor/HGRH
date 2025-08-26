


import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

# —— 全局字体设置（放在 import 后、绘图前）——
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 18

plt.rcParams.update({
    "font.size": FONT_SIZE,          # 基础字号
    "axes.unicode_minus": False,
    "font.family": "serif",
    "font.serif": [FONT_FAMILY],     # 指定 Times New Roman
    "mathtext.fontset": "stix",      # 数学文本更接近 Times
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})

# ——— Parameters ———
MIN_COUNT = 0  # minimum months required (>=360)
order = ['Reference', 'Homogenization']

# set time periods for each dataset - both 1973-2024
start_years = {'Reference': 1973, 'Homogenization': 1973}
end_years = {'Reference': 2024, 'Homogenization': 2024}

paths = {
    'Reference': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/参考_距平_Global.nc',
    'Homogenization': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/均一化_Global.nc'
}

# ——— Display names mapping ———
display_names = {
    'Reference': 'Reference (1973-2024)',
    'Homogenization': 'Homogenization (1973-2024)'
}

# ——— Colors for the two datasets ———
colors = {'Reference': '#e73618', 'Homogenization': '#79add6'}

def reorder_data_with_lon(lon, data):
    lon2 = np.where(lon > 180, lon - 360, lon)
    idx = np.argsort(lon2)
    return lon2[idx], data[..., idx]

def calculate_slope_p(data, min_count=360):
    nt, nlat, nlon = data.shape
    t = np.arange(nt)
    valid_count = np.sum(~np.isnan(data), axis=0)
    slope = np.full((nlat, nlon), np.nan)
    pval = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            if valid_count[i, j] >= min_count:
                y = data[:, i, j]
                ok = ~np.isnan(y)
                if np.sum(ok) >= min_count:
                    res = linregress(t[ok], y[ok])
                    slope[i, j] = res.slope * 120.0  # trend per decade
                    pval[i, j] = res.pvalue
    return slope, pval, valid_count

def calculate_annual_mean(data):
    nyears = data.shape[0] // 12
    return np.nanmean(
        data[:nyears * 12].reshape(nyears, 12, *data.shape[1:]),
        axis=1
    )

# ——— Load and process each dataset ———
datasets = {}
for name in order:
    ds = netCDF4.Dataset(paths[name])

    # Assume the variable name is 'rhum' for anomaly data
    varname = 'rhum'

    rh = ds.variables[varname][:].astype(float)
    fill = getattr(ds.variables[varname], '_FillValue', None)
    if fill is not None:
        rh[rh == fill] = np.nan

    # Get coordinates
    lat = ds.variables.get('latitude', ds.variables.get('lat'))[:]
    lon = ds.variables.get('longitude', ds.variables.get('lon'))[:]
    lon2, rh = reorder_data_with_lon(lon, rh)
    if lat[0] > lat[-1]:
        lat, rh = lat[::-1], rh[:, ::-1, :]

    # Store coords once
    if 'lats' not in globals():
        lats, lons = lat, lon2

    # Calculate expected data length based on time period
    expected_months = (end_years[name] - start_years[name] + 1) * 12
    actual_months = rh.shape[0]

    # Truncate data if necessary to match the expected time period
    if actual_months > expected_months:
        rh = rh[:expected_months, :, :]

    # Since data is already anomaly, use it directly
    anom = rh

    # Trend computation
    slope, pval, valid_count = calculate_slope_p(anom, min_count=MIN_COUNT)

    # Annual mean anomaly
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
    arr = np.where(mask_short[None, ...], np.nan, arr)

    ts = []
    for year_slice in arr:
        w2d = np.broadcast_to(weights[:, None], year_slice.shape)
        num = np.nansum(year_slice * w2d)
        den = np.nansum(w2d[~np.isnan(year_slice)])
        if den > 0:
            ts.append(num / den)
        else:
            ts.append(np.nan)

    years = np.arange(start_years[name], start_years[name] + len(ts))
    time_series[name] = (years, np.array(ts))

# ——— Plot in a 2×1 grid for maps + 1 time series ———
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.6], hspace=-0.05, wspace=0.16)

labels = ['(a)', '(b)']
pcm_list = []

# Plot spatial distribution maps
for idx, name in enumerate(order):
    ax = fig.add_subplot(gs[0, idx], projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90])

    # Grid labels without grid lines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0,  # no grid lines
        crs=ccrs.PlateCarree()
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    gl.xlocator = plt.MultipleLocator(60)
    gl.ylocator = plt.MultipleLocator(30)

    ax.coastlines()

    lon2d, lat2d = np.meshgrid(lons, lats)
    slope_map = np.where(datasets[name]['valid_count'] < MIN_COUNT,
                         np.nan, datasets[name]['slope'])
    pval_map = np.where(datasets[name]['valid_count'] < MIN_COUNT,
                        np.nan, datasets[name]['pval'])

    pcm = ax.pcolormesh(
        lon2d, lat2d, slope_map,
        transform=ccrs.PlateCarree(),
        cmap='RdBu', vmin=-1.5, vmax=1.5
    )

    # Significance markers
    sig = (pval_map < 0.05)
    ax.scatter(lon2d[sig], lat2d[sig], s=0.5, color='black', alpha=0.5,
               transform=ccrs.PlateCarree())

    # Area-weighted global trend
    w2d = np.broadcast_to(weights[:, None], slope_map.shape)
    global_trend = np.nansum(slope_map * w2d) / np.nansum(w2d[~np.isnan(slope_map)])

    ax.set_title(f"{labels[idx]} {display_names[name]}", loc='left')
    ax.text(-170, -80,
            f"Global mean: {global_trend:.2f} %/decade",
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=16)
    pcm_list.append(pcm)

# Shared colorbar for the maps
cax = fig.add_axes([0.13, 0.48, 0.76, 0.02])
cbar = fig.colorbar(pcm_list[0], cax=cax, orientation='horizontal', extend='both')
cbar.set_label('Trend (% / decade)')
cbar.set_ticks(np.linspace(-1.5, 1.5, 7))

# Time series panel
axf = fig.add_subplot(gs[1, :])
for name in order:
    yrs, ts = time_series[name]
    axf.plot(yrs, ts, label=display_names[name], color=colors[name], linewidth=2)

axf.set_xticks(np.arange(1973, 2028, 5))
axf.set_xlabel('Year')
axf.set_ylabel('RH anomaly (%)')
axf.set_title('(c) Global RH anomaly', loc='left')

# Legend
axf.legend(frameon=False, loc='lower left')

plt.savefig(
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/Reference_Homogenization_空间分布图和时间分布图.svg',
    format='svg',
    bbox_inches='tight'
)
plt.show()

# ——— Print time-series trends ———
print("Linear trend of global area-weighted series (% per decade):")
for name in order:
    yrs, ts = time_series[name]
    ok = ~np.isnan(ts)
    if np.sum(ok) > 1:
        res = linregress(yrs[ok], ts[ok])
        print(f"  {name}: slope = {res.slope:.3f} %/decade, p-value = {res.pvalue:.4f}")
    else:
        print(f"  {name}: insufficient data for trend calculation")