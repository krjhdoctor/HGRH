import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set default font (to display English correctly)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# year range
start_year = 1973
end_year   = 2024
years      = np.arange(start_year, end_year + 1)

# specify each country’s instrument change start/end years
# end_year = None means only a single line
vertical_lines = {
    'UNITED KINGDOM': (2008, 2009),
    'BELGIUM'       : (2005, 2010),
    'DENMARK'       : (2000, 2002),
    'BRAZIL'        : (2000, 2003),
    'UNITED STATES' : (2002, 2006),
    'JAPAN'         : (1994, 2010),
    'CHINA'         : (2003, 2006),
    'SPAIN'         : (2000, 2010),
    'RUSSIA'        : (2014, None),
    'NETHERLANDS'    : (2000, None),
    'ITALY'         : (2005, 2010),
}

# function to compute global area‐weighted annual series (cos(lat) weights)
def get_global_annual_series(nc_file):
    ds   = nc.Dataset(nc_file)
    rhum = ds.variables['rhum'][:]   # [time, lat, lon]
    lat  = ds.variables['lat'][:]    # [lat]
    ds.close()

    if np.ma.is_masked(rhum):
        rhum = rhum.filled(np.nan)

    n_years = end_year - start_year + 1
    rhum_yearly = np.nanmean(
        rhum.reshape(n_years, 12, *rhum.shape[1:]), axis=1
    )

    weights = np.cos(np.deg2rad(lat))
    w2d     = np.broadcast_to(weights[:, None], rhum_yearly.shape[1:])

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

# root directory containing continent folders
root_rhtest = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/RHtest国家/'

# new output folder
output_folder = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/国家和更换仪器信息图片'
os.makedirs(output_folder, exist_ok=True)

# loop through each specified country (searching all continents)
for country, (v_start, v_end) in vertical_lines.items():
    country_path = None
    for continent in os.listdir(root_rhtest):
        cpath = os.path.join(root_rhtest, continent, country)
        if os.path.isdir(cpath):
            country_path = cpath
            break
    if country_path is None:
        print(f"{country}: folder not found, skipping")
        continue

    # find observed and homogenized .nc files
    obs_file = next((os.path.join(country_path, f)
                     for f in os.listdir(country_path)
                     if f.startswith('观测_') and f.endswith('.nc')), None)
    hom_file = next((os.path.join(country_path, f)
                     for f in os.listdir(country_path)
                     if f.startswith('均一化_') and f.endswith('.nc')), None)
    if obs_file is None or hom_file is None:
        print(f"{country}: missing observed or homogenized file, skipping")
        continue

    # compute series and plot
    series_obs = get_global_annual_series(obs_file)
    series_hom = get_global_annual_series(hom_file)

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6), dpi=300)

    sns.lineplot(x=series_obs.index, y=series_obs.values,
                 marker='o', label='Observed', linewidth=2)
    sns.lineplot(x=series_hom.index, y=series_hom.values,
                 marker='s', label='Homogenized', linewidth=2)

    # instrument change start/end vertical lines
    if v_start is not None:
        plt.axvline(x=v_start, color='red', linestyle='--', linewidth=1.5,
                    label=f'Instrument change start ({v_start})')
    if v_end is not None:
        plt.axvline(x=v_end, color='red', linestyle='--', linewidth=1.5,
                    label=f'Instrument change end ({v_end})')

    # lighter vertical grid lines every 5 years
    for yr in range(start_year, end_year + 1, 5):
        plt.axvline(x=yr, color='lightgray', linestyle=':', linewidth=0.5, alpha=0.7)

    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Global Annual Mean Relative Humidity (%)', fontsize=14)
    plt.title(f"{start_year}–{end_year} {country} Global Annual Mean RH (area‐weighted)", fontsize=16)
    plt.xticks(np.arange(start_year, end_year + 1, 5), rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()

    out_fig = os.path.join(output_folder, f"Global_RH_{country}.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"{country} plot saved to: {out_fig}")

print('All country plots generation complete.')
print(f'Plots are saved in: {output_folder}')