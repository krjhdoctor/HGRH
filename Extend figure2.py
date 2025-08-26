import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


# === 全局字体设置 ===
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.size': 18,        # 全部字体大小
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})

# 1) Read station metadata
stations_csv = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/merged_station_info.csv'
df_st = pd.read_csv(stations_csv, dtype={'id': str}).set_index('id')

# 2) Wind speed data paths
wind_paths = {
    '10m': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/10m风速/',
    '1.5m': '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/重新计算相对湿度_根据干湿球系数/1.5m风速/'
}


def process_wind_data(wind_path, height_label):
    """Process wind speed data for a given height"""
    station_wind_speeds = {}

    if not os.path.exists(wind_path):
        print(f"Warning: Path {wind_path} does not exist!")
        return station_wind_speeds

    for file in os.listdir(wind_path):
        if file.startswith('.'):
            continue

        station_id = os.path.splitext(file)[0]
        file_path = os.path.join(wind_path, file)

        try:
            # Read wind speed data (注意：虽然列名是'rh'，但实际是风速数据)
            data = pd.read_csv(file_path, header=None, names=['year', 'month', 'day', 'rh'], sep=' ')

            # Filter 1990-1999 data
            filtered_data = data[(data['year'] >= 1990) & (data['year'] <= 1999)]

            if not filtered_data.empty:
                # Calculate mean wind speed (excluding NaN)
                mean_wind_speed = filtered_data['rh'].mean()
                if not np.isnan(mean_wind_speed):
                    station_wind_speeds[station_id] = mean_wind_speed

        except Exception as e:
            print(f"Error processing {file} for {height_label}: {e}")
            continue

    print(f"Processed {len(station_wind_speeds)} stations with valid {height_label} wind speed data")
    return station_wind_speeds


def prepare_plot_data(station_wind_speeds, df_st):
    """Prepare data for plotting"""
    plot_data = {'lons': [], 'lats': [], 'wind_speeds': [], 'station_ids': []}

    for station_id, wind_speed in station_wind_speeds.items():
        if station_id in df_st.index:
            rec = df_st.loc[station_id]
            lon, lat = rec['lon'], rec['lat']
            plot_data['lons'].append(lon)
            plot_data['lats'].append(lat)
            plot_data['wind_speeds'].append(wind_speed)
            plot_data['station_ids'].append(station_id)

    return plot_data
def create_wind_speed_subplot(ax, plot_data, height_label, subplot_label=None):
    """Create wind speed subplot"""
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.set_global()

    gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.MultipleLocator(60)
    gl.ylocator = plt.MultipleLocator(30)
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    if len(plot_data['wind_speeds']) == 0:
        ax.set_title(f'{height_label} Wind Speed - No Data Available',
                     fontsize=18, pad=10)
        return None, None

    scatter = ax.scatter(
        plot_data['lons'], plot_data['lats'],
        c=plot_data['wind_speeds'],
        s=8, alpha=0.8,
        cmap='viridis',
        transform=ccrs.PlateCarree(),
        zorder=3,
        edgecolors='black',
        linewidths=0.1,
        vmin=0, vmax=5
    )

    # ✅ 去掉括号 + 去掉加粗
    if subplot_label:
        ax.set_title(f'{subplot_label} {height_label} Wind Speed Distribution (1990–1999)',
                     fontsize=18, pad=10)
    else:
        ax.set_title(f'{height_label} Wind Speed Distribution (1990–1999)',
                     fontsize=18, pad=10)

    return scatter, np.array(plot_data['wind_speeds'])


# 3) Process wind data for both heights
wind_data = {}
for height, path in wind_paths.items():
    wind_data[height] = process_wind_data(path, height)

# 4) Prepare plot data for both heights
plot_data_10m = prepare_plot_data(wind_data['10m'], df_st)
plot_data_1_5m = prepare_plot_data(wind_data['1.5m'], df_st)

print(f"Found coordinates for {len(plot_data_10m['lons'])} stations (10m)")
print(f"Found coordinates for {len(plot_data_1_5m['lons'])} stations (1.5m)")

# Print statistics for both heights
for height, plot_data in [('10m', plot_data_10m), ('1.5m', plot_data_1_5m)]:
    if len(plot_data['wind_speeds']) > 0:
        wind_speeds_array = np.array(plot_data['wind_speeds'])
        print(f"\n{height} Wind speed statistics:")
        print(f"  Min: {wind_speeds_array.min():.2f} m/s")
        print(f"  Max: {wind_speeds_array.max():.2f} m/s")
        print(f"  Mean: {wind_speeds_array.mean():.2f} m/s")
        print(f"  Std: {wind_speeds_array.std():.2f} m/s")

# 只处理 1.5m 风速
plot_data_1_5m = prepare_plot_data(wind_data['1.5m'], df_st)

print(f"Found coordinates for {len(plot_data_1_5m['lons'])} stations (1.5m)")

if len(plot_data_1_5m['wind_speeds']) > 0:
    wind_speeds_array = np.array(plot_data_1_5m['wind_speeds'])
    print(f"\n1.5m Wind speed statistics:")
    print(f"  Min: {wind_speeds_array.min():.2f} m/s")
    print(f"  Max: {wind_speeds_array.max():.2f} m/s")
    print(f"  Mean: {wind_speeds_array.mean():.2f} m/s")
    print(f"  Std: {wind_speeds_array.std():.2f} m/s")


fig = plt.figure(figsize=(12, 8))

ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
scatter, wind_speeds_1_5m = create_wind_speed_subplot(ax, plot_data_1_5m, '1.5m', )

if wind_speeds_1_5m is not None:
    # 固定颜色范围 0–3
    scatter.set_clim(0, 3)

    # 单独 colorbar，指定刻度 0,1,2,3
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Average Wind Speed at 1.5m (m/s)', fontsize=18, )
    cbar.set_ticks([0, 1, 2, 3])   # ✅ 设置刻度
    cbar.ax.tick_params(labelsize=14)



plt.tight_layout()
output_path = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/Wind_Speed_1_5m_1990_1999.png'
plt.savefig(output_path, format='png', dpi=800, bbox_inches='tight')



plt.show()