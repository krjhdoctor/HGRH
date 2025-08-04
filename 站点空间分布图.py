# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import xarray as xr
# import cmocean
# import numpy as np
# # ——— 基本设置 ———
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 1) ETOPO2 高程数据路径
# etopo_path = '/Users/yincj/PycharmProjects/pythonProject/example/ETOPO2v2g_f4_netCDF/ETOPO2v2g_f4.nc'
#
# # 2) 读入高程
# ds_elev = xr.open_dataset(etopo_path)
# # 通常变量名是 'z' 或者 'ETOPO2v2g_f4'; 这里尝试取第一个非坐标变量
# varname = [v for v in ds_elev.data_vars][0]
# elev = ds_elev[varname]
# # 经度/纬度坐标名可能是 'lon','lat' 或 'x','y'
# if 'lon' in elev.coords:
#     lon = elev['lon']
#     lat = elev['lat']
# else:
#     lon = elev['x']
#     lat = elev['y']
#
# # 如果经度是 0-360，需要转换到 -180–180
# if lon.max() > 180:
#     lon2 = (lon + 180) % 360 - 180
#     # 重排
#     idx = np.argsort(lon2)
#     lon = lon2[idx]
#     elev = elev.isel({elev.dims[-1]: idx})
#
# # 3) 读入站点元信息
# stations_csv = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/merged_station_info.csv'
# df_st = pd.read_csv(stations_csv, dtype={'id': str}).set_index('id')
#
# # 4) “六大洲”根目录 和 配色
# root = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/六大洲'
# continent_mapping = {
#     'Africa': '非洲',
#     'Asia': '亚洲',
#     'Europe': '欧洲',
#     'North_America': '北美洲',
#     'South_America': '南美洲',
#     'Oceania': '大洋洲'
# }
# continent_colors = {
#     'Asia': '#B4555A',
#     'Europe': '#B9944F',
#     'Africa': '#9D8EA5',
#     'North_America': '#A6BD66',
#     'South_America': '#D1863D',
#     'Oceania': '#E9A7C0'
# }
#
# # ——— 开始绘图 ———
# fig = plt.figure(figsize=(15,10))
# ax = plt.axes(projection=ccrs.Robinson())
#
# # 先画高程
# # pcolormesh 需要 2D 网格
# lon2d, lat2d = np.meshgrid(lon, lat)
# elev2d = elev.values
# pcm = ax.pcolormesh(
#     lon2d, lat2d, elev2d,
#     transform=ccrs.PlateCarree(),
#     cmap=cmocean.cm.topo,
#     shading='auto',
#     zorder=0
# )
# # 添加 colorbar
# cb = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.02, fraction=0.04)
# cb.set_label('Elevation (m)')
#
# # 再加海陆边界
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=1)
# ax.set_global()
#
# # 画经纬网格
# gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, zorder=2)
# gl.top_labels = False
# gl.right_labels = False
# gl.xlocator = plt.MultipleLocator(30)
# gl.ylocator = plt.MultipleLocator(30)
# # …（前面部分保持不变）…
#
# # 叠加各大洲的观测站点
# handles, labels = [], []
# for cont_en, cont_cn in continent_mapping.items():
#     obs_folder = os.path.join(root, cont_en, '观测数据')
#     if not os.path.isdir(obs_folder):
#         continue
#
#     lons, lats = [], []
#     for fname in os.listdir(obs_folder):
#         if fname.startswith('.'):
#             continue
#         sid, _ = os.path.splitext(fname)
#         if sid in df_st.index:
#             rec = df_st.loc[sid]
#             lons.append(rec['lon'])
#             lats.append(rec['lat'])
#
#     if lons:
#         sc = ax.scatter(
#             lons, lats,
#             s=20, alpha=0.8,
#             color=continent_colors[cont_en],
#             label=cont_cn,
#             transform=ccrs.PlateCarree(),
#             zorder=3
#         )
#         handles.append(sc)
#         labels.append(cont_cn)
#
# # 图例 & 标题（去掉 zorder 参数）
# ax.legend(handles, labels,
#           loc='lower left', ncol=2,
#           frameon=True, title='大洲')
#
# plt.title('观测站点空间分布（叠加高程）', fontsize=18, pad=12)
# plt.tight_layout()
#
# # 保存
# plt.savefig('/Users/yincj/Desktop/站点分布图_带高程.svg',
#             dpi=800, bbox_inches='tight')
# plt.show()




import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ——— Basic settings ———
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1) Read station metadata
stations_csv = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/merged_station_info.csv'
df_st = pd.read_csv(stations_csv, dtype={'id': str}).set_index('id')

# 2) “六大洲” root folder
root = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/六大洲'

# 3) Mapping and colors
continent_mapping = {
    'Africa':       'Africa',
    'Asia':         'Asia',
    'Europe':       'Europe',
    'North_America':'North America',
    'South_America':'South America',
    'Oceania':      'Oceania'
}
continent_colors = {
    'Africa':       '#9D8EA5',
    'Asia':         '#B4555A',
    'Europe':       '#B9944F',
    'North_America':'#A6BD66',
    'South_America':'#D1863D',
    'Oceania':      '#E9A7C0'
}

# ——— Collect stations by continent and record matches ———
stations_by_cont = {c: {'lons': [], 'lats': [], 'ids': []}
                    for c in continent_mapping}

matched_records = []
for cont_en in continent_mapping:
    obs_folder = os.path.join(root, cont_en, '观测数据')
    if not os.path.isdir(obs_folder):
        continue
    for fname in os.listdir(obs_folder):
        if fname.startswith('.'):
            continue
        sid, ext = os.path.splitext(fname)
        full = os.path.join(obs_folder, fname)
        # only consider data files
        if ext.lower() not in ('.dat', '.nc', '.csv', '.txt'):
            continue
        if sid in df_st.index:
            rec = df_st.loc[sid]
            lon, lat = rec['lon'], rec['lat']
            stations_by_cont[cont_en]['lons'].append(lon)
            stations_by_cont[cont_en]['lats'].append(lat)
            stations_by_cont[cont_en]['ids'].append(sid)
            matched_records.append({
                'id': sid,
                'continent': continent_mapping[cont_en],
                'lon': lon,
                'lat': lat
            })

# ——— Save matched info to Excel ———
df_matched = pd.DataFrame(matched_records)
output_excel = '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/六大洲站点分布.xlsx'
df_matched.to_excel(output_excel, index=False)
print(f"Matched station info saved to {output_excel}")

# ——— Start plotting ———
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection=ccrs.Robinson())
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.set_global()

# gridlines
gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = plt.MultipleLocator(30)
gl.ylocator = plt.MultipleLocator(30)

# scatter per continent
handles, labels = [], []
for cont_en, cont_cn in continent_mapping.items():
    data = stations_by_cont[cont_en]
    if not data['lons']:
        continue
    sc = ax.scatter(
        data['lons'], data['lats'],
        s=5, alpha=0.8,
        color=continent_colors[cont_en],
        label=cont_cn,
        transform=ccrs.PlateCarree(),
        zorder=3
    )
    handles.append(sc)
    labels.append(cont_cn)

# legend under map
ax.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=6,
    frameon=True
)

# title
plt.title('Global Station Distribution', fontsize=18, pad=12)
plt.tight_layout()

# save as SVG
plt.savefig(
    '/Users/yincj/Desktop/GSOD均一化/HadISD均一化2/Global_Station_Distribution.png',
    format='png',
    dpi=100,
    bbox_inches='tight'
)
plt.show()