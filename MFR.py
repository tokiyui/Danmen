import datetime
import numpy as np
import pygrib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
import matplotlib.dates as mdates
import pandas as pd
import requests
import os
import datetime

# 現在のUTC時刻を取得
now_utc = datetime.datetime.utcnow()

# 8時間引く
adjusted_time = now_utc - datetime.timedelta(hours=4)

# 12時間単位で切り捨て
truncated_hour = (adjusted_time.hour // 12) * 12
dt = adjusted_time.replace(hour=truncated_hour, minute=0, second=0, microsecond=0)

# 個別に取り出したい場合
i_year = dt.year
i_month = dt.month
i_day = dt.day
i_hourZ = dt.hour

print("IT(UTC):", dt)

## 時間範囲：作成するデータセットのftの期間を指定(ft_s:期間始まりのFT ft_e:期間終端のFT ft_step:読み込む時間間隔)
ft_s = 0
ft_e = 78
ft_step = 3
fts = list(range(ft_s, ft_e+1, ft_step))

## 空間範囲：GPVの切り出し領域の指定：(lonW,latS)-(lonE,latN)の矩形
(latS, latN, lonW, lonE) = (20, 50, 120, 150)

## 気圧面範囲：上端の気圧面を指定
tagHp=100

# データの格納先フォルダー名
##!!! GRIB2データの保存先をFolderを指定すること !!!
data_fld="https://object.data.gouv.fr/meteofrance-pnt/pnt/{0:4d}-{1:02d}-{2:02d}T{3:02d}:00:00Z/arpege/025/"

# 読み込むGRIB2形式GSMのファイル名
gsm_fn_t1="arpege__025__IP1__{0:s}__{1:4d}-{2:02d}-{3:02d}T{4:02d}:00:00Z.grib2"
gsm_fn_t2="arpege__025__IP2__{0:s}__{1:4d}-{2:02d}-{3:02d}T{4:02d}:00:00Z.grib2"

# ### データフレームを作成するための４次元配列を確保する
# - 試しにGPVを読み込み、必要な配列を確保するlevelやlat,lonのサイズを求める
# - list形式: fts, levels,lats,lons
# - sizeの値: f_size, l_size, lat_size, lon_size
# - 要素別の4次元配列を確保（値は0に初期化）: aryHt, aryTm など

### 事前に一度データを読み込み  配列のサイズを決めるため
ft = fts[0]
dat_fld = data_fld.format(i_year,i_month,i_day,i_hourZ)
gr_fn = gsm_fn_t1.format('000H024H',i_year,i_month,i_day,i_hourZ)

# HTTPでファイルダウンロード
response = requests.get(dat_fld + "IP1/" + gr_fn)
response.raise_for_status()  # ダウンロードに失敗した場合、エラーを発生させる

# ダウンロードしたコンテンツをローカルに保存
with open(gr_fn, 'wb') as f:
    f.write(response.content)

# データOpen
grbs = pygrib.open(gr_fn)
print(gr_fn)

# 要素別に読み込み（tagHpの等圧面から下部のデータを全て）
grbTm = grbs(shortName="t",forecastTime=0,typeOfLevel='isobaricInhPa',level=lambda l:l >= 300)
valHt, latHt, lonHt = grbTm[0].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

## データの大きさを調べる
# レベルの数
levels = np.array([g['level'] for g in grbTm])
l_size = levels.size
# 平面のsize
(lat_size, lon_size) = valHt.shape
lats = latHt[:,0]
lons = lonHt[0,:]
# 時間の数
dts = []
t_size = len(fts)
print("t_size:{:3d} {}".format(t_size, fts))

## 配列確保(0で初期化)
aryHt = np.zeros([t_size, l_size, lat_size, lon_size])
aryTm = np.zeros([t_size, l_size, lat_size, lon_size])
aryTd = np.zeros([t_size, l_size, lat_size, lon_size])
aryWu = np.zeros([t_size, l_size, lat_size, lon_size])
aryWv = np.zeros([t_size, l_size, lat_size, lon_size])
aryOmg = np.zeros([t_size, l_size, lat_size, lon_size])

# ### 4次元配列にデータを格納
dts=[]
for i in range(t_size): # ft の時間ループ
    ft = fts[i]

    if ft <= 24:
        date = "000H024H"
    elif ft <= 48:
        date = "025H048H"
    elif ft <= 72:
        date = "049H072H"
    else:
        date = "073H102H"

    # ファイル名作成
    gr_fn1 = gsm_fn_t1.format(date,i_year,i_month,i_day,i_hourZ)
    gr_fn2 = gsm_fn_t2.format(date,i_year,i_month,i_day,i_hourZ)
    print("FT:{:02d}:{:04d} {}, {}".format(ft, ft,gr_fn1,gr_fn2))

    # HTTPでファイルダウンロード
    file_path = os.path.join(dat_fld, gr_fn1)
    if not os.path.exists(gr_fn1):
        print(f"ダウンロード開始: {gr_fn1}")
        response = requests.get(dat_fld + "IP1/" + gr_fn1)
        response.raise_for_status()  # エラーがあれば例外を発生

    with open(gr_fn1, 'wb') as f:
        f.write(response.content)
    grbs1 = pygrib.open(gr_fn1)
    all_grbs = list(grbs1)
    
    # 要素別に読み込み（tagHpの等圧面から下部のデータを全て）    
    grbTm = sorted([g for g in all_grbs if g.shortName == "t" and g.forecastTime == ft and g.typeOfLevel == "isobaricInhPa" and g.level >= tagHp], key=lambda g: g.level, reverse=True)
    grbWu = sorted([g for g in all_grbs if g.shortName == "u" and g.forecastTime == ft and g.typeOfLevel == "isobaricInhPa" and g.level >= tagHp], key=lambda g: g.level, reverse=True)
    grbWv = sorted([g for g in all_grbs if g.shortName == "v" and g.forecastTime == ft and g.typeOfLevel == "isobaricInhPa" and g.level >= tagHp], key=lambda g: g.level, reverse=True)

    # HTTPでファイルダウンロード
    file_path = os.path.join(dat_fld, gr_fn2)
    if not os.path.exists(gr_fn2):
        print(f"ダウンロード開始: {gr_fn2}")
        response = requests.get(dat_fld + "IP2/" + gr_fn2)
        response.raise_for_status()  # エラーがあれば例外を発生
        
    with open(gr_fn2, 'wb') as f:
        f.write(response.content)
    grbs2 = pygrib.open(gr_fn2)
    all_grbs = list(grbs2)
    grbTd = sorted([g for g in all_grbs if g.shortName == "dpt" and g.forecastTime == ft and g.typeOfLevel == "isobaricInhPa" and g.level >= tagHp], key=lambda g: g.level, reverse=True)

    # 読み込んだデータの時刻取得
    dts.append(grbTm[0].validDate)

    # 要素毎に3次元配列作成
    for l in range(l_size):
        valTm, _, _ = grbTm[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
        valWu, _, _ = grbWu[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
        valWv, _, _ = grbWv[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
        valTd, _, _ = grbTd[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

        ## 4次元配列に代入
        aryTm[i][l] = valTm
        aryWu[i][l] = valWu
        aryWv[i][l] = valWv
        aryTd[i][l] = valTd

    os.remove(gr_fn1)
    os.remove(gr_fn2)

# ### Xarry データセットに変換
# GPVをmetpyを利用して物理量を計算するには、numpyの配列データからXarrayのデータセットを作成する
# その後で、必要な物理量を計算する
# Xarray data set作成
ds = xr.Dataset(
    {
        "temperature": (["time","level","lat", "lon"], aryTm * units('K')),
        "dewpoint_temperature": (["time","level","lat", "lon"], aryTd * units('K')),
        "u_wind": (["time","level","lat", "lon"], aryWu * units('m/s')),
        "v_wind": (["time","level","lat", "lon"], aryWv * units('m/s')),
    },
    coords={
        "time": dts,
        "level": levels,
        "lat": lats,
        "lon": lons,
    },
)

ds['temperature'].attrs['units']='K'
ds['dewpoint_temperature'].attrs['units']='K'
ds['u_wind'].attrs['units']='m/s'
ds['v_wind'].attrs['units']='m/s'
ds['level'].attrs['units'] = 'hPa'
ds['lat'].attrs['units'] = 'degrees_north'
ds['lon'].attrs['units'] = 'degrees_east'
ds['level'] = np.sort(ds['level'])[::-1]

## 必要な物理量を計算する
#ds['dewpoint_temperature'] = mpcalc.dewpoint_from_specific_humidity(ds['level'] * units.hPa, ds['specific_humidity'] * 1000 * units('g/kg'))
ds['ttd'] = ds['temperature'] - ds['dewpoint_temperature']

# 表示のために単位を変換する
ds['u_wind'] = (ds['u_wind']).metpy.convert_units('knots')
ds['v_wind'] = (ds['v_wind']).metpy.convert_units('knots')
ds['temperature'] = (ds['temperature']).metpy.convert_units('degC')

## Parse full dataset
dsp= ds.metpy.parse_cf()

dsp['time'] = dsp['time'] + pd.Timedelta(hours=9)

## 時系列図作成
# ポイント指定
i_lat, i_lon= 57, 79  # 羽田 35.5554N　139.754E
print("Point {}N,{}E".format(lats[i_lat],lons[i_lon]))

# 図の大きさを指定
fig = plt.figure(1, figsize=(16, 9))
ax = plt.axes()

# X軸の指定
ax.set_xlim([dts[0] + datetime.timedelta(hours=9) - datetime.timedelta(hours=1), dts[-1] + datetime.timedelta(hours=9) + datetime.timedelta(hours=1)])

# Y軸をlog指定、ラベル指定、目盛り指定
###ax.set_yscale('symlog')
ax.set_ylim(1050.0, 280.0)
ax.set_yticks(np.arange(1000, 290, -100))
ax.set_yticklabels(np.arange(1000, 200, -100))

# CAPE（自由対流高度）およびLCL（持ち上げ凝結高度）の計算
lcl_heights_list = []
lcl_heights = []
el_heights_list = []
el_heights = []
wb_heights_list = []
wb_heights = []

for i in range(t_size):
    # 時間に対応するデータを取得
    temperature = dsp['temperature'][:, :, i_lat, i_lon]
    dewpoint = dsp['dewpoint_temperature'][:, :, i_lat, i_lon]
    pressure = dsp['level'].values

    # 湿球温度
    wb = mpcalc.wet_bulb_temperature(dsp['level'] * units.hPa, temperature[i, :], dewpoint[i, :])
    wb_heights_list.append(wb)

    # LCL (持ち上げ凝結高度)を計算
    lcl = mpcalc.lcl(pressure[0] * units.hPa, temperature[i, 0], dewpoint[i, 0])
    lcl_heights_list.append(lcl[0].magnitude)

    # CAPE (自由対流高度)を計算
    el = mpcalc.el(pressure * units.hPa, temperature[i, :], dewpoint[i, :])
    el_heights_list.append(el[0].magnitude)

# 計算結果をリストに追加
lcl_heights = np.array(lcl_heights_list) * units.hPa
el_heights = np.array(el_heights_list) * units.hPa
wb_heights = np.array(wb_heights_list) * units.hPa

# 時系列図にLCLとCAPEをプロット
ax.plot(dts, lcl_heights, color='blue', label='LCL (hPa)', linestyle='', marker='o', markersize=12)
ax.plot(dts, el_heights, color='red', label='EL (hPa)', linestyle='', marker='o', markersize=12)

## T-TDの時系列鉛直分布表示
ax.contourf(dsp['time'],  dsp['level'].values, dsp['ttd'][:,:,i_lat,i_lon].values.T, [3,15], colors=["lime","white","yellow"], extend='both', alpha = 0.4)

# 気温(橙実線)とラベル
tmp_contour = ax.contour(dsp['time'],  dsp['level'].values, dsp['temperature'][:,:,i_lat,i_lon].values.T, levels=np.arange(-60, 45, 3), colors='red', linewidths=1, linestyles='solid')
tmp_contour.clabel(tmp_contour.levels[1::2], fontsize=16, colors='red', inline=1, inline_spacing=8, fmt='%i', rightside_up=False, use_clabeltext=True)

# 湿球温度
ax.contour(dsp['time'],  dsp['level'].values, wb_heights.T, levels=np.arange(1.5, 100, 100), colors='purple', linewidths=2)

# 風の矢羽
ax.barbs(dsp['time'], dsp['level'].values, dsp['u_wind'][:,:,i_lat,i_lon].values.T, dsp['v_wind'][:,:,i_lat,i_lon].values.T)

plt.gca().invert_xaxis()

# X軸をJSTの形式に変更（日時のフォーマット）
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %HJST'))

# X軸の目盛り間隔を6時間に設定
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6,18]))

# ユニークな日付を取得（時刻は0:00にする）
dates = pd.to_datetime(dsp['time']).normalize().unique()

# 各日付の0:00 JSTに縦線を引く
for date in dates:
    ax.axvline(date, linestyle='--', linewidth=1)

fig.text(0.5,0.01,"{:.2f}N, {:.2f}E MFR FT{:03d}-{:03d} {}Z Initial".format(lats[i_lat],lons[i_lon],fts[0],fts[-1],dt), ha='center',va='bottom', size=15)
plt.savefig('mfr.png')
