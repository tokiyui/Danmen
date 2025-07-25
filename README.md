# [気象モデル鉛直ー時間断面図](https://tokiyui.github.io/Danmen/)
![](gsm.png)

* GSM（JMA）、MSM（JMA）、GFS（NCEP）、IFS（ECMWF）の4つの気象モデルによる、東京付近の鉛直ー時間断面図を毎日00Z・12Zの2初期値で自動更新しています。(上はGSMによる図)
* 最も予報時間の短いMSMに合わせて、78時間（3日ちょっと）先まで描画しています。天気は西から東に変わるため、先の時刻が左にくるように描画しています。
* 定型の注意事項ではありますが、これらの図は天気予報そのものではありませんので、発表されている天気予報を説明する材料と考えてください。
* バグの指摘や質問、要望などがありましたらTwitterの @cho_tokisen までお問い合わせください。

* もし専門的な知識のある方がいれば、この図単体で天気予報をするというよりは、上層と下層の対応関係や各要素の時間変化およびモデルごとの差を把握し、「何に着目すべきか」を考えるのが理想的だと思います。
  - 例1)上層トラフと寒気の通過に合わせて、下層の南風が強まり湿域が流入するので、大気の状態が不安定となるだろう→850hPa相当温位や下層収束など、不安定降水にかかわる要素の分布を確認
  - 例2)モデル間で下層の湿りの表現に差があり、下層風ベクトルにも差がある→地上の気圧配置（低気圧の位置や発達の程度）に着目し、じょう乱のメインシナリオに合った気象モデルを採用
  - このあたりの利用方法は作者にとっても試行錯誤中ではあります。

* 表の見方
  - 赤線：等温線（3℃ごと）
  - 矢羽根：各高度における風
  - 緑塗：湿数（気温-露点温度）<3℃の湿域
  - 黄塗：湿数>15℃の乾燥域
  - 青点：LFC（自由対流高度、存在しない場合もある）
  - 赤点：EL（平衡高度、存在しない場合もある）
  - 紫線：湿球温度0℃線（降雪の可能性がある目安）

* モデルの特性
  - GSM：一番基本となる、気象庁の全球モデル。
  - MSM：気象庁による、地形をGSMより細かく反映したモデル。
  - GFS：米国の気象局による気象モデル。湿域の表現が過大といわれることがあるらしい。
  - IFS：ヨーロッパの気象局による気象モデル。一般的に世界で最も精度が良いとされる。
 
* 今後の追加予定
  - MFR（フランス）、CMC（カナダ）は比較的容易に追加できそう。
  - AI気象モデルのAIFS（欧州）、GraphCast（google）の追加も目指す。
  - UKM（イギリス）、DWD（ドイツ）はデータ形式が特殊なため保留中。
  - そもそもどのモデルが精度がよいかを調べる必要がある。
