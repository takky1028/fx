
#-- OANDA API取得

#!pip install git+https://github.com/oanda/oandapy.git
#!pip install git+https://github.com/matplotlib/mpl_finance.git

#-- 必要なオアッケージ一覧
import oandapy
import pandas as pd
import datetime
from datetime import datetime, timedelta
import pytz
import configparser
import os 
import numpy as np
import gzip
import matplotlib
import matplotlib.pyplot as plt
import mpl_finance
from matplotlib import ticker
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go



#-- API接続key
account_id = ""
api_key    = ""

#-- API接続
oanda = oandapy.API(environment="practice", access_token=api_key)

#-- 取得メソッド

def iso_to_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = date.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date

# datetime -> 表示用文字列
def date_to_str(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')
#    return date.strftime('%Y/%m/%d')
#    date = date.strftime('%Y-%m-%d')
#    date = date.strftime('%Y-%m-%d')
#    return pd.to_datetime(date , format = "%Y-%m-%d")

#配列setting
#get_array = ["AUD_JPY","AUD_USD","EUR_AUD","EUR_CHF","EUR_GBP","EUR_JPY","EUR_USD","GBP_CHF","GBP_JPY","GBP_USD","NZD_USD","USD_CAD","USD_CHF","USD_JPY","XAU_USD","XAG_USD"]
#time_array =["S5","S10","S15","S30","M1","M2","M3","M4","M5","M10","M15","M30","H1","H2","H3","H4","H6","H8","H12","D","W","M"]

get_array = ["GBP_JPY"]
time_array =["M5"]
'''
S5	5秒間
S10	10秒間
S15	15秒間
S30	30秒間
M1	1分間
M2	2分間
M3	3分間
M4	4分間
M5	5分間
M10	10分間
M15	15分間
M30	30分間
H1	1時間
H2	2時間
H3	3時間
H4	4時間
H6	6時間
H8	8時間
H12	12時間
D	1日
W	1週間
M	1ヶ月
'''


for currency in get_array :

    for time_ in time_array :
      
        #try :
            #5分足に変更してレートを取得してみる
            res_hist = oanda.get_history(instrument=currency, granularity=time_ , count="5000")

            # データフレーム形式へ変換
            res_hist = pd.DataFrame(res_hist['candles'])

            # 日付をISOから変換
            res_hist['time'] = res_hist['time'].apply(lambda x: iso_to_jp(x))
            res_hist['time'] = res_hist['time'].apply(lambda x: date_to_str(x))

            df = res_hist

            #日ごとの変動幅
            df["diff"] = df["closeAsk"] - df["openAsk"]
            df["pre_diff"] = df["diff"].diff()
            df["pre_change"] = df["diff"].pct_change()

        #MA
            df["close_MA_10"] = df['closeAsk'].rolling(window=10, center=False).mean()
            df["close_MA_20"] = df['closeAsk'].rolling(window=20, center=False).mean()
            df["close_MA_30"] = df['closeAsk'].rolling(window=30, center=False).mean()
            df["close_MA_40"] = df['closeAsk'].rolling(window=40, center=False).mean()
            df["close_MA_50"] = df['closeAsk'].rolling(window=50, center=False).mean()
            df["close_MA_60"] = df['closeAsk'].rolling(window=60, center=False).mean()
            df["close_MA_70"] = df['closeAsk'].rolling(window=70, center=False).mean()
            df["close_MA_80"] = df['closeAsk'].rolling(window=80, center=False).mean()
            df["close_MA_90"] = df['closeAsk'].rolling(window=90, center=False).mean()
            df["close_MA_100"] = df['closeAsk'].rolling(window=100, center=False).mean()
            df["close_MA_150"] = df['closeAsk'].rolling(window=150, center=False).mean()
            df["close_MA_200"] = df['closeAsk'].rolling(window=200, center=False).mean()

        #EMA
            df["close_EMA_10"] = df['closeAsk'].ewm(span=10).mean()
            df["close_EMA_20"] = df['closeAsk'].ewm(span=20).mean()
            df["close_EMA_30"] = df['closeAsk'].ewm(span=30).mean()
            df["close_EMA_40"] = df['closeAsk'].ewm(span=40).mean()
            df["close_EMA_50"] = df['closeAsk'].ewm(span=50).mean()
            df["close_EMA_60"] = df['closeAsk'].ewm(span=60).mean()
            df["close_EMA_70"] = df['closeAsk'].ewm(span=70).mean()
            df["close_EMA_80"] = df['closeAsk'].ewm(span=80).mean()
            df["close_EMA_90"] = df['closeAsk'].ewm(span=90).mean()
            df["close_EMA_100"] = df['closeAsk'].ewm(span=100).mean()
            df["close_EMA_150"] = df['closeAsk'].ewm(span=150).mean()
            df["close_EMA_200"] = df['closeAsk'].ewm(span=200).mean()

        #MAとの乖離
            df["close_MA_10_diff"] = abs(df['closeAsk'] - df["close_MA_10"])
            df["close_MA_20_diff"] = abs(df['closeAsk'] - df["close_MA_20"])
            df["close_MA_30_diff"] = abs(df['closeAsk'] - df["close_MA_30"])
            df["close_MA_40_diff"] = abs(df['closeAsk'] - df["close_MA_40"])
            df["close_MA_50_diff"] = abs(df['closeAsk'] - df["close_MA_50"])
            df["close_MA_60_diff"] = abs(df['closeAsk'] - df["close_MA_60"])
            df["close_MA_70_diff"] = abs(df['closeAsk'] - df["close_MA_70"])
            df["close_MA_80_diff"] = abs(df['closeAsk'] - df["close_MA_80"])
            df["close_MA_90_diff"] = abs(df['closeAsk'] - df["close_MA_90"])
            df["close_MA_100_diff"] = abs(df['closeAsk'] - df["close_MA_100"])
            df["close_MA_150_diff"] = abs(df['closeAsk'] - df["close_MA_150"])
            df["close_MA_200_diff"] = abs(df['closeAsk'] - df["close_MA_200"])

        #EMAとの乖離
            df["close_EMA_10_diff"] = abs(df['closeAsk'] - df["close_EMA_10"])
            df["close_EMA_20_diff"] = abs(df['closeAsk'] - df["close_EMA_20"])
            df["close_EMA_30_diff"] = abs(df['closeAsk'] - df["close_EMA_30"])
            df["close_EMA_40_diff"] = abs(df['closeAsk'] - df["close_EMA_40"])
            df["close_EMA_50_diff"] = abs(df['closeAsk'] - df["close_EMA_50"])
            df["close_EMA_60_diff"] = abs(df['closeAsk'] - df["close_EMA_60"])
            df["close_EMA_70_diff"] = abs(df['closeAsk'] - df["close_EMA_70"])
            df["close_EMA_80_diff"] = abs(df['closeAsk'] - df["close_EMA_80"])
            df["close_EMA_90_diff"] = abs(df['closeAsk'] - df["close_EMA_90"])
            df["close_EMA_100_diff"] = abs(df['closeAsk'] - df["close_EMA_100"])
            df["close_EMA_150_diff"] = abs(df['closeAsk'] - df["close_EMA_150"])
            df["close_EMA_200_diff"] = abs(df['closeAsk'] - df["close_EMA_200"])

        #BB_upper
            df['close_10_BB_Upper'] = df['closeAsk'].ewm(span=10).mean() + (df['closeAsk'].rolling(window=10).std() * 2)
            df['close_20_BB_Upper'] = df['closeAsk'].ewm(span=20).mean() + (df['closeAsk'].rolling(window=20).std() * 2)
            df['close_30_BB_Upper'] = df['closeAsk'].ewm(span=30).mean() + (df['closeAsk'].rolling(window=30).std() * 2)

            df['close_10_BB_Upper_diff'] = df['closeAsk'] - df['close_10_BB_Upper']
            df['close_20_BB_Upper_diff'] = df['closeAsk'] - df['close_20_BB_Upper']
            df['close_30_BB_Upper_diff'] = df['closeAsk'] - df['close_30_BB_Upper']

        #BB_Lower
            df['close_10_BB_Lower'] = df['closeAsk'].ewm(span=10).mean() + (df['closeAsk'].rolling(window=10).std() * 2)
            df['close_20_BB_Lower'] = df['closeAsk'].ewm(span=20).mean() + (df['closeAsk'].rolling(window=20).std() * 2)
            df['close_30_BB_Lower'] = df['closeAsk'].ewm(span=30).mean() + (df['closeAsk'].rolling(window=30).std() * 2)
            df['close_10_BB_Lower_diff'] = df['closeAsk'] - df['close_10_BB_Lower']
            df['close_20_BB_Lower_diff'] = df['closeAsk'] - df['close_20_BB_Lower']
            df['close_30_BB_Lower_diff'] = df['closeAsk'] - df['close_30_BB_Lower']

        #RSI

            #値上がり幅、値下がり幅をシリーズへ切り分け
            up, down = df["pre_diff"].copy(), df["pre_diff"].copy()
            up[up < 0] = 0
            down[down > 0] = 0

            # 値上がり幅/値下がり幅の単純移動平均を処理
            up_sma_7 = up.rolling(window=7, center=False).mean()
            down_sma_7 = down.abs().rolling(window=7, center=False).mean()
            up_sma_10 = up.rolling(window=10, center=False).mean()
            down_sma_10 = down.abs().rolling(window=10, center=False).mean()
            up_sma_14 = up.rolling(window=14, center=False).mean()
            down_sma_14 = down.abs().rolling(window=14, center=False).mean()

            # RSIの計算
            RS_7 = up_sma_7 / down_sma_7
            df["RSI_7"] = 100.0 - (100.0 / (1.0 + RS_7))
            RS_10 = up_sma_10 / down_sma_10
            df["RSI_10"] = 100.0 - (100.0 / (1.0 + RS_10))
            RS_14 = up_sma_14 / down_sma_14
            df["RSI_14"] = 100.0 - (100.0 / (1.0 + RS_14))
        
        #MACD
            df['macd_12_26'] = df['closeAsk'].ewm(span=12).mean() - df['closeAsk'].ewm(span=26).mean()
            df['macd_12_26_signal_9'] = df['macd_12_26'].ewm(span=9).mean()
            df['macd_12_26_signal_9_hist'] = df['macd_12_26'] - df['macd_12_26_signal_9'] 

            #陽線判定
            df["up_flg"] = df['diff'].apply(lambda x : 1 if x >= 0 else 0)

            #欠損存在したら行削除
            df2 = df.dropna()

            #CSV出力
            df2.to_csv("/Users/takky/Documents/my-pj/fx/output/" + str(currency) + "_" + str(time_) + ".csv" )

            print("OK: "+ "/Users/takky/Documents/my-pj/fx/output/" + str(currency) + "_" + str(time_) + ".csv" )

        #except:

            #print("NG: "+ "/Users/takky/Documents/my-pj/fx/output/"  + str(currency) + "_" + str(time_) + ".csv" )

#-----
#乖離量の表
#-----

df_kairi = df2.loc[:,['close_MA_10_diff',
                    'close_MA_20_diff',
                    'close_MA_30_diff',
                    'close_MA_40_diff',
                    'close_MA_50_diff',
                    'close_MA_60_diff',
                    'close_MA_70_diff',
                    'close_MA_80_diff',
                    'close_MA_90_diff',
                    'close_MA_100_diff',
                    'close_MA_150_diff',
                    'close_MA_200_diff',
                    'close_EMA_10_diff',
                    'close_EMA_20_diff',
                    'close_EMA_30_diff',
                    'close_EMA_40_diff',
                    'close_EMA_50_diff',
                    'close_EMA_60_diff',
                    'close_EMA_70_diff',
                    'close_EMA_80_diff',
                    'close_EMA_90_diff',
                    'close_EMA_100_diff',
                    'close_EMA_150_diff',
                    'close_EMA_200_diff']]

df_kairi['close_MA_10_diff_max'] = max(df_kairi['close_MA_10_diff'])
df_kairi['close_MA_20_diff_max'] = max(df_kairi['close_MA_20_diff'])
df_kairi['close_MA_30_diff_max'] = max(df_kairi['close_MA_30_diff'])
df_kairi['close_MA_40_diff_max'] = max(df_kairi['close_MA_40_diff'])
df_kairi['close_MA_50_diff_max'] = max(df_kairi['close_MA_50_diff'])
df_kairi['close_MA_60_diff_max'] = max(df_kairi['close_MA_60_diff'])
df_kairi['close_MA_70_diff_max'] = max(df_kairi['close_MA_70_diff'])
df_kairi['close_MA_80_diff_max'] = max(df_kairi['close_MA_80_diff'])
df_kairi['close_MA_90_diff_max'] = max(df_kairi['close_MA_90_diff'])
df_kairi['close_MA_100_diff_max'] = max(df_kairi['close_MA_100_diff'])
df_kairi['close_MA_150_diff_max'] = max(df_kairi['close_MA_150_diff'])
df_kairi['close_MA_200_diff_max'] = max(df_kairi['close_MA_200_diff'])

df_kairi['close_EMA_10_diff_max'] = max(df_kairi['close_EMA_10_diff'])
df_kairi['close_EMA_20_diff_max'] = max(df_kairi['close_EMA_20_diff'])
df_kairi['close_EMA_30_diff_max'] = max(df_kairi['close_EMA_30_diff'])
df_kairi['close_EMA_40_diff_max'] = max(df_kairi['close_EMA_40_diff'])
df_kairi['close_EMA_50_diff_max'] = max(df_kairi['close_EMA_50_diff'])
df_kairi['close_EMA_60_diff_max'] = max(df_kairi['close_EMA_60_diff'])
df_kairi['close_EMA_70_diff_max'] = max(df_kairi['close_EMA_70_diff'])
df_kairi['close_EMA_80_diff_max'] = max(df_kairi['close_EMA_80_diff'])
df_kairi['close_EMA_90_diff_max'] = max(df_kairi['close_EMA_90_diff'])
df_kairi['close_EMA_100_diff_max'] = max(df_kairi['close_EMA_100_diff'])
df_kairi['close_EMA_150_diff_max'] = max(df_kairi['close_EMA_150_diff'])
df_kairi['close_EMA_200_diff_max'] = max(df_kairi['close_EMA_200_diff'])

df_kairi2 = df_kairi.loc[:,['close_MA_10_diff_max',
                    'close_MA_20_diff_max',
                    'close_MA_30_diff_max',
                    'close_MA_40_diff_max',
                    'close_MA_50_diff_max',
                    'close_MA_60_diff_max',
                    'close_MA_70_diff_max',
                    'close_MA_80_diff_max',
                    'close_MA_90_diff_max',
                    'close_MA_100_diff_max',
                    'close_MA_150_diff_max',
                    'close_MA_200_diff_max',
                    'close_EMA_10_diff_max',
                    'close_EMA_20_diff_max',
                    'close_EMA_30_diff_max',
                    'close_EMA_40_diff_max',
                    'close_EMA_50_diff_max',
                    'close_EMA_60_diff_max',
                    'close_EMA_70_diff_max',
                    'close_EMA_80_diff_max',
                    'close_EMA_90_diff_max',
                    'close_EMA_100_diff_max',
                    'close_EMA_150_diff_max',
                    'close_EMA_200_diff_max']]

         
#-----
#dash
#-----
'''
def generate_table(dataframe, max_row=1):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_row))]
    )

app = dash.Dash()

app.layout = html.Div(children=[
    html.H4(children='US Agriculture Exports (2011)'),
    generate_table(df_kairi2)
])

if __name__ == '__main__':
    app.run_server(debug=True)

import utils
 
trace1 = go.Scatter(
    x = df2["time"],
    y = df2["closeAsk"],
    mode = 'lines',
    name = 'PRICE',
    yaxis="y2"
)
trace2 = go.Scatter(
    x = df2["time"],
    y = df2["close_MA_200"],
    mode = 'lines',
    name = '200EMA'
) 
trace3 = go.Scatter(
    x = df2["time"],
    y = df2["close_MA_100"],
    mode = 'lines',
    name = '100EMA'
) 
trace4 = go.Bar(
    x=df2["time"], 
    y=df2["up_flg"], 
    name="up_flg",
    marker = dict(
        color = utils.prep_color_string('light_green'),
    )
    )

#data = [trace1,trace2,trace3,trace4]  # assign traces to data
data = [trace1,trace4]  # assign traces to data

p_layout = go.Layout(
    title="memfile",
    xaxis=dict(
        title="time"
    ),
    yaxis=dict(
        title="up_flg",
    ),
    yaxis2=dict(
        title="price",
        overlaying="y",
        side="right"
    ),
)

layout = go.Layout(
    title = 'Line chart showing three different modes'
)
fig = go.Figure(data=data,layout=p_layout)
pyo.plot(fig, filename='line0.html')