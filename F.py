import sys
import subprocess
# output = subprocess.check_output(['hostname', '-I'])
# sys.path.append('/home/fin/phu_tv/nas/SSI/ssi_fc_data')
sys.path.append('/home/fin/temp/nas/Data/SSI/ssi_fc_data')
# from getdata import intraday_ohlc
import config
import requests
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from datetime import timedelta, time, date
from time import mktime
import matplotlib.pyplot as plt
import sqlalchemy
# from get_data_mul import get_data

class Query_realtime():
    def __init__(self):
        self.db_realtime=sqlalchemy.create_engine('postgresql://client1:finpros2022@192.168.110.15:6431/db_data')
        self.db_history=sqlalchemy.create_engine('postgresql://client2:finpros2022@192.168.110.15:6431/db_ps')
        
    def history_realtime(self, day=datetime.datetime.now().strftime('%Y_%m_%d')):
        query=f'''select "Time",
                "Close_Price",
                "Total_Volume"
                from  realtime_{day}
                where "Open_Interest"=(select max("Open_Interest") from realtime_{day}) 
                order by "Time" desc '''
        df=pd.read_sql_query(query,self.db_realtime)
        df.set_index('Time',inplace=True)
        df.index=pd.to_datetime(df.index)
        return df
    
    def query_his_real(self, list_day=[datetime.datetime.now().strftime('%Y_%m_%d')], duration=5):
        list_df = []
        for day in list_day:
            try:
                list_df.append(self.history_realtime(day))
            except:
                pass

        df3=pd.concat(list_df).dropna()
        df4 = df3.resample(f'{duration}Min',closed='right').last().dropna()
        df4['Open'] = df3.resample(f'{duration}Min',closed='right').Close_Price.first()
        df4['High'] = df3.resample(f'{duration}Min',closed='right').Close_Price.max()
        df4['Low'] = df3.resample(f'{duration}Min',closed='right').Close_Price.min()
        df4 = df4.reset_index().loc[~df4.reset_index()['Time'].dt.time.isin([time(14,31), time(14,32), time(14,33), time(14,34), \
                                                                            time(14,35), time(14,36), time(14,37), time(14,38), \
                                                                            time(14,39), time(14,40), time(14,41), time(14,42), \
                                                                            time(14,43), time(14,44), time(8,55), time(8, 50), \
                                                                            time(8, 39), time(8, 40), time(8, 41),time(8, 42),time(8, 43), \
                                                                            time(8, 44), time(8, 45), time(8, 46),time(8, 47),time(8, 48), \
                                                                            time(8, 49),time(8, 51),time(8, 52),time(8, 53),time(8, 54),\
                                                                            time(8, 56),time(8, 57),time(8, 58),time(8, 59),time(8, 0)])]
        df4['Volume'] = df4.Total_Volume.diff()
        df4.loc[df4['Time'].dt.time == time(9,0), 'Volume'] = df4.Total_Volume
        # df4.loc[df4.index[0], 'Volume'] = df4.loc[df4.index[0], 'Total_Volume']
        return df4.rename(columns={'Time':'Datetime', 'Close_Price':'Close'}).set_index('Datetime')
    
def get_stock_data(ticker, start_date='2000-01-01'):
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                         
        'Close': 'last',                                                                                                    
        'Volume': 'sum'
        }
    try:
        today_date = int(mktime(pd.Timestamp(start_date).timetuple()))
        end_date = int(mktime((date.today() + pd.Timedelta('1D')).timetuple()))
        url = 'https://dchart-api.vndirect.com.vn/dchart/history?resolution=1&symbol={}&from={}&to={}'.format(today_date,end_date)
        headers = {'Accept':'application/json, text/plain',
        'Accept-Language':'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5,zh-CN;q=0.4,zh;q=0.3',
        'Connection':'keep-alive',
        'Host':'dchart-api.vndirect.com.vn',
        'Origin':'https://dchart.vndirect.com.vn',
        'Referer':'https://dchart.vndirect.com.vn/',
        'Sec-Ch-Ua':'"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'Sec-Ch-Ua-Platform':'"Windows"',
        'Sec-Fetch-Dest':'empty',
        'Sec-Fetch-Mode':'cors',
        'Sec-Fetch-Site':'same-site',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        r = requests.get(url=url,headers=headers)
        df = pd.DataFrame(r.json())
        df.columns = ['Date','Close','Open','High','Low','Volume','s']
        df['Date'] = pd.to_datetime(df['Date'].astype(int), unit='s') + timedelta(hours=7)
        df = df[['Date','Open','High','Low','Close','Volume']].sort_values(by='Date')
        df['Value'] = df['Close'] * df['Volume']
        df['day'] = df['Date'].dt.date
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.resample('1D').apply(ohlc_dict).dropna().apply(lambda x: round(x, 1))
        print('VNDIRECT')

    except:
        try:
            headers = {
            'authority': 'services.entrade.com.vn',
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'dnt': '1',
            'origin': 'https://banggia.dnse.com.vn',
            'referer': 'https://banggia.dnse.com.vn/',
            'sec-ch-ua': '"Edge";v="114", "Chromium";v="114", "Not=A?Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1788.0'
            }
            today_date = int(mktime(pd.Timestamp(start_date).timetuple()))
            end_date = int(mktime((date.today() + pd.Timedelta('1D')).timetuple()))
            link = 'https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from={}&to={}&symbol={}&resolution=1'.format(today_date,end_date,ticker)
            dict_f = requests.get(url=link, headers=headers).json()
            df = pd.DataFrame(dict_f)
            df.columns = ['Date','Open','High','Low','Close','Volume','nt']
            df['Date'] = pd.to_datetime(df['Date'].astype(int).apply(lambda x: datetime.datetime.fromtimestamp(x)))
            df['Volume'] = dict_f['v']
            df['Value'] = df['Close'] * df['Volume']
            df['day'] = df['Date'].dt.date
            df['stock'] = ticker
            df = df.sort_values(by='Date',ascending=True).set_index('Date')
            df.index = pd.to_datetime(df.index)
            df = df.resample('1D').apply(ohlc_dict).dropna()
            print('DNSE')
        except:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%d/%m/%Y')
            df = pd.DataFrame(intraday_ohlc(config, ticker, start_date, datetime.datetime.now().strftime('%d/%m/%Y'), 1, 9999, True, 0)['data'])
            df['Date'] = pd.to_datetime(df['TradingDate'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df.set_index('Date', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            df = df.resample('1D').apply(ohlc_dict).dropna()
            df['Open'] = df['Open']/1000
            df['High'] = df['High']/1000
            df['Low'] = df['Low']/1000
            df['Close'] = df['Close']/1000
            print('SSI')
    return df.reset_index()

def portfolio_pnl_single(position, close):
    close_price = close[close.index.isin(position.index)]
    intitial_capital = position.iloc[0]*close.iloc[0]
    cash = (position.diff(1)*close_price)
    cash[0] = intitial_capital
    cash_cs = cash.cumsum()
    portfolio_value = (position*close_price)
    return (portfolio_value - cash_cs).iloc[1:]

def DumpCSV_and_MesToTele_CS(name, tickers, path_position, path_close, path_gain, Position, Close, token, id, capital=945000):
    ''' Ghi file csv và gửi tin nhắn đến telegram 
        Input: name: tên của chiến thuật
               tickers: danh sách mã chứng khoán
               path_position: đường dẫn file csv position hằng ngày
               path_close: đường dẫn file csv close hằng ngày
               path_gain: đường dẫn file csv gain hằng ngày
               Position: Dataframe vị thế của chiến thuật 
               Close: Dataframe giá khớp lệnh
               token: token của bot telegram
               id: id của chat group telegram
               capital: vốn ban đầu (nghìn vnd) Ví dụ: capital = 945000 tương đương 945tr'''
    ip_address = output.decode().strip()
# Close
    try:
        df_close = pd.read_csv(path_close)
        dict_close = {
            'Datetime': df_close.Datetime.tolist(),
        }
        for ticker in tickers:
            dict_close[ticker] = df_close[ticker].tolist()
        df_close = pd.DataFrame(data=dict_close).ffill()
    except:
        dict_close = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')],
        }
        for ticker in tickers:
            dict_close[ticker] = [Close[ticker].iloc[-2]]
    dict_close['Datetime'].append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for ticker in tickers:
        dict_close[ticker][-1] = Close[ticker].iloc[-2]
        dict_close[ticker].append(Close[ticker].iloc[-1])
    df_close = pd.DataFrame(data=dict_close).ffill()
    df_close.to_csv(path_close, index=False)

# Position
    try:
        df_position = pd.read_csv(path_position)
        dict_position = {
            'Datetime': df_position.Datetime.tolist(),
        }
        for ticker in tickers:
            dict_position[ticker] = df_position[ticker].tolist()
        df_position = pd.DataFrame(data=dict_position).ffill()
    except:
        dict_position = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')],
        }
        for ticker in tickers:
            dict_position[ticker] = [Position[ticker].iloc[-2]]
    dict_position['Datetime'].append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for ticker in tickers:
        dict_position[ticker].append(Position[ticker].iloc[-1])
    df_position = pd.DataFrame(data=dict_position).ffill()
    df_position[tickers] = df_position[tickers]
    print(df_position)
    df_position.to_csv(path_position, index=False)

# Gain
    try:
        df_gain = pd.read_csv(path_gain)
        dict_gain = {
            'Datetime': df_gain.Datetime.tolist(),
            'gain': df_gain.gain.tolist(),
            'total_gain': df_gain.total_gain.tolist(),
            'return': df_gain['return'].tolist()
        }
        for ticker in tickers:
            dict_gain[ticker] = df_gain[ticker].tolist()
        df_gain = pd.DataFrame(data=dict_gain)
    except:
        dict_gain = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')],
            'gain': [0],
            'total_gain': [0],
            'return': [0]
        }
        for ticker in tickers:
            dict_gain[ticker] = [0]

    dict_gain['Datetime'].append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    num_profit = 0
    num_loss = 0
    profit_today = 0
    for ticker in tickers:
        df = pd.DataFrame()
        df['Close'] = df_close[ticker]
        df['signal_long'] = df_position[ticker]
        df['total_gain'] = portfolio_pnl_single(df['signal_long'], df['Close'])
        df['total_gain'] = df['total_gain'].ffill().fillna(0)/1000
        df['return'] = df['total_gain']/capital*1000
        dict_gain[ticker].append(df.total_gain.diff().iloc[-1])
        profit_today += np.round(df.total_gain.diff().iloc[-1]*100)/100
        if np.round(df.total_gain.diff().iloc[-1]*100)/100 > 0:
            num_profit += 1
        elif np.round(df.total_gain.diff().iloc[-1]*100)/100 < 0:
            num_loss += 1

    return_today = profit_today/capital*1000
    dict_gain['gain'].append(profit_today)
    dict_gain['total_gain'].append(dict_gain['total_gain'][-1] + profit_today)
    dict_gain['return'].append(dict_gain['total_gain'][-1]*1000/capital + 1)
    df_gain = pd.DataFrame(data=dict_gain)
    df_gain.to_csv(path_gain, index=False)

    time_now = datetime.datetime.now()
    mes = f'{ip_address}\n{name} (capital {capital/1000}tr):\nNumber of Instruments: {len(tickers)} \nNumber of profit: {num_profit}\nNumber of loss: {num_loss}\nProfit today: {np.round(df_gain.gain.iloc[-1]*100)/100}tr\nReturn today: {np.round(return_today*10000)/100}%'
    send_to_telegram(mes, token, id)

    print(name)
    print(time_now)
    print(f'Number of profit: {num_profit}')
    print(f'Number of loss: {num_loss}')
    print(f'Profit today: {np.round(df_gain.gain.iloc[-1]*100)/100}')
    print(f'Return today: {np.round(return_today*10000)/100}%')
    print('\n')

def get_data_SSI(days=60):
    
    df_final = pd.DataFrame()
    while days > 30:
        
        df = pd.DataFrame(intraday_ohlc(config, 'VN30F1M', (datetime.datetime.now()-timedelta(days=days)).strftime('%d/%m/%Y'), (datetime.datetime.now()-timedelta(days=days-30)).strftime('%d/%m/%Y'), 1, 9999, True, 0)['data'])
        df['Datetime'] = pd.to_datetime(df['TradingDate'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df_final = pd.concat([df_final, df])
        days -= 30

    df = pd.DataFrame(intraday_ohlc(config, 'VN30F1M', (datetime.datetime.now()-timedelta(days=days)).strftime('%d/%m/%Y'), datetime.datetime.now().strftime('%d/%m/%Y'), 1, 9999, True, 0)['data'])
    df['Datetime'] = pd.to_datetime(df['TradingDate'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df_final = pd.concat([df_final, df])

    return df_final[['Datetime','Open','High','Low','Close','Volume']].set_index('Datetime').astype('float')

def get_data_realtime(days=1, duration=5, time_origin='9:00'):
    ''' lấy data realtime
    days: số ngày lấy data
    duration: timeslide của data'''
    
    ohlc_dict = {                                                                                                             
            'Open': 'first',                                                                                                    
            'High': 'max',                                                                                                       
            'Low': 'min',                                                         
            'Close': 'last',                                                                                                    
            'Volume': 'sum'
            }

    ### url của lấy phái sinh
    url = 'http://192.168.110.166:3123/ps/{}'.format(days)
    df = requests.get(url)
    r = pd.DataFrame(df.json())
    r['Date'] = pd.to_datetime(r['Date'])
    r['day'] = r['Date'].dt.date
    r['Value'] = r['Volume'] * r['Close']
    df_new = r[['Date','Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Date':'Datetime'}).set_index('Datetime')
    df_new.index = pd.to_datetime(df_new.index)
    
    return df_new.resample(str(duration)+'Min', origin=time_origin).apply(ohlc_dict).dropna().astype(float)

def get_vn30(duration, fromtimestamp=0):
    def vn30():                                     
        return requests.get(f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index?from={fromtimestamp}&resolution=1&symbol=VN30&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    s = pd.read_csv('../VN30.csv')
    s['Date'] = pd.to_datetime(s['Date']) + timedelta(hours=7)
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                         
        'Close': 'last',                                                                                                    
        'Volume': 'sum',}
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = pd.concat([process_data(vn30fm), process_data(s)]).sort_values('Date').drop_duplicates('Date').sort_values('Date')
    return vn30f_base

def portfolio_pnl_future(position_long, position_short, Close):
    ''' tính PNL của một chiến thuật 
    position_long: series position long
    position_short: series position short'''
    intitial_capital_long = (position_long.iloc[0])*(Close.iloc[0])
    cash_long = (position_long.diff(1)*Close)
    cash_long[0] = intitial_capital_long
    cash_cs_long = cash_long.cumsum()
    portfolio_value_long = (position_long*Close)

    intitial_capital_short = (position_short.iloc[0])*(Close.iloc[0])
    cash_short = (position_short.diff(1)*Close)
    cash_short[0] = intitial_capital_short
    cash_cs_short = cash_short.cumsum()
    portfolio_value_short = (position_short*Close)

    backtest = (portfolio_value_long - cash_cs_long) + (cash_cs_short - portfolio_value_short)
    backtest.fillna(0, inplace=True)
    cash_max = (cash_long + cash_short).max()
    pnl =  backtest/cash_max
    
    ''' return PNL, lần vào lệnh lớn nhất, PNL tương đối theo % '''
    return backtest, cash_max, pnl

def Sharp(pnl):
    ''' Tính Sharp ratio '''
    r = pnl.diff(1)
    return r.mean()/r.std() * np.sqrt(252)

def maximum_drawdown_future(gain, cash_max):
    ''' Tính maximum drawdown theo điểm, theo % '''
    return (gain.cumsum().cummax() - gain.cumsum()).max(), (gain.cumsum().cummax() - gain.cumsum()).max()/cash_max

def Margin(test):
    ''' Tính Margin '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test['inLong'] = test.signal_long.diff()[test.signal_long.diff() > 0].astype(int)
    test['inShort'] = test.signal_short.diff()[test.signal_short.diff() > 0].astype(int)
    test['outLong'] = -test.signal_long.diff()[test.signal_long.diff() < 0].astype(int)
    test['outShort'] = -test.signal_short.diff()[test.signal_short.diff() < 0].astype(int)
    test.loc[test.index[0], 'inLong'] = test.signal_long.iloc[0]
    test.loc[test.index[0], 'inShort'] = test.signal_short.iloc[0]
    test.fillna(0, inplace=True)

    ''' return dataframe chưa thêm các cột inLong, inShort, outLong, outShort và Margin '''
    return test, test.total_gain.iloc[-1]/(test.inLong * test.Close + test.inShort * test.Close + test.outLong * test.Close + test.outShort * test.Close).sum()*10000

def HitRate(test):
    ''' Tính Hit Rate '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test = Margin(test)[0]
    test = test[(test.outLong > 0) | (test.outShort > 0) | (test.inLong > 0) | (test.inShort > 0)]
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test.fillna(0, inplace=True)
    test['gain'] = test.total_gain.diff()
    test.fillna(0, inplace=True)
    test['gain'] = np.where(np.abs(test.gain) < 0.00001, 0, test.gain)
    try:
        ''' return dataframe thu gọn và Hit Rate'''
        return test, len(test[test.gain > 0])/(len(test[test.inLong > 0]) + len(test[test.inShort > 0]))
    except:
        return 0

def test_live(duration, fromtimestamp=1651727820):
    ''' Lấy dữ liệu từ API '''
    ''' Input: duration: sample dữ liệu theo phút '''
    def vn30f():
        return requests.get(f"https://services.entrade.com.vn/chart-api/chart?from={fromtimestamp}&resolution=1&symbol=VN30F1M&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    s = pd.read_csv('../Data/DataMinute/VN30F1M.csv', index_col=0, parse_dates=True)[datetime.datetime.fromtimestamp(fromtimestamp).strftime('%Y-%m-%d'):].reset_index()
    s['Date'] = pd.to_datetime(s['Date']) + timedelta(hours =7)
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                                                                        
        'Close': 'last',                                                                                                    
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()#change s
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = pd.concat([process_data(vn30fm), process_data(s)]).sort_values('Date').drop_duplicates('Date').sort_values('Date')
    return vn30f_base

def test_live_realtime(duration, fromtimestamp=1651727820):
    ''' Lấy dữ liệu từ API '''
    ''' Input: duration: sample dữ liệu theo phút '''
    def vn30f():
        return requests.get(f"https://services.entrade.com.vn/chart-api/chart?from={fromtimestamp}&resolution=1&symbol=VN30F1M&to={int(datetime.datetime.now().timestamp())}").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                                                                        
        'Close': 'last',                                                                                                    
        'Volume': 'sum',}
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()#change s
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = process_data(vn30fm).sort_values('Date').drop_duplicates('Date')
    return vn30f_base

def send_to_telegram(message, token='5683073192:AAHOAHjiRwk3pbNWI4dPFfURa4YaySvbfLY', id='-879820435'):
    ''' Gửi tin nhắn đến telegram '''
    ''' Input: message: tin nhắn muốn gửi
               token: token của bot
                id: id của chat group '''
    apiToken = token
    chatID = id
    try:
        apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage?chat_id={chatID}&text={message}"
        requests.get(apiURL).json()
    except Exception as e:
        print(e)

def position_input(position, path_Po='G:/alpha_live_pos/PHU/PS13_PHU.txt'):
    ''' ghi file position input (cho chiến thuật chạy live) '''
    ''' Input: position: vị thế của chiến thuật 
               path_Po: đường dẫn file position input'''
    if position != 0:
        with open(path_Po, "w") as f:
            info = "pos={}\ntime=5".format(position)
            f.write(info)

def position_report(position, path_CP='G:/alpha_live_pos/PHU/PS13_PHU_CP.txt'):
    ''' ghi file position report (vị thế hiện tại) (cho chiến thuật chạy live) 
        Input: position: vị thế của chiến thuật
               path_CP: đường dẫn file position report'''
    with open(path_CP, "w") as f:
        pos_rp = "pos={}".format(position)
        f.write(pos_rp)

def Check_expiry():
    today = date.today()
    third_thursday = today.replace(day=1)

    while third_thursday.weekday() != 3:  # Thursday is 3
        third_thursday += timedelta(days=1)

    for i in range(2):
        third_thursday += datetime.timedelta(days=7)

    if today == third_thursday:
        return True
    else:
        return False

def DumpCSV_and_MesToTele(name, path_csv_intraday, Position, Close, token, id, position_input=1, fee=0.4):
    ''' Ghi file csv và gửi tin nhắn đến telegram 
        Input: name: tên của chiến thuật
               path_csv_intraday: đường dẫn file csv intraday
               Position: Series vị thế của chiến thuật 
               Close: Series giá khớp lệnh
               token: token của bot telegram
               id: id của chat group telegram
               position_input: số hợp đồng vào mỗi lệnh'''
    ip_address = output.decode().strip()
    try:
        df = pd.read_csv(path_csv_intraday)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'Position': df.Position.tolist(),
            'Close': df.Close.tolist(),
            'total_gain': df.total_gain.tolist(),
            'gain': df.gain.tolist(),
        }
        try:
            dict_data['Datetime'] = pd.to_datetime(dict_data['Datetime']).to_list()
        except:
            for i in range(len(dict_data['Datetime'])):
                dict_data['Datetime'][i] = pd.to_datetime(dict_data['Datetime'][i])
            dict_data['Datetime'] = list(dict_data['Datetime'])
        df = pd.DataFrame(data=dict_data)
    except:
        dict_data = {
            'Datetime': [pd.to_datetime((datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))],
            'Position': [0],
            'Close': [0],
            'total_gain': [0],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_intraday, index=False)

    Close = Close.iloc[-1]
    new_Pos = int(Position.iloc[-1])
    time_now = datetime.datetime.now()
    profit = 0
    profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
    mes = f'{ip_address}\n{name}:'
    
    if new_Pos != dict_data['Position'][-1] or time_now.time() >= datetime.time(14, 45):

        if time_now.time() >= datetime.time(14, 45):
            new_Pos = dict_data['Position'][-1]
        
        if Check_expiry() and time_now.time() >= datetime.time(14, 45):
            new_Pos = 0

        inputPos = int(new_Pos - dict_data['Position'][-1])
        dict_data['Datetime'].append(pd.to_datetime(time_now.strftime('%Y-%m-%d %H:%M:%S')))
        dict_data['Close'].append(Close)
        dict_data['Position'].append(new_Pos)
        dict_data['total_gain'].append(0)
        dict_data['gain'].append(0)

        df = pd.DataFrame(data=dict_data)
        df['signal_long'] = np.where(df.Position > 0, df.Position, 0)
        df['signal_short'] = np.where(df.Position < 0, np.abs(df.Position), 0)
        df['total_gain'] = portfolio_pnl_future(df['signal_long'], df['signal_short'], df.Close)[0]
        df['gain'] = df.total_gain.diff()
        df.fillna(0, inplace=True)
        df['gain'] = np.where(np.abs(df.gain.to_numpy()) < 0.00001, 0, df.gain.to_numpy())
        df.loc[df['Position'].diff().fillna(0) != 0, 'gain'] = df.loc[df['Position'].diff() != 0, 'gain'] - np.abs(fee/2 * inputPos)
        # df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] = df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] - fee
        df['total_gain'] = df.gain.cumsum()
        profit = df.gain.iloc[-1]
        profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
        
        if inputPos > 0:
            mes = f'{ip_address}\n{name}:\nLong {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        elif inputPos < 0:
            mes = f'{ip_address}\n{name}:\nShort {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        else:
            mes = f'{ip_address}\n{name}:\nClose at {Close}, Current Pos: {new_Pos*position_input}'

        if np.round(profit*10)/10 != 0:
            mes += f'\nProfit: {np.round(profit*10)/10}'
        mes += f'\nProfit today: {np.round(profit_today*10)/10}'

        df.drop(columns=['signal_long', 'signal_short'], inplace=True)
        send_to_telegram(mes, token, id)
        df.to_csv(path_csv_intraday, index=False)

    else:
        inputPos = 0

    profit_today = np.round(profit_today*10)/10
    print(name)
    print(time_now)
    print(Close)
    print('Input Position:', inputPos*position_input)
    print('Current Position:', new_Pos*position_input)
    if np.round(profit*10)/10 != 0:
        print(f'Profit: {np.round(profit*10)/10}')
    print(f'Profit today: {profit_today}')
    print('\n')

    ''' return dataframe intraday, input position, current position'''
    df['profit_today'] = profit_today
    return df, inputPos, new_Pos

def PNL_per_day(path_csv_daily, profit_today):
    ''' Ghi file csv PNL theo ngày
        Input: path_csv_daily: đường dẫn file csv PNL theo ngày
               profit_today: Series profit_today của chiến thuật
               (Lấy ra từ dataframe df của hàm DumpCSV_and_MesToTele)'''
    try:
        df = pd.read_csv(path_csv_daily)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'gain': df.gain.tolist(),
        }
    except:
        dict_data = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_daily, index=False)

    gain = profit_today.iloc[-1]
    time_now = datetime.datetime.now()

    if time_now.strftime('%Y-%m-%d') != pd.to_datetime(dict_data['Datetime'][-1]).strftime('%Y-%m-%d'):
        if gain != dict_data['gain'][-1]:
            dict_data['Datetime'].append(time_now.strftime('%Y-%m-%d'))
            dict_data['gain'].append(gain)
            df = pd.DataFrame(data=dict_data)
    else:
        dict_data['gain'][-1] = gain
        df = pd.DataFrame(data=dict_data)

    df['total_gain'] = df['gain'].cumsum()
    df['total_gain'].apply(lambda x: np.round(x*10)/10)
    df.fillna(0, inplace=True)

    df.to_csv(path_csv_daily, index=False)
    ''' return dataframe PNL theo ngày '''
    return df

class BacktestInformation:
    ''' Thông tin backtest của chiến thuật 
        Input: Datetime: Series Datetime
                Position: Series Position
                Close: Series Close '''
    ''' CHÚ Ý: Nên dùng class này để lấy được các thông tin của chiến thuật chứ không nên dùng các hàm riêng lẻ
               vì các hàm riêng lẻ phía trên có thể có định dạng position không đồng nhất với class này '''
    def __init__(self, Datetime, Position, Close, fee=0.4):
        signal_long = np.where(Position >= 0, Position, 0)
        signal_short = np.where(Position <= 0, np.abs(Position), 0)
        try:
            Datetime = pd.to_datetime(Datetime)
        except:
            Datetime = Datetime.to_list()
            for i in range(len(Datetime)):
                Datetime[i] = pd.to_datetime(Datetime[i])
        self.df = pd.DataFrame(data={'Datetime': Datetime, 'Position': Position, 'signal_long': signal_long, 'signal_short': signal_short, 'Close': Close})
        self.df.set_index('Datetime', inplace=True)
        self.hold_overnight = not (self.df.resample('D').last().dropna()['Position'] == 0).all()
        self.df.index = pd.to_datetime(self.df.index)
        self.df_brief = HitRate(self.df)[0]
        self.fee = fee + 0.22*self.Trading_per_day()
    
    def PNL(self):
        ''' Tính PNL của chiến thuật '''
        total_gain, cash_max, pnl = portfolio_pnl_future(self.df.signal_long, self.df.signal_short, self.df.Close) 

        ''' return Series PNL, cash_max '''
        return total_gain, cash_max, pnl
    
    def Sharp(self):
        ''' Tính Sharp của chiến thuật '''
        return Sharp(self.PNL()[0].resample("1D").last().dropna())
    
    def Sharp_after_fee(self):
        ''' Tính Sharp sau khi trừ phí của chiến thuật '''
        return Sharp(self.Plot_PNL(plot=False)['total_gain_after_fee'].resample("1D").last().dropna())
    
    def Margin(self):
        ''' Tính Margin của chiến thuật '''
        return Margin(self.df_brief)[1]
    
    def MDD(self):
        ''' Tính MDD của chiến thuật '''
        return maximum_drawdown_future(self.Plot_PNL(plot=False)['total_gain_after_fee'].astype(float).resample("1D").last().diff().dropna(), self.PNL()[1])
    
    def Hitrate(self):
        ''' Tính Hitrate của chiến thuật '''
        return HitRate(self.df_brief)[1]
    
    def Number_of_trade(self):
        ''' Tính số lần giao dịch của chiến thuật '''
        return len(self.df_brief[self.df_brief.inLong != 0]) + len(self.df_brief[self.df_brief.inShort != 0])
    
    def Profit_per_trade(self):
        ''' Tính Profit trung bình của 1 giao dịch '''
        return self.Plot_PNL(plot=False)['total_gain'].iloc[-1]/self.Number_of_trade()
    
    def Profit_after_fee(self):
        ''' Tính Profit sau khi trừ phí '''
        return np.round(self.Plot_PNL(plot=False)['total_gain_after_fee'].iloc[-1]*10)/10
    
    def Profit_per_day(self):
        ''' Tính Profit trung bình theo ngày '''
        return self.Profit_after_fee()/len(self.PNL()[0].resample("1D").last().dropna())
    
    def Trading_per_day(self):
        ''' Tính số lần giao dịch trung bình theo ngày '''
        return self.Number_of_trade()/len(self.PNL()[0].resample("1D").last().dropna())
    
    def Hitrate_per_day(self):
        ''' Tính Hitrate theo ngày '''
        if self.PNL()[0].resample("1D").last().dropna().iloc[0] != 0:
            Profit = self.PNL()[0].resample("1D").last().dropna().diff()
            Profit.loc[Profit.index[0]] = self.PNL()[0].resample("1D").last().dropna().iloc[0]
        else:
            Profit = self.PNL()[0].resample("1D").last().dropna().iloc[1:].diff()
            Profit.loc[Profit.index[0]] = self.PNL()[0].resample("1D").last().dropna().iloc[1:].iloc[0]
        return Profit, len(Profit[Profit > 0])/len(Profit)

    def Return(self):
        ''' Tính Return trung bình mỗi năm (theo %) của chiến thuật '''
        cash_max = self.PNL()[1]
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)/cash_max    
    
    def Profit_per_year(self):
        ''' Tính Profit trung bình theo năm '''
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)

    def Plot_PNL(self, window_MA=None, plot=True):
        ''' Print thông tin và Vẽ biểu đồ PNL của chiến thuật 
            Input: after_fee: bool, True: plot có trừ phí, False: plot không trừ phí'''

        total_gain, cash_max, pnl = self.PNL()
        total_gain = pd.DataFrame(total_gain.to_numpy(), index=total_gain.index, columns=['total_gain'])
        total_gain.loc[self.df['Position'].diff().fillna(0) != 0, 'fee'] = np.abs(self.fee/2 * self.df['Position'].diff().fillna(0))
        total_gain['fee'] = total_gain['fee'].fillna(0).cumsum()
        total_gain['total_gain_after_fee'] = total_gain['total_gain'] - total_gain['fee']

        if total_gain['total_gain'].resample('1D').last().dropna().iloc[0] != 0:
            total_gain.reset_index(inplace=True)
            previous_day = pd.DataFrame(total_gain.iloc[0].to_numpy(), index=total_gain.columns).T
            previous_day.loc[previous_day.index[0], 'Datetime'] = pd.to_datetime(previous_day['Datetime'].iloc[0]) - timedelta(days = 1) 
            previous_day.loc[previous_day.index[0], 'total_gain'] = 0
            total_gain = pd.concat([previous_day, total_gain]).set_index('Datetime')

        if plot:

            print('Margin:',Margin(self.df_brief)[1])
            print(f'MDD: {self.MDD()}\n')

            data = [('Total trading quantity', self.Number_of_trade()),
                    ('Profit per trade',self.Profit_per_trade()),
                    ('Total Profit', np.round(total_gain.total_gain.iloc[-1]*10)/10),
                    ('Profit after fee', self.Profit_after_fee()),
                    ('Trading quantity per day', self.Number_of_trade()/len(total_gain.total_gain.resample("1D").last().dropna())),
                    ('Profit per day after fee', self.Profit_per_day()),
                    ('Return', self.Return()),
                    ('Profit per year', self.Profit_per_year()),
                    ('HitRate', self.Hitrate()),
                    ('HitRate per day', self.Hitrate_per_day()[1]),
                    ]
            for row in data:
                print('{:>25}: {:>1}'.format(*row))

            # total_gain[f'MA{window_MA}'] = total_gain['total_gain'].rolling(window_MA).mean().fillna(0)
            (total_gain.total_gain.resample("1D").last().dropna()).plot(figsize=(15, 4), label=f'{Sharp(total_gain.total_gain.resample("1D").last().dropna())}')
            (total_gain.total_gain_after_fee.resample("1D").last().dropna()).plot(figsize=(15, 4), label=f'{Sharp(total_gain.total_gain_after_fee.resample("1D").last().dropna())}')
            
            if window_MA != None:
                (total_gain.total_gain.resample("1D").last().dropna().rolling(window_MA).mean()).plot(figsize=(15, 4), label=f'MA{window_MA}')
            plt.grid()
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('PNL')
            plt.show()

            plt.figure()
            (1 + pnl).plot(figsize=(15, 4), label=f'{self.Sharp()}')
            plt.legend()
            plt.grid()
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.show()

        # self.df.set_index('Datetime', inplace=True)
        total_gain['Position'] = self.df['Position']
        total_gain['Close'] = self.df['Close']
        total_gain['Return'] = 1 + pnl
        total_gain.reset_index(inplace=True)

        return total_gain.set_index('Datetime')