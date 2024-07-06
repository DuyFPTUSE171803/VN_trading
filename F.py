import numpy as np
import pandas as pd
import datetime
from datetime import timedelta, date
from time import mktime
import matplotlib.pyplot as plt

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