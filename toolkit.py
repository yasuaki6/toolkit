import csv
import pandas
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import sys
import time
import pickle
import pandas_datareader as pdr
import datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import io 
from PIL import Image
import imagehash
import datetime
import tkinter as tk
from tkinter import Frame
from matplotlib.backends.backend_tkagg import  FigureCanvasTkAgg
import japanize_matplotlib



def stook_code():
    df = pandas.read_excel('data.xls') 
    cnt = 0
    strat = 4192
    for i,j in df.iterrows():
        cnt+=1
        if strat > cnt:
            continue
        stock_code  = j['コード']
        stock_name = j['銘柄名']
        yield stock_code,stock_name
        
class stock_dowload(object):
    def __init__(self) -> None:
        pass
    
    def yahoo(stock_code,period=None,
              frequency=None):   
          
            my_share = share.Share(str(stock_code)+'.T')
            symbol_data = None
            
            Period = {'day':share.PERIOD_TYPE_DAY,
                        'week':share.PERIOD_TYPE_WEEK,
                       'month':share.PERIOD_TYPE_MONTH,
                       'year':share.PERIOD_TYPE_YEAR}
            
            Frequency = {'day':share.FREQUENCY_TYPE_DAY,
                        'week':share.FREQUENCY_TYPE_WEEK,
                       'month':share.FREQUENCY_TYPE_MONTH,
                       }
            
            period =  share.PERIOD_TYPE_WEEK if period == None else Period[period]
            frequency = share.FREQUENCY_TYPE_DAY if frequency == None else Frequency[frequency] 
            
            try:
                symbol_data = my_share.get_historical(period,
                                                1,
                                                frequency,
                                                1)
            except YahooFinanceError as e:
                print(e.message)
                return[]
                
            symbol_data = pandas.DataFrame(symbol_data)
            symbol_data['timestamp'] = pandas.to_datetime(symbol_data.timestamp, unit = 'ms')        
            symbol_data = symbol_data.set_index(['timestamp'])
            symbol_data = symbol_data.tz_localize('utc').tz_convert('Asia/Tokyo')
            return symbol_data

    def stooq(stock_code,start=None,end=None):
        start = datetime.datetime.now() - datetime.timedelta(days=30) if start == None else start
        end = datetime.datetime.now()  if end == None else end
         
        tg = str(stock_code) + '.JP'
        df = pdr.DataReader(tg,data_source='stooq',start=start,end=end).rename(columns=str.lower).tz_localize('utc').tz_convert('Asia/Tokyo').iloc[::-1]
        return df

    def target_dowload(start,end,target):
        tg = str(target) + '.JP'
        df = pdr.DataReader(tg,data_source='stooq',start=start,end=end).rename(columns=str.lower).tz_localize('utc').tz_convert('Asia/Tokyo').iloc[::-1]
        if end == datetime.date.today():
            df = stock_dowload.merge(df,stock_dowload.yahoo(target))
        return df

    def merge(df1,df2):
        if len(df2):
            return df1
        else:
            return pandas.concat([df1, df2]).drop_duplicates()


class Load():
    def __init__(self,period='mouth'):
        self.period = period
        self.checklist = []
        self.add_check()
        
    def add_check(self):
        #読み込まなくていい銘柄リスト
        self.checklist.extend([i for i in range(224)])
        self.checklist.extend([i for i in range(357,385)])
        self.checklist.extend(i for i in range(568,607))
        self.checklist.extend([i for i in range(626,655)])
        self.checklist.extend([i for i in range(747,765)])
        self.checklist.extend([i for i in range(3731,3754)])
        
    def all_load():
        with open('stockchart for year','rb',) as f:
            for i in range(4166):
                d = pickle.load(f)
                yield d  
                
    def check_load(self):
           with open('stockchart for year','rb',) as f:
            for i in range(4166):
                if i in self.checklist:
                    continue
                d = pickle.load(f)
                yield d 
                
    def check_load_dict(self):   
        with open('stockchart for dict','rb',) as f:
            all_df = pickle.load(f)
            for i,j in enumerate(all_df.items()):
                if i in self.checklist:
                    continue
                name,df = j
                yield name,df
            

class Scaler():
    def __init__(self,df):
        self.df = df
    def encode_graph_as_bits(df=[]):
        #df = df if len(df) else self.df
        List = []
        df_close = df["close"][0]
        if df['open'][0] > df['close'][0]:
            List.append(0)
        else:
            List.append(1)
        for i in range(1,len(df)):
            if df_close < df['close'][i]:
                List.append(1)
            else:
                List.append(0)
            df_close = df['close'][i]
        return np.array(List,dtype=int)

    def rate(df=[]):
        import copy 
        #df = df if len(df) else self.df
        sub = copy.deepcopy(df['close'])
        for i in range(1,len(df)):
            sub[i] -= df['close'][i-1]
        sub[0] -= int(df['open'][0])
        sub = round((sub / df['close']) * 100,1)
        return np.array(sub)  
               
    def percent(df=[],index=None):
        #df = self.df  if df == None else df
        if index == None:
            return (((df['close'] - df['close'][0]) / df['close']) *100)
        else:
            return (((df['close'] - df['close'][index]) / df['close']) *100)
    
    def normalize(df):
        #df = df if len(df) else self.df
        return (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
    
    def normalize_col(col):
        return (col - col.min()) / (col.max() - col.min())
    
    
class Bias():

    def Exponential(df,a=2):
        f = lambda x:x**a   
        return list(map(f,df))
    
    def self_maid(df):
        List =[]
        for i in df :
            if i < 1:
                List.append(i)
            else:
                List.append(i*5)
        return List
    
    
class Comparer():   
    def __init__(self,org) :
        self.org = org
        self.collection_rate = {}
        self.collection_encode = {}
        self.result_rate = {}
        self.result_encode = {}
    
    def rate_comparison(self,name,sub_cps):
        org = Scaler.rate(self.org)
        sub = Scaler.rate(sub_cps)
        minbox = []
        for i in range(len(sub)-len(org)):
            bond = abs((org - sub[i:len(org)+i]))
            bond = Bias.Exponential(bond)

            minbox.append(round(sum(bond),1))
        try: 
            i = minbox.index(min(minbox))
        except ValueError:
            return
        self.collection_rate[name] = [i,len(org)+i]
        #self.collection_df[name] = sub_cps[i:len(self.org)+i]
        self.result_rate[name] = min(minbox) 
    
    def encode_comparison(self,name,sub_cps):
        org = Scaler.encode_graph_as_bits(self.org)
        sub = Scaler.encode_graph_as_bits(sub_cps)
        
        if  not len(sub):
            return
        minbox = []
        for i in range(len(sub)-len(org)):
            bond = org + sub[i:len(org)+i]
            bond = bond % 2
            minbox.append(sum(bond))
        try: 
            i = minbox.index(min(minbox))
        except ValueError:
            return
        self.collection_encode[name] = [i,len(org)+i]
        self.result_encode[name] = min(minbox)     
        
def img_hash(a):
    hash_return = imagehash.average_hash(Image.open(a))
    # hash_return = imagehash.phash(Image.open(a))
    # hash_return = imagehash.dhash(Image.open(a))
    # hash_return = imagehash.whash(Image.open(a))
    return hash_return   


class Plot():
    #抽象クラス
    def __init__(self,fig,ax):
        self.ax = ax
        self.fig = fig  
    
    def candles_idx(self,df):#-> tow detaframe
        idx_higt = df.index[df["close"]>=df["open"]]
        idx_low = df.index[df["close"]<df["open"]]
        return idx_higt,idx_low
              
    def cla(self):
        plt.cla()  
        self.canvas.draw()

    def candle_plot(self,df):
        idx_higt,idx_low = self.candles_idx(df)
        df = df.apply(Scaler.normalize_col)
        df["body"] = df["close"] - df["open"]


        kwags_bar = {"width":1*0.5,
                "color":"#33c076",
                "linewidth":1 ,
                "bottom":df.loc[idx_higt,"open"],
                "zorder":2}
        kwags_bar2 = {"width":1*0.5,#0.041 一時間足
                "color":"#ff5050",
                "linewidth":1 ,
                "bottom":df.loc[idx_low,"open"],
                "zorder":2}   

        self.ax.bar(idx_higt,df.loc[idx_higt,"body"],**kwags_bar)
        self.ax.bar(idx_low,df.loc[idx_low,"body"],**kwags_bar2)
        self.ax.vlines(df.index,df["low"],df["high"],zorder=1)   
        
        
    def line_plot(self,df,name):
        self.ax.set_ylim([0,1])
        df = df.apply(Scaler.normalize_col)
        df['index'] = [i for i in range(len(df))]
        df.plot('index','close',ax = self.ax,label=name)

    """class img_seve():
    def __init__(self) -> None:
        seve_img = []
        
    def img_seve(fig,ax,df):
        df = df.reset_index()
        fig.patch.set_facecolor('black') 
        ax.patch.set_facecolor('black')
        #line_plot(ax,df['scal'])
        img_seve.candle_plot(ax,df)
        buf = io.BytesIO()
        plt.savefig(buf,fromat='png')
        plt.cla() 
        return buf"""


class DataProcessor(Comparer):
    def __init__(self,org):
        super().__init__(org)
        self.database = {i:j.apply(Scaler.normalize_col) for i,j in Load().check_load_dict()}
        self.ordinal = {}
        self.mode_dict = {None:ValueError, 
                          'exponential':self.collection_rate,
                          'encode':self.collection_encode}
        self.mode = None
             
    def exponential_function(self):
        if len(self.result_rate):
            self.ordinal = self.result_rate
        else:
            load = Load()
            load.add_check()
            for name, df in load.check_load_dict():
                super().rate_comparison(name,df)
            self.ordinal = sorted(self.result_rate.items(), key=lambda x:x[1])
            self.mode = 'exponential'
           

    def encode_function(self):
        if len(self.result_encode):
            self.ordinal = self.result_encode
        else:
            load = Load()
            load.add_check()
            for name,df in load.check_load_dict():
                super().encode_comparison(name,df)
            self.ordinal = dict(sorted(self.result_encode.items(), key=lambda x:x[1]))
            self.mode = 'encode'
          

    def hash_function(self):
        print("ハッシュ関数が選択されました")

    def get(self):
        index = self.mode_dict[self.mode]
        for name, score in self.ordinal:
            result = self.database[name].iloc[index[name][0]:index[name][1]]
            yield result,name

class Menu(tk.Menu):
    def __init__(self,root):
        global database
        global canvas
        tk.Menu.__init__(self, root)
        # ファイルメニューの作成
        file_menu = tk.Menu(self, tearoff=0)
        file_menu.add_command(label="指数関数",command=self.on_exponential_function)
        file_menu.add_command(label="符号関数",command=self.on_encode_function)
        file_menu.add_command(label="ハッシュ関数")
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=root.quit)
        self.add_cascade(label="評価指標", menu=file_menu)  
        
    def on_exponential_function(self):
        database.exponential_function()
        canvas.line()
        print("指数関数が選択されました")

    def on_encode_function(self):
        database.encode_function()
        canvas.line()
        print("符号関数が選択されました")

    def on_hash_function(self):
        print("ハッシュ関数が選択されました")

class Canvas(Plot):
    def __init__(self,parent,org_df):
        global database
        self.fig, self.ax = plt.subplots()
        self.org = org_df
        self.database = database.get()
        super().__init__(self.fig, self.ax)
        self.canvas = FigureCanvasTkAgg(figure=self.fig, master=parent)
        self.canvas_wid = self.canvas.get_tk_widget()
        self.canvas_wid.bind('<Button-3>',self.line)
        self.canvas_wid.bind('<Button-1>',self.line)
        self.canvas.draw()
        self.canvas_wid.pack(fill=tk.BOTH, expand=1)


    def candle(self,event=None):
        super().cla()
        super().candle_plot(self.org)
        for i in range(5):
            df = next(database.get())
            super().candle_plot(df)
        self.canvas.draw()
        
    def line(self,event=None):
        super().cla()
        super().line_plot(self.org,'orgnal')
        for i in range(5):
            df,name = next(self.database)     
            super().line_plot(df,name)
        self.canvas.draw()
    

def result_generator(result, volume):
    name_list = []
    for cnt,valuse in enumerate(result, start=1):
        name_list.append(valuse[0]) 
        if cnt % volume == 0:
            yield result_generator2(name_list)
            
def result_generator2(name_list):
    for _ in range(len(name_list)):
        yield name_list.pop()


def on_closing():
    root.destroy()
    plt.close() 

if __name__ == '__main__':
    start = dt.date(2023,4,16)
    #end = dt.date(2023,4,13)
    end = datetime.date.today()
    target = 7116
    org_df = stock_dowload.target_dowload(start,end,target)
    database = DataProcessor(org_df)
    com = Comparer(org_df)
    root = tk.Tk()
    menu = Menu(root)
    canvas = Canvas(root,org_df)
    canvas.candle_plot(org_df)
    root.config(menu=menu)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
        
    

