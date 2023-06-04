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
import multiprocessing
import logging
import threading
import queue
import mysql

def stock_code(start=1):
    df = pandas.read_excel('data.xls') 
    cnt = 0
    for i,j in df.iterrows():
        cnt+=1
        if start > cnt:
            continue
        stock_code  = j['コード']
        stock_name = j['銘柄名']
        yield stock_code,stock_name     
        

class stock_dowload():
    def yahoo(stock_code,period=None,frequency=None):   
            my_share = share.Share(str(stock_code)+'.T')
            symbol_data = None
            
            Period = {'day':share.PERIOD_TYPE_DAY,
                        'week':share.PERIOD_TYPE_WEEK,
                       'month':share.PERIOD_TYPE_MONTH,
                       'year':share.PERIOD_TYPE_YEAR}
            
            Frequency = {'minute':share.FREQUENCY_TYPE_MINUTE,
                        'day':share.FREQUENCY_TYPE_DAY,
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
            if not len(symbol_data): return []
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

    def target_dowload(target,start=datetime.datetime.today() - datetime.timedelta(days=7) ,end=datetime.datetime.today()):
        tg = str(target) + '.JP'
        df = pdr.DataReader(tg,data_source='stooq',start=start,end=end).rename(columns=str.lower).tz_localize('utc').tz_convert('Asia/Tokyo').iloc[::-1]
        
        if len(df) == 0:
            df=stock_dowload.yahoo(target,'year','day')
            
            try:
                df=df.loc[start:end]
            except AttributeError:
                print('データが見つかりませんでした。')
                exit()
            
        elif end == datetime.date.today():
            df = stock_dowload.merge(df,stock_dowload.yahoo(target))
            
        return df

    def merge(df1,df2):
        
        if len(df2):
            return df1
        
        else:
            return pandas.concat([df1, df2]).drop_duplicates()
        
            
class Scaler():
    '''スケールに関してのクラス'''
        
    def encode_graph_as_bits(df):
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

    def rate(df):
        import copy 
        sub = copy.deepcopy(df['close'])
        
        for i in range(1,len(df)):
            sub[i] -= df['close'][i-1]
            
        sub[0] -= int(df['open'][0])
        sub = round((sub / df['close']) * 100,1)
        return np.array(sub)  
               
    def percent(df, index=0, columns=['close']):
        #columnsはリスト型にしてください。arrayが対応しない型は値を変更しません。
        result = pandas.DataFrame()
        for col in columns:
            try:
                result[col] = round(((df[col] - df[col][index]) / df[col][index]) * 100, 2)
            except TypeError:
                result[col] = df[col]
        return result

    def normalize_col(col):
        #df.apply関数用
        return (col - col.min()) / (col.max() - col.min())

    
class Bias():

    def Exponential(df,a=2):
        f = lambda x:x**a   
        return list(map(f,df))
    
    
class Comparer():
    """データを比較するためのクラス、初期設定で対象を設定します。
    各比較関数に比較対象のデータを渡し、スコア化、保存するクラス。
        arges
        org(pandas.dataframe):対象のデータフレーム"""   
    def __init__(self,org) :
        self.org = org
        org_to_hash = Plot()
        org_to_hash.candle_plot(org[-7:])
        self.org_hash = org_to_hash.seve_plot()
        org_to_hash.plot_close()
        self.index_rate = {}
        self.index_encode = {}
        self.index_hash = {}
        self.score_rate = {}
        self.score_encode = {}
        self.score_hash = {}
        
    def rate_comparison(self,name,sub_cps):
        org = Scaler.rate(self.org)
        sub = Scaler.rate(sub_cps)
        minbox = []
        
        if len(sub) == len(org):
            bond = abs(org-sub)
            bond = Bias.Exponential(bond)
            minbox.append(round(sum(bond),1))  
        else:            
            for i in range(len(sub)-len(org)):
                bond = abs((org - sub[i:len(org)+i]))
                bond = Bias.Exponential(bond)
                minbox.append(round(sum(bond),1))  
                        
        try: 
            i = minbox.index(min(minbox))
        except ValueError as e:
            print(str(e))
            return  
        
        self.index_rate[name] = [i,len(org)+i]
        self.score_rate[name] = min(minbox) 
    
    def encode_comparison(self,name,sub_cps):
        org = Scaler.encode_graph_as_bits(self.org)
        sub = Scaler.encode_graph_as_bits(sub_cps)
        if  not len(sub):
            return
        minbox = []
        
        if len(sub) == len(org):
            bond = abs(org-sub)
            bond = Bias.Exponential(bond)
            minbox.append(sum(bond))
        else:            
            for i in range(len(sub)-len(org)):
                bond = org + sub[i:len(org)+i]
                bond = bond % 2
                minbox.append(sum(bond))
            
        try: 
            i = minbox.index(min(minbox))
        except ValueError as e:
            print(str(e))
            return
        
        self.index_encode[name] = [i,len(org)+i]
        self.score_encode[name] = min(minbox) 
        
    def hash_comparison(self,name,sub_cps):
        #現在の保存している画像は1week用です。
        self.org_value = self.img_hash(self.org_hash)
        self.sub_value = self.img_hash(sub_cps)
        score = self.org_value - self.sub_value
        self.score_hash[name] = score
        self.index_hash[name] = [None,None] #フラグ
        
    def img_hash(self,img):
        hash_return = imagehash.average_hash(Image.open(img))
        # hash_return = imagehash.phash(Image.open(a))
        # hash_return = imagehash.dhash(Image.open(a))
        # hash_return = imagehash.whash(Image.open(a))
        return hash_return   


class Plot():
    def __init__(self):
        self.fig,self.ax = plt.subplots()
    
    def candles_idx(self,df):#-> tow detaframe
        idx_higt = df.index[df["close"]>=df["open"]]
        idx_low = df.index[df["close"]<df["open"]]
        return idx_higt,idx_low
              
    def cla(self):
        plt.cla()  
        self.canvas.draw()

    def candle_plot(self,df,width=None,color=None,linewidh=None):
        idx_higt,idx_low = self.candles_idx(df)
        
        df = Scaler.percent(df,columns=['open','close','high','low'])
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
        
    def line_plot(self,df,name='orgnal'):
        df = Scaler.percent(df)
        df['index'] = [i for i in range(len(df))]
        df.plot('index','close',ax = self.ax,label=name)
    
    def plotseve_setting(self):
        self.fig.patch.set_facecolor('black')
        self.ax.set_ylim([-75,75])
        
    def seve_plot(self):
        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        self.ax.clear()
        buf.seek(0)
        return buf
    
    def plot_close(self):
        plt.close()


class ChangeError(Exception):
    #ジェネレーターを終了させ、新しくジェネレーターを作成するためのクラス
    pass


class create_setting():
    """各種設定の基底クラス。
    args stock_code(int) :株コード
        start(datetime)  :指定期間の始点
        end(datetime)　   :指定期間の終点
        priode(str): day week month year updataクラスに使用する予定
        frequency(str):  minutes day week"""
    def __init__(self,org_df,start=datetime.date.today()-datetime.timedelta(days=7),
                 end=datetime.date.today(),period='month',frequency='day',forecast_period=0,created_at=None):
        self.org_df = org_df
        self.start = start 
        self.end = end
        if self.end.weekday() == 5 or 6:
            self.end -= datetime.timedelta(days=(self.end.weekday() % 4)) # 土日は存在しないので金曜日に調整しています。
        self.frequency = frequency
        self.period = period
        self.created_at = self.end if created_at == None else created_at
        self.created_at = self.created_at.strftime('%Y-%m-%d')
        self.forecast_period =  forecast_period # 比較したデータを表示する際のデータの長さ
        self.checklist = [] #飛ばす銘柄 エクセルのindexで指定。
        
    def add_check(self):
        #読み込まなくていい銘柄リスト
        self.checklist.extend([i for i in range(224)])
        self.checklist.extend([i for i in range(357,385)])
        self.checklist.extend(i for i in range(568,607))
        self.checklist.extend([i for i in range(626,655)])
        self.checklist.extend([i for i in range(747,765)])
        self.checklist.extend([i for i in range(3731,3754)])
        #self.org = stock_dowload.target_dowload(stock_code,start=start,end=end,frequency=frequency)
   
        
import mysql.connector
class handle_mysql():
    """テーブルの設計　(datas_for_dfy)
    CREATE TABLE `datas_for_day` (
  `companyname` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `data_open` float DEFAULT NULL,
  `data_high` float DEFAULT NULL,
  `data_low` float DEFAULT NULL,
  `data_close` float DEFAULT NULL,
  `data_volume` int DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  UNIQUE KEY `companyname` (`companyname`,`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """
    def __init__(self,setting):
        self.db = mysql.connector.connect(host='host',user='root',
                                              password='password',database='datas')
        self.corsor = self.db.cursor()
        self.setting = setting
        
    def write_datas(self,datas):
        """ディクトで作成されたdataをmysqlにインサートする関数。
        args datas(dict):{name:pandas.datafreme}"""
        for name,df in datas.items():
            for timestamp,data in df.iterrows():
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                query = 'INSERT INTO datas_for_day (companyname,data_open,data_high,data_low,data_close,data_volume,created_at)\
                VALUES("{}",{},{},{},{},{},"{}")'.format(name,data['open'],
                    data['high'],data['low'],data['close'],
                    data['volume'],timestamp)
            
            try:
                self.corsor.execute(query)
                self.db.commit()
                print('正常に挿入されました。')
                
            except Exception as e:
                print(timestamp)
                self.db.rollback()
                print('error',str(e))       
    
    def write_imgs(self,datas):
        '''imgseveされたbuf情報をimgsに保存します。
        こちらもデータから作成可能かつ可変性があり、保存には適していないと考えますが、現在のツールの起動時間短縮ため、
        apiインターバルの時間を活用して、作成してみようと思います。
        imgsサイズ　日足の7本とします。
        ※bufが''または""で囲んだ際にbuf自身が"",''を使用しており、エラーが起こります。対策を考えています。
        
        args imgs(dict):{name:(buf)}
        '''
        for name,buf in datas.items():
            timestamp = self.setting.end.strftime('%Y-%m-%d %H:%M:%S')
            query = "INSERT INTO imgs (companyname,img,created_at) VALUES(%s,%b,%s)"
            try:
                self.corsor.execute(query,(name,buf,timestamp))
                self.db.commit()
                print('正常に挿入されました。')
                
            except Exception as e:
                self.db.rollback()
                print('error',str(e))    
                    
    def load_datas(self,start=None,end=None):
        """databaseのdatasテーブルの情報を取得する関数
        return dict{companyname:pandas.dataframe}
        """
        if start == None:start = self.setting.start
        if end == None:end = self.setting.end 
        result = {}

        self.corsor.execute(('SELECT * FROM datas_for_day WHERE created_at >= "{}" and created_at <= "{}"'
                             ).format(start,end))
        for row in self.corsor:
            if row[0] not in result:
                result[row[0]] = pandas.DataFrame(columns=['open','high','low','close','volume','timestmap'])
            result[row[0]].loc[len(result[row[0]])] = (row[1],row[2],row[3],row[4],row[5],row[6])
            
        return result
    
    def load_imgs(self,created = None):
        '''databaseのimgsテーブルから情報を取得する関数
        args created(datatime or timestamp):作成日
        retrun dict{companyname:buf}'''
        result = {}
        if created == None:created = self.setting.start
        self.corsor.execute(('SELECT * FROM imgs WHERE created_at = "{}"').format(created))
        for name,buf,created_at in self.corsor:
            result[name] = buf
        
    def __del__(self):
        self.db.close()


def generator_datas(datas):
    for name,df in datas.items():
        yield name,df
    
def generator_imgs(imgs):
    for name, buf in imgs.items():
        yield name,buf
     
def load_img(day):
    with open('dict_for_hash{}'.format(str(day)),'rb') as f:
        d = pickle.load(f)
        for i,(name,value) in enumerate(d.items()):
            yield name,value

class DataProcessor(Comparer):
    """settingクラスのデータと入力データを比較し、スコアを保存するクラス
    args
    setting(classobject):created_setting
    datas(dict):{name:pd.datafreme}"""
    
    def __init__(self,setting,datas):
        super().__init__(setting.org_df)
        self.setting = setting
        self.datas = datas
        load = generator_datas(datas)
        self.database = {i:j for i,j in load}
        self.mode = ''
        self.ordinal = {}
             
    def exponential_function(self):
        #logging.debug('start')
        if len(self.score_rate):
            self.mode = 'exponential'
            self.ordinal = self.score_rate
            self.index = self.index_rate
        else:
            load = generator_datas(self.datas) 
            for name, df in load:
                super().rate_comparison(name,df)
                        
            return ("exponential", self.index_rate, self.score_rate)

    def encode_function(self):
        #logging.debug('start')
        if len(self.score_encode):
            self.mode = 'encode'
            self.ordinal = self.score_encode
            self.index = self.index_encode
        else:
            load = generator_datas(self.datas)
            for name, df in load:
                super().encode_comparison(name,df)
            return ("encode", self.index_encode, self.score_encode)
          
    def hash_function(self,imgs=None):
        #今後の課題、スケーリングしているので短期的な期間だと下落率や高騰率などを適切に評価できない。
        #logging.debug('start')
        if len(self.score_hash):
            self.mode = 'hash'
            self.ordinal = self.score_hash
            self.index = self.index_hash
        else:
            load = load_img(self.setting.created_at)
            for name, img in load:
                img = io.BytesIO(img)
                super().hash_comparison(name,img)
                        
            return ("hash", self.index_hash, self.score_hash)
        
    def get(self):
        self.ordinal_sorted = sorted(self.ordinal.items(), key=lambda x:x[1])
        mode = self.mode
        for name, score in self.ordinal_sorted:
            if mode != self.mode:
                raise ChangeError('modeをchangeする')
            if self.index[name][1] == None: #elseの処理だと[-7:]ができないため。
                result = self.database[name].iloc[-7:]
            else: 
                result = self.database[name].iloc[self.index[name][0] :self.index[name][1] + self.setting.forecast_period]
            yield name,result
            
    def updeta_processor(self,category):
        self.index_rate,self.score_rate = category['exponential'][0],category['exponential'][1]
        self.index_encode,self.score_encode = category['encode'][0],category['encode'][1]
        self.index_hash,self.score_hash = category['hash'][0],category['hash'][1]


class Menu(tk.Menu):
    """tkinterのメニューを作成するクラス"""
    def __init__(self,root,mune_fanc_evaluation):
        global database
        global canvas
        tk.Menu.__init__(self, root)
        # ファイルメニューの作成
        file_menu = tk.Menu(self, tearoff=0)
        file_menu.add_command(label="指数関数",command=mune_fanc_evaluation.on_exponential_function)
        file_menu.add_command(label="符号関数",command=mune_fanc_evaluation.on_encode_function)
        file_menu.add_command(label="ハッシュ関数",command=mune_fanc_evaluation.on_hash_function)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=root.quit)
        self.add_cascade(label="評価指標", menu=file_menu)  
        

class mune_fanc_Evaluation():
    def on_exponential_function(self):
        database.exponential_function()
        canvas.line()
        print("指数関数が選択されました")

    def on_encode_function(self):
        database.encode_function()
        canvas.line()
        print("符号関数が選択されました")

    def on_hash_function(self):
        database.hash_function()
        canvas.line()
        print("ハッシュ関数が選択されました")      


class Canvas(Plot):
    """tkinterのfigの作成、設定をおこなうクラス"""
    def __init__(self,parent,setting):
        global database
        
        self.setting = setting
        self.database = database.get()
        super().__init__()
        self.canvas = FigureCanvasTkAgg(figure=self.fig, master=parent)
        self.canvas_wid = self.canvas.get_tk_widget()
        self.canvas_wid.bind('<Button-3>',self.line)
        self.canvas_wid.bind('<Button-1>',self.line)
        self.canvas.draw()
        self.canvas_wid.pack(fill=tk.BOTH, expand=1)


    def candle(self,event=None):
        #この関数は現状は使うことはない。
        super().cla()
        super().candle_plot(self.setting.org_df)
        for i in range(5):
            
            try:
                name,df = next(database.get())   
                  
            except StopIteration:
                self.database = database.get()
                break
                         
            except ChangeError:
                self.database = database.get()
                name,df = next(self.database)
                
            except AttributeError:
                print('評価指標を設定してください')
                break
            
            super().candle_plot(df)
        self.canvas.draw()
        
    def line(self,event=None):
        super().cla()
        super().line_plot(self.setting.org_df)
        for i in range(5):
            
            try:
                name,df = next(self.database) 
                  
            except StopIteration:
                self.database = database.get()
                break
            
            except ChangeError:
                self.database = database.get()
                name,df = next(self.database)
            
            except AttributeError:
                print('評価指標を設定してください')
                return
            
            super().line_plot(df,name)

        self.canvas.draw()
    

def on_closing():
    root.destroy()
    plt.close()         
    
def worker():
    #並列処理関数
    global database
    
    category = {'exponential':(database.index_rate, database.score_rate),
     'encode':(database.index_encode, database.score_encode),
     'hash':(database.index_hash,database.score_hash)}
    
    tasks = [database.exponential_function, database.encode_function, database.hash_function]
    
    with multiprocessing.Pool(3) as p:
        results = [p.apply_async(task) for task in tasks]
        output = [result.get() for result in results]
        
        for category_valuse, index, score in output:
            category[category_valuse] = index,score

    database.updeta_processor(category)
       

class update(Plot):
    """databaseにdataを挿入、imgbinary―をfileにクラス"""
    def __init__(self,setting,days=-1):
        self.days = days
        self.setting = setting
        self.setting.add_check()
        mpl.use('Agg')
        super().__init__()
        self.plotseve_setting()
        self.checklist = []
        self.stockchart_for_dict = {}
        self.hash_for_dict = {}
        self.data_queue = queue.Queue()
        
    def threading_of_update(self):
        t1 = threading.Thread(target=self.producer,args=(self.days,))
        t2 = threading.Thread(target=self.consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        sql = handle_mysql(self.setting)
        sql.write_datas(self.stockchart_for_dict)
        self.hash_write(self.hash_for_dict)
        
    def producer(self,days):
        load = self.csv_load()
        for name, code in load:
            result_df = stock_dowload.yahoo(code,'week','day')
            if len(result_df):
                self.stockchart_for_dict[name] = result_df.iloc[days:]
                self.data_queue.put((name,result_df))
                time.sleep(3)
            else:
                continue
        self.data_queue.put((None,None))
    
    def consumer(self):
        while True:
            name,df = self.data_queue.get()
            if name == None:
                break
            buffer = self.seve_img(df)
            image_data = buffer.getvalue()
            self.hash_for_dict[name] = image_data
            
    def csv_load(self,strat=0):
        cnt = 0
        end = 4193
        df = pandas.read_excel("data.xls") 
        for self.i,self.j in df.iterrows():
            cnt += 1
            print(cnt,'/',end)
            if strat >= cnt or cnt in self.setting.checklist:
                continue
            else:
                yield self.j["銘柄名"],self.j["コード"]
                
    def hash_write(self,hs):        
        with open('dict_for_hash{}'.format(str(self.setting.end)),'wb') as f:
            pickle.dump(hs,f)
        
    
    def seve_img(self,df):
        df = df.reset_index()
        self.candle_plot(df)
        self.ax.patch.set_facecolor('black')
        buf = self.seve_plot()
        return buf
             
                
            
logging.basicConfig(level=logging.DEBUG,format='%(processName)s:%(message)s')

if __name__ == '__main__':
    #setting = create_setting(None)
    #up = update(setting)
    #up.threading_of_update()
    start = dt.date(2023,4,30)
    end = dt.date(2023,5,11)
    target = 3856
    org_df = stock_dowload.target_dowload(target)
    setting = create_setting(org_df,forecast_period=2)
    sql = handle_mysql(setting)
    datas = sql.load_datas()
    database = DataProcessor(setting,datas)
    worker()
    root = tk.Tk()
    mune_fanc_evaluation = mune_fanc_Evaluation()
    menu = Menu(root,mune_fanc_evaluation)
    canvas = Canvas(root,setting)
    canvas.candle_plot(org_df)
    root.config(menu=menu)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

