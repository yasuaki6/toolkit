import numpy as np
import matplotlib.pyplot as plt
import pandas


#カルキュラムに沿ったaiを作成していきます。
    
#回帰モデルは誤差に二乗和誤差、活性化関数に恒等関数を使用します。
#出力層
class regression_outputLayer:
    def __init__(self, n_upper, n):
        '''args:
        n_upper,上の層の数
        n,出力層の数'''
        #重み、バイヤスを生成。
        wb_widht = 0.01
        self.w = wb_widht * np.random.randn(n_upper,n)
        self.b = wb_widht * np.random.randn(n)
    
    def forward(self,x): #順伝播
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u #恒等関数
    
    def backward(self,t):  #逆伝播
        delta = self.y - t

        self.grad_w  = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta, axis=0) #パッチサイズに対応するためsum関数を使用。
        
        self.grad_x = np.dot(delta,self.w.T)
        
    def updata(self,eta): #重みとバイヤスの更新
        '''args eta:学習係数'''
        
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        
        
#分類モデルは誤差に二乗和誤差、活性化関数にソフトマックス関数を使用します。        
class classification_outputLayer:
    def __init__(self, n_upper, n):
        #重み、バイヤスを生成。
        wb_widht = 0.01
        self.w = wb_widht * np.random.randn(n_upper,n)
        self.b = wb_widht * np.random.randn(n)
    
    def forward(self,x): #順伝播
        self.x = x
        u = np.dot(x, self.w) + self.b
        
        #出力層の合計値を分母、出力値を分子にして合計値が一になるようにし、その値の符号に関わらず非負の値にするために、exp関数を使用している。
        #またバッチに対応させるため、次元を保持させます。
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True) #ソフトマックス関数
        
    def backward(self,t):  #逆伝播
        delta = self.y - t

        self.grad_w  = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta,self.w.T)
        
    def updata(self,eta): #重みとバイヤスの更新
        
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        

class MiddleLayer:
    def __init__(self,n_upper,n):
        wb_widht = 0.01
        self.w = wb_widht * np.random.randn(n_upper,n)
        self.b = wb_widht * np.random.randn(n)
    
    def forward(self,x): #順伝播
        self.x = x
        u = np.dot(x, self.w) + self.b
        
        #活性化関数にシグモイド関数を使用します。
        self.y = 1/(1+np.exp(-u)) 
        
    def backward(self,grad_y):  #逆伝播
        delta = grad_y * (1-self.y) * self.y #シグモイド関数の微分の参考url https://qiita.com/yosshi4486/items/d111272edeba0984cef2#%E5%8F%82%E8%80%83
        
        self.grad_w  = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta,axis=0)
        
        self.grad_x = np.dot(delta,self.w.T)
        
    def updata(self,eta): 
        
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

#回帰の実装
class regression(regression_outputLayer,MiddleLayer):
    def __init__(self) -> None:
        #学習データ　
        #入力にlinspaceで等間隔に値を生成し、ある関数の出力を学ばせます。今回はサイン関数を学ばせていこうと思います。
        self.input_data = np.linspace(-np.pi,np.pi) #入力
        self.correct_data = np.sin(self.input_data)  # 正解
        self.n_data = len(self.input_data)

        #ニューロンの設定します。
        #層の数
        n_in = 1 
        n_mid = 3
        n_out = 1


        self.eta = 0.1 
        self.epoch = 2001
        self.interval = 100
        self.middle_layer = MiddleLayer(n_in, n_mid)
        self.output_layer = regression_outputLayer(n_mid,n_out)
    
    def start(self):
        for i in range(self.epoch):
            
            #インデックスをシャッフル関数でシャッフルし、適当な順番で与える。
            index_random = np.arange(self.n_data) 
            np.random.shuffle(index_random)
            
            # 結果の表示
            total_error = 0
            plot_x = []
            plot_y = []
            
            for idx in index_random:
                x = self.input_data[idx:idx+1]
                t = self.correct_data[idx:idx+1]
                
                #順伝播
                self.middle_layer.forward(x.reshape(1,1))
                self.output_layer.forward(self.middle_layer.y)
                
                #逆伝播
                self.output_layer.backward(t)
                self.middle_layer.backward(self.output_layer.grad_x)
                
                #重みとバイヤスの更新
                self.middle_layer.updata(self.eta)
                self.output_layer.updata(self.eta)
                
                if i%self.interval == 0:
                    
                    y = self.output_layer.y.reshape(-1)
                    
                    total_error += 1.0/2.0*np.sum(np.square(y-t))
                    
                    plot_x.append(x)
                    plot_y.append(y)
                
            if i % self.interval == 0:
                plt.plot(self.input_data, self.correct_data, linestyle='dashed')
                plt.scatter(plot_x,plot_y,marker='+')
                plt.show()
                
                print('Epoch' + str(i) + '/' + str(self.epoch), 'Error' + str(total_error/self.n_data))
                
#test = regression()
#test.start()    
#分類の実装            
class classification(MiddleLayer,classification_outputLayer):
    def __init__(self):
        #ニューロンの設定
        n_in = 2
        n_mid = 6
        n_out = 2

        self.eta = 0.1
        self.epoch = 101 
        self.interval = 10
        
        self.middle_layer = MiddleLayer(n_in,n_mid)
        self.output_layer = classification_outputLayer(n_mid,n_out)
        
        #入力座標がsin関数より大きい値か小さい値か分類していきます。
        #座標
        self.X = np.arange(-1.0,1.1,0.1)
        self.Y = np.arange(-1.0,1.1,0.1)
        
        #入力データ、正解データの作成
        input_data = []
        self.correct_data = []
        
        for x in self.X:
            for y in self.Y:
                input_data.append([x,y])
                
                if y < np.sin(np.pi * x): 
                    self.correct_data.append([0,1])
                else:
                    self.correct_data.append([1,0])
    

        self.input_data = np.array(input_data)
        self.correct_data = np.array(self.correct_data)
        self.n_data = len(self.correct_data) 
    
    def start(self):
        sin_data = np.sin(np.pi * self.X)
        for i in range(self.epoch):
            
            #インデックスのシャッフル
            index_random = np.arange(self.n_data)
            np.random.shuffle(index_random)
            
            #結果の表示用
            self.total_error = 0
            x_1 = []
            y_1 = []
            x_2 = []
            y_2 = []
            
            for idx in index_random:
                x = self.input_data[idx]
                t = self.correct_data[idx]
                
                #準伝播
                self.middle_layer.forward(x.reshape(1,2))    
                self.output_layer.forward(self.middle_layer.y)
                
                #逆伝播
                self.output_layer.backward(t.reshape(1,2))
                self.middle_layer.backward(self.output_layer.grad_x)

                #重みとバイヤスの更新
                self.middle_layer.updata(self.eta)
                self.output_layer.updata(self.eta)
                
                if i % self.interval == 0:
                    
                    y = self.output_layer.y.reshape(-1)
                    
                    self.total_error += np.sum(t * np.log(y + 1e-7))
                    
                    if y[0] > y[1]:
                        x_1.append(x[0])
                        y_1.append(x[1])
                    else:
                        x_2.append(x[0])
                        y_2.append(x[1])
            
            if i % self.interval == 0:
                
                plt.plot(self.X, sin_data, linestyle='dashed')
                plt.scatter(x_1,y_1,marker='+')
                plt.scatter(x_2,y_2,marker='x')
                plt.show()
                
                print('Epoch:' + str(i) + '/' + str(self.epoch),'Error' + str(self.total_error/self.n_data))
            
            
test = classification()
test.start()