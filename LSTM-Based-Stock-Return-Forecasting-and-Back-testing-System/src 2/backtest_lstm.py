# backtest_lstm.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import backtrader as bt, torch, pandas as pd, numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import libs.stock_a as stock_a

SEQ_LEN = 60               # 用最近 60 根 K 线做预测

# ---------- 1) 加载 PyTorch 模型 ----------
ckpt   = torch.load('models/lstm_model.pth', map_location='cpu')
scaler = ckpt['scaler']

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

net = LSTMModel(ckpt['input_size'], ckpt['hidden_size'])
net.load_state_dict(ckpt['state_dict'])
net.eval()

# ---------- 2) Backtrader 策略 ----------
class LSTMStrategy(bt.Strategy):
    params = dict(seq_len=SEQ_LEN,
                  buy_thresh=0.015,   # 预测收益 > 0.15% 时买入
                  sell_thresh=-0)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.volume    = self.datas[0].volume
        self.order     = None
        self.prices    = []          # 缓存收盘价
        self.volumes   = []          # 缓存成交量

    def next(self):
        # 1) 缓存
        self.prices.append(self.dataclose[0])
        self.volumes.append(self.volume[0])
        if len(self.prices) < self.p.seq_len:
            return
        if len(self.prices) > self.p.seq_len:
            self.prices.pop(0)
            self.volumes.pop(0)
        if self.data.regime[-1] <= 1:
            print("熊市不操作")
            return
        # 打印当前仓位情况
        print(f"当前仓位：{self.position.size}， 仓位价格：{self.position.price}")

        # 2) 构造 60×5 的 OHLCV
        ohlcv = np.column_stack([
            [self.data.open[-i]   for i in reversed(range(self.p.seq_len))],
            [self.data.high[-i]   for i in reversed(range(self.p.seq_len))],
            [self.data.low[-i]    for i in reversed(range(self.p.seq_len))],
            [self.data.close[-i]  for i in reversed(range(self.p.seq_len))],
            [self.data.volume[-i] for i in reversed(range(self.p.seq_len))]
        ])  # shape = (60, 5)
        # 3) 归一化 & 预测
        ohlcv_scaled = np.c_[[self.data.regime[-i]  for i in reversed(range(self.p.seq_len))], scaler.transform(ohlcv)[-self.p.seq_len:]]  # 仍为 60×5
   
        x = torch.tensor(ohlcv_scaled, dtype=torch.float32).unsqueeze(0)  # (1,60,5)

        with torch.no_grad():
            pred_ret = net(x).item()
        
        #print(f"{self.data.datetime.date(0)}  pred_ret={pred_ret:.5f}")
        # 4) 交易逻辑
        if not self.position and pred_ret > self.p.buy_thresh:
            print("买入信号",self.p.buy_thresh)
            self.order = self.buy(size=400)
        elif self.position.price * 1.05 < self.data.close[0] and self.position.size > 0:
            print(f"超过3%盈利，见好就收，卖出仓位：{self.position.size}，仓位价格：{self.position.price} 当前价格:{self.data.close[0]}")
            self.order = self.sell(size = self.position.size)
        elif self.position.price * 0.97 > self.data.close[0] and self.position.size > 0:
            print(f"超过3%亏损，及时止损卖出，卖出仓位：{self.position.size}，仓位价格：{self.position.price} 当前价格:{self.data.close[0]}")
            self.order = self.sell(size = self.position.size)
        elif self.position and pred_ret < self.p.sell_thresh:
            print("sell")
            self.order = self.sell(size = self.position.size)
        

class PandasDataEx(bt.feeds.PandasData):
    lines = ('regime',)
    params = (('regime', 5),)    # regime 在 DataFrame 第 7 列（0 开始）

# ---------- 3) 回测引擎 ----------
if __name__ == '__main__':
    thresholds = np.load('thresholds.npz')
    #buy_range  = thresholds['buy']
    #sell_range = thresholds['sell']

    cerebro = bt.Cerebro(optreturn=False, maxcpus=1)


    data = stock_a.fetch_data(stock_code='sh600372', start_date='20220701', end_date='20240630')


    # 过去 60 天收益率 = 今天 / 60 天前
    ret60 = data['close'] / data['close'].shift(60) - 1

    # 过去 60 天波动率
    vol60 = data['close'].pct_change().rolling(window=60, min_periods=60).std()

    data = data.iloc[60:]

    # 定义分位点
    data['ret_q']  = pd.qcut(ret60, 3, labels=['bear','neut','bull'])
    data['vol_q']  = pd.qcut(vol60, 2, labels=['low','high'])

    data['regime'] = data['ret_q'].astype(str) + '_' + data['vol_q'].astype(str)
    # 得到 6 个状态：bear_low, bear_high, neut_low, ...
    # 可再映射成 0~5
    regime_map = {'bear_low': 0, 'bear_high': 1,
              'neut_low': 2, 'neut_high': 3,
              'bull_low': 4, 'bull_high': 5}
    data['regime'] = data['regime'].map(regime_map).astype('float64')

    data = data.dropna(subset=['regime'])      # 关键：把 NaN 行扔掉
    cols = [c for c in data.columns if c != 'regime'] + ['regime']
    data = data[cols]
    del data['ret_q']
    del data['vol_q']
    print(data)
    #data['regime'] = pd.Categorical(data['regime']).codes
    feed = PandasDataEx(dataname=data,
                        fromdate=data.index[0],
                        todate=data.index[-1])
    #feed = bt.feeds.PandasData(dataname=data, fromdate=data.index[0], todate=data.index[-1])
    cerebro.adddata(feed)
    cerebro.addstrategy(LSTMStrategy)
    
    # 用 optstrategy 代替 addstrategy
    # thresh_grid = np.linspace(0.000, 0.020, 21)   # 21 个点
    # cerebro.optstrategy(
    #     LSTMStrategy,
    #     buy_thresh=thresh_grid,
    #     sell_thresh=[-t for t in thresh_grid]
    # )
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='ret')

    print('Start Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]

    # best = max(results,
    #         key=lambda r: r[0].analyzers.sharpe.get_analysis().get('sharperatio', -np.inf))
    # print('Best buy_thresh  : %.4f' % best[0].params.buy_thresh)
    # print('Best sell_thresh : %.4f' % best[0].params.sell_thresh)
    # print('Best Sharpe      : %.2f' % best[0].analyzers.sharpe.get_analysis().get('sharperatio', 0.0))
    print('End   Portfolio Value: %.2f' % cerebro.broker.getvalue())
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    print('Sharpe Ratio : %.2f' % (sharpe if sharpe is not None else 0.0))
    print('Max Drawdown : %.2f %%' % strat.analyzers.dd.get_analysis().max.drawdown)
    cerebro.plot(style='candlestick')