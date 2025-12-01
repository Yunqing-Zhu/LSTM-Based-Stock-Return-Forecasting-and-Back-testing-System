import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import libs.stock_a as stock_a
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler

# 获取股票数据
ticker = 'AAPL'
data = stock_a.fetch_data(stock_code='sh600372', start_date='20220701', end_date='20240630')
# df 必须包含 'close'
# data['regime'] = 0
# data['sma_fast'] = talib.SMA(data['close'], timeperiod=20)
# data['sma_slow'] = talib.SMA(data['close'], timeperiod=120)
# data.loc[data['sma_fast'] > data['sma_slow'] * 1.02, 'regime'] = 2   # 牛市
# data.loc[data['sma_fast'] < data['sma_slow'] * 0.98, 'regime'] = 0   # 熊市


# 过去 60 天收益率 = 今天 / 60 天前
ret60 = data['close'] / data['close'].shift(60) - 1

# 过去 60 天波动率
vol60 = data['close'].pct_change().rolling(window=60, min_periods=60).std()

data = data.iloc[60:]
# 定义分位点
data['ret_q']  = pd.qcut(ret60, 3, labels=['bear','neut','bull'])
data['vol_q']  = pd.qcut(vol60, 2, labels=['low','high'])

data['regime'] = data['ret_q'].astype(str) + '_' + data['vol_q'].astype(str)
regime_map = {'bear_low': 0, 'bear_high': 1,
              'neut_low': 2, 'neut_high': 3,
              'bull_low': 4, 'bull_high': 5}
data['regime'] = data['regime'].map(regime_map).astype('float64')
# 得到 6 个状态：bear_low, bear_high, neut_low, ...
# 可再映射成 0~5
#data['regime'] = pd.Categorical(data['regime']).codes

# 创建特征和标签
data['Return'] = data['close'].pct_change()
data.dropna(inplace=True)
X = data[[ 'open', 'high', 'low', 'close', 'volume']].values
y = data['Return'].values
print(len(X))
# 数据标准化
scaler = StandardScaler()
X_scaled = np.c_[data['regime'], scaler.fit_transform(X)]

print(X_scaled)
# 将数据转换为 PyTorch 张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size=6, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 180
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor.unsqueeze(1))
    loss = criterion(outputs.squeeze(), y_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 预测
model.eval()
with torch.no_grad():
    predictions = model(X_tensor.unsqueeze(1)).squeeze().numpy()



# 保存阈值网格
buy_range  = np.quantile(predictions, np.arange(0.55, 0.96, 0.05))   # 55%~95% 分位
sell_range = np.quantile(predictions, np.arange(0.05, 0.46, 0.05))   # 5%~45% 分位
np.savez('thresholds.npz', buy=buy_range, sell=sell_range)

import matplotlib.pyplot as plt
plt.hist(predictions, bins=50, edgecolor='k')
plt.xlabel('predicted return'); plt.ylabel('count'); plt.show()

# 将预测结果添加到数据框中
data['Predicted_Return'] = predictions

# 输出结果
print(data[['close', 'Return', 'Predicted_Return']])

torch.save({
    'state_dict': model.state_dict(),
    'scaler':     scaler,
    'input_size': 6,
    'hidden_size': 50
}, 'models/lstm_model.pth')