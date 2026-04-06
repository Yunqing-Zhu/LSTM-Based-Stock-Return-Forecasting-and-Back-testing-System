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
from torch.utils.data import Dataset, DataLoader

# 获取股票数据
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
regime_map = {'bear_low': 0, 'bear_high': 1,
              'neut_low': 2, 'neut_high': 3,
              'bull_low': 4, 'bull_high': 5}
data['regime'] = data['regime'].map(regime_map).astype('float64')

# 创建特征和标签
data['Return'] = data['close'].pct_change()
data.dropna(inplace=True)

# 特征工程
feature_cols = ['open', 'high', 'low', 'close', 'volume']
X_raw = data[feature_cols].values
y = data['Return'].values
regime = data['regime'].values

print(f"总样本数: {len(X_raw)}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 将 regime 添加为特征
X_scaled = np.c_[regime, X_scaled]  # 形状: (N, 6)

# ========== 构建 LSTM 序列样本 ==========
SEQ_LEN = 60  # 用过去60天预测明天

def create_sequences(X, y, seq_len):
    """创建序列样本: 过去 seq_len 天 -> 预测下一天"""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])  # 过去 seq_len 天
        y_seq.append(y[i])            # 第 i 天的收益（明天相对今天）
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)
print(f"序列样本数: {len(X_seq)}, 序列形状: {X_seq.shape}")

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X_seq, dtype=torch.float32)  # (N, 60, 6)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)  # (N,)

# 创建 DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = StockDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后时刻的隐藏状态
        return out

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = LSTMModel(input_size=6, hidden_size=50, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 500
print("开始训练...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

# 预测
model.eval()
with torch.no_grad():
    X_pred = X_tensor.to(device)
    predictions = model(X_pred).squeeze().cpu().numpy()

print(f"\n预测结果统计:")
print(f"  最小值: {predictions.min():.6f}")
print(f"  最大值: {predictions.max():.6f}")
print(f"  均值: {predictions.mean():.6f}")
print(f"  标准差: {predictions.std():.6f}")

# 保存阈值网格
os.makedirs('models', exist_ok=True)

buy_range  = np.quantile(predictions, np.arange(0.55, 0.96, 0.05))
sell_range = np.quantile(predictions, np.arange(0.05, 0.46, 0.05))
np.savez('models/thresholds.npz', buy=buy_range, sell=sell_range)

# 绘制预测分布
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(predictions, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('predicted return')
plt.ylabel('count')
plt.title('Prediction Distribution')
plt.axvline(0, color='r', linestyle='--', label='zero')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_seq, label='actual', alpha=0.7)
plt.plot(predictions, label='predicted', alpha=0.7)
plt.xlabel('time')
plt.ylabel('return')
plt.title('Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.savefig('prediction_result.png', dpi=150)
plt.show()

# 保存模型
torch.save({
    'state_dict': model.state_dict(),
    'scaler': scaler,
    'input_size': 6,
    'hidden_size': 50,
    'seq_len': SEQ_LEN
}, 'models/lstm_model.pth')

print("\n模型已保存到 models/lstm_model.pth")
