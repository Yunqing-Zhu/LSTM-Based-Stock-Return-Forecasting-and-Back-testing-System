import akshare as ak
from datetime import datetime
import pandas as pd

def save_pandas_data(df, file_path):
    df.to_csv(file_path, index=True)
# =============================================
# 1. 使用 akshare 下载历史数据
# =============================================
def fetch_data(stock_code='sh510330', start_date='20200101', end_date='20241231'):
    file_path = f"stock_a_data/history/{stock_code}_{start_date}_{end_date}.csv"
    try:
        df = pd.read_csv(file_path)
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        #df['date'] = pd.to_datetime(df['date']).dt.date
        print("load data from cache")
        return df
    except FileNotFoundError:
        pass
    df = ak.fund_etf_hist_sina(symbol=stock_code)
    df['date'] = pd.to_datetime(df['date']).dt.date
    start_date_dt = datetime.strptime(start_date, "%Y%m%d").date()
    end_date_dt = datetime.strptime(end_date, "%Y%m%d").date()
    # 筛选日期范围
    df = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
    
    # 重命名列名以适应 backtrader 的要求
    df.rename(columns={
        'date': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }, inplace=True)
    
    # 设置索引
    df.set_index('datetime', inplace=True)
    
    # 转换为 datetime 类型
    df.index = pd.to_datetime(df.index)
    save_pandas_data(df, file_path)
    return df

def get_em_concept():
    """获取东方财富概念板块数据"""
    # 获取概念板块行情
    concept_quotes =ak.stock_board_concept_name_em()
    return concept_quotes

def get_em_concept_stocks(symbol="BK0918"):
    # 获取特定概念板块成分股
    concept_stocks = ak.stock_board_concept_cons_em(symbol=symbol)
    return concept_stocks