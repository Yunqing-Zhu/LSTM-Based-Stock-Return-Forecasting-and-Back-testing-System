import akshare as ak
import pandas as pd
from datetime import datetime

def get_em_concept():
    """获取东方财富概念板块数据"""
    # 获取概念板块行情
    concept_quotes =ak.stock_board_concept_name_em()
    print("东方财富概念板块行情:")
    print(concept_quotes)
    # 获取特定概念板块成分股
    concept_stocks = ak.stock_board_concept_cons_em(symbol="BK0918")
    print("\n车联网概念股:")
    print(concept_stocks)
#调用示例
get_em_concept()
