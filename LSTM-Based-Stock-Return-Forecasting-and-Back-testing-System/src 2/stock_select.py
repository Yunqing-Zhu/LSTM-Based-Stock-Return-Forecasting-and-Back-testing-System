from libs.stock_a import save_pandas_data, fetch_data, get_em_concept, get_em_concept_stocks
import akshare as ak
from datetime import datetime, timedelta
import threading
import queue
import sys
import pandas as pd
import numpy as np
import os
import talib as ta
from tqdm import tqdm
class QueueWriter:
    def __init__(self, out_queue):
        self.out_queue = out_queue

    def write(self, msg):
        if msg:
            self.out_queue.put(msg)

    def flush(self):
        pass  # No-op for compatibility

class CommandExecutor:
    cached_data = {}
    def __init__(self):
        self.commands = {}
        self.data_keys  = {}

    def register(self, name, func, data_key="default"):
        self.commands[name] = func
        self.data_keys[name] = data_key

    def execute(self, line):
        parts = line.strip().split()
        if not parts:
            print("No command entered.")
            return
        cmd, *args = parts
        if cmd in self.commands:
            result = self.commands[cmd](*args)
            if result is None:
                print(f"No data returned for command '{cmd}'.")
            elif isinstance(result, str):
                print(result)
            elif isinstance(result, bool):
                print(f"Command '{cmd}' executed with result: {result}")
            elif isinstance(result, pd.DataFrame):
                self.cached_data[self.data_keys[cmd]] = result
                print(f"Data cached to ['{self.data_keys[cmd]}'].")
        else:
            print(f"Unknown command: {cmd}")


executor = CommandExecutor()
trade_date = ak.tool_trade_date_hist_sina()['trade_date'].max()

def download_stock_codes():
    """下载股票代码列表"""
    stock_codes = ak.stock_zh_a_spot_em()[['代码', '名称']]
    stock_codes = stock_codes[
        ~stock_codes['代码'].str.contains('ST|退|sh688', na=False) &  # 排除科创板股票
        ~stock_codes['代码'].str.startswith('8')  # 排除北交所        
    ]
    stock_codes.to_csv("stock_a_data/stock_codes.csv", index=False, encoding='utf-8')
    print("Stock codes downloaded successfully.")
    return stock_codes

def load_stock_codes():
    """加载股票代码列表"""
    file_path = "stock_a_data/stock_codes.csv"
    if os.path.exists(file_path):
        stock_codes = pd.read_csv(file_path, encoding='utf-8')
        print("Stock codes loaded successfully.")
        return stock_codes
    else:
        print("No stock codes found. Please download first.")
        return None
    
def download_stock_data(stock_code='sh510330'):
    """下载股票数据并保存到 CSV 文件"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = datetime.now().replace(year=datetime.now().year - 2).strftime('%Y%m%d')
    df = fetch_data(stock_code, start_date, end_date)
    print(f"Data for {stock_code} from {start_date} to {end_date} downloaded successfully.")
    return df
    
def load_stock_data(stock_code='sh510330'):
    """加载股票数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = datetime.now().replace(year=datetime.now().year - 2).strftime('%Y%m%d')
    file_path = f"stock_a_data/{stock_code}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
        print(f"Data for {stock_code} loaded from {file_path}.")
        return df
    else:
        print(f"No data found for {stock_code} in the specified date range.")
        return None

def download_concept_data():
    """下载东方财富概念板块数据"""
    concept_data = get_em_concept()
    concept_data.to_csv("stock_a_data/东方财富板块.csv", index=False, encoding='utf-8')
    print("Concept data downloaded successfully.")
    

def download_concept_stocks(symbol="BK0918"):
    """下载特定概念板块成分股数据"""
    if not os.path.exists(f"stock_a_data/{symbol}_stocks.csv"):
        print(f"Downloading concept stocks for {symbol}...")
    else:
        print(f"Concept stocks for {symbol} already exist. Loading from cache.")
        concept_stocks = pd.read_csv(f"stock_a_data/{symbol}_stocks.csv", encoding='utf-8')
        return concept_stocks
    concept_stocks = get_em_concept_stocks(symbol)
    concept_stocks.to_csv(f"stock_a_data/{symbol}_stocks.csv", index=False, encoding='utf-8')
    print(f"Concept stocks for {symbol} downloaded successfully.")

def kdj_analysis():
    """计算 KDJ 指标"""
    stock_codes = executor.cached_data.get('stock_codes', None)
    if stock_codes is None:
        print("No stock codes available. Please download stock codes first.")
        return None
    for code in stock_codes['代码']:
        df = load_stock_data(code)
        if df is None:
            return None
        df['K'], df['D'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=9, slowk_period=3, slowd_period=3)
        df['J'] = 3 * df['K'] - 2 * df['D']
        df = df[['K', 'D', 'J']].dropna()
    
    return df

def evaluate_single_stock(code):
    try:      
        ROE_threshold = 3.5
        PE_threshold = 30     
        if os.path.exists(f"stock_a_data/fin/{code}_financial_data.csv"):
            df_fin = pd.read_csv(f"stock_a_data/fin/{code}_financial_data.csv", encoding='utf-8')
        else:
            df_fin = ak.stock_financial_analysis_indicator(symbol=code,start_year="2000")
            if df_fin.empty:
                #print(f"No financial data found for {code}.")
                return False
            else:
                df_fin.to_csv(f"stock_a_data/fin/{code}_financial_data.csv", index=False, encoding='utf-8')
        # 指标1：ROE > 14%
        roe = pd.to_numeric(df_fin['净资产收益率(%)'], errors='coerce').dropna()
        avg_roe = roe[-5:].mean()
        var1 = avg_roe > ROE_threshold
        if not var1:
            #print(f"ROE for {code} is below 14%: {avg_roe:.2f}%")
            return False

        # 指标2：PE 0 < pe < 30
        pe_df = ak.stock_a_lg_indicator(symbol=code)
        pe = pd.to_numeric(pe_df['pe'], errors='coerce').dropna().iloc[-1]
        var2 = 0 < pe < PE_threshold
        if not var2:
            #print(f"PE for {code} is not in the range (0, 30): {pe:.2f}")
            return False
        # 指标3：每股经营现金流 > 0
        cash_flow = pd.to_numeric(df_fin['每股经营性现金流(元)'], errors='coerce').dropna().iloc[0]
        var3 = cash_flow > 0
        if not var3:
            #print(f"Operating cash flow per share for {code} is not positive: {cash_flow:.2f}")
            return False
        # 指标4：最新净利润 > 过去5年最大值
        net_profit = pd.to_numeric(df_fin['扣除非经常性损益后的净利润(元)'], errors='coerce').dropna()
        latest = net_profit.iloc[0]
        max_5 = net_profit.iloc[1:6].max()
        var4 = latest > max_5
        if not var4:
            #print(f"Latest net profit for {code} is not greater than the maximum of the past 5 years: {latest:.2f} <= {max_5:.2f}")
            return False
        return var1 and var2 and var3 and var4
    except:
        return False

# 评估函数
def evaluate_stocks():
    try:
        codes = executor.cached_data.get('stock_codes', None)        

        if codes is None:
            print("No stock codes available. Please download stock codes first.")
            return False
        evaluated_stocks = pd.DataFrame(columns=['代码', '名称', '莫伦卡模型结论'])
        for code in tqdm(codes['代码'], desc="Evaluating stocks"):
            if evaluate_single_stock(code):
                stock_name = codes[codes['代码'] == code]['名称'].values[0]
                evaluated_stocks = evaluated_stocks.append({'代码': code, '名称': stock_name, '莫伦卡模型结论': '通过'}, ignore_index=True)
                #print(f"Stock {code} ({stock_name}) passed the evaluation criteria.")
            #else:
                #print(f"Stock {code} did not pass the evaluation criteria.")
        # 获取财务指标
        if evaluated_stocks.empty:
            print("No stocks passed the evaluation criteria.")
            return False
        evaluated_stocks.to_csv("stock_a_data/evaluated_stocks.csv", index=False, encoding='utf-8')
        return evaluated_stocks
    except:
        return False

# 3. CANSLIM 打分函数
def canslim_score(code: str):  
    # C 当季净利润增速
    #q_profit = ak.stock_profit_sheet_by_quarterly_em(symbol=code)
    # print(df_q.iloc[0][['净利润', '净利润同比增长率']])
    if code.startswith('60'):
        code = 'SH' + code
    if code.startswith('00'):
        code = 'SZ' + code
    if not code.startswith(('SH', 'SZ')):
        print(f"Invalid stock code format: {code}. Must start with 'SH' or 'SZ'.")
        return None
    print(f"Calculating CANSLIM score for {code}...")
    report_cached_file = f"stock_a_data/report/{code}_quarterly_profit.csv"
    if os.path.exists(report_cached_file):
        q_profit = pd.read_csv(report_cached_file, encoding='utf-8')
    else:
        q_profit = ak.stock_profit_sheet_by_report_em(symbol=code)
        if q_profit.empty:
            print(f"No quarterly profit data found for {code}.")
            return None
        q_profit.to_csv(f"stock_a_data/report/{code}_quarterly_profit.csv", index=False, encoding='utf-8')
    q_profit = q_profit.iloc[0]
    # TOTAL_PROFIT_BALANCE_YOY 净利润同比增长率
    q_np_growth = float(q_profit['TOTAL_PROFIT_BALANCE_YOY']) if q_profit['TOTAL_PROFIT_BALANCE_YOY'] != '--' else np.nan

    # A 连续 3 年净利润复合增速
    report_cached_file = f"stock_a_data/report/{code}_yearly_profit.csv"
    if os.path.exists(report_cached_file):
        yr_profit = pd.read_csv(report_cached_file, encoding='utf-8')
    else:
        yr_profit = ak.stock_profit_sheet_by_yearly_em(symbol=code).head(3)
        if yr_profit.empty:
            print(f"No yearly profit data found for {code}.")
            return None
        yr_profit.to_csv(f"stock_a_data/report/{code}_yearly_profit.csv", index=False, encoding='utf-8')
    # TOTAL_PROFIT 净利润
    yr_np = pd.to_numeric(yr_profit['TOTAL_PROFIT'], errors='coerce')
    if len(yr_np.dropna()) < 2: 
        return None
    ann_growth = (yr_np.iloc[0] / yr_np.iloc[-1]) ** (1 / (len(yr_np) - 1)) - 1

    # N 新产品/管理层/高送专（简化为营收增速>30%） 
    # TOTAL_OPERATE_INCOME_YOY 营业总收入同比增长率
    q_revenue_growth = float(q_profit['TOTAL_OPERATE_INCOME_YOY']) if q_profit['TOTAL_OPERATE_INCOME_YOY'] != '--' else np.nan

    # S 供给股本小（流通市值<300亿）
    report_cached_file = f"stock_a_data/basic/stock_zh_a_spot.csv"
    if os.path.exists(report_cached_file):
        basic = pd.read_csv(report_cached_file, encoding='utf-8')
    else:
        basic = ak.stock_zh_a_spot()
        if basic.empty:
            print(f"No basic data found for {code}.")
            return None
        basic.to_csv(f"stock_a_data/basic/stock_zh_a_spot.csv", index=False, encoding='utf-8')
    
    mkt_cap = basic.loc[basic['代码'] == code, '总市值'].iloc[0]
    if pd.isna(mkt_cap): 
        return None
    mkt_cap = float(mkt_cap)


    # I 机构持股：北向资金持股
    report_cached_file = f"stock_a_data/north/stock_hsgt_hold_stock_em.csv"
    if os.path.exists(report_cached_file):
        north = pd.read_csv(report_cached_file, encoding='utf-8')
    else:
        north = ak.stock_hsgt_hold_stock_em()
        if north.empty:
            print(f"No northbound holdings data found for {code}.")
            return None
        north.to_csv(f"stock_a_data/north/stock_hsgt_hold_stock_em.csv", index=False, encoding='utf-8')
    hsgt = north[north['股票代码'] == code]['持股数量'].sum() > 0

    # M 大盘趋势（沪深300 在年线之上）
    param_start_date = (datetime.today()-timedelta(days=300)).strftime('%Y%m%d')
    report_cached_file = f"stock_a_data/index/hs300_{param_start_date}.csv"
    if os.path.exists(report_cached_file):
        hs300 = pd.read_csv(report_cached_file, encoding='utf-8')
    else:
        hs300 = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=param_start_date)
        if hs300.empty:
            print("No historical data found for HS300.")
            return None
        hs300.to_csv(f"stock_a_data/index/hs300_{param_start_date}.csv", index=False, encoding='utf-8')
        #hs300 = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=(datetime.today()-timedelta(days=300)).strftime('%Y%m%d'))
    hs300['ma250'] = hs300['收盘'].rolling(250).mean()
    market_ok = hs300.iloc[-1]['收盘'] > hs300.iloc[-1]['ma250']

    score = 0
    if q_np_growth > 25: score += 1
    if ann_growth > 15: score += 1
    if q_revenue_growth > 30: score += 1
    if mkt_cap < 300: score += 1
    if hsgt: score += 1
    if market_ok: score += 1
    return score


# 4. 计算 250 日 RPS
def calc_rps(code_list):
    end = datetime.today()
    start = end - timedelta(days=300)
    prices = {}
    for code in tqdm(code_list, desc="下载行情"):
        try:
            prices[code] = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.strftime('%Y%m%d'))['收盘']
        except:
            continue
    price_df = pd.DataFrame(prices).ffill()
    ret250 = price_df.pct_change(250).iloc[-1]
    rps = ret250.rank(pct=True) * 100
    return rps

def calism_rps():
    stock_list = executor.cached_data.get('stock_codes', None)
    if stock_list is None:
        print("No stock codes available. Please download stock codes first.")
        return None

    # 先算 CANSLIM
    stock_list['CANSLIM'] = [canslim_score(str(c)) for c in tqdm(stock_list['代码'], desc="CANSLIM")]
    stock_list = stock_list[stock_list['CANSLIM'] >= 4]  # 至少满足4条
    if stock_list.empty:
        print("No stocks passed the CANSLIM criteria.")
        return None
    # 再算 RPS
    rps_series = calc_rps(stock_list['代码'].tolist())
    stock_list = stock_list.set_index('代码').join(rps_series.rename('RPS')).dropna()
    final = stock_list[stock_list['RPS'] >= 85]  # RPS 前15%

    # 6. 结果
    final = final.sort_values('RPS', ascending=False)
    final['weight'] = 1 / len(final)  # 等权
    print('\n==== CANSLIM + 动量组合 ====')
    print(final[['名称', 'CANSLIM', 'RPS', 'weight']])

def help(name="help"):
    lines = [
        "Available commands:",
        "  download_stock_data <stock_code> <start_date> <end_date> - Download stock data",
        "  download_concept_data - Download concept board data",
        "  download_concept_stocks <symbol> - Download stocks in a specific concept board",
        "  help - Show this help message"
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    # stock_profit_sheet_by_report_em_df = ak.stock_profit_sheet_by_report_em(symbol="SH600519")
    # print(stock_profit_sheet_by_report_em_df)
    executor.register("download_concept_data", download_concept_data, data_key="concept_data")
    executor.register("download_stock_data", download_stock_data, data_key="stock_data")
    executor.register("download_concept_stocks", download_concept_stocks, data_key="stock_codes")
    executor.register("kdj_analysis", kdj_analysis, data_key="stock_data")
    executor.register("evaluate_stocks", evaluate_stocks, data_key="evaluate_stocks")
    executor.register("download_stock_codes", download_stock_codes, data_key="stock_codes")
    executor.register("load_stock_codes", load_stock_codes, data_key="stock_codes")
    executor.register("load_stock_data", load_stock_data, data_key="stock_data")
    executor.register("canslim_score", canslim_score, data_key="canslim_scores")
    executor.register("calism_rps", calism_rps, data_key="calism_rps")
    executor.register("help", help)
    lines = ["download_concept_stocks BK0490",
             "calism_rps"]
    for line in lines:
        executor.execute(line)
    print("Enter commands (type 'exit' to quit):")
    while True:
        line = input("> ")
        if line.strip() == "exit":
            break
        executor.execute(line)
