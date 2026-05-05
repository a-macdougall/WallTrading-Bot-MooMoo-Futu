"""
By: LukeLab
Created on 09/20/2023
Version: 1.0
Last Modified: 09/27/2023

Major Updated: 04/04/2024, decision and order function furnish
Still in testing

updated: 04/09/2024, output formatting

# updated: 11/17/2024, final version for open source only
# Version 2.0
# for more info, please visit: https://www.patreon.com/LookAtWallStreet

"""
import numpy as np
import yfinance as yf
from moomoo import *
import math
import time
# finbert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# marketaux
import requests
from env.NewSecret import marketaux_api_key

from strategy.Strategy import Strategy
import pandas as pd
# from ta.trend import SMAIndicator
# import pandas_ta as pta
from utils.dataIO import read_json_file, write_json_file, logging_info, get_current_time
from utils.time_tool_new_york import is_market_hours
from typing import Dict

#import streamlit as st
#import matplotlib.pyplot as plt
#import plotly.figure_factory as ff
#import plotly.express as px

class NewStrategy(Strategy):
    """
    This is an example strategy class, you can define your own strategy here.
    """

    def __init__(self, trader):
        super().__init__(trader)
        self.max_buy_value = None
        self.strategy_name = "New_Trading_Bot_Strategy"

        # streamlit graph shown at http://localhost:8501

        # Initialize an empty DataFrame
        #data = pd.DataFrame(columns=["Time", "Value"])
        #print("graph plot by steamlit is running on http://localhost:8501")

        """⬇️⬇️⬇️ Strategy Settings ⬇️⬇️⬇️"""

        # self.stock_trading_list_by_sector = {
        #     #'Technology' : ['AAPL'],
        #     'Technology' : ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'CRM', 'ORCL'],
        #     'Financial Services': ['JPM', 'BAC', 'WFC', 'V', 'MA'],
        #     'Healthcare & Pharmaceuticals': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY'],
        #     'Consumer Goods & Retail': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX'],
        #     'Industrial & Manufacturing': ['BA', 'CAT', 'GE', 'MMM'], #'HON',
        #     'Energy & Utilities': ['XOM', 'CVX', 'NEE'],
        #     'Telecommunications': ['VZ', 'T'],
        # }

        self.stock_trading_list_by_sector = {
            'Technology': ['NVDA', 'MSFT', 'AMZN', 'GOOGL', 'AVGO'],
            'Financial Services': ['JPM', 'V', 'MA', 'GS', 'BAC'],
            'Healthcare & Pharmaceuticals': ['LLY', 'UNH', 'NVO', 'ABBV', 'MRK', 'PFE'],
            'Consumer Goods & Retail': ['WMT', 'HD', 'COST', 'MCD', 'PG', 'SBUX', 'NKE'],
            'Industrial & Manufacturing': ['GEV', 'GE', 'LMT', 'RTX', 'CAT'],
            'Energy & Utilities': ['VST', 'CEG', 'XOM', 'CVX', 'NEE'],
            'Telecommunications': ['META', 'TMUS', 'T', 'VZ', 'CMCSA']
        }

        self.stock_trading_list = [stock for stocks in self.stock_trading_list_by_sector.values() for stock in stocks]

        self.trading_confirmation = True    # True to enable trading confirmation

        # please add any other settings here based on your strategy

        # --- FinBERT init ---
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.finbert_model.to(self.device)

        if self.device.type == "cuda":
            self.finbert_model = self.finbert_model.half()

        self.finbert_model.eval()

        print(f"FinBERT running on: {self.device}")

        # --- Marketaux API ---
        self.marketaux_api_key = marketaux_api_key

        # --- News cache ---
        self.news_cache = {}

        """⬆️⬆️⬆️ Strategy Settings ⬆️⬆️⬆️"""

        print(f"Strategy {self.strategy_name} initialized...")

    def strategy_decision(self):
        print("Strategy Decision running...")
        """ ⬇️⬇️⬇️ Simple Example Strategy starts here ⬇️⬇️⬇️"""

        # check the current buying power first
        acct_ret, acct_info = self.trader.get_account_info()
        if acct_ret != RET_OK:
            print('Trader: Get Account Info failed: ', acct_info)
            return False
        current_cash = acct_info['cash']

        # check current position
        pos_ret, position_data = self.trader.get_positions()
        if pos_ret != RET_OK:
            return False

        current_portfolio = current_cash

        # for stock in position_data:
        #     qty = position_data[stock]['qty']
        #     if qty > 0:
        #         current_portfolio = current_portfolio + position_data[stock]['market_val']

        for stock, pos in position_data.items():
            qty = pos.get("qty", 0)
            if qty > 0:
                price = pos.get("market_price", pos.get("cost_price", 0))
                current_portfolio += qty * price

        # max buy value 20% of total portfolio
        # self.max_buy_value = current_cash * 0.2
        self.max_buy_value = current_portfolio * 0.2

        print(f'---------------------------------------------------')
        print(f'Current cash: ${current_cash}')
        print(f'Current portfolio: ${current_portfolio:.2f}')
        print(f'Max buy value: ${self.max_buy_value:.2f}')
        print(f'---------------------------------------------------')

        prices = {}

        data = yf.download(
            self.stock_trading_list,
            interval="1h",
            group_by="ticker",
            auto_adjust=False,
            threads=True
        )

        # STEP 1: collect everything first
        for index, stock in enumerate(self.stock_trading_list, start=1):

            if stock not in data.columns.get_level_values(0):
                continue

            df = data[stock].dropna()
            if len(df) < 30:
                continue

            # --- cached news ---
            if stock not in self.news_cache:
                try:
                    url = "https://api.marketaux.com/v1/news/all"
                    params = {
                        "symbols": stock,
                        "language": "en",
                        "limit": 5,
                        "api_token": self.marketaux_api_key
                    }

                    response = requests.get(url, params=params, timeout=5)
                    json_data = response.json()

                    self.news_cache[stock] = [
                        article["title"]
                        for article in json_data.get("data", [])
                        if "title" in article
                    ]

                except Exception:
                    self.news_cache[stock] = []

        stock_text_pairs = []

        for stock, texts in self.news_cache.items():
            for t in texts[:3]:
                stock_text_pairs.append((stock, t))

        texts = [t for _, t in stock_text_pairs]
        stocks = [s for s, _ in stock_text_pairs]

        sentiment_map = self.get_batch_sentiment_score(texts, stocks)

        # please modify the following code to match your own strategy
        for index, stock in enumerate(self.stock_trading_list, start=1):
            try:
                # 1. get the stock data from quoter, return a pandas dataframe
                # ticker = yf.Ticker(stock)
                # info = ticker.info
                # df = ticker.history(interval="1h", actions=False, prepost=False, raise_errors=True)
                # fast_info = ticker.fast_info
                # market_cap = fast_info.get("market_cap", None)

                if stock not in data.columns.get_level_values(0):
                    continue
                df = data[stock].dropna()
                if len(df) < 30:
                    continue

                # price = info.get('currentPrice', 0) # more accurate price
                price = df['Close'].iloc[-1]

                prices[stock] = {'current_price': price}

                sentiment_score = sentiment_map.get(stock)
                if sentiment_score is None:
                    sentiment_score = 0

                if stock in position_data:
                    position = position_data.get(stock, {"qty": 0})
                    qty = position["qty"]
                    already_own = qty > 0
                    if qty > 0:
                        holding = position_data.get(stock, {})
                        # price = holding['nominal_price'] # more accurate price
                        cost_price = holding['cost_price'] # average_cost = cost_price same when only making one purchase per stock
                        profit_loss = (price - cost_price) / cost_price * 100
                        profit_loss_amount = - qty * cost_price + qty * price
                    else:
                        profit_loss = 0
                        profit_loss_amount = 0
                else:
                    qty = 0
                    already_own = False
                    # price = df['Close'].iloc[-1]
                    # price = info['currentPrice'] # more accurate price
                    profit_loss = 0
                    profit_loss_amount = 0

                # qty = self.trading_qty[stock]

                # quote_ret, quote_data = self.trader.get_quote(stock)
                # if quote_ret != RET_OK:
                #     return False

                if len(df) > 1:
                    price_change_1d = (price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                else:
                    price_change_1d = 0

                df_1w = df.tail(7 * 6)  # ~7 trading days * 6.5 hours
                df_1m = df.tail(30 * 6)
                # df_1w = df.last("7D")
                # df_1m = df.last("30D")

                price_change_1w = (price - df_1w['Close'].iloc[0]) / df_1w['Close'].iloc[0] * 100 if len(df_1w) > 1 else 0
                price_change_1m = (price - df_1m['Close'].iloc[0]) / df_1m['Close'].iloc[0] * 100 if len(df_1m) > 1 else 0

                avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                if np.isnan(avg_volume) or avg_volume == 0:
                    avg_volume = df['Volume'].mean()

                stock_data = {
                    'current_price': price,
                    'profit_loss': profit_loss,
                    'profit_loss_amount': profit_loss_amount,
                    'price_change_1d': price_change_1d,
                    'price_change_1w': price_change_1w,
                    'price_change_1m': price_change_1m,
                    'volatility': df['Close'].pct_change().std() * np.sqrt(252 * 6.5) * 100,
                    #'volume': df['Volume'].iloc[-1],
                    #'volume': info.get('volume', 0), # more accurate
                    # 'volume': info.get('volume', 0),
                    # 'avg_volume': df['Volume'].mean(),
                    # 'market_cap': info.get('marketCap', 0),
                    # 'pe_ratio': info.get('trailingPE', 0),
                    'volume': df['Volume'].iloc[-1],
                    'avg_volume': avg_volume,
                    'market_cap': None,  # removed (not available without .info)
                    # 'market_cap': fast_info.get("market_cap", None),
                    'pe_ratio': None,  # removed (not available without .info)
                    'sector': self.get_sector(stock),
                    'data': df
                }

                indicator = "* " if already_own else ""
                # 2. calculate the indicator
                print(f'{indicator}Analysing {stock_data["sector"]} - {stock} {index}/{len(self.stock_trading_list)}')

                analysis = self.analyze_stock(stock, stock_data, position_data, sentiment_score)
                recommendation = analysis['recommendation']
                reason = ', '.join(analysis['reasons'][:3])
                print(f'    Recommendation {recommendation} - {reason}')

                if already_own:
                    print(f'Current profit/loss {profit_loss:.2f}% ${profit_loss_amount:.2f}')

                # 3. check the signal and place order
                if analysis['recommendation'] == "BUY":
                    # buy up to 20% of total portfolio of shares
                    # existing_value = qty * price if already_own else 0
                    position = position_data.get(stock, {})
                    qty_held = position.get("qty", 0)

                    # if qty_held > 0:
                    #     buy_qty = 0
                    # else:
                    cost_price = position.get("cost_price", price)
                    existing_value = qty_held * cost_price
                    remaining_cap = max(0, self.max_buy_value - existing_value)
                    buy_qty = math.floor(min(current_cash, remaining_cap) / price)

                    print('BUY Signals')
                    print(f"\n🟢 {stock} ({stock_data['sector']})")
                    print(f"   Current Price: ${analysis['current_price']:.2f}")
                    print(f"   Target Price:  ${analysis['target_price']:.2f}")
                    print(f"   Confidence:    {analysis['confidence']:.0f}%")
                    print(f"   Reasons: {', '.join(analysis['reasons'][:3])}")

                    if already_own:
                        print('BUT already own')
                    elif buy_qty > 0:
                        self.strategy_make_trade(action='BUY', stock=stock, price=price, qty=buy_qty, position_data = position_data)   # place order
                    else:
                        print('BUT not enough cash')

                if analysis['recommendation'] == "SELL":
                    # # sell 100% of available shares
                    # if already_own:
                    #     qty = position_data[stock]["qty"]
                    # else:
                    #     qty = 0

                    print('SELL Signals')
                    print(f"\n🔴 {stock} ({stock_data['sector']})")
                    print(f"   Current Price: ${analysis['current_price']:.2f}")
                    print(f"   Target Price:  ${analysis['target_price']:.2f}")
                    print(f"   Confidence:    {analysis['confidence']:.0f}%")
                    print(f"   Reasons: {', '.join(analysis['reasons'][:3])}")

                    if not already_own:
                        print("BUT don't own")
                    else:
                        # sell_qty = position_data[stock]["qty"]
                        # sell_qty = position_data.get(stock, {}).get("qty", 0)
                        # if qty > 0:
                        sell_qty = qty
                        self.strategy_make_trade(action='SELL', stock=stock, price=price, qty=sell_qty, position_data = position_data) # place order

                time.sleep(1)  # sleep 1 second to avoid the quote limit
            except Exception as e:
                print(f"Strategy Error: {e}")
                logging_info(f'{self.strategy_name}: {e}')

        self.print_portfolio(prices)

        # self.trader.unsubscribe()
        # self.trader.close_quote_context()

        """ ⏫⏫⏫ New Strategy ends here ⏫⏫⏫ """

        print('---------------------------------------------------')
        print("Strategy checked... Waiting next decision called...")
        print('---------------------------------------------------')
        return None

    def get_batch_sentiment_score(self, texts, stocks):
        if not texts:
            return {}

        inputs = self.finbert_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        stock_scores = {}
        stock_counts = {}

        for i, p in enumerate(probs):
            neg, neu, pos = p.tolist()

            # FIX: damp neutral noise (important)
            score = pos - neg - 0.2 * neu

            stock = stocks[i]

            stock_scores[stock] = stock_scores.get(stock, 0) + score
            stock_counts[stock] = stock_counts.get(stock, 0) + 1

        for stock in stock_scores:
            stock_scores[stock] /= stock_counts[stock]

        return stock_scores

    """ ⬇️⬇️⬇️ Order related functions ⬇️⬇️⬇️"""

    def strategy_make_trade(self, action, stock, qty, price, position_data):
        if self.trading_confirmation:
            # check if trading confirmation is enabled first
            if action == 'BUY':
                # check the current buying power first
                acct_ret, acct_info = self.trader.get_account_info()
                if acct_ret != RET_OK:
                    print('Trader: Get Account Info failed: ', acct_info)
                    return False
                current_cash = acct_info['cash']

                if current_cash > qty * price:
                    # before buy action, check if it has enough cash
                    if is_market_hours():
                        # market order
                        ret, data = self.trader.market_buy(stock, qty, price)
                    else:
                        # limit order for extended hours
                        ret, data = self.trader.limit_buy(stock, qty, price)

                    if ret == RET_OK:
                        # order placed successfully:
                        print(data)
                        self.save_order_history(data)
                        print('make trade success, show latest position:')
                        print(self.get_current_position())  # show the latest position after trade
                    else:
                        print('Trader: Buy failed: ', data)
                        logging_info(f'{self.strategy_name}: Buy failed: {data}')
                else:
                    print('Trader: Buy failed: Not enough cash to buy')
                    logging_info(f'{self.strategy_name}: Buy failed: Not enough cash to buy')

            if action == 'SELL':
                if qty <= position_data.get(stock, {}).get("qty", 0):
                    # before sell action, check if it has enough position to sell
                    if is_market_hours():
                        # market order
                        ret, data = self.trader.market_sell(stock, qty, price)
                    else:
                        # limit order for extended hours
                        ret, data = self.trader.limit_sell(stock, qty, price)
                    if ret == RET_OK:
                        print(data)
                        logging_info(f'{self.strategy_name}: {data}')
                        self.save_order_history(data)
                        print('make trade success, show latest position:')
                        print(self.get_current_position())  # show the latest position after trade
                    else:
                        print('Trader: Sell failed: ', data)
                        logging_info(f'{self.strategy_name}: Sell failed: {data}')
                else:
                    print('Trader: Sell failed: Not enough position to sell')
                    logging_info(f'{self.strategy_name}: Sell failed: Not enough position to sell')
        return None

    def save_order_history(self, data):
        file_data = read_json_file("order_history.json")
        data_dict = data.to_dict()
        new_dict = {}
        for key, v in data_dict.items():
            new_dict[key] = v[0]
        logging_info(f'{self.strategy_name}: {str(new_dict)}')

        if file_data:
            file_data.append(new_dict)
        else:
            file_data = [new_dict]
        write_json_file("order_history.json", file_data)

    # add any other functions you need here

    def get_sector(self, symbol: str) -> str:
        """Get sector for a stock symbol"""
        for sector, stocks in self.stock_trading_list_by_sector.items():
            if symbol in stocks:
                return sector
        return 'Other'

    def calculate_technical_indicators(self, stock: str, stock_data: Dict) -> Dict:
        """
        Calculate technical indicators for a stock

        Args:
            stock: The stock
            stock_data: Stock Dictionary

        Returns:
            Dictionary with technical indicators
        """

        data = stock_data['data']

        ema_12_series = data['Close'].ewm(span=12).mean()
        ema_26_series = data['Close'].ewm(span=26).mean()

        indicators = {'sma_20': data['Close'].rolling(20).mean().iloc[-1],
                      'sma_50': data['Close'].rolling(50).mean().iloc[-1],
                      'ema_12': ema_12_series.iloc[-1],
                      'ema_26': ema_26_series.iloc[-1]}

        # Moving averages

        # MACD
        macd_series = ema_12_series - ema_26_series
        macd_signal_series = macd_series.ewm(span=9).mean()
        macd_hist = macd_series - macd_signal_series
        macd_hist_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        indicators['macd'] = macd_series.iloc[-1]
        indicators['macd_signal'] = macd_signal_series.iloc[-1]
        if len(macd_series) < 2:
            indicators['macd_prev'] = macd_series.iloc[-1]
            indicators['macd_signal_prev'] = macd_signal_series.iloc[-1]
        else:
            indicators['macd_prev'] = macd_series.iloc[-2]
            indicators['macd_signal_prev'] = macd_signal_series.iloc[-2]
        indicators['macd_hist'] = macd_hist.iloc[-1]
        indicators['macd_hist_prev'] = macd_hist_prev

        # # RSI
        # delta = data['Close'].diff()
        # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        # avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        # # rs = gain / (loss + 1e-9)
        # rs = avg_gain / (avg_loss + 1e-9)
        # rsi = 100 - (100 / (1 + rs))
        # # indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        # indicators['rsi'] = rsi

        # RSI (Wilder smoothing)
        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi_series.iloc[-1]

        # Bollinger Bands
        bb_period = 20
        bb_std = data['Close'].rolling(bb_period).std().iloc[-1]
        bb_middle = data['Close'].rolling(bb_period).mean().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower
        # indicators['bb_position'] = (data['Close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        bb_range = (bb_upper - bb_lower)
        if bb_range == 0:
            indicators['bb_position'] = 0.5
        else:
            indicators['bb_position'] = (stock_data['current_price'] - bb_lower) / bb_range
        if bb_middle == 0:
            indicators['bb_width'] = 0
        else:
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # ATR
        high = stock_data['data']['High']
        low = stock_data['data']['Low']
        close = stock_data['data']['Close']
        prev_close = close.shift(1)
        hl = high - low
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()
        tr = pd.DataFrame({"hl": hl, "hc": hc, "lc": lc}).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else tr.mean()
        indicators['atr'] = atr

        logging_info(f'{self.strategy_name}: stock = {stock}, indicators calculated')
        return indicators

    def analyze_stock(self, stock: str, stock_data: Dict, position_data: Dict, sentiment_score: float = 0) -> Dict:
        """
        Analyze a single stock and generate buy/sell recommendation

        Args:
            stock: Stock symbol
            stock_data: Stock data dictionary
            position_data: Position dictionary
            sentiment_score: Float

        Returns:
            Analysis dictionary
        """

        current_price = stock_data['current_price']

        analysis = {
            'symbol': stock,
            'sector': stock_data['sector'],
            'current_price': current_price,
            'recommendation': 'HOLD',
            'confidence': 0,
            'reasons': [] #,
            # 'target_price': current_price,
            # 'stop_loss': current_price * 0.9, # ATR suggested but not actually using this value
        }

        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(stock, stock_data)

        atr = indicators['atr']
        analysis['target_price'] = current_price + 3 * atr
        analysis['stop_loss'] = current_price - 2 * atr
        analysis['force_sell'] = current_price <= analysis["stop_loss"]
        analysis['force_take_profit'] = current_price >= analysis["target_price"]

        score = 0
        reasons = []

        # --- FinBERT sentiment impact ---
        if sentiment_score > 0.2:
            score += sentiment_score * 2
            reasons.append("Positive sentiment")
        elif sentiment_score < -0.2:
            score += sentiment_score * 2
            reasons.append("Negative sentiment")
        else:
            reasons.append("Neutral sentiment")

        sma20 = indicators['sma_20']
        sma50 = indicators['sma_50']

        # Technical Analysis Scoring

        # Moving Average Analysis
        ma_trend = 0

        if current_price > sma20:
            ma_trend += 0.5
        elif current_price < sma20:
            ma_trend -= 0.5

        if current_price > sma50:
            ma_trend += 0.5
        elif current_price < sma50:
            ma_trend -= 0.5

        if sma20 > sma50:
            ma_trend += 0.5
        elif sma20 < sma50:
            ma_trend -= 0.5

        score += ma_trend

        if ma_trend > 0:
            reasons.append("Bullish MA structure")
        elif ma_trend < 0:
            reasons.append("Bearish MA structure")
        else:
            reasons.append("Neutral MA structure")

        # RSI Analysis
        if indicators['rsi'] < 30:
            score += 1.5
            reasons.append("RSI oversold (potential bounce)")
        elif indicators['rsi'] > 70:
            score -= 1.5
            reasons.append("RSI overbought (potential correction)")
        else:
            reasons.append("RSI in neutral zone")

        # if stock_data['volatility'] > 60:
        #     score -= 1
        #     reasons.append("High volatility risk")

        # Bollinger Bands Analysis
        low_vol = indicators['bb_width'] < 0.02
        if low_vol:
            score -= 1
            reasons.append("Low volatility (no trade zone)")
        else:
            if indicators['bb_position'] < 0.2:
                score += 1
                reasons.append("Near lower Bollinger Band")
            elif indicators['bb_position'] > 0.8:
                score -= 1
                reasons.append("Near upper Bollinger Band")

        # MACD Analysis
        # MACD Analysis crossover
        macd_hist_prev = indicators['macd_hist_prev']
        macd_hist = indicators['macd_hist']
        # crossover
        if macd_hist_prev < 0 < macd_hist:
            score += 1
            reasons.append("MACD bullish crossover")
        elif macd_hist_prev > 0 > macd_hist:
            score -= 1
            reasons.append("MACD bearish crossover")

        # Momentum Analysis
        if stock_data['price_change_1w'] > 10:
            score += 1
            reasons.append("Strong weekly momentum")
        elif stock_data['price_change_1w'] < -10:
            score -= 1
            reasons.append("Weak weekly momentum")
        else:
            reasons.append("Neutral momentum")

        # Volume Analysis
        avg_vol = stock_data.get('avg_volume', 0)

        if avg_vol < 1e5:
            score -= 1
            reasons.append("Low liquidity warning")
        else:
            volume_ratio = stock_data['volume'] / avg_vol
            if volume_ratio > 1.5:
                if current_price > sma20:
                    score += 1
                    reasons.append("Bullish volume expansion")
                else:
                    score -= 1
                    reasons.append("Bearish volume expansion")

        if stock in position_data:
            profit_loss = stock_data['profit_loss']
            position = position_data.get(stock, {"qty": 0})
            qty = position["qty"]
            if qty > 0:
                if profit_loss >= 20: # and indicators['rsi'] > 70:
                    score -= 3
                    reasons.append(f"Large profit ({profit_loss:.1f}%) - consider taking profits")
                # elif profit_loss >= 20:
                #     score -= 3
                #     reasons.append(f"Medium profit ({profit_loss:.1f}%) - consider taking profits")
                elif profit_loss <= -20:
                    score -= 3
                    reasons.append(f"Large loss ({profit_loss:.1f}%) - consider cutting losses")
                elif profit_loss <= -10:
                    score -= 2
                    reasons.append(f"Medium loss ({profit_loss:.1f}%) - consider cutting losses")

        if analysis["force_sell"] or analysis["force_take_profit"]:
            analysis['recommendation'] = "SELL"
            analysis['confidence'] = 100
            analysis['reasons'] = reasons # + ["Stop-loss triggered"]
            analysis['score'] = score
            return analysis

        # if analysis["force_take_profit"]:
        #     score -= 2
        #     reasons.append("Take-profit zone (ATR)")

        score = max(min(score, 8), -8)

        # Generate recommendation
        if score >= 4:
            analysis['recommendation'] = 'BUY'
            analysis['confidence'] = min(50 + abs(score) * 6, 95)
        elif score <= -4:
            analysis['recommendation'] = 'SELL'
            analysis['confidence'] = min(50 + abs(score) * 6, 95)
        else:
            analysis['recommendation'] = 'HOLD'
            analysis['confidence'] = 50

        analysis['reasons'] = reasons
        analysis['score'] = score

        return analysis

    def print_portfolio(self, prices):
        # check the current buying power
        acct_ret, acct_info = self.trader.get_account_info()
        if acct_ret != RET_OK:
            print('Trader: Get Account Info failed: ', acct_info)
            return False
        current_cash = acct_info['cash']

        # check current position
        pos_ret, position_data = self.trader.get_positions()
        if pos_ret != RET_OK:
            return False

        print(f'---------------------------------------------------')
        print(f'PORTFOLIO')
        print(f'---------------------------------------------------')

        formatted_time = get_current_time()
        print(f'At {formatted_time}')

        current_portfolio = current_cash

        for stock in position_data:
            qty = position_data[stock]['qty']
            if qty > 0:
                # price = prices[stock]['current_price']
                price = prices.get(stock, {}).get('current_price', position_data[stock].get("cost_price", 0))
                # price = pos.get("market_price", pos.get("cost_price", 0))
                amount = qty * price
                current_portfolio = current_portfolio + amount
                print(f'{stock} Qty {qty:.0f} Price ${price:.2f} Amount ${amount:.2f}')

        print(f'Cash {current_cash}')
        print(f'Portfolio {current_portfolio:.2f}')
        return None

    # def plot(self):
    #     #matplotlib.use('tkagg')
    #     #plt.show(block=False)
    #
    #     positions_data = self.trader.get_positions()
    #     if not positions_data:
    #         return False
    #
    #     fig, ax = plt.subplots()
    #     ax.clear()
    #
    #     for symbol in positions_data:
    #         data = positions_data.loc[symbol]
    #         ax.plot(data.index, data['average_cost'], label=symbol)
    #
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Stock Value')
    #     ax.set_title('Yearly')
    #     ax.legend(loc='upper left')
    #     ax.tick_params(axis='x', rotation=45)
    #     #plt.show(block=True)
    #     st.plotly_chart(fig)
    #     return None
