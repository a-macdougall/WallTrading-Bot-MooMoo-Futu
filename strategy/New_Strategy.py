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
from numpy.ma.core import floor

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

        for stock in position_data:
            qty = position_data[stock]['qty']
            if qty > 0:
                current_portfolio = current_portfolio + position_data[stock]['market_val']

        # max buy value 20% of total portfolio
        self.max_buy_value = current_portfolio * 0.2

        print(f'---------------------------------------------------')
        print(f'Current cash: ${current_cash}')
        print(f'Current portfolio: ${current_portfolio:.2f}')
        print(f'Max buy value: ${self.max_buy_value:.2f}')
        print(f'---------------------------------------------------')

        prices = {}

        # please modify the following code to match your own strategy
        for index, stock in enumerate(self.stock_trading_list, start=1):
            try:
                # 1. get the stock data from quoter, return a pandas dataframe
                ticker = yf.Ticker(stock)
                info = ticker.info
                df = ticker.history(interval="1h", actions=False, prepost=False, raise_errors=True)

                price = info.get('currentPrice', 0) # more accurate price

                prices[stock] = {'current_price': price}

                if stock in position_data:
                    qty = position_data[stock]["qty"]
                    already_own = qty > 0
                    if qty > 0:
                        holding = position_data[stock]
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

                stock_data = {
                    'current_price': price,
                    'profit_loss': profit_loss,
                    'profit_loss_amount': profit_loss_amount,
                    'price_change_1d': ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100),
                    'price_change_1w': (
                            (price - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(
                        df) >= 5 else 0,
                    'price_change_1m': (
                            (price - df['Close'].iloc[-22]) / df['Close'].iloc[-22] * 100) if len(
                        df) >= 22 else 0,
                    'volatility': df['Close'].pct_change().std() * np.sqrt(252) * 100,
                    #'volume': df['Volume'].iloc[-1],
                    'volume': info.get('volume', 0), # more accurate
                    'avg_volume': df['Volume'].mean(),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'sector': self.get_sector(stock),
                    'data': df
                }

                # 2. calculate the indicator
                print(f'Analysing {stock_data["sector"]} - {stock} {index}/{len(self.stock_trading_list)}')

                analysis = self.analyze_stock(stock, stock_data, position_data)
                recommendation = analysis['recommendation']
                reason = ', '.join(analysis['reasons'][:3])
                print(f'Recommendation {recommendation} - {reason}')

                if already_own:
                    print(f'Current profit/loss {profit_loss:.2f}% ${profit_loss_amount:.2f}')

                # 3. check the signal and place order
                if analysis['recommendation'] == "BUY":
                    # buy up to 20% of total portfolio of shares
                    buy_qty = floor(self.max_buy_value / price)

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
                    elif -10 < profit_loss < 20:
                        print("BUT want to make a profit")
                    else:
                        self.strategy_make_trade(action='SELL', stock=stock, price=price, qty=qty, position_data = position_data) # place order

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
                if qty <= position_data[stock]["qty"]:
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

        indicators = {'sma_20': data['Close'].rolling(20).mean().iloc[-1],
                      'sma_50': data['Close'].rolling(50).mean().iloc[-1],
                      'ema_12': data['Close'].ewm(span=12).mean().iloc[-1],
                      'ema_26': data['Close'].ewm(span=26).mean().iloc[-1]}

        # Moving averages

        # MACD
        macd = indicators['ema_12'] - indicators['ema_26']
        macd_signal = pd.Series(macd).ewm(span=9).mean()
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]

        # Bollinger Bands
        bb_period = 20
        bb_std = data['Close'].rolling(bb_period).std().iloc[-1]
        bb_middle = data['Close'].rolling(bb_period).mean().iloc[-1]
        indicators['bb_upper'] = bb_middle + (2 * bb_std)
        indicators['bb_lower'] = bb_middle - (2 * bb_std)
        # indicators['bb_position'] = (data['Close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        indicators['bb_position'] = (stock_data['current_price'] - indicators['bb_lower']) / (
                    indicators['bb_upper'] - indicators['bb_lower']) # more accurate price

        logging_info(f'{self.strategy_name}: stock = {stock}, indicators calculated')
        return indicators

    def analyze_stock(self, stock: str, stock_data: Dict, position_data: Dict) -> Dict:
        """
        Analyze a single stock and generate buy/sell recommendation

        Args:
            stock: Stock symbol
            stock_data: Stock data dictionary
            position_data: Position dictionary

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
            'reasons': [],
            'target_price': current_price,
            'stop_loss': current_price * 0.9,
        }

        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(stock, stock_data)

        score = 0
        reasons = []

        # Technical Analysis Scoring

        # Moving Average Analysis
        if current_price > indicators['sma_20']:
            score += 1
            reasons.append("Price above 20-day SMA")
        elif current_price < indicators['sma_20']:
            score -= 1
            reasons.append("Price below 20-day SMA")

        if current_price > indicators['sma_50']:
            score += 1
            reasons.append("Price above 50-day SMA")
        elif current_price < indicators['sma_50']:
            score -= 1
            reasons.append("Price below 50-day SMA")

        # MACD Analysis
        if indicators['macd'] > indicators['macd_signal']:
            score += 1
            reasons.append("MACD bullish crossover")
        elif indicators['macd'] < indicators['macd_signal']:
            score -= 1
            reasons.append("MACD bearish crossover")

        # RSI Analysis
        if indicators['rsi'] < 30:
            score += 2
            reasons.append("RSI oversold (potential bounce)")
        elif indicators['rsi'] > 70:
            score -= 2
            reasons.append("RSI overbought (potential correction)")
        else:
            reasons.append("RSI in neutral zone")

        # Bollinger Bands Analysis
        if indicators['bb_position'] < 0.2:
            score += 1
            reasons.append("Near lower Bollinger Band")
        elif indicators['bb_position'] > 0.8:
            score -= 1
            reasons.append("Near upper Bollinger Band")

        # Momentum Analysis
        if stock_data['price_change_1w'] > 10:
            score += 1
            reasons.append("Strong weekly momentum")
        elif stock_data['price_change_1w'] < -15:
            score -= 1
            reasons.append("Weak weekly momentum")
        else:
            reasons.append("Neutral momentum")

        # Volume Analysis
        volume_ratio = stock_data['volume'] / stock_data['avg_volume']
        if volume_ratio > 1.5:
            score += 1
            reasons.append("Above average volume")

        if stock in position_data:
            profit_loss = stock_data['profit_loss']
            qty = position_data[stock]['qty']
            if qty > 0:
                if profit_loss >= 25:
                    score -= 3
                    reasons.append(f"Large profit ({profit_loss:.1f}%) - consider taking profits")
                elif profit_loss >= 20:
                    score -= 2
                    reasons.append(f"Medium profit ({profit_loss:.1f}%) - consider taking profits")
                elif profit_loss <= -15:
                    score -= 3
                    reasons.append(f"Large loss ({profit_loss:.1f}%) - consider cutting losses")
                elif profit_loss <= -10:
                    score -= 2
                    reasons.append(f"Medium loss ({profit_loss:.1f}%) - consider cutting losses")

        # Generate recommendation
        if score >= 4:
            analysis['recommendation'] = 'BUY'
            analysis['confidence'] = min(score * 15, 95)
            analysis['target_price'] = current_price * 1.3
        elif score <= -3:
            analysis['recommendation'] = 'SELL'
            analysis['confidence'] = min(abs(score) * 15, 95)
            analysis['target_price'] = current_price * 0.8
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
                price = prices[stock]['current_price']
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
