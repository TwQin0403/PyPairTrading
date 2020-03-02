import pandas 
import numpy
import backtrader as bt

class PairTradingStrategy_basic(bt.Strategy):
    params = dict(
        period=30,
        printout=True,
        threshold=1,
        portfolio_value=10000,
    )

    def log(self, txt, dt=None):
        pass

    def notify_order(self, order):
        pass

    def __init__(self):
        pass

    def next(self):
        pass