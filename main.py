from ib_insync import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ALMA_CrossSignal import ALMA_CrossSignal
# Connect to IB TWS
ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=1)  # Adjust host/port/clientId as needed

# Define the contract you want data for
contract = Stock('AAPL', 'SMART', 'USD')  # Example: Apple on SMART routing

# Request historical data (e.g., 1 day of 1-hour bars)
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='5 D',
    barSizeSetting='1 hour',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=False
)

# Convert to DataFrame
df = util.df(bars)
# Rename and ensure columns match expected:
# The df from util.df(bars) typically has: date, open, high, low, close, volume, barCount, WAP
# We just need: open, high, low, close, volume.
df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
df.set_index('date', inplace=True)

# Instantiate the strategy with desired parameters (adjust as needed)
strategy = ALMA_CrossSignal(
    df,
    shortlen=1000,
    useRes=True,
    intRes=1,
    basisType="ALMA",
    basisLen=12,
    offsetSigma=3,
    offsetALMA=0.85,
    rsiLengthInput=20,
    rsiSourceInput='close',   # or 'open', 'high', etc.
    maTypeInput="EMA",
    maLengthInput=14,
    MAvalue=35,
    MAvalue1=65,
    rsivalue=50,
    rsivalue1=50
)

# Run the strategy
result_df = strategy.run()

# result_df now has columns: 'xlong', 'xshort', etc.

# Generate a plot to visualize the signals
plt.figure(figsize=(12,6))
plt.plot(result_df.index, result_df['close'], label='Close Price', color='blue')

# Plot buy signals (xlong) as green up arrows
buy_signals = result_df[result_df['xlong'] == True]
plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Long Signal')

# Plot sell signals (xshort) as red down arrows
sell_signals = result_df[result_df['xshort'] == True]
plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Short Signal')

plt.title("ALMA Cross Signals")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# If you want alerts, you can print them or handle as desired
if not buy_signals.empty:
    print("ALERT: Long signal(s) found at:")
    print(buy_signals.index)

if not sell_signals.empty:
    print("ALERT: Short signal(s) found at:")
    print(sell_signals.index)

ib.disconnect()
