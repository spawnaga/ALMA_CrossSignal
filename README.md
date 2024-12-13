# ALMA_CrossSignal Project

This project aims to fetch OHLCV data from Interactive Brokers TWS using `ib_insync`, and apply indicators and logic similar to a given TradingView (PineScript) strategy. The code attempts to replicate the original PineScript strategy’s signals as closely as possible in Python. However, due to differences in how PineScript and external data sources handle data (especially alternate resolutions and lookahead capabilities), the signals may not be a perfect 1:1 match.

## Dependencies

- Python 3.7+ recommended
- `ib_insync`
- `pandas`
- `numpy`
- `matplotlib` (optional, for plotting)
- `talib` (Technical Analysis library for Python)

## Installing TA-Lib

**Windows Example:**

1. Download `ta-lib-0.4.0-msvc.zip` from [SourceForge](https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-msvc.zip/download?use_mirror=phoenixnap)  
2. Unzip to `C:\ta-lib`

This is a 32-bit binary release. If you want to use 64-bit Python, you will need to build a 64-bit version of the library. Some unofficial (and unsupported) instructions for building on 64-bit Windows 10 can be found online. For reference, you’ll need a compatible C compiler, and you’ll build and place the resulting `.dll` in a directory accessible to Python.

After you have `C:\ta-lib` set up, install `TA-Lib` for Python:
```bash
pip install TA-Lib
```

If you encounter errors, ensure that:
- The TA-Lib `.dll` files are in `C:\ta-lib\lib`
- The include files are in `C:\ta-lib\include`

## Running the Application

1. Make sure TWS or IB Gateway is running and you are connected via `ib_insync`.
2. Edit the script (for example `main.py`) to specify your contract details, timeframe, and other parameters.
3. Run the script. The code will fetch data from IB, apply the indicators, and produce a `DataFrame` with signals.

## A Note on Accuracy and TradingView Alignment

While we have taken steps to replicate the PineScript logic in Python, certain features (like `lookahead=barmerge.lookahead_on` and `security()` calls for alternate resolutions) are not straightforward to emulate. Therefore, the signals in Python may differ from those generated directly on TradingView.
