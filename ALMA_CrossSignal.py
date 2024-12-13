import pandas as pd
import numpy as np
import talib as ta
from math import sqrt, exp, cos


class ALMA_CrossSignal:
    def __init__(
            self,
            df,
            # Parameters matching the original PineScript
            shortlen=1000,
            useRes=True,
            intRes=1,
            basisType="ALMA",
            basisLen=12,
            offsetSigma=3,
            offsetALMA=0.85,
            rsiLengthInput=20,
            rsiSourceInput='close',
            maTypeInput="EMA",  # Will be ignored for final RSI MA step, always EMA used
            maLengthInput=14,
            MAvalue=35,
            MAvalue1=65,
            rsivalue=50,
            rsivalue1=50,
            # Additional parameters needed for replication
            mainTimeframe="1h",  # e.g. "1h", "1D", "1W", "1M"
            ib=None,  # ib_insync.IB instance
            contract=None  # ib_insync contract object
    ):
        """
        Initialize the strategy with user-defined parameters.
        df: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'.
        mainTimeframe: String representing the primary timeframe (e.g., "1h", "1D", "1W", "1M").
        ib: An instance of ib_insync.IB connected to IBKR.
        contract: The IB contract to fetch data for alternate resolutions if useRes is True.
        """
        self.df = df.copy()
        self.shortlen = shortlen
        self.longlen = 2 * shortlen
        self.useRes = useRes
        self.intRes = intRes
        self.basisType = basisType
        self.basisLen = basisLen
        self.offsetSigma = offsetSigma
        self.offsetALMA = offsetALMA
        self.rsiLengthInput = rsiLengthInput
        self.rsiSourceInput = rsiSourceInput
        self.maTypeInput = maTypeInput
        self.maLengthInput = maLengthInput
        self.MAvalue = MAvalue
        self.MAvalue1 = MAvalue1
        self.rsivalue = rsivalue
        self.rsivalue1 = rsivalue1
        self.mainTimeframe = mainTimeframe
        self.ib = ib
        self.contract = contract

    @staticmethod
    def sma(series, length):
        return series.rolling(length).mean()

    @staticmethod
    def ema(series, length):
        return ta.EMA(series, timeperiod=length)

    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def alma(series, length=9, offset=0.85, sigma=6):
        if length < 1:
            return series
        m = offset * (length - 1)
        s = length / sigma
        idx = np.arange(length)
        w = np.exp(-((idx - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        return series.rolling(length).apply(lambda x: (x * w).sum(), raw=True)

    @staticmethod
    def linreg(series, length, offset=0):
        x = np.arange(length)

        def linreg_calc(x_series):
            y = x_series.values
            m = ((x * y).mean() - x.mean() * y.mean()) / ((x ** 2).mean() - (x.mean() ** 2))
            b = y.mean() - m * x.mean()
            return m * (length - 1 - offset) + b

        return series.rolling(length).apply(linreg_calc, raw=False)

    @staticmethod
    def rma(series, length):
        alpha = 1.0 / length
        r = np.zeros(len(series))
        r[0] = series.iloc[0]
        for i in range(1, len(series)):
            r[i] = alpha * series.iloc[i] + (1 - alpha) * r[i - 1]
        return pd.Series(r, index=series.index)

    def tema(self, series, length):
        e1 = self.ema(series, length)
        e2 = self.ema(e1, length)
        e3 = self.ema(e2, length)
        return 3 * e1 - 3 * e2 + e3

    def dema(self, series, length):
        e1 = self.ema(series, length)
        e2 = self.ema(e1, length)
        return 2 * e1 - e2

    def ssf(self, series, length):
        a1 = exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = -a1 ** 2
        c1 = 1 - c2 - c3
        result = np.zeros(len(series))
        for i in range(len(series)):
            if i < 2:
                result[i] = series.iloc[i]
            else:
                result[i] = c1 * (series.iloc[i] + series.iloc[i - 1]) / 2 + c2 * result[i - 1] + c3 * result[i - 2]
        return pd.Series(result, index=series.index)

    def vwma(self, price, volume, length):
        pv = price * volume
        return pv.rolling(length).sum() / volume.rolling(length).sum()

    def variant(self, basisType, source, length, offSig, offALMA):
        if basisType == "ALMA":
            return self.alma(source, length, offALMA, offSig)
        elif basisType == "EMA":
            return self.ema(source, length)
        elif basisType == "DEMA":
            return self.dema(source, length)
        elif basisType == "TEMA":
            return self.tema(source, length)
        elif basisType == "WMA":
            return self.wma(source, length)
        elif basisType == "VWMA":
            return self.vwma(source, self.df['volume'], length)
        elif basisType == "SMMA":
            return self.rma(source, length)
        elif basisType == "HullMA":
            half_len = max(1, length // 2)
            wma1 = self.wma(source, half_len)
            wma2 = self.wma(source, length)
            hull_input = 2 * wma1 - wma2
            return self.wma(hull_input, int(sqrt(length)))
        elif basisType == "LSMA":
            return self.linreg(source, length, offSig)
        elif basisType == "SMA":
            return self.sma(source, length)
        elif basisType == "TMA":
            return self.sma(self.sma(source, length), length)
        elif basisType == "SSMA":
            return self.ssf(source, length)
        else:
            # default ALMA if unknown
            return self.alma(source, length, offALMA, offSig)

    @staticmethod
    def rsi_func(series, length=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = ALMA_CrossSignal.rma(up.fillna(0), length)
        ma_down = ALMA_CrossSignal.rma(down.fillna(0), length)
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        return rsi

    @staticmethod
    def crossover(series1, series2):
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    @staticmethod
    def crossunder(series1, series2):
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

    def determine_stratRes(self):
        """
        Attempt to replicate PineScript logic for stratRes:
        stratRes = timeframe.ismonthly ? "###M"
                  : timeframe.isweekly ? "###W"
                  : timeframe.isdaily ? "###D"
                  : timeframe.isintraday ? "####"
                  : '60'
        """
        # Parse mainTimeframe to determine multiplier and type
        tf = self.mainTimeframe.lower()
        if tf.endswith('m'):  # monthly
            # Assuming 1M means monthly
            timeframe_isMonthly = True
            timeframe_isWeekly = False
            timeframe_isDaily = False
            timeframe_isIntraday = False
            timeframe_multiplier = 1
        elif tf.endswith('w'):  # weekly
            timeframe_isMonthly = False
            timeframe_isWeekly = True
            timeframe_isDaily = False
            timeframe_isIntraday = False
            timeframe_multiplier = 1
        elif tf.endswith('d'):  # daily
            timeframe_isMonthly = False
            timeframe_isWeekly = False
            timeframe_isDaily = True
            timeframe_isIntraday = False
            timeframe_multiplier = 1
        else:
            # Assume intraday, extract minutes
            timeframe_isMonthly = False
            timeframe_isWeekly = False
            timeframe_isDaily = False
            timeframe_isIntraday = True
            # If tf='1h' or '60', convert to minutes:
            if 'h' in tf:
                # Convert hours to minutes
                val = int(tf.replace('h', ''))
                timeframe_multiplier = val * 60
            else:
                timeframe_multiplier = int(tf)  # minutes directly

        if timeframe_isMonthly:
            # ###M
            return f"{timeframe_multiplier * self.intRes}M"
        elif timeframe_isWeekly:
            # ###W
            return f"{timeframe_multiplier * self.intRes}W"
        elif timeframe_isDaily:
            # ###D
            return f"{timeframe_multiplier * self.intRes}D"
        elif timeframe_isIntraday:
            # #### (just minutes)
            return f"{timeframe_multiplier * self.intRes}"
        else:
            return "60"

    def fetch_data_at_resolution(self, resolution_str):
        """
        Placeholder function to fetch data at the alternate resolution.
        In PineScript, `security()` is used; here we must emulate.

        resolution_str might look like '60', '1D', '1W', etc.

        You must implement this logic to fetch from IB using ib_insync:
        - Convert resolution_str to IB barSizeSetting and durationStr.
        - Request historical data.
        For demonstration, we return a df identical to self.df but you need real logic.
        """

        # Example: if resolution_str is '60', assume 1 hour bars
        # This is just a placeholder. In real code, you'd map '60' -> '1 hour',
        # '1D' -> '1 day', etc., then fetch with ib.reqHistoricalData.
        # Make sure the returned df has columns open,high,low,close,volume with a datetime index.

        # In a real scenario:
        # bars = self.ib.reqHistoricalData(self.contract, ..., barSizeSetting based on resolution_str)
        # alt_df = util.df(bars)
        # alt_df must be processed just like the main df
        # For now, just return self.df as a placeholder
        alt_df = self.df.copy()
        return alt_df

    def reso(self, series, use, res_str):
        """
        Emulate PineScript reso() function:
        security_1 = security(syminfo.tickerid, res, exp, gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_on)
        use ? security_1 : exp

        We'll fetch alternate resolution data and try to align it.
        Then we apply lookahead by shifting series back by 1 bar.
        """
        if not use:
            return series
        alt_df = self.fetch_data_at_resolution(res_str)

        # Recompute the variant basis at the alternate resolution from alt_df.
        # We must know if series is closeSeries or openSeries. We'll assume we pass already computed series.
        # Actually, we should recalculate the same variant(basisType...) for alt_df.
        # For simplicity, we assume `series` is either closeSeries or openSeries and we do the same calc:
        if series.name == 'closeSeries':
            alt_series = self.variant(self.basisType, alt_df['close'], self.basisLen, self.offsetSigma, self.offsetALMA)
        else:
            alt_series = self.variant(self.basisType, alt_df['open'], self.basisLen, self.offsetSigma, self.offsetALMA)

        # Align alt_series with main df's index
        alt_series = alt_series.reindex(self.df.index, method='ffill')

        # Emulate lookahead=barmerge.lookahead_on: shift alt_series backward by 1 bar so future data is seen now
        alt_series = alt_series.shift(-1)

        return alt_series

    def run(self):
        df = self.df

        # Short/Long MAs for market condition
        df['short'] = self.sma(df['close'], self.shortlen)
        df['long'] = self.sma(df['close'], self.longlen)

        # Basis close/open series on main timeframe
        closeSeries = self.variant(self.basisType, df['close'], self.basisLen, self.offsetSigma, self.offsetALMA)
        closeSeries.name = 'closeSeries'
        openSeries = self.variant(self.basisType, df['open'], self.basisLen, self.offsetSigma, self.offsetALMA)
        openSeries.name = 'openSeries'

        # Determine stratRes and get alternate resolution series if useRes is True
        stratRes = self.determine_stratRes()
        df['closeSeriesAlt'] = self.reso(closeSeries, self.useRes, stratRes)
        df['openSeriesAlt'] = self.reso(openSeries, self.useRes, stratRes)

        # RSI
        rsi_source = df[self.rsiSourceInput]
        df['rsi'] = self.rsi_func(rsi_source, self.rsiLengthInput)

        # rsiMA always EMA to match PineScript provided function ma() => ema()
        df['rsiMA'] = self.ema(df['rsi'], self.maLengthInput)

        # Conditions for rsiMA
        df['MAoption'] = df['rsiMA'] < self.MAvalue
        df['MAoption1'] = df['rsiMA'] > self.MAvalue1

        # RSI conditions (with default length=14)
        df['rsi_14'] = self.rsi_func(df['close'], 14)
        df['rsioption'] = df['rsi_14'] < self.rsivalue
        df['rsioption1'] = df['rsi_14'] > self.rsivalue1

        # Signals
        df['xlong'] = self.crossover(df['closeSeriesAlt'], df['openSeriesAlt']) & df['rsioption'] & df['MAoption'] & (
                    df['short'] < df['long'])
        df['xshort'] = self.crossunder(df['closeSeriesAlt'], df['openSeriesAlt']) & df['rsioption1'] & df[
            'MAoption1'] & (df['long'] < df['short'])

        # Market condition check
        df['_value'] = self.sma(df['close'], 1000) - self.sma(df['close'], 2000)
        df['market_condition'] = np.where(df['_value'] > 0, "bull market",
                                          np.where(df['_value'] < 0, "bear market", "trend change"))

        return df
