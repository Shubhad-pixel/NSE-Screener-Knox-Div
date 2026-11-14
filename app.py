from concurrent.futures import ThreadPoolExecutor
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Configure the web page
st.set_page_config(page_title='NSE ADX + Knoxville Screener', layout='wide')
st.title('📈 NSE Trending & Reversal Stock Screener')
st.markdown(
    'Filters trending NSE stocks (**ADX > 25**) using **Knoxville Divergence**'
    ' and **MACD-Histogram** confirmation.'
)


@st.cache_data(ttl=86400)
def get_nse_symbols():
    """Fetches active equity symbols from NSE India."""
    url = 'https://nseindia.com'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(response.text))
        return [f'{sym.strip()}.NS' for sym in df['SYMBOL'].dropna().unique()]
    except Exception:
        return [
            'RELIANCE.NS',
            'TCS.NS',
            'HDFCBANK.NS',
            'INFY.NS',
            'ICICIBANK.NS',
            'SBIN.NS',
            'BHARTIARTL.NS',
            'ITC.NS',
            'LT.NS',
            'AXISBANK.NS',
        ]


def compute_rsi(close, period=21):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_adx(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0
    )

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (
        pd.Series(plus_dm, index=close.index)
        .ewm(alpha=1 / period, min_periods=period, adjust=False)
        .mean()
        / atr
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=close.index)
        .ewm(alpha=1 / period, min_periods=period, adjust=False)
        .mean()
        / atr
    )

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def detect_knoxville_divergence(
    df, lookback=30, mom_period=20, rsi_period=21, ob=70, os=30
):
    if len(df) < lookback + mom_period:
        return False, False

    close, high, low = df['Close'], df['High'], df['Low']
    mom = close.diff(mom_period)
    rsi = compute_rsi(close, period=rsi_period)

    curr_idx = len(df) - 1
    bearish_kd, bullish_kd = False, False

    if high.iloc[curr_idx] >= high.iloc[curr_idx - lookback : curr_idx + 1].max():
        for i in range(5, lookback + 1):
            past_idx = curr_idx - i
            if (
                mom.iloc[curr_idx] < mom.iloc[past_idx]
                and rsi.iloc[past_idx : curr_idx + 1].max() >= ob
            ):
                bearish_kd = True
                break

    if low.iloc[curr_idx] <= low.iloc[curr_idx - lookback : curr_idx + 1].min():
        for i in range(5, lookback + 1):
            past_idx = curr_idx - i
            if (
                mom.iloc[curr_idx] > mom.iloc[past_idx]
                and rsi.iloc[past_idx : curr_idx + 1].min() <= os
            ):
                bullish_kd = True
                break

    return bullish_kd, bearish_kd


def analyze_stock(params):
    symbol, adx_threshold, min_volume = params
    try:
        df = yf.download(symbol, period='6mo', interval='1d', progress=False)
        if df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Liquidity filter
        avg_vol = df['Volume'].tail(20).mean()
        if avg_vol < min_volume:
            return None

        # 1. ADX Filter
        adx_series = compute_adx(df['High'], df['Low'], df['Close'], period=14)
        latest_adx = adx_series.iloc[-1]
        if pd.isna(latest_adx) or latest_adx < adx_threshold:
            return None

        # 2. Knoxville Divergence
        bull_kd, bear_kd = detect_knoxville_divergence(df)
        if not (bull_kd or bear_kd):
            return None

        # 3. MACD Histogram Confirmation
        macd_hist = compute_macd_hist(df['Close'], fast=12, slow=26, signal=9)
        hist_curr = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-2]

        signal = None
        if bull_kd and hist_curr > hist_prev:
            signal = 'BUY (Bullish KD + Rising MACD-H)'
        elif bear_kd and hist_curr < hist_prev:
            signal = 'SELL (Bearish KD + Falling MACD-H)'

        if signal:
            return {
                'Symbol': symbol.replace('.NS', ''),
                'Close (₹)': round(df['Close'].iloc[-1], 2),
                'ADX (14)': round(latest_adx, 2),
                'MACD Hist': round(hist_curr, 4),
                '20D Avg Vol': int(avg_vol),
                'Signal': signal,
            }
    except Exception:
        return None
    return None


# Sidebar Controls
st.sidebar.header('Screener Settings')
scan_limit = st.sidebar.slider(
    'Number of NSE Stocks to Scan',
    min_value=50,
    max_value=2200,
    value=150,
    step=50,
)
adx_min = st.sidebar.slider(
    'Minimum ADX (Trend Strength)', min_value=15, max_value=40, value=25
)
min_vol = st.sidebar.number_input(
    'Minimum 20-Day Avg Volume', value=100000, step=50000
)

if st.button('🚀 Run Market Scan', type='primary'):
    symbols = get_nse_symbols()[:scan_limit]
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    tasks = [(sym, adx_min, min_vol) for sym in symbols]

    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, res in enumerate(executor.map(analyze_stock, tasks)):
            if res:
                results.append(res)
            progress_bar.progress((idx + 1) / len(tasks))
            status_text.text(f'Scanning {idx + 1} of {len(tasks)} stocks...')

    status_text.empty()
    progress_bar.empty()

    if results:
        st.success(f'Found {len(results)} matching stocks!')
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            '📥 Download Signals as CSV',
            data=csv,
            file_name='nse_signals.csv',
            mime='text/csv',
        )
    else:
        st.warning(
            'No stocks matched all criteria today. Try lowering the ADX'
            ' threshold or increasing the scan limit.'
        )
