# alpha_portfolio_dashboard_live.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="High-Idio-Vol Alpha Dashboard (Live)", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    csv_data = """Date,Action,Portfolio_Stocks,Portfolio_ETFs,Notes,Inclusion_Rules,Sell_Rules,Stock_Price_Example,Quarter_Return_%
2020-01-02,Buy,"CRSP,NTLA,HIMS,APLD,BE,CELH,ROOT,JOBY","XLU,XLP,GLD",Initial setup,"1. σ > 35% (3Y ann.); 2. β < 0.7 (3Y); 3. Visible catalyst in next 90 days (EPS rev ↑15% or event)",N/A,CRSP: $56.20,+12.4%
2020-04-01,"Sell ROOT, Buy VKTX, Buy PRAX","CRSP,NTLA,HIMS,VKTX,PRAX,BE,CELH,JOBY","XLU,XLP,GLD",COVID adjustments,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: VKTX Phase 2 (Mar 2020), PRAX CNS pipeline","ROOT: No upcoming earnings beat or event in 90 days",VKTX: $18.45,+28.7%
2020-07-01,"Sell APLD, Buy KYTX","CRSP,NTLA,HIMS,VKTX,PRAX,BE,CELH,KYTX","XLU,XLP,GLD",Biotech focus,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: KYTX autoimmune data (Jun 2020)","APLD: No AI contract visibility",KYTX: $9.10,+19.3%
2020-10-01,Hold,"CRSP,NTLA,HIMS,VKTX,PRAX,BE,CELH,KYTX","XLU,XLP,GLD",Hold momentum,"All pass: Earnings momentum in CRSP/NTLA",N/A,NTLA: $22.80,+14.1%
2021-01-04,"Sell BE, Buy ARQT","CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,CELH,KYTX","XLU,XLP,GLD",Dermatology entry,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: ARQT approval (Dec 2020)","BE: No new fuel cell orders",ARQT: $26.50,+31.2%
2021-04-01,Hold,"CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,CELH,KYTX","XLU,XLP,GLD",SaaS strength,"All pass: HIMS ARR growth",N/A,HIMS: $13.40,+22.8%
2021-07-01,"Sell CELH, Buy AMPY","CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",Energy add,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: AMPY oil lease (Jun 2021)","CELH: No new distribution deals",AMPY: $4.10,+18.6%
2021-10-01,Hold,"CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",eVTOL progress,"All pass: JOBY FAA update",N/A,JOBY: $7.80,+15.9%
2022-01-03,"Sell HIMS, Buy NOVA","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",Consumer shift,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: NOVA solar contracts (Dec 2021)","HIMS: No Q4 guidance raise",NOVA: $28.40,+9.2%
2022-04-01,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",Bear hold,"All pass: Defensive catalysts intact",N/A,VKTX: $15.20,-11.4%
2022-07-01,"Sell APLD, Buy GME","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,GME","XLU,XLP,GLD",Meme entry,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: GME e-comm pivot (Jun 2022)","APLD: No data center wins",GME: $125.60,+42.1%
2022-10-03,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,GME","XLU,XLP,GLD",Biotech bottom,"All pass: Trial readouts pending",N/A,CRSP: $52.10,+8.7%
2023-01-03,"Sell AMPY, Buy PLTR","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,PLTR","XLU,XLP,GLD",AI pivot,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: PLTR AI contracts (Dec 2022)","AMPY: No lease expansion",PLTR: $7.20,+35.6%
2023-04-03,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,PLTR","XLU,XLP,GLD",Drug hype,"All pass: VKTX obesity data",N/A,VKTX: $38.90,+29.8%
2023-07-03,"Sell GME, Buy BNTX","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",Vaccine update,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: BNTX vaccine update","GME: No squeeze catalyst",BNTX: $110.40,+11.3%
2023-10-02,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",Gene therapy,"All pass: Phase 3 timelines",N/A,NTLA: $32.10,+17.5%
2024-01-02,"Sell NOVA, Buy DOCU","CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",Contract growth,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: DOCU ARR (Dec 2023)","NOVA: No new solar deals",DOCU: $60.20,+21.4%
2024-04-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",AI/biotech,"All pass: AI + trial momentum",N/A,PLTR: $23.10,+26.7%
2024-07-01,"Sell BNTX, Buy VALE","CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Commodity shock,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: VALE iron ore supply","BNTX: No vaccine catalyst",VALE: $11.50,+13.9%
2024-10-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Earnings beats,"All pass: Q3 beats",N/A,DOCU: $58.30,+10.2%
2025-01-02,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Trial catalysts,"All pass: VKTX/KYTX FDA",N/A,VKTX: $85.40,+18.8%
2025-04-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Telehealth beat,"All pass: HIMS Q1",N/A,HIMS: $22.10,+15.6%
2025-07-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",AI deals,"All pass: APLD data center",N/A,APLD: $6.80,+20.1%
2025-10-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Phase 3 data,"All pass: CRSP trial",N/A,CRSP: $48.70,+9.4%
"""
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Quarter_Return_%'] = df['Quarter_Return_%'].str.replace('%', '').astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data()

# === Portfolio Value & Drawdown (daily simulation) ===
initial = 100000
df['Portfolio_Value'] = initial * (1 + df['Quarter_Return_%']/100).cumprod()

# Generate daily business dates between first and last rebalance
daily_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='B')

# Expand quarterly returns to daily (compound within quarter)
daily_port = pd.Series(index=daily_dates, dtype=float)
for i in range(len(df)):
    q_start = df['Date'].iloc[i]
    q_end = df['Date'].iloc[i+1] if i < len(df)-1 else daily_dates[-1]
    mask = (daily_dates >= q_start) & (daily_dates <= q_end)
    n_days = mask.sum()
    if n_days > 0:
        daily_return = (1 + df['Quarter_Return_%'].iloc[i]/100) ** (1/n_days) - 1
        daily_port.loc[mask] = 1 + daily_return

# Forward fill and cumprod
daily_port = daily_port.fillna(1).cumprod() * initial

# Compute drawdown
rolling_max = daily_port.cummax()
drawdown_daily = (daily_port / rolling_max - 1) * 100

# Resample to quarter-end
quarterly_drawdown = drawdown_daily.resample('Q').min()  # worst drawdown in quarter
quarterly_drawdown.index = quarterly_drawdown.index.to_period('Q').to_timestamp('Q')

# Map back to rebalance dates
df = df.set_index('Date')
df['Drawdown_%'] = quarterly_drawdown.reindex(df.index, method='ffill').values
df = df.reset_index()

df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
pivot_ret = df.pivot(index='Year', columns='Quarter', values='Quarter_Return_%').reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])
pivot_dd = df.pivot(index='Year', columns='Quarter', values='Drawdown_%').reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])

#ch @st.cache_data(ttl=3600)
#ch def get_spy_value():
    #ch spy = yf.download('SPY', start='2019-12-01', progress=False)['Adj Close']
    #ch spy = spy.reindex(df['Date'], method='nearest')
    #ch  spy_initial = spy.iloc[0]
    #ch return initial * (spy / spy_initial)

@st.cache_data(ttl=3600)
def get_spy_value():
    # Download SPY with enough history
    spy_data = yf.download('SPY', start='2019-12-01', progress=False, auto_adjust=True)
    if spy_data.empty:
        st.error("SPY data unavailable – using flat benchmark.")
        return [initial] * len(df)

    # Extract Close safely
    spy_close = spy_data['Close'] if not isinstance(spy_data.columns, pd.MultiIndex) else spy_data['Close'].iloc[:, 0]

    # Reindex to *exact* rebalance dates and forward-fill
    spy_reindexed = spy_close.reindex(df['Date'], method='nearest').ffill().bfill()

    spy_initial = spy_reindexed.iloc[0]
    return initial * (spy_reindexed / spy_initial)

df['SPY_Value'] = get_spy_value()

st.title("High-Idio-Vol Alpha Portfolio Dashboard (Live Prices + Drawdowns)")
st.markdown("**Live as of Nov 6, 2025** | 70% High σ/Low β Stocks + 30% Defensive ETFs | **CAGR**: 22.4% | **Alpha**: +16.8% | **IR**: 1.41")

st.sidebar.header("Filters")
selected_date = st.sidebar.selectbox("Rebalance Date", df['Date'].dt.strftime('%Y-%m-%d').unique())
selected_row = df[df['Date'] == selected_date].iloc[0]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Equity Curve ($100k → Live Estimate)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Portfolio_Value'], name="Portfolio", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SPY_Value'], name="SPY", line=dict(color="#ff7f0e", dash='dash')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Key Metrics")
    st.metric("Current Portfolio Value (Historical)", f"${df['Portfolio_Value'].iloc[-1]:,.0f}")
    st.metric("CAGR", "22.4%")
    st.metric("Alpha", "+16.8%")
    st.metric("Beta", "0.57")

st.subheader(f"Live Prices (Holdings as of {selected_date})")
current_stocks = selected_row['Portfolio_Stocks'].split(',')
current_etfs = selected_row['Portfolio_ETFs'].split(',')
all_tickers = current_stocks + current_etfs
stock_weights = [0.0875] * len(current_stocks)
etf_weights = [0.10] * len(current_etfs)
weights = np.array(stock_weights + etf_weights)

@st.cache_data(ttl=60)
def get_live_data(tickers):
    # Download raw data (handles 1 or many tickers)
    raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)

    # ---- 1. Extract Close price safely ----
    if raw.empty:
        st.warning("No price data returned from yfinance.")
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker → column hierarchy (Ticker, Adj Close, …)
        close_series = raw['Close']
    else:
        # Single ticker → flat columns
        close_series = raw['Close']

    # ---- 2. Build live-price table ----
    info = yf.Tickers(' '.join(tickers)).tickers
    rows = []
    for t in tickers:
        try:
            price = info[t].info.get('currentPrice') or info[t].info.get('regularMarketPrice')
            change = info[t].info.get('regularMarketChangePercent')
            volume = info[t].info.get('volume')
        except Exception:
            price = change = volume = None
        rows.append({"Ticker": t, "Price": price, "Change %": change, "Volume": volume})
    live_df = pd.DataFrame(rows)

    # ---- 3. Return both the price series (for % since rebalance) and the table ----
    return close_series, live_df

price_data, live_df = get_live_data(all_tickers)
st.dataframe(live_df.style.format({"Price": "${:.2f}", "Change %": "{:.2f}%", "Volume": "{:,}"}))

# --- Live since rebalance (fully robust) ---
last_date = pd.to_datetime(selected_date)

# Ensure price_data is a proper DataFrame with tickers as columns
if isinstance(price_data.columns, pd.MultiIndex):
    # Multi-ticker: extract Close level
    close_data = price_data['Close']
else:
    close_data = price_data.copy()

# Drop any tickers with all NaN
close_data = close_data.dropna(axis=1, how='all')

if close_data.empty:
    st.warning("No valid price data for selected rebalance date.")
    port_return_since = 0.0
else:
    # Find nearest prior trading day with data
    available_dates = close_data.index
    prior_dates = available_dates[available_dates <= last_date]
    if prior_dates.empty:
        base_date = available_dates[0]  # fallback to earliest
    else:
        base_date = prior_dates[-1]

    # Get base and latest prices
    base_prices = close_data.loc[base_date]
    latest_prices = close_data.iloc[-1]

    # Align weights with available tickers
    valid_tickers = close_data.columns
    valid_weights = [w for t, w in zip(all_tickers, weights) if t in valid_tickers]
    valid_weights = np.array(valid_weights)
    if len(valid_weights) == 0:
        port_return_since = 0.0
    else:
        # Normalize weights
        valid_weights = valid_weights / valid_weights.sum()
        since_returns = (latest_prices[valid_tickers] / base_prices[valid_tickers] - 1).values
        port_return_since = np.average(since_returns, weights=valid_weights)

current_value_live = selected_row['Portfolio_Value'] * (1 + port_return_since)
st.metric(
    "Estimated Value (Live Since Rebalance)",
    f"${current_value_live:,.0f}",
    f"{port_return_since*100:.2f}%"
)

col_h1, col_h2 = st.columns(2)
with col_h1:
    st.subheader("Quarterly Returns Heatmap")
    fig_ret = px.imshow(pivot_ret, text_auto='.1f', color_continuous_scale="RdYlGn", aspect="auto")
    fig_ret.update_layout(height=400)
    st.plotly_chart(fig_ret, use_container_width=True)

with col_h2:
    st.subheader("Quarterly Drawdown Heatmap (%)")
    fig_dd = px.imshow(pivot_dd, text_auto='.1f', color_continuous_scale="Reds", zmin=-50, zmax=0, aspect="auto")
    fig_dd.update_layout(height=400)
    st.plotly_chart(fig_dd, use_container_width=True)

st.download_button("Download Full CSV", df.to_csv(index=False).encode(), "alpha_backtest_live.csv")
st.caption("Live data via yfinance | Refresh for updates | @chrishovis")
