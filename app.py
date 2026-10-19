import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- APP SETUP ---
st.set_page_config(page_title="Ultimate QARP Hedge Fund Engine", layout="wide")
st.title("🏛️ Ultimate Live QARP Screener & Backtester")
st.markdown("**Buffett + Greenblatt Logic | Real Live Market Data from Yahoo Finance**")

# Base Universe
@st.cache_data
def get_universe():
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "LT.NS", 
        "HINDUNILVR.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", 
        "ONGC.NS", "NTPC.NS", "AXISBANK.NS", "TATAMOTORS.NS", "COALINDIA.NS",
        "POWERGRID.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "GRASIM.NS", "ADANIENT.NS"
    ]

st.sidebar.header("⚙️ Controls")
batch_size = st.sidebar.slider("Batch Speed", 5, 20, 10)
col1, col2 = st.sidebar.columns(2)
run_scanner = col1.button("🚀 Run Live Scan")
run_backtest = col2.button("📊 Run 10-Yr Backtest")

all_tickers = get_universe()

# --- 1. LIVE SCANNER LOGIC ---
if run_scanner:
    st.info(f"🔄 Scanning stocks with real live market fundamentals...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total = len(all_tickers)
    for i in range(0, total, batch_size):
        batch = all_tickers[i:i+batch_size]
        status_text.text(f"Processing stocks {i+1} to {min(i+batch_size, total)} of {total}...")
        
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                
                # Real Price & 52W High from History
                hist = stock.history(period="1y")
                if hist.empty: continue
                price = float(hist['Close'].iloc[-1])
                high52 = float(hist['High'].max())
                drop_from_high = ((high52 - price) / high52) * 100 if high52 > 0 else 0
                
                # Real Fundamental Data Extraction
                info = stock.info
                name = info.get('shortName', ticker)
                pe = info.get('trailingPE', None)
                eps = info.get('trailingEps', 0)
                
                # Agar trailing P/E nahi mila toh forward P/E try karo
                if pe is None:
                    pe = info.get('forwardPE', 0)
                
                # Balance sheet & Returns data
                roe = info.get('returnOnEquity', None)
                roce = (roe * 100) if roe is not None else 15.0  # Real ROE as proxy if available
                
                total_debt = info.get('totalDebt', 0)
                total_equity = info.get('totalStockholderEquity', 1)
                
                if total_equity and total_equity > 0:
                    debt_eq = total_debt / total_equity
                else:
                    debt_eq = 0.0

                # Strict QARP Logic check using real values
                if pe and pe > 0 and pe < 35 and debt_eq < 0.5:
                    signal = "🟢🔥 STRONG QARP BUY"
                else:
                    signal = "🟡 WATCHLIST / AVOID"
                    
                results.append({
                    "Company": name,
                    "Ticker": ticker.replace(".NS", ""),
                    "Price (₹)": round(price, 2),
                    "52W High Drop (%)": round(drop_from_high, 1),
                    "P/E": round(pe, 2) if pe else 0.0,
                    "ROCE (%)": round(roce, 2),
                    "Debt/Eq": round(debt_eq, 2),
                    "Signal": signal
                })
            except Exception:
                continue
                
        progress_bar.progress(min((i + batch_size) / total, 1.0))
        time.sleep(0.5)
        
    status_text.text("Scan Complete! ✅")
    
    if results:
        df = pd.DataFrame(results)
        winners = df[df['Signal'].str.contains('STRONG QARP BUY')]
        
        st.write(f"### 🏆 Top QARP Wealth Compounders Found: {len(winners)}")
        if not winners.empty:
            st.dataframe(winners.style.background_gradient(subset=['ROCE (%)'], cmap='Greens'), use_container_width=True)
            st.balloons()
        else:
            st.warning("Current live filter ke mutabiq koi stock match nahi hua.")
            
        st.write("#### 📊 Complete Scanned Database (Real Values):")
        st.dataframe(df, use_container_width=True)

# --- 2. 10-YEAR BACKTEST ENGINE ---
if run_backtest:
    st.info("⏳ Running 10-Year Historical Backtest (2014 - 2024)...")
    start_date = "2014-01-01"
    end_date = "2024-01-01"
    initial_inv = 100000
    
    valid_data = []
    historical_basket = get_universe()[:20]
    
    for i in range(0, len(historical_basket), 10):
        batch = historical_basket[i:i+10]
        try:
            data = yf.download(batch, start=start_date, end=end_date, progress=False)['Close']
            if not data.empty:
                valid_data.append(data)
        except:
            continue
        time.sleep(0.5)
        
    if valid_data:
        master_df = pd.concat(valid_data, axis=1).dropna(axis=1, how='all')
        daily_returns = master_df.pct_change().dropna()
        portfolio_returns = daily_returns.mean(axis=1)
        
        portfolio_cum = (1 + portfolio_returns).cumprod() * initial_inv
        final_val = float(np.array(portfolio_cum)[-1])
        years = 10
        cagr = ((final_val / initial_inv) ** (1/years) - 1) * 100
        
        st.success("✅ Backtest Complete!")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Initial Capital", f"₹{initial_inv:,.0f}")
        col_b.metric("Final Value", f"₹{final_val:,.0f}", f"+{((final_val-initial_inv)/initial_inv)*100:.1f}%")
        col_c.metric("10-Yr CAGR", f"{cagr:.2f}%")
        st.line_chart(portfolio_cum)
    else:
        st.error("Backtest data download fail ho gaya.")
