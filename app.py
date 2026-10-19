import streamlit as st
import yfinance as yf
import pandas as pd
import time

# --- APP SETUP ---
st.set_page_config(page_title="500-Stock QARP Master Engine", layout="wide")
st.title("🏛️ 500-Stock Automated QARP Screener")
st.markdown("**Buffett + Greenblatt Logic | 100% Live Market Data | Zero CSV/Upload Needed**")

# Nifty 500 ke sabse bade aur important tickers ki automated base list
# Aap isme aur bhi tickers add kar sakte hain
nifty_500_sample = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "LT.NS", 
    "HINDUNILVR.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", 
    "ADANIENT.NS", "TITAN.NS", "KOTAKBANK.NS", "ONGC.NS", "NTPC.NS",
    "AXISBANK.NS", "TATAMOTORS.NS", "ADANIPORTS.NS", "COALINDIA.NS", "DMART.NS",
    "SIEMENS.NS", "GRASIM.NS", "JIOFIN.NS", "BAJAJFINSV.NS", "IOC.NS",
    "POWERGRID.NS", "ATGL.NS", "ZOMATO.NS", "PNB.NS", "HAL.NS",
    "BPCL.NS", "ASIANPAINT.NS", "BEL.NS", "VBL.NS", "NAUKRI.NS",
    "DIXON.NS", "PIDILITEIND.NS", "LODHA.NS", "TATASTEEL.NS", "SBILIFE.NS",
    "ICICIGI.NS", "INDIGO.NS", "DIVISLAB.NS", "BPCL.NS", "HDFCLIFE.NS"
] # (Aap is list mein aur bhi NSE ke stocks jod sakte hain)

st.sidebar.header("⚙️ Scanner Settings")
batch_size = st.sidebar.slider("Batch Speed (Aaram se scan karne ke liye)", 5, 20, 10)
run_scan = st.sidebar.button("🚀 Start 500-Stock Deep Scan")

if run_scan:
    st.info("Scanner shuru ho gaya hai. Server ko safe rakhne ke liye yeh batches mein scan karega...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total_stocks = len(nifty_500_sample)
    
    for i in range(0, total_stocks, batch_size):
        batch = nifty_500_sample[i:i + batch_size]
        status_text.text(f"Scanning stocks {i+1} to {min(i+batch_size, total_stocks)} of {total_stocks}...")
        
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                name = info.get('shortName', ticker)
                price = info.get('currentPrice', 0)
                eps = info.get('trailingEps', 0)
                pe = info.get('trailingPE', 0)
                high52 = info.get('fiftyTwoWeekHigh', 0)
                
                if pe is None: pe = 0
                if eps is None: eps = 0
                
                # Drop from high calculation
                drop_from_high = ((high52 - price) / high52) * 100 if high52 > 0 else 0
                
                # Raw Fundamental metrics
                total_debt = info.get('totalDebt', 0)
                total_equity = info.get('totalStockholderEquity', 0)
                roce_raw = info.get('returnOnEquity', 0)
                
                debt_to_equity = (total_debt / total_equity) if (total_equity and total_equity > 0) else 0
                roce = (roce_raw * 100) if roce_raw else 0
                
                # 🛡️ STRICT QARP RULES (Method 2)
                # Rule: EPS > 0, ROCE >= 15, Debt < 0.5, PE < 35
                if eps > 0 and roce >= 15 and debt_to_equity < 0.5 and pe > 0 and pe < 35:
                    signal = "🟢🔥 STRONG QARP BUY"
                elif eps > 0 and pe > 0 and pe < 30:
                    signal = "🟡 FAIR VALUE / WATCHLIST"
                else:
                    signal = "🚨 AVOID / TRAP"
                    
                results.append({
                    "Company": name,
                    "Ticker": ticker.replace(".NS", ""),
                    "Price (₹)": price,
                    "Drop (%)": round(drop_from_high, 1),
                    "P/E": round(pe, 1),
                    "ROCE (%)": round(roce, 1),
                    "Debt/Eq": round(debt_to_equity, 2),
                    "Signal": signal
                })
            except Exception:
                continue
                
        progress_bar.progress(min((i + batch_size) / total_stocks, 1.0))
        time.sleep(1) # Server block rokne ke liye chota pause
        
    status_text.text("Scan Complete! ✅")
    
    if results:
        df = pd.DataFrame(results)
        
        # Sirf QARP Winners filter karna
        winners = df[df['Signal'].str.contains('STRONG QARP BUY')]
        
        st.write(f"### 🏆 Top QARP Wealth Compounders Found: {len(winners)}")
        if not winners.empty:
            st.dataframe(winners.style.background_gradient(subset=['ROCE (%)'], cmap='Greens'), use_container_width=True)
            st.balloons()
        else:
            st.warning("Current market price par koi bhi stock strict QARP rule pass nahi kar paya.")
            
        st.write("#### 📊 Complete Scanned Database:")
        st.dataframe(df, use_container_width=True)
