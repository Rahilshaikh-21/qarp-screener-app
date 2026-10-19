import io
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Buffett-Greenblatt Hybrid Screener", layout="wide")
st.title("🏛️ Buffett-Greenblatt Hybrid Stock Screener")
st.caption("Greenblatt Magic Formula (ROCE + Earnings Yield) + Buffett Safety (Low Debt + Consistent Growth)")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
NSE = "https://archives.nseindia.com/content"
URLS = {
    "nifty500": f"{NSE}/indices/ind_nifty500list.csv",
    "micro250": f"{NSE}/indices/ind_niftymicrocap250_list.csv",
    "all": f"{NSE}/equities/EQUITY_L.csv",
}


# ------------------------------------------------------------------ helpers
def get_symbols(universe, limit):
    def read(url):
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))

    if universe == "Nifty 500 + Microcap 250 (~750 companies)":
        d = pd.concat([read(URLS["nifty500"]), read(URLS["micro250"])])
        syms = d["Symbol"].astype(str).str.strip().unique().tolist()
    else:
        d = read(URLS["all"])
        d.columns = d.columns.str.strip()
        d = d[d["SERIES"].str.strip() == "EQ"]
        syms = d["SYMBOL"].astype(str).str.strip().unique().tolist()
        random.Random(42).shuffle(syms)  # alphabetical bias hatane ke liye
        syms = syms[:limit]
    return syms


def pct(x):
    return None if x is None or pd.isna(x) else float(x) * 100


def fetch_stage1(sym):
    """Fast data from yfinance .info"""
    try:
        i = yf.Ticker(sym + ".NS").info
        de = i.get("debtToEquity")
        ev, ebitda = i.get("enterpriseValue"), i.get("ebitda")
        return {
            "Symbol": sym,
            "Name": i.get("shortName", sym),
            "Price": i.get("currentPrice"),
            "MCap_Cr": (i.get("marketCap") or 0) / 1e7,
            "PE": i.get("trailingPE"),
            "ROE": pct(i.get("returnOnEquity")),
            "DE": None if de is None else de / 100,  # yfinance percent deta hai
            "Rev_Growth": pct(i.get("revenueGrowth")),
            "EPS_Growth": pct(i.get("earningsGrowth")),
            "EV_EBITDA_Yield": (ebitda / ev * 100) if ev and ebitda and ev > 0 else None,
        }
    except Exception:
        return None


def _row(df, names):
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _cagr(series):
    s = series.dropna()
    n = len(s) - 1
    if n >= 2 and s.iloc[0] > 0 and s.iloc[-1] > 0:
        return ((s.iloc[0] / s.iloc[-1]) ** (1 / n) - 1) * 100
    return None


def _up_years(series):
    s = series.dropna().iloc[::-1]  # purana -> naya
    return int((s.diff().dropna() > 0).sum()), max(len(s) - 1, 0)


def fetch_stage2(sym):
    """Accurate ROCE + multi-year growth from financial statements"""
    out = {"Symbol": sym, "ROCE": None, "Sales_CAGR": None, "Profit_CAGR": None, "Sales_Up_Years": ""}
    try:
        t = yf.Ticker(sym + ".NS")
        fin, bs = t.financials, t.balance_sheet
        ebit = _row(fin, ["EBIT", "Operating Income"])
        ta = _row(bs, ["Total Assets"])
        cl = _row(bs, ["Current Liabilities", "Total Current Liabilities"])
        if ebit is not None and ta is not None and cl is not None:
            cap = ta.iloc[0] - cl.iloc[0]
            if cap > 0:
                out["ROCE"] = ebit.iloc[0] / cap * 100
        rev = _row(fin, ["Total Revenue", "Operating Revenue"])
        ni = _row(fin, ["Net Income", "Net Income Common Stockholders"])
        if rev is not None:
            out["Sales_CAGR"] = _cagr(rev)
            up, tot = _up_years(rev)
            out["Sales_Up_Years"] = f"{up}/{tot}"
        if ni is not None:
            out["Profit_CAGR"] = _cagr(ni)
    except Exception:
        pass
    return out


def run_parallel(fn, items, label, workers=16):
    results, bar = [], st.progress(0.0, text=label)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, s) for s in items]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                results.append(r)
            bar.progress(k / len(items), text=f"{label} ({k}/{len(items)})")
    bar.empty()
    return results


def rank_score(df, use_growth=True):
    df = df.copy()
    df["Earnings_Yield"] = 100 / df["PE"]
    df["R_ROCE"] = df["ROCE"].rank(ascending=False)
    df["R_EY"] = df["Earnings_Yield"].rank(ascending=False)
    df["Score"] = df["R_ROCE"] + df["R_EY"]
    if use_growth and "Profit_CAGR" in df and df["Profit_CAGR"].notna().any():
        df["R_Growth"] = df["Profit_CAGR"].rank(ascending=False)
        df["Score"] += df["R_Growth"]
    return df.sort_values("Score")


# ------------------------------------------------------------------ sidebar
st.sidebar.header("⚙️ Strategy Settings")
roce_min = st.sidebar.slider("Min ROCE (%)", 10, 40, 20)
de_max = st.sidebar.slider("Max Debt/Equity", 0.0, 1.5, 0.5, 0.1)
pe_max = st.sidebar.slider("Max P/E", 10, 80, 35)
growth_min = st.sidebar.slider("Min Sales & Profit CAGR (%)", 0, 25, 10)
mcap_min = st.sidebar.number_input("Min Market Cap (₹ Crore)", value=500, step=100)
top_n = st.sidebar.slider("Kitne top stocks dikhane hain", 5, 30, 15)

tab1, tab2 = st.tabs(["🚀 Auto Scan (Live Data)", "📁 CSV Upload (Screener.in)"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    c1, c2 = st.columns([2, 1])
    universe = c1.selectbox(
        "Kaunsi companies scan karni hain?",
        ["Nifty 500 + Microcap 250 (~750 companies)", "All NSE listed (random sample)"],
    )
    limit = c2.number_input("All NSE mode me kitni companies", 200, 2500, 1000, 100)

    if st.button("🚀 SCAN START - Top Stocks Dhundo", type="primary", use_container_width=True):
        try:
            with st.spinner("NSE se company list la raha hoon..."):
                symbols = get_symbols(universe, int(limit))
            st.info(f"{len(symbols)} companies mili. Scan shuru (5-10 minute lag sakte hain)...")
        except Exception as e:
            st.error(f"NSE list download nahi hui: {e}. Tab 2 me CSV upload use karein.")
            st.stop()

        # Stage 1: quick data
        s1 = pd.DataFrame(run_parallel(fetch_stage1, symbols, "Stage 1: Basic data"))
        for c in ["MCap_Cr", "PE", "ROE", "DE", "Rev_Growth", "EPS_Growth"]:
            s1[c] = pd.to_numeric(s1[c], errors="coerce")

        pre = s1[
            (s1["MCap_Cr"] >= mcap_min)
            & (s1["PE"] > 0) & (s1["PE"] < pe_max)
            & (s1["ROE"] >= roce_min * 0.6)   # loose pre-filter
            & (s1["DE"].fillna(0) <= de_max)
        ].copy()
        st.write(f"Stage 1 ke baad **{len(pre)}** companies bachi.")

        # Quick rank, top 80 ka hi detailed data nikalo
        pre["Earnings_Yield"] = 100 / pre["PE"]
        pre["q"] = pre["ROE"].rank(ascending=False) + pre["Earnings_Yield"].rank(ascending=False)
        shortlist = pre.sort_values("q").head(80)

        # Stage 2: accurate ROCE + growth
        s2 = pd.DataFrame(run_parallel(fetch_stage2, shortlist["Symbol"].tolist(), "Stage 2: ROCE & growth check"))
        merged = shortlist.merge(s2, on="Symbol")
        final = merged[
            (merged["ROCE"] >= roce_min)
            & (merged["Sales_CAGR"] >= growth_min)
            & (merged["Profit_CAGR"] >= growth_min)
        ]
        st.session_state["result"] = rank_score(final).head(top_n) if len(final) else final
        st.session_state["all"] = s1

    if "result" in st.session_state:
        res = st.session_state["result"]
        if len(res) == 0:
            st.warning("Koi stock sab filters pass nahi kar paya. Sidebar me ROCE ya growth thoda kam karke dobara try karein.")
        else:
            st.subheader(f"🏆 Top {len(res)} Buffett-Greenblatt Stocks")
            cols = ["Symbol", "Name", "Price", "MCap_Cr", "PE", "Earnings_Yield", "ROCE", "DE",
                    "Sales_CAGR", "Profit_CAGR", "Sales_Up_Years", "Score"]
            show = res[cols].round(2).reset_index(drop=True)
            show.index += 1
            st.dataframe(show.style.background_gradient(subset=["ROCE"], cmap="Greens"),
                         use_container_width=True)
            st.download_button("⬇️ CSV download", show.to_csv().encode(), "top_stocks.csv")
            st.balloons()
        with st.expander("Poora scanned data dekhein"):
            st.dataframe(st.session_state["all"], use_container_width=True)

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.write("Screener.in export (columns: Name, CMP, P/E, ROCE, Debt to equity, Sales growth 3Years, Profit growth 3Years).")
    f = st.file_uploader("CSV/Excel upload karein", type=["csv", "xlsx"])
    if f:
        df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        df.columns = df.columns.str.strip()
        sg = next((c for c in df.columns if "sales" in c.lower() and "growth" in c.lower()), None)
        pg = next((c for c in df.columns if "profit" in c.lower() and "growth" in c.lower()), None)
        df = df.rename(columns={"P/E": "PE", "Debt to equity": "DE"})
        if sg: df = df.rename(columns={sg: "Sales_CAGR"})
        if pg: df = df.rename(columns={pg: "Profit_CAGR"})
        for c in ["PE", "ROCE", "DE", "Sales_CAGR", "Profit_CAGR"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        need = {"ROCE", "PE", "DE"}
        if not need.issubset(df.columns):
            st.error("ROCE, P/E aur Debt to equity columns zaroori hain.")
        else:
            m = (df["ROCE"] >= roce_min) & (df["DE"] <= de_max) & (df["PE"] > 0) & (df["PE"] < pe_max)
            if "Sales_CAGR" in df: m &= df["Sales_CAGR"] >= growth_min
            if "Profit_CAGR" in df: m &= df["Profit_CAGR"] >= growth_min
            out = rank_score(df[m]).head(top_n)
            st.subheader(f"🏆 Top {len(out)} stocks")
            st.dataframe(out, use_container_width=True)
