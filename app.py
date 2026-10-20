"""CoffeeCan + GARP Quant Scanner & 15-Year Backtester (NSE). Run: streamlit run app.py"""
import io
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
NAN = float("nan")
RF = 0.06  # liquid-fund / risk-free proxy

BUILTIN = ("RELIANCE TCS HDFCBANK INFY ICICIBANK HINDUNILVR ITC BAJFINANCE ASIANPAINT TITAN PIDILITIND HAVELLS "
           "NESTLEIND BRITANNIA MARICO DABUR COLPAL PAGEIND BERGEPAINT ABBOTINDIA SUNPHARMA DIVISLAB TATAELXSI "
           "PERSISTENT COFORGE POLYCAB ASTRAL SUPREMEIND DIXON TRENT CDSL CAMS HDFCAMC MUTHOOTFIN SBILIFE HDFCLIFE "
           "EICHERMOT HEROMOTOCO MARUTI TVSMOTOR PIIND DEEPAKNTR SRF ATUL SIEMENS ABB CUMMINSIND KEI BLUESTARCO "
           "VOLTAS IRCTC GODREJCP TATACONSUM VBL RELAXO CRISIL MPHASIS NAUKRI LALPATHLAB ALKEM TORNTPHARM AJANTPHARM "
           "CIPLA LUPIN KOTAKBANK AXISBANK LT WIPRO HCLTECH TECHM BAJAJ-AUTO DMART INDIAMART").split()
NSE_URLS = {"Nifty 500 (NSE archive)": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "Nifty Total Market (NSE archive)": "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"}

ALIASES = {
    "Ticker": ["nse code", "ticker", "symbol", "nse"], "Name": ["name", "company"],
    "Sector": ["sector", "industry"], "CMP": ["cmp rs.", "cmp", "current price", "price"],
    "MCap": ["mar cap rs.cr.", "market cap", "mcap", "mar cap"], "ROE": ["roe", "return on equity", "roe %"],
    "ROCE": ["roce", "return on capital employed", "roce %"],
    "SalesCAGR": ["sales growth 5years", "sales growth 5 years", "sales cagr 5y", "sales growth 3years"],
    "ProfitCAGR": ["profit growth 5years", "profit growth 5 years", "profit cagr 5y", "profit growth 3years"],
    "DE": ["debt to equity", "debt/equity", "d/e"], "PE": ["p/e", "pe", "stock p/e", "price to earning"],
    "ICR": ["interest coverage ratio", "icr", "int coverage"], "CFO_PAT": ["cfo/pat"],
    "NPM": ["npm", "net profit margin", "npm %"], "PB": ["pb", "price to book value", "p/b"],
}


# ----------------------------------------------------------------------------- scoring
def lin(x, lo, hi):
    """0..1 linear ramp; missing data gets a neutral 0.5."""
    if x is None or pd.isna(x):
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def score_stock(m):
    fin = bool(m.get("Fin"))
    if fin:
        cap = 0.6 * lin(m.get("ROE"), 8, 15) + 0.4 * lin(m.get("ROA"), 0.8, 1.5)
        solv = 1 - lin(m.get("PB"), 3, 6)
        cash = 0.5
    else:
        dup = 0.6 * lin(m.get("NPM"), 4, 12) + 0.4 * lin(m.get("AssetTurn"), 0.3, 0.8)
        cap = 0.4 * lin(m.get("ROE"), 8, 15) + 0.4 * lin(m.get("ROCE"), 10, 20) + 0.2 * dup
        solv = 0.6 * (1 - lin(m.get("DE"), 0.5, 1.5)) + 0.4 * lin(m.get("ICR"), 1, 5)
        fcf = m.get("FCF")
        fcf_s = 0.5 if (fcf is None or pd.isna(fcf)) else float(fcf > 0)
        cash = 0.6 * lin(m.get("CFO_PAT"), 0.3, 0.8) + 0.4 * fcf_s
    growth = 0.5 * lin(m.get("SalesCAGR"), 4, 12) + 0.5 * lin(m.get("ProfitCAGR"), 5, 15)
    peg = m.get("PEG")
    if peg is None or pd.isna(peg) or peg <= 0:
        garp = 0.2
    elif peg < 0.5:
        garp = lin(peg, 0.2, 0.5)
    elif peg <= 1.5:
        garp = 1.0
    else:
        garp = 1 - lin(peg, 1.5, 3.0)
    total = 30 * cap + 25 * growth + 20 * cash + 15 * solv + 10 * garp
    return round(total, 1)


# ----------------------------------------------------------------------------- fundamentals
def _row(df, names):
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            return df.loc[n].dropna().astype(float).sort_index()
    return pd.Series(dtype=float)


def _cagr(s):
    s = s[s > 0]
    if len(s) < 2:
        return NAN
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs >= 1 else NAN


def _last(s):
    return float(s.iloc[-1]) if len(s) else NAN


def fetch_one(tk, retries=3):
    err = ""
    for i in range(retries):
        try:
            t = yf.Ticker(tk)
            info = t.info or {}
            if not info or ("currentPrice" not in info and "regularMarketPrice" not in info):
                raise ValueError("no data")
            inc, bs, cf = t.income_stmt, t.balance_sheet, t.cashflow
            rev = _row(inc, ["Total Revenue", "Operating Revenue"])
            ni = _row(inc, ["Net Income", "Net Income Common Stockholders"])
            ebit = _row(inc, ["EBIT", "Operating Income"])
            intr = _row(inc, ["Interest Expense", "Interest Expense Non Operating"])
            ta = _row(bs, ["Total Assets"])
            cl = _row(bs, ["Current Liabilities", "Total Current Liabilities"])
            eq = _row(bs, ["Stockholders Equity", "Common Stock Equity"])
            cfo = _row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
            fcfs = _row(cf, ["Free Cash Flow"])
            sector = info.get("sector", "NA")
            fin = sector == "Financial Services"
            pe = info.get("trailingPE", NAN)
            pcagr = _cagr(ni)
            de = info.get("debtToEquity", NAN)
            de = de / 100 if de is not None and not pd.isna(de) else NAN
            roe = info.get("returnOnEquity")
            roe = roe * 100 if roe is not None else (_last(ni) / _last(eq) * 100 if len(eq) and len(ni) else NAN)
            capemp = _last(ta) - _last(cl) if len(ta) and len(cl) else NAN
            ni_l = _last(ni)
            return {
                "Ticker": tk, "Name": info.get("shortName", tk), "Sector": sector,
                "CMP": info.get("currentPrice", info.get("regularMarketPrice", NAN)),
                "MCap": (info.get("marketCap") or NAN) / 1e7, "Fin": fin,
                "ROE": roe, "ROA": (info.get("returnOnAssets") or NAN) * 100,
                "ROCE": _last(ebit) / capemp * 100 if capemp and capemp > 0 else NAN,
                "NPM": _last(ni) / _last(rev) * 100 if len(rev) and len(ni) else NAN,
                "AssetTurn": _last(rev) / _last(ta) if len(rev) and len(ta) else NAN,
                "SalesCAGR": _cagr(rev), "ProfitCAGR": pcagr, "DE": de,
                "ICR": abs(_last(ebit) / _last(intr)) if len(intr) and len(ebit) and _last(intr) != 0 else NAN,
                "CFO_PAT": _last(cfo) / ni_l if len(cfo) and ni_l and ni_l > 0 else NAN,
                "FCF": _last(fcfs) if len(fcfs) else info.get("freeCashflow", NAN),
                "PE": pe, "PB": info.get("priceToBook", NAN),
                "PEG": pe / pcagr if pe and not pd.isna(pe) and pcagr and pcagr > 0 else NAN,
            }
        except Exception as e:  # rate limit / missing data
            err = str(e)[:80]
            time.sleep(1.5 ** (i + 1))
    return {"Ticker": tk, "error": err or "failed"}


def run_scan(tickers):
    cache = st.session_state.setdefault("fcache", {})
    todo = [t for t in tickers if t not in cache]
    bar, txt = st.progress(0.0), st.empty()
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch_one, t): t for t in todo}
        for f in as_completed(futs):
            cache[futs[f]] = f.result()
            done += 1
            bar.progress(done / max(len(todo), 1))
            txt.caption(f"Fetched {done}/{len(todo)} (cached: {len(tickers) - len(todo)})")
    bar.empty()
    txt.empty()
    rows = [cache[t] for t in tickers if t in cache]
    good = [r for r in rows if "error" not in r]
    return pd.DataFrame(good), len(rows) - len(good)


def parse_screener(df):
    cols = {c.strip().lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for key, names in ALIASES.items():
        src = next((cols[n] for n in names if n in cols), None)
        out[key] = pd.to_numeric(df[src], errors="coerce") if src and key not in ("Ticker", "Name", "Sector") \
            else (df[src] if src else (NAN if key not in ("Ticker", "Name", "Sector") else ""))
    if "Ticker" not in out or (out["Ticker"] == "").all():
        raise ValueError("CSV needs an 'NSE Code' / 'Ticker' / 'Symbol' column")
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False) + ".NS"
    out["Name"] = out["Name"].where(out["Name"] != "", out["Ticker"])
    out["Sector"] = out["Sector"].replace("", "NA")
    out["CFO_PAT"] = out["CFO_PAT"].apply(lambda v: v / 100 if pd.notna(v) and v > 5 else v)
    out["Fin"] = out["Sector"].astype(str).str.contains("Bank|Financ|NBFC|Insur", case=False)
    out["PEG"] = out["PE"] / out["ProfitCAGR"].where(out["ProfitCAGR"] > 0)
    out["ROA"] = NAN
    out["AssetTurn"] = NAN
    out["FCF"] = NAN
    return out[out["Ticker"] != "NAN.NS"].drop_duplicates("Ticker")


def csv_template():
    return pd.DataFrame([{"Name": "Pidilite Industries", "NSE Code": "PIDILITIND", "Sector": "Chemicals",
                          "CMP Rs.": 2900, "Mar Cap Rs.Cr.": 147000, "ROE": 23.0, "ROCE": 30.0,
                          "Sales growth 5Years": 14.0, "Profit growth 5Years": 18.0, "Debt to equity": 0.05,
                          "P/E": 75.0, "Interest Coverage Ratio": 40.0, "CFO/PAT": 0.95, "NPM": 15.0, "PB": 15.0}])


# ----------------------------------------------------------------------------- technicals
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def add_ind(df):
    df = df.copy()
    c = df["Close"]
    df["EMA50"], df["EMA200"] = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    df["SMA50"], df["SMA200"] = c.rolling(50).mean(), c.rolling(200).mean()
    df["RSI"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"], df["MSIG"] = macd, macd.ewm(span=9, adjust=False).mean()
    df["MHIST"] = df["MACD"] - df["MSIG"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["BBU"], df["BBL"] = mid + 2 * sd, mid - 2 * sd
    df["VOLX"] = df["Volume"] / df["Volume"].rolling(20).mean()
    w = c.resample("W-FRI").last()
    df["RSIW"] = rsi(w).reindex(df.index, method="ffill")
    df["SMA20W"] = w.rolling(20).mean().reindex(df.index, method="ffill")
    wk_dma = df["SMA200"].resample("W-FRI").last()
    df["WKBELOW"] = (w < 0.97 * wk_dma).reindex(df.index, method="ffill").fillna(False).astype(bool)
    return df


def tech_series(df, score, peg):
    """Vectorised technical-only conditions (fundamentals held at current values) for chart markers."""
    c = df["Close"]
    near = ((c / df["SMA50"] - 1).abs() <= 0.05) | ((c / df["SMA20W"] - 1).abs() <= 0.05)
    buy = (score >= 75) & (c > df["SMA200"]) & near & df["RSI"].between(40, 58) & (not pd.isna(peg) and peg <= 1.35)
    sell = df["WKBELOW"] & (df["SMA50"] < df["SMA200"]) & (df["RSI"] < 40)
    macd_bear = (df["MHIST"] < 0) & (df["MHIST"].rolling(6).max() > 0)
    trim = ((df["RSI"] > 78) & macd_bear) | ((c > 1.25 * df["SMA200"]) & (df["RSI"] > 70))
    return buy, trim, sell


def signal_for(df, m):
    r, reasons = df.iloc[-1], []
    score, peg, fin = m.get("Score", 0), m.get("PEG", NAN), bool(m.get("Fin"))
    # Hard sell: fundamental breakdown
    if fin:
        if pd.notna(m.get("ROE")) and m["ROE"] < 10:
            reasons.append("ROE < 10%")
    else:
        if pd.notna(m.get("ROCE")) and m["ROCE"] < 15:
            reasons.append("ROCE < 15%")
        if pd.notna(m.get("DE")) and m["DE"] > 1.0:
            reasons.append("D/E > 1.0")
        if pd.notna(m.get("CFO_PAT")) and m["CFO_PAT"] < 0.6:
            reasons.append("CFO/PAT < 0.60")
    if r["WKBELOW"] and r["SMA50"] < r["SMA200"] and r["RSI"] < 40:
        reasons.append("Weekly close >3% below 200-DMA + Death Cross + RSI<40")
    if reasons:
        return "🔴 HARD SELL", reasons
    c = r["Close"]
    macd_bear = r["MHIST"] < 0 and df["MHIST"].iloc[-6:].max() > 0
    if (pd.notna(peg) and peg > 2.2) or (r["RSI"] > 78 and macd_bear) or (c > 1.25 * r["SMA200"] and r["RSI"] > 70):
        return "🟡 TRIM 25-30%", ["PEG > 2.2 / overheated RSI+MACD / >25% above 200-DMA"]
    near = abs(c / r["SMA50"] - 1) <= 0.05 or abs(c / r["SMA20W"] - 1) <= 0.05
    if score >= 75 and c > r["SMA200"] and near and 40 <= r["RSI"] <= 58 and pd.notna(peg) and peg <= 1.35:
        return "🟢 STRONG BUY", ["Quality + uptrend + pullback to support + RSI 40-58 + PEG<=1.35"]
    if score >= 75 and c > r["SMA50"] and 58 < r["RSI"] <= 70 and pd.notna(peg) and 1.35 < peg <= 1.75:
        return "🔵 ACCUMULATE", ["Quality + above 50-DMA + RSI 58-70 + PEG 1.35-1.75"]
    return "⚪ HOLD / WATCH", ["No trigger"]


def get_hist(tk, period="3y"):
    cache = st.session_state.setdefault("hcache", {})
    key = (tk, period)
    if key not in cache:
        d = yf.Ticker(tk).history(period=period, auto_adjust=True)
        if d.empty:
            return None
        d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
        cache[key] = add_ind(d)
    return cache[key]


def hist_pe_median(tk, cur_pe):
    try:
        t = yf.Ticker(tk)
        eps = _row(t.income_stmt, ["Diluted EPS", "Basic EPS"])
        px = t.history(period="6y", auto_adjust=False)["Close"]
        px.index = px.index.tz_localize(None)
        pes = []
        for d, e in eps.items():
            if e > 0:
                pes.append(px[:d].iloc[-1] / e)
        return float(np.median(pes)) if pes else NAN
    except Exception:
        return NAN


# ----------------------------------------------------------------------------- backtest
def fetch_prices(tickers, start):
    raw = yf.download(tickers + ["^NSEI"], start=start, end=date.today().isoformat(), auto_adjust=True,
                      progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px.index = px.index.tz_localize(None) if px.index.tz is not None else px.index
    px = px.dropna(axis=1, how="all")
    return px.drop(columns=["^NSEI"]), px["^NSEI"].dropna()


def backtest(prices, bench, capital, sip, freq, trend, cost=0.0025):
    idx = bench.index
    prices = prices.reindex(idx)
    av = prices.notna().values
    rets = prices.pct_change(fill_method=None).fillna(0).values
    n, T = prices.shape[1], len(idx)
    if trend:
        sma = prices.rolling(200, min_periods=200).mean()
        sig = (prices > sma).astype(float).where(sma.notna())
        ip = sig.resample("W-FRI").last().reindex(idx, method="ffill").shift(1).fillna(1).values > 0.5
    else:
        ip = np.ones((T, n), dtype=bool)
    yr, mo = idx.year.values, idx.month.values
    key = {"Annual": yr, "Semi-Annual": yr * 2 + (mo > 6), "Buy-and-Hold": np.zeros(T)}[freq]
    rebal = np.r_[False, key[1:] != key[:-1]] if freq != "Buy-and-Hold" else np.zeros(T, bool)
    sipday = np.r_[False, mo[1:] != mo[:-1]] if sip > 0 else np.zeros(T, bool)
    cr = RF / 252
    vals, cash, inv, prev_post = np.zeros(n), float(capital), float(capital), capital
    tw, eq, invested = np.ones(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        act = av[t]
        if t > 0:
            vals = vals * (1 + np.where(ip[t], rets[t], cr))
            cash *= 1 + cr
            tog = (ip[t] != ip[t - 1]) & act
            vals = vals - vals * tog * cost
            tw[t] = (vals.sum() + cash) / prev_post
            if sipday[t]:
                cash += sip
                inv += sip
        total = vals.sum() + cash
        if act.any() and (t == 0 or rebal[t]):
            tgt = np.where(act, total / act.sum(), 0.0)
            fee = cost * np.abs(tgt - vals).sum()
            vals, cash = tgt * (1 - fee / total), 0.0
        elif act.any() and cash > 1e-9:
            vals = vals + np.where(act, cash / act.sum(), 0.0) * (1 - cost)
            cash = 0.0
        prev_post = vals.sum() + cash
        eq[t], invested[t] = prev_post, inv
    b = bench.values
    units, beq = capital * (1 - cost) / b[0], np.zeros(T)
    for t in range(T):
        if t > 0 and sipday[t]:
            units += sip * (1 - cost) / b[t]
        beq[t] = units * b[t]
    btw = np.r_[1.0, b[1:] / b[:-1]]
    return {"idx": idx, "s_curve": pd.Series(np.cumprod(tw), idx), "b_curve": pd.Series(np.cumprod(btw), idx),
            "s_val": eq[-1], "b_val": beq[-1], "invested": invested[-1]}


def metrics(curve):
    rr = curve.pct_change().dropna()
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    dd = curve / curve.cummax() - 1
    ex = rr - RF / 252
    down = rr[rr < RF / 252] - RF / 252
    sharpe = ex.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else NAN
    sortino = ex.mean() / np.sqrt((down ** 2).mean()) * np.sqrt(252) if len(down) else NAN
    mdd = dd.min()
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Sortino": sortino, "Calmar": cagr / abs(mdd) if mdd else NAN,
            "dd": dd}


def yearly(curve):
    y = curve.groupby(curve.index.year).last()
    return (y / y.shift(1).fillna(1.0) - 1) * 100


# ----------------------------------------------------------------------------- UI
def fig_style(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", legend=dict(orientation="h", y=1.08))
    return fig


def tab_scanner(sb):
    st.subheader("Top 15 Coffee Can + GARP Scanner")
    if st.button("🚀 Run Scan", type="primary"):
        t0 = time.time()
        if sb["mode"] == "Screener.in CSV":
            if sb["file"] is None:
                st.error("Upload a Screener.in CSV in the sidebar first.")
                return
            df = parse_screener(pd.read_csv(sb["file"]))
            errs = 0
        else:
            tickers = sb["tickers"][:sb["limit"]]
            if not tickers:
                st.error("Ticker universe is empty.")
                return
            df, errs = run_scan(tickers)
        if df.empty:
            st.error("No usable data returned (possibly rate-limited). Try fewer tickers or the Screener CSV mode.")
            return
        if not sb["fin"]:
            df = df[~df["Fin"].astype(bool)]
        df["Score"] = df.apply(lambda r: score_stock(r.to_dict()), axis=1)
        df = df[df["Score"] >= sb["min_score"]].sort_values("Score", ascending=False).reset_index(drop=True)
        top = df.head(15).copy()
        sigs, why = [], []
        with st.spinner("Computing technical signals for Top 15..."):
            with ThreadPoolExecutor(max_workers=8) as ex:
                hist = dict(zip(top["Ticker"], ex.map(lambda t: _safe_hist(t), top["Ticker"])))
            for _, r in top.iterrows():
                h = hist.get(r["Ticker"])
                if h is None:
                    sigs.append("n/a")
                    why.append("no price data")
                else:
                    st.session_state.setdefault("hcache", {})[(r["Ticker"], "3y")] = h
                    s, w = signal_for(h, r.to_dict())
                    sigs.append(s)
                    why.append("; ".join(w))
        top["Signal"], top["Why"] = sigs, why
        st.session_state["scan"], st.session_state["top"] = df, top
        st.success(f"Scanned {len(df)} qualifying stocks in {time.time() - t0:.1f}s ({errs} failed fetches).")
    top = st.session_state.get("top")
    if top is None:
        st.info("Pick a universe in the sidebar and press **Run Scan**.")
        return
    show = top[["Ticker", "Sector", "CMP", "MCap", "ROE", "ROCE", "SalesCAGR", "ProfitCAGR", "DE", "CFO_PAT",
                "PE", "PEG", "Score", "Signal"]].copy()
    show["CFO_PAT"] = show["CFO_PAT"] * 100
    st.dataframe(show, use_container_width=True, hide_index=True, column_config={
        "CMP": st.column_config.NumberColumn("CMP ₹", format="%.1f"),
        "MCap": st.column_config.NumberColumn("Mkt Cap ₹Cr", format="%.0f"),
        "ROE": st.column_config.NumberColumn("ROE %", format="%.1f"),
        "ROCE": st.column_config.NumberColumn("ROCE %", format="%.1f"),
        "SalesCAGR": st.column_config.NumberColumn("Sales CAGR %", format="%.1f"),
        "ProfitCAGR": st.column_config.NumberColumn("Profit CAGR %", format="%.1f"),
        "DE": st.column_config.NumberColumn("D/E", format="%.2f"),
        "CFO_PAT": st.column_config.NumberColumn("CFO/PAT %", format="%.0f"),
        "PE": st.column_config.NumberColumn("P/E", format="%.1f"),
        "PEG": st.column_config.NumberColumn("PEG", format="%.2f"),
        "Score": st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%.1f")})
    with st.expander("Signal reasons"):
        st.dataframe(top[["Ticker", "Signal", "Why"]], hide_index=True, use_container_width=True)
    st.download_button("Download full ranked universe (CSV)", st.session_state["scan"].to_csv(index=False),
                       "ranked_universe.csv")


def _safe_hist(tk):
    try:
        d = yf.Ticker(tk).history(period="3y", auto_adjust=True)
        if d.empty:
            return None
        d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
        return add_ind(d)
    except Exception:
        return None


def tab_backtest(sb):
    st.subheader("15-Year Backtest vs Nifty 50")
    st.warning("⚠️ Look-ahead / survivorship bias: the portfolio is today's Top 15 applied to history, so results "
               "are optimistic. Use this as a stress-test of the style, not as expected returns. Weights are "
               "equal; fundamentals are not re-scored historically (Yahoo has only ~4 years of statements).")
    top = st.session_state.get("top")
    if top is None:
        st.info("Run the scanner first.")
        return
    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Downloading 15 years of prices..."):
            prices, bench = fetch_prices(list(top["Ticker"]), sb["start"])
        if prices.empty or bench.empty:
            st.error("Price download failed.")
            return
        st.session_state["bt"] = backtest(prices, bench, sb["capital"], sb["sip"], sb["freq"], sb["trend"],
                                          sb["cost"] / 100)
        st.session_state["bt_n"] = prices.shape[1]
    bt = st.session_state.get("bt")
    if bt is None:
        return
    s, b = metrics(bt["s_curve"]), metrics(bt["b_curve"])
    passed = s["CAGR"] > b["CAGR"] and s["Sharpe"] > b["Sharpe"] and s["MDD"] >= b["MDD"] - 0.05
    (st.success if passed else st.error)(
        f"{'✅ STRATEGY PASSED' if passed else '❌ STRATEGY FAILED'} — CAGR {s['CAGR']:.1%} vs {b['CAGR']:.1%}, "
        f"Sharpe {s['Sharpe']:.2f} vs {b['Sharpe']:.2f}, MaxDD {s['MDD']:.1%} vs {b['MDD']:.1%}")
    c = st.columns(4)
    c[0].metric("Strategy CAGR", f"{s['CAGR']:.1%}", f"Alpha {100 * (s['CAGR'] - b['CAGR']):+.1f} pts")
    c[1].metric("Nifty 50 CAGR", f"{b['CAGR']:.1%}")
    c[2].metric("Final Value", f"₹{bt['s_val']:,.0f}", f"Invested ₹{bt['invested']:,.0f}", delta_color="off")
    c[3].metric("Nifty Final Value", f"₹{bt['b_val']:,.0f}")
    st.dataframe(pd.DataFrame({"Strategy": s, "Nifty 50": b}).drop("dd").assign(
        Strategy=lambda d: d["Strategy"].astype(float).round(3), **{"Nifty 50": lambda d: d["Nifty 50"].astype(float).round(3)}),
        use_container_width=True)
    f = go.Figure()
    f.add_scatter(x=bt["idx"], y=bt["s_curve"], name="Strategy (TWR)", line=dict(color="#00d4aa"))
    f.add_scatter(x=bt["idx"], y=bt["b_curve"], name="Nifty 50", line=dict(color="#ff9f1c"))
    f.update_layout(title="Growth of ₹1 (time-weighted, SIP-neutral)", yaxis_type="log")
    st.plotly_chart(fig_style(f), use_container_width=True)
    f = go.Figure()
    f.add_scatter(x=bt["idx"], y=s["dd"] * 100, name="Strategy", fill="tozeroy", line=dict(color="#ef476f"))
    f.add_scatter(x=bt["idx"], y=b["dd"] * 100, name="Nifty 50", line=dict(color="#ff9f1c"))
    f.update_layout(title="Underwater plot (drawdown %)")
    st.plotly_chart(fig_style(f, 320), use_container_width=True)
    ys, yb = yearly(bt["s_curve"]), yearly(bt["b_curve"])
    z = [ys.values, yb.values, (ys - yb).values]
    f = go.Figure(go.Heatmap(z=z, x=[str(y) for y in ys.index], y=["Strategy", "Nifty 50", "Excess"],
                             colorscale="RdYlGn", zmid=0, text=np.round(z, 1), texttemplate="%{text}",
                             colorbar=dict(title="%")))
    f.update_layout(title="Year-by-year returns (%)")
    st.plotly_chart(fig_style(f, 280), use_container_width=True)


def tab_deepdive(sb):
    st.subheader("Single-Stock Deep Dive")
    top = st.session_state.get("top")
    opts = list(top["Ticker"]) if top is not None else []
    c1, c2 = st.columns([2, 2])
    pick = c1.selectbox("From Top 15", [""] + opts)
    typed = c2.text_input("...or any NSE ticker (e.g. TCS)").strip().upper()
    tk = (typed + ("" if typed.endswith(".NS") else ".NS")) if typed else pick
    if not tk:
        return
    with st.spinner("Loading..."):
        m = (top[top["Ticker"] == tk].iloc[0].to_dict() if top is not None and tk in set(top["Ticker"])
             else fetch_one(tk))
        if "error" in m:
            st.error(f"Could not fetch fundamentals: {m['error']}")
            return
        m.setdefault("Score", score_stock(m))
        h = get_hist(tk, "5y")
    if h is None:
        st.error("No price history.")
        return
    sig, why = signal_for(h, m)
    pe_med = hist_pe_median(tk, m.get("PE"))
    c = st.columns(5)
    c[0].metric("Signal", sig)
    c[1].metric("Score", f"{m['Score']:.0f}/100")
    c[2].metric("PEG", "n/a" if pd.isna(m.get("PEG")) else f"{m['PEG']:.2f}")
    c[3].metric("P/E vs hist. median", "n/a" if pd.isna(m.get("PE")) else f"{m['PE']:.1f}",
                None if pd.isna(pe_med) else f"median {pe_med:.1f}", delta_color="off")
    c[4].metric("RSI d / w", f"{h['RSI'].iloc[-1]:.0f} / {h['RSIW'].iloc[-1]:.0f}")
    st.caption("Reasons: " + "; ".join(why))
    buy, trim, sell = tech_series(h, m["Score"], m.get("PEG", NAN))
    ent = lambda s: s & ~s.shift(1, fill_value=False)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.17, 0.18], vertical_spacing=0.02)
    fig.add_candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"], name="Price", row=1, col=1)
    for col, clr in [("EMA50", "#00b4d8"), ("EMA200", "#ff9f1c"), ("SMA20W", "#c77dff"), ("BBU", "#555"), ("BBL", "#555")]:
        fig.add_scatter(x=h.index, y=h[col], name=col, line=dict(color=clr, width=1), row=1, col=1)
    for s_, nm, sym, clr, off in [(ent(buy), "BUY", "triangle-up", "#06d6a0", 0.97), (ent(trim), "TRIM", "diamond", "#ffd166", 1.03),
                                  (ent(sell), "SELL", "triangle-down", "#ef476f", 1.03)]:
        fig.add_scatter(x=h.index[s_], y=h["Close"][s_] * off, mode="markers", name=nm, row=1, col=1,
                        marker=dict(symbol=sym, size=12, color=clr, line=dict(width=1, color="white")))
    vc = np.where(h["Close"] >= h["Open"], "#06d6a0", "#ef476f")
    fig.add_bar(x=h.index, y=h["Volume"], marker_color=vc, name="Volume", row=2, col=1)
    fig.add_scatter(x=h.index, y=h["RSI"], name="RSI", line=dict(color="#00b4d8"), row=3, col=1)
    for lv in (30, 70):
        fig.add_hline(y=lv, line_dash="dot", line_color="#666", row=3, col=1)
    fig.add_bar(x=h.index, y=h["MHIST"], name="MACD hist", marker_color=np.where(h["MHIST"] >= 0, "#06d6a0", "#ef476f"), row=4, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(title=f"{tk} — technical-only markers (fundamentals held at current values)")
    st.plotly_chart(fig_style(fig, 820), use_container_width=True)
    st.caption("Volume breakout today: " + ("YES" if h["VOLX"].iloc[-1] > 1.5 else "no") +
               f" ({h['VOLX'].iloc[-1]:.1f}x 20d avg)")


def sidebar():
    sb = {}
    st.sidebar.header("⚙️ Universe")
    sb["mode"] = st.sidebar.radio("Mode", ["Live scan (Yahoo)", "Screener.in CSV"])
    sb["tickers"], sb["file"], sb["limit"] = [], None, 100
    if sb["mode"] == "Live scan (Yahoo)":
        src = st.sidebar.selectbox("Universe", ["Built-in list"] + list(NSE_URLS) + ["Upload ticker list"])
        sb["limit"] = st.sidebar.slider("Max stocks", 10, 1000, 100, 10)
        if src == "Built-in list":
            sb["tickers"] = [t + ".NS" for t in BUILTIN]
        elif src in NSE_URLS:
            try:
                req = urllib.request.Request(NSE_URLS[src], headers={"User-Agent": "Mozilla/5.0"})
                d = pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=20).read()))
                sb["tickers"] = [s.strip() + ".NS" for s in d["Symbol"]]
            except Exception as e:
                st.sidebar.error(f"NSE download failed ({e}). Upload the CSV manually.")
        else:
            up = st.sidebar.file_uploader("CSV with Symbol/Ticker column", type="csv", key="tk")
            if up is not None:
                d = pd.read_csv(up)
                col = next((c for c in d.columns if c.lower() in ("symbol", "ticker", "nse code")), d.columns[0])
                sb["tickers"] = [str(s).strip().upper().replace(".NS", "") + ".NS" for s in d[col]]
    else:
        sb["file"] = st.sidebar.file_uploader("Screener.in export (CSV)", type="csv", key="scr")
        st.sidebar.download_button("⬇ CSV template", csv_template().to_csv(index=False), "screener_template.csv")
    sb["min_score"] = st.sidebar.slider("Min composite score", 0, 90, 40)
    sb["fin"] = st.sidebar.checkbox("Include Banks/NBFC (ROE + P/B scoring)", True)
    st.sidebar.header("📈 Backtest")
    sb["capital"] = st.sidebar.number_input("Initial capital ₹", 10000, 100000000, 100000, 10000)
    sb["sip"] = st.sidebar.number_input("Monthly SIP ₹", 0, 10000000, 15000, 1000)
    sb["freq"] = st.sidebar.selectbox("Rebalance", ["Annual", "Semi-Annual", "Buy-and-Hold"])
    sb["trend"] = st.sidebar.checkbox("200-DMA weekly trend filter (exit to Liquid @6%)", False)
    sb["cost"] = st.sidebar.number_input("Friction per trade (%)", 0.0, 2.0, 0.25, 0.05)
    sb["start"] = st.sidebar.text_input("Start date", "2011-01-01")
    return sb


def main():
    st.set_page_config(page_title="CoffeeCan + GARP Quant Scanner", page_icon="☕", layout="wide")
    st.markdown("<style>.stApp{background:#0e1117;color:#e6e6e6}[data-testid=stMetricValue]{font-size:1.4rem}</style>",
                unsafe_allow_html=True)
    st.title("☕ CoffeeCan + GARP Quant Scanner & 15-Year Backtester")
    st.caption("NSE • Educational research tool, not investment advice.")
    sb = sidebar()
    t1, t2, t3, t4 = st.tabs(["🔎 Scanner", "📊 Backtest", "🕯 Deep Dive", "📄 CSV Template"])
    with t1:
        tab_scanner(sb)
    with t2:
        tab_backtest(sb)
    with t3:
        tab_deepdive(sb)
    with t4:
        st.write("Export your Screener.in screen with these columns (names are matched case-insensitively):")
        st.dataframe(csv_template(), hide_index=True)
        st.download_button("Download template", csv_template().to_csv(index=False), "screener_template.csv", key="t4")


if __name__ == "__main__":
    main()
