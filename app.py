import io
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Buffett-Greenblatt Hybrid Screener", layout="wide")
st.title("🏛️ Buffett-Greenblatt Hybrid Stock Screener")
st.caption("Greenblatt Magic Formula (ROCE + Earnings Yield) + Buffett Safety (Low Debt + Consistent Growth) "
           "+ Fair Value & Buy/Sell Signal + 10-Year Backtest")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
NSE = "https://archives.nseindia.com/content"
URLS = {
    "nifty500": f"{NSE}/indices/ind_nifty500list.csv",
    "micro250": f"{NSE}/indices/ind_niftymicrocap250_list.csv",
    "all": f"{NSE}/equities/EQUITY_L.csv",
}
RISK_FREE = 0.065  # India 10Y approx, Sharpe ke liye


# ================================================================== DATA
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
        random.Random(42).shuffle(syms)
        syms = syms[:limit]
    return syms


def pct(x):
    return None if x is None or pd.isna(x) else float(x) * 100


def fetch_stage1(sym):
    """Fast data from yfinance .info (with retry)"""
    i = None
    for attempt in range(3):
        try:
            i = yf.Ticker(sym + ".NS").info
            if i and (i.get("marketCap") or i.get("currentPrice")):
                break
        except Exception:
            pass
        i = None
        time.sleep(0.7 * (attempt + 1))
    if not i:
        return None
    try:
        de = i.get("debtToEquity")
        ev, ebitda = i.get("enterpriseValue"), i.get("ebitda")
        return {
            "Symbol": sym,
            "Name": i.get("shortName", sym),
            "Price": i.get("currentPrice"),
            "MCap_Cr": (i.get("marketCap") or 0) / 1e7,
            "PE": i.get("trailingPE"),
            "ROE": pct(i.get("returnOnEquity")),
            "DE": None if de is None else de / 100,
            "Rev_Growth": pct(i.get("revenueGrowth")),
            "EPS_Growth": pct(i.get("earningsGrowth")),
            "EV_EBITDA_Yield": (ebitda / ev * 100) if ev and ebitda and ev > 0 else None,
            "EPS": i.get("trailingEps"),
            "BVPS": i.get("bookValue"),
            "FCF": i.get("freeCashflow"),
            "Shares": i.get("sharesOutstanding"),
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
    s = series.dropna().iloc[::-1]
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


def run_parallel(fn, items, label, workers=8):
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


# ================================================================== VALUATION
def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def dcf_per_share(fcf, shares, growth_pct, disc=0.12, term=0.05, years=10):
    """Conservative 10-yr DCF: growth 5 saal tak g1 (max 15%), phir 6% tak fade, terminal 5%."""
    fcf, shares = _num(fcf), _num(shares)
    if not fcf or not shares or fcf <= 0 or shares <= 0:
        return None
    g1 = min(max(growth_pct, 0), 15) / 100
    cf, pv = fcf, 0.0
    for t in range(1, years + 1):
        g = g1 if t <= 5 else g1 + (0.06 - g1) * (t - 5) / (years - 5)
        cf *= 1 + g
        pv += cf / (1 + disc) ** t
    pv += (cf * (1 + term) / (disc - term)) / (1 + disc) ** years
    return pv / shares


def fair_values(r):
    """3 models: Graham Number, Lynch PEG, DCF. Fair value = median of available."""
    eps, bvps = _num(r.get("EPS")), _num(r.get("BVPS"))
    g = _num(r.get("Profit_CAGR"))
    if g is None:
        g = _num(r.get("EPS_Growth"))
    if g is None:
        g = 10.0
    out = {"Graham": None, "Lynch_PEG": None, "DCF": None}
    if eps and bvps and eps > 0 and bvps > 0:
        out["Graham"] = (22.5 * eps * bvps) ** 0.5
    if eps and eps > 0:
        out["Lynch_PEG"] = eps * min(max(g, 8), 25)  # PEG=1, fair P/E 8..25 ke beech
    out["DCF"] = dcf_per_share(r.get("FCF"), r.get("Shares"), g)
    vals = [v for v in out.values() if v]
    out["Fair_Value"] = float(np.median(vals)) if vals else None
    out["Models_Used"] = len(vals)
    return out


def get_signal(price, fv, roce, de, pcagr):
    price, fv = _num(price), _num(fv)
    roce, de, pcagr = _num(roce), _num(de), _num(pcagr)
    breaks = []
    if roce is not None and roce < 12: breaks.append("ROCE < 12%")
    if de is not None and de > 1: breaks.append("Debt/Equity > 1")
    if pcagr is not None and pcagr < 0: breaks.append("Profit CAGR negative")
    if not price or not fv:
        return "⚪ NO DATA", "Fair value nikalne layak data nahi"
    ratio = price / fv
    if breaks:
        return "🔴 SELL / REVIEW", "Quality toot rahi hai: " + ", ".join(breaks)
    if ratio > 1.3:
        return "🔴 SELL / TRIM", f"Price fair value se {100*(ratio-1):.0f}% upar"
    if ratio > 1.1:
        return "🟠 OVERVALUED", "Naya mat khareedo, hold karein"
    if ratio > 0.85:
        return "🟡 HOLD / FAIR", "Fair value ke aas-paas"
    if ratio > 0.7:
        return "🟢 BUY", f"Fair value se {100*(1-ratio):.0f}% neeche"
    return "🟢 STRONG BUY", f"Fair value se {100*(1-ratio):.0f}% neeche (margin of safety)"


def add_valuation(df):
    df = df.copy()
    rows = []
    for _, r in df.iterrows():
        v = fair_values(r)
        sig, why = get_signal(r.get("Price"), v["Fair_Value"], r.get("ROCE"), r.get("DE"), r.get("Profit_CAGR"))
        fv, p = v["Fair_Value"], _num(r.get("Price"))
        rows.append({
            "Graham": v["Graham"], "Lynch_PEG": v["Lynch_PEG"], "DCF": v["DCF"],
            "Fair_Value": fv,
            "Upside_%": ((fv / p - 1) * 100) if fv and p else None,
            "Signal": sig, "Reason": why,
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ================================================================== BACKTEST
def rebalance_curve(prices):
    """Equal-weight, har calendar year ke start par rebalance. prices: DataFrame (date x stock)."""
    ret = prices.pct_change().iloc[1:]
    pieces, value = [], 1.0
    for _, ch in ret.groupby(ret.index.year):
        valid = ch.columns[ch.iloc[0].notna()]
        if len(valid) == 0:
            continue
        cum = (1 + ch[valid].fillna(0)).cumprod().mean(axis=1) * value
        pieces.append(cum)
        value = cum.iloc[-1]
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def perf_metrics(curve):
    curve = curve.dropna()
    curve = curve / curve.iloc[0]
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    dd = (curve / curve.cummax() - 1).min()
    vol = curve.pct_change().dropna().std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE) / vol if vol > 0 else np.nan
    return {"CAGR %": cagr * 100, "Max Drawdown %": dd * 100, "Volatility %": vol * 100,
            "Sharpe": sharpe, "Total Return %": (curve.iloc[-1] - 1) * 100}


def yearly_returns(curve):
    curve = curve.dropna()
    ye = curve.groupby(curve.index.year).last()
    yr = ye.pct_change()
    yr.iloc[0] = ye.iloc[0] / curve.iloc[0] - 1
    return yr * 100


def run_backtest(symbols, years=10):
    tickers = [s + ".NS" for s in symbols]
    data = yf.download(tickers + ["^NSEI"], period=f"{years}y", auto_adjust=True,
                       progress=False, threads=True)["Close"]
    data = data.dropna(axis=1, how="all").ffill()
    if "^NSEI" not in data.columns:
        raise ValueError("Nifty 50 ka data nahi mila")
    bench_px = data["^NSEI"]
    stocks = data.drop(columns=["^NSEI"])
    port = rebalance_curve(stocks)
    bench = (1 + bench_px.pct_change().iloc[1:]).cumprod()
    both = pd.concat([port.rename("Portfolio"), bench.rename("Nifty 50")], axis=1).dropna()
    both = both / both.iloc[0]
    return both, stocks.shape[1]


# ================================================================== SIDEBAR
st.sidebar.header("⚙️ Strategy Settings")
roce_min = st.sidebar.slider("Min ROCE (%)", 10, 40, 20)
de_max = st.sidebar.slider("Max Debt/Equity", 0.0, 1.5, 0.5, 0.1)
pe_max = st.sidebar.slider("Max P/E", 10, 80, 35)
growth_min = st.sidebar.slider("Min Sales & Profit CAGR (%)", 0, 25, 10)
mcap_min = st.sidebar.number_input("Min Market Cap (₹ Crore)", value=500, step=100)
top_n = st.sidebar.slider("Kitne top stocks dikhane hain", 5, 30, 15)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Auto Scan", "🔎 Stock Checker (Buy/Sell)", "📁 CSV Upload", "📚 Philosophy & Limits"])

# ================================================================== TAB 1
with tab1:
    c1, c2 = st.columns([2, 1])
    universe = c1.selectbox(
        "Kaunsi companies scan karni hain?",
        ["Nifty 500 + Microcap 250 (~750 companies)", "All NSE listed (random sample)"],
    )
    limit = c2.number_input("All NSE mode me kitni companies", 200, 2500, 1000, 100)

    if st.button("🚀 SCAN START - Top Stocks Dhundo", type="primary", use_container_width=True):
        for k in ("result", "all", "bt"):
            st.session_state.pop(k, None)
        try:
            with st.spinner("NSE se company list la raha hoon..."):
                symbols = get_symbols(universe, int(limit))
            st.info(f"{len(symbols)} companies mili. Scan shuru (5-10 minute lag sakte hain)...")
        except Exception as e:
            st.error(f"NSE list download nahi hui: {e}. Tab 3 me CSV upload use karein.")
            st.stop()

        raw = run_parallel(fetch_stage1, symbols, "Stage 1: Basic data", workers=8)
        st.write(f"Yahoo se data mila: **{len(raw)} / {len(symbols)}** companies")
        if len(raw) < 0.3 * len(symbols):
            st.error("Yahoo Finance ne zyada tar companies ka data nahi diya (rate-limit/block). "
                     "10-15 min baad try karein, app laptop par chalayein, ya Tab 3 me CSV upload karein.")
            st.stop()
        s1 = pd.DataFrame(raw)
        for c in ["MCap_Cr", "PE", "ROE", "DE", "Rev_Growth", "EPS_Growth", "EPS", "BVPS", "FCF", "Shares"]:
            s1[c] = pd.to_numeric(s1[c], errors="coerce")

        pre = s1[
            (s1["MCap_Cr"] >= mcap_min)
            & (s1["PE"] > 0) & (s1["PE"] < pe_max)
            & (s1["ROE"] >= roce_min * 0.6)
            & (s1["DE"].fillna(0) <= de_max)
        ].copy()
        st.write(f"Stage 1 ke baad **{len(pre)}** companies bachi.")
        if len(pre) == 0:
            st.warning("Stage 1 filters ne sab hata diye. Sidebar me ROCE/P/E/Debt dheela karke try karein.")
            st.stop()

        pre["Earnings_Yield"] = 100 / pre["PE"]
        pre["q"] = pre["ROE"].rank(ascending=False) + pre["Earnings_Yield"].rank(ascending=False)
        shortlist = pre.sort_values("q").head(80)

        s2 = pd.DataFrame(run_parallel(fetch_stage2, shortlist["Symbol"].tolist(), "Stage 2: ROCE & growth check"))
        merged = shortlist.merge(s2, on="Symbol")
        final = merged[
            (merged["ROCE"] >= roce_min)
            & (merged["Sales_CAGR"] >= growth_min)
            & (merged["Profit_CAGR"] >= growth_min)
        ]
        if len(final):
            st.session_state["result"] = add_valuation(rank_score(final).head(top_n))
        else:
            st.session_state["result"] = final
        st.session_state["all"] = s1

    if "result" in st.session_state:
        res = st.session_state["result"]
        if len(res) == 0:
            st.warning("Koi stock sab filters pass nahi kar paya. Sidebar me ROCE ya growth thoda kam karein.")
        else:
            only_cheap = st.checkbox("Sirf Fair Value se sasta (BUY / STRONG BUY) wale dikhao")
            view = res[res["Signal"].str.contains("BUY")] if only_cheap else res
            st.subheader(f"🏆 Top {len(view)} Buffett-Greenblatt Stocks")
            cols = ["Symbol", "Name", "Price", "Fair_Value", "Upside_%", "Signal", "PE", "Earnings_Yield",
                    "ROCE", "DE", "Sales_CAGR", "Profit_CAGR", "Sales_Up_Years", "MCap_Cr", "Reason"]
            show = view[cols].round(2).reset_index(drop=True)
            show.index += 1
            try:
                st.dataframe(show.style.background_gradient(subset=["ROCE"], cmap="Greens"),
                             use_container_width=True)
            except Exception:  # matplotlib na ho to bina colour ke dikhao
                st.dataframe(show, use_container_width=True)
            st.download_button("⬇️ CSV download", show.to_csv().encode(), "top_stocks.csv")
            with st.expander("Fair value ke 3 models (Graham / Lynch PEG / DCF)"):
                st.dataframe(res[["Symbol", "Price", "Graham", "Lynch_PEG", "DCF", "Fair_Value"]].round(1),
                             use_container_width=True)

            st.divider()
            st.subheader("📈 10 Saal Backtest")
            st.warning("⚠️ Yeh test **aaj ke chune hue stocks** ko pichle 10 saal par chalata hai. "
                       "Aaj ke winners pehle se pata hone ki wajah se result **zyada achha dikhega** "
                       "(look-ahead bias). Ise 'strategy pakki jeetegi' ka saboot na samjhein. "
                       "Details 'Philosophy & Limits' tab me.")
            if st.button("📈 10 Saal Backtest Chalao"):
                with st.spinner("10 saal ka price data la raha hoon..."):
                    try:
                        st.session_state["bt"] = run_backtest(res["Symbol"].tolist(), 10)
                    except Exception as e:
                        st.error(f"Backtest fail: {e}")
            if "bt" in st.session_state:
                both, n = st.session_state["bt"]
                st.line_chart(both)
                m = pd.DataFrame({"Portfolio": perf_metrics(both["Portfolio"]),
                                  "Nifty 50": perf_metrics(both["Nifty 50"])}).round(2)
                st.dataframe(m, use_container_width=True)
                yr = pd.DataFrame({"Portfolio %": yearly_returns(both["Portfolio"]),
                                   "Nifty 50 %": yearly_returns(both["Nifty 50"])}).round(1)
                st.write("Saal-dar-saal return:")
                st.dataframe(yr.T, use_container_width=True)
                st.caption(f"{n} stocks, equal-weight, har saal rebalance. Jin stocks ki history 10 saal "
                           "puri nahi, wo listing ke baad se judte hain. Brokerage/tax/slippage shamil nahi.")

        with st.expander("Poora scanned data dekhein"):
            st.dataframe(st.session_state["all"], use_container_width=True)

# ================================================================== TAB 2
with tab2:
    st.write("Apna koi bhi NSE stock daalein (jaise `TCS`, `ITC`, `ASIANPAINT`). Fair value, signal aur "
             "sell-triggers dekhein.")
    sym = st.text_input("NSE Symbol", value="").strip().upper()
    if st.button("🔎 Check Karo") and sym:
        with st.spinner("Data la raha hoon..."):
            a = fetch_stage1(sym)
            b = fetch_stage2(sym)
        if not a:
            st.error("Data nahi mila. Symbol check karein ya thodi der baad try karein.")
        else:
            r = {**a, **b}
            v = fair_values(r)
            sig, why = get_signal(r["Price"], v["Fair_Value"], r["ROCE"], r["DE"], r["Profit_CAGR"])
            st.subheader(f"{r['Name']} ({sym})")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Price", f"₹{r['Price']:.1f}" if r["Price"] else "NA")
            k2.metric("Fair Value", f"₹{v['Fair_Value']:.1f}" if v["Fair_Value"] else "NA")
            if v["Fair_Value"] and r["Price"]:
                k3.metric("Upside / (Downside)", f"{(v['Fair_Value']/r['Price']-1)*100:.0f}%")
            k4.metric("Signal", sig)
            st.info(why)

            st.write("**Fair value models**")
            st.dataframe(pd.DataFrame([{k: (round(x, 1) if x else None) for k, x in v.items()
                                         if k in ("Graham", "Lynch_PEG", "DCF", "Fair_Value")}]),
                         use_container_width=True)

            st.write("**Sell-trigger checklist**")
            def chk(ok, text): st.write(("✅ " if ok else "❌ ") + text)
            chk(r["ROCE"] is not None and r["ROCE"] >= 15, f"ROCE >= 15%  (abhi: {_num(r['ROCE']) and round(r['ROCE'],1)}%)")
            chk(r["DE"] is not None and r["DE"] <= 0.5, f"Debt/Equity <= 0.5  (abhi: {_num(r['DE']) and round(r['DE'],2)})")
            chk(r["Profit_CAGR"] is not None and r["Profit_CAGR"] >= 10, f"Profit CAGR >= 10%  (abhi: {_num(r['Profit_CAGR']) and round(r['Profit_CAGR'],1)}%)")
            chk(r["Sales_CAGR"] is not None and r["Sales_CAGR"] >= 10, f"Sales CAGR >= 10%  (abhi: {_num(r['Sales_CAGR']) and round(r['Sales_CAGR'],1)}%)")
            chk(r["PE"] is not None and 0 < r["PE"] < 40, f"P/E < 40  (abhi: {_num(r['PE']) and round(r['PE'],1)})")

            try:
                h = yf.Ticker(sym + ".NS").history(period="2y")["Close"]
                chart = pd.DataFrame({"Price": h, "200-DMA": h.rolling(200).mean()}).dropna()
                st.write("**Price vs 200-DMA** (sirf trend ka reference, sell rule nahi)")
                st.line_chart(chart)
            except Exception:
                pass

# ================================================================== TAB 3
with tab3:
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
        if not {"ROCE", "PE", "DE"}.issubset(df.columns):
            st.error("ROCE, P/E aur Debt to equity columns zaroori hain.")
        else:
            m = (df["ROCE"] >= roce_min) & (df["DE"] <= de_max) & (df["PE"] > 0) & (df["PE"] < pe_max)
            if "Sales_CAGR" in df: m &= df["Sales_CAGR"] >= growth_min
            if "Profit_CAGR" in df: m &= df["Profit_CAGR"] >= growth_min
            out = rank_score(df[m]).head(top_n)
            st.subheader(f"🏆 Top {len(out)} stocks")
            st.dataframe(out, use_container_width=True)

# ================================================================== TAB 4
with tab4:
    st.markdown("""
### Yeh strategy kya hai
- **Greenblatt Magic Formula:** high ROCE (achha business) + high Earnings Yield (sasta daam), dono ranks jodo.
- **Buffett-style safety:** kam karza, saalon se badhti sales/profit. Yahan **growth rank** bhi jodi gayi hai.
- **Fair value:** 3 models (Graham Number, Lynch PEG, 10-yr DCF) ka median. Yeh **andaza** hai, sachchai nahi.

### Signal kaise banta hai
| Price / Fair Value | Signal |
|---|---|
| <= 0.70 | 🟢 STRONG BUY |
| 0.70 – 0.85 | 🟢 BUY |
| 0.85 – 1.10 | 🟡 HOLD |
| 1.10 – 1.30 | 🟠 OVERVALUED |
| > 1.30 | 🔴 SELL / TRIM |
| ROCE < 12, Debt/Equity > 1 ya Profit CAGR < 0 | 🔴 SELL / REVIEW (price kuch bhi ho) |

### Kab sell karein (Buffett/Greenblatt ke hisaab se)
1. **Business bigad gaya** (ROCE giri, karza badha, profit ghat raha) → sabse bada sell reason.
2. **Bahut mehnga ho gaya** (price fair value se 30%+ upar) → thoda profit book karein.
3. **Usse behtar stock mila** → badlein (Greenblatt: saal me ek baar rebalance).
4. Sirf price girne par sell nahi, pehle dekhein ki business kharab hua ya sirf mood.

### Limits (zaroor padhein)
- **Backtest me look-ahead bias:** aaj ke chune stocks par purana test hai, isliye result zyada achha dikhta hai.
- **Banks/NBFC/Insurance** par Debt/Equity aur ROCE ke rules fit nahi baithte. Yeh screen unhe galat tareeke se hata ya chun sakta hai.
- **Cyclical companies** (metals, commodities) peak earnings par sasta dikhti hain (low P/E) aur phir girti hain.
- Yahoo ka free data kabhi galat/adhura hota hai. Kharidne se pehle Screener.in aur annual report dekhein.
- Greenblatt ka formula bhi **3-5 saal ke kamzor daur** de sakta hai. Sabr aur diversification (15-30 stocks) zaroori hai.
- Yeh educational tool hai, investment advice nahi.
""")
