"""
SmallCap Fund Ranking Engine
─────────────────────────────
Advanced multi-factor ranking using:
  • Rolling Returns (3Y / 5Y windows)
  • Downside & Upside Capture Ratios
  • Sortino Ratio
  • Maximum Drawdown
  • Composite scoring with configurable weights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmallCap Ranking Engine",
    layout="wide",
    page_icon="🔬",
)

# ═══════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #05080f;
    --surface: #0c1220;
    --surface2: #111a2e;
    --border: #1a2744;
    --border-accent: #243656;
    --text: #e2e8f0;
    --text-dim: #7a8baa;
    --green: #22c55e;
    --green-bg: rgba(34,197,94,0.08);
    --red: #ef4444;
    --red-bg: rgba(239,68,68,0.08);
    --amber: #eab308;
    --amber-bg: rgba(234,179,8,0.08);
    --blue: #3b82f6;
    --blue-bg: rgba(59,130,246,0.08);
    --purple: #a855f7;
    --cyan: #06b6d4;
}

.stApp {
    background: var(--bg);
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #05080f 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    font-family: 'Space Grotesk', sans-serif !important;
}

h1 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
     background: linear-gradient(135deg, #3b82f6, #06b6d4, #22c55e); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; font-size: 2rem !important; letter-spacing: -0.5px; }
h2 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
     color: var(--text) !important; font-size: 1.3rem !important; letter-spacing: -0.3px; }
h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text-dim) !important; }

.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 1rem 0; }
.m-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; position: relative; overflow: hidden;
}
.m-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--blue), var(--cyan));
}
.m-card.green::before { background: linear-gradient(90deg, #22c55e, #06b6d4); }
.m-card.red::before { background: linear-gradient(90deg, #ef4444, #f97316); }
.m-card.amber::before { background: linear-gradient(90deg, #eab308, #f59e0b); }
.m-card.purple::before { background: linear-gradient(90deg, #a855f7, #6366f1); }
.m-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.m-value { font-size: 26px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; color: var(--text); }
.m-sub { font-size: 12px; color: var(--text-dim); margin-top: 4px; }

.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
.tbl thead th {
    background: var(--surface2); color: var(--text-dim); padding: 12px 14px; text-align: left;
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
    border-bottom: 2px solid var(--border); position: sticky; top: 0; z-index: 1;
}
.tbl thead th:first-child { border-radius: 10px 0 0 0; }
.tbl thead th:last-child { border-radius: 0 10px 0 0; }
.tbl tbody td {
    padding: 11px 14px; border-bottom: 1px solid var(--border); color: var(--text);
    font-family: 'IBM Plex Mono', monospace; font-size: 12.5px;
}
.tbl tbody tr:hover td { background: var(--surface2); }
.tbl tbody tr:last-child td:first-child { border-radius: 0 0 0 10px; }
.tbl tbody tr:last-child td:last-child { border-radius: 0 0 10px 0; }

.rk { display: inline-flex; align-items: center; justify-content: center;
      width: 30px; height: 30px; border-radius: 8px; font-weight: 700; font-size: 13px; }
.rk-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #0a0e17; }
.rk-2 { background: linear-gradient(135deg, #d1d5db, #9ca3af); color: #0a0e17; }
.rk-3 { background: linear-gradient(135deg, #cd7f32, #b8860b); color: white; }
.rk-n { background: var(--surface); color: var(--text-dim); border: 1px solid var(--border); }

.sig { padding: 3px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 600;
       font-family: 'Space Grotesk', sans-serif; display: inline-block; }
.sig-strong { background: var(--green-bg); color: var(--green); border: 1px solid rgba(34,197,94,0.25); }
.sig-good { background: rgba(6,182,212,0.1); color: var(--cyan); border: 1px solid rgba(6,182,212,0.25); }
.sig-neutral { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(234,179,8,0.2); }
.sig-weak { background: rgba(249,115,22,0.1); color: #f97316; border: 1px solid rgba(249,115,22,0.2); }
.sig-avoid { background: var(--red-bg); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }

.pill { display: inline-block; padding: 2px 10px; border-radius: 14px; font-size: 11px;
        font-family: 'IBM Plex Mono', monospace; font-weight: 500; }
.pill-g { background: var(--green-bg); color: var(--green); }
.pill-r { background: var(--red-bg); color: var(--red); }

.info-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 20px; margin: 12px 0; border-left: 3px solid var(--blue);
}
.info-box p { color: var(--text-dim); font-size: 13px; line-height: 1.6; margin: 0; }
.info-box strong { color: var(--text); }

div[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: var(--surface); border-radius: 12px;
    padding: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 10px 20px; font-family: 'Space Grotesk', sans-serif;
    font-weight: 500; font-size: 13.5px; color: var(--text-dim);
}
.stTabs [aria-selected="true"] { background: var(--blue) !important; color: white !important; }
.stDownloadButton button {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stDownloadButton button:hover { border-color: var(--blue) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_all_data():
    """Load and clean NAV, AUM, and Expense Ratio data."""
    raw = pd.read_excel("smallcapfinalrank.xlsx")
    fund_names = raw.iloc[1, 1:].dropna().tolist()
    nav = raw.iloc[3:, : len(fund_names) + 1].copy()
    nav.columns = ["Date"] + fund_names
    nav = nav[pd.to_datetime(nav["Date"], errors="coerce").notna()].copy()
    nav["Date"] = pd.to_datetime(nav["Date"])
    nav = nav.sort_values("Date").reset_index(drop=True)
    for f in fund_names:
        nav[f] = pd.to_numeric(nav[f], errors="coerce")

    # Build benchmark proxy: equal-weight of all available funds
    valid = [f for f in fund_names if nav[f].notna().sum() > 252]
    nav["Benchmark"] = nav[valid].mean(axis=1)

    # AUM
    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
    aum_latest = (
        aum.sort_values("Month_End", ascending=False)
        .groupby("Fund").first().reset_index()[["Fund", "AUM"]]
    )

    # Expense Ratio
    er_raw = pd.read_excel("smallcap_expense_ratio.xlsx")
    er_recs = []
    cur = None
    for _, row in er_raw.iterrows():
        v = str(row.iloc[0])
        if v.startswith("Scheme Name:"):
            cur = v.replace("Scheme Name: ", "").strip()
        elif cur:
            try:
                er_recs.append({"Fund": cur, "ER": float(row.iloc[1])})
                cur = None
            except:
                pass
    df_er = pd.DataFrame(er_recs).drop_duplicates(subset="Fund", keep="first")

    return nav, fund_names, aum_latest, df_er


# ═══════════════════════════════════════════════════════════════════
# METRICS COMPUTATION ENGINE
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_all_metrics(_nav, fund_names, aum_latest, df_er):
    """Compute rolling returns, capture ratios, Sortino, drawdowns for every fund."""
    nav = _nav.copy()
    latest_date = nav["Date"].max()

    # Monthly returns for all funds + benchmark
    monthly = nav.set_index("Date").resample("ME").last()
    monthly_ret = monthly.pct_change().dropna(how="all")
    bench_ret = monthly_ret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 252:
            continue

        fund_monthly = monthly_ret[fund].dropna()
        common_idx = fund_monthly.index.intersection(bench_ret.dropna().index)
        if len(common_idx) < 12:
            continue
        f_ret = fund_monthly.loc[common_idx]
        b_ret = bench_ret.loc[common_idx]

        # ──────────────────────────────────
        # 1. ROLLING RETURNS (3Y & 5Y)
        # ──────────────────────────────────
        daily_fd = fd.set_index("Date")[fund]

        def rolling_cagr(series, years, window_months):
            """Compute rolling CAGR over monthly NAV."""
            m_nav = series.resample("ME").last().dropna()
            if len(m_nav) < window_months:
                return None, None, None, None
            cagrs = []
            for i in range(window_months, len(m_nav)):
                start_nav = m_nav.iloc[i - window_months]
                end_nav = m_nav.iloc[i]
                if start_nav > 0:
                    cagr = (end_nav / start_nav) ** (1 / years) - 1
                    cagrs.append(cagr * 100)
            if not cagrs:
                return None, None, None, None
            arr = np.array(cagrs)
            return np.mean(arr), np.median(arr), np.min(arr), np.max(arr)

        r3_mean, r3_med, r3_min, r3_max = rolling_cagr(daily_fd, 3, 36)
        r5_mean, r5_med, r5_min, r5_max = rolling_cagr(daily_fd, 5, 60)

        # Rolling returns that beat benchmark
        def rolling_win_rate(fund_series, bench_series, years, window_months):
            fm = fund_series.resample("ME").last().dropna()
            bm = bench_series.resample("ME").last().dropna()
            common = fm.index.intersection(bm.index)
            fm, bm = fm.loc[common], bm.loc[common]
            if len(fm) < window_months:
                return None
            wins = 0
            total = 0
            for i in range(window_months, len(fm)):
                fs = fm.iloc[i - window_months]
                fe = fm.iloc[i]
                bs = bm.iloc[i - window_months]
                be = bm.iloc[i]
                if fs > 0 and bs > 0:
                    f_cagr = (fe / fs) ** (1 / years) - 1
                    b_cagr = (be / bs) ** (1 / years) - 1
                    if f_cagr > b_cagr:
                        wins += 1
                    total += 1
            return (wins / total * 100) if total > 0 else None

        bench_daily = nav.set_index("Date")["Benchmark"]
        win_3y = rolling_win_rate(daily_fd, bench_daily, 3, 36)
        win_5y = rolling_win_rate(daily_fd, bench_daily, 5, 60)

        # ──────────────────────────────────
        # 2. DOWNSIDE & UPSIDE CAPTURE RATIOS
        # ──────────────────────────────────
        down_months = b_ret[b_ret < 0]
        up_months = b_ret[b_ret > 0]

        # Downside capture
        if len(down_months) > 3:
            f_down = f_ret.loc[down_months.index]
            downside_capture = (f_down.mean() / down_months.mean()) * 100
        else:
            downside_capture = None

        # Upside capture
        if len(up_months) > 3:
            f_up = f_ret.loc[up_months.index]
            upside_capture = (f_up.mean() / up_months.mean()) * 100
        else:
            upside_capture = None

        # Capture ratio (upside / downside)
        capture_ratio = None
        if upside_capture and downside_capture and downside_capture != 0:
            capture_ratio = upside_capture / downside_capture

        # ──────────────────────────────────
        # 3. SORTINO RATIO
        # ──────────────────────────────────
        rf_monthly = 0.06 / 12  # 6% annual
        excess = f_ret - rf_monthly
        neg_excess = excess[excess < 0]
        downside_std = np.sqrt(np.mean(neg_excess ** 2)) if len(neg_excess) > 3 else None

        sortino = None
        if downside_std and downside_std > 0:
            sortino = (f_ret.mean() - rf_monthly) / downside_std

        # Category average Sortino (computed later, placeholder)
        # Sortino over last 3 years
        f_3y = f_ret.tail(36)
        neg_3y = (f_3y - rf_monthly)
        neg_3y = neg_3y[neg_3y < 0]
        ds_3y = np.sqrt(np.mean(neg_3y ** 2)) if len(neg_3y) > 3 else None
        sortino_3y = (f_3y.mean() - rf_monthly) / ds_3y if ds_3y and ds_3y > 0 else None

        # ──────────────────────────────────
        # 4. MAXIMUM DRAWDOWN (full history & 3Y)
        # ──────────────────────────────────
        prices = fd[fund].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd_full = dd.min() * 100

        recent_3y = fd[fd["Date"] >= latest_date - timedelta(days=365 * 3)]
        if len(recent_3y) > 50:
            p3 = recent_3y[fund].values
            cm3 = np.maximum.accumulate(p3)
            dd3 = (p3 - cm3) / cm3
            max_dd_3y = dd3.min() * 100
        else:
            max_dd_3y = None

        # Current drawdown from ATH
        current_dd = (prices[-1] - prices.max()) / prices.max() * 100

        # ──────────────────────────────────
        # 5. ADDITIONAL METRICS
        # ──────────────────────────────────
        # Sharpe (annualised)
        vol_ann = f_ret.std() * np.sqrt(12)
        sharpe = ((f_ret.mean() - rf_monthly) * 12) / (vol_ann * np.sqrt(12)) if vol_ann > 0 else None
        # actually: sharpe = (ann_ret - rf) / ann_vol
        ann_ret = f_ret.mean() * 12
        sharpe = (ann_ret - 0.06) / (f_ret.std() * np.sqrt(12)) if f_ret.std() > 0 else None

        # Calmar ratio (ann return / max dd)
        calmar = abs(ann_ret * 100 / max_dd_full) if max_dd_full != 0 else None

        # Information ratio vs benchmark
        active_ret = f_ret - b_ret.loc[f_ret.index]
        tracking_err = active_ret.std() * np.sqrt(12)
        info_ratio = (active_ret.mean() * 12) / tracking_err if tracking_err > 0 else None

        # 1Y simple return
        latest_nav = fd.iloc[-1][fund]
        ld = fd.iloc[-1]["Date"]
        n12m_mask = (nav["Date"] >= ld - timedelta(days=375)) & (nav["Date"] <= ld - timedelta(days=355))
        n12m_df = nav[n12m_mask].dropna(subset=[fund])
        ret_1y = None
        if len(n12m_df) > 0:
            n12m = n12m_df.iloc[(n12m_df["Date"] - (ld - timedelta(days=365))).abs().argsort().iloc[0]][fund]
            ret_1y = (latest_nav / n12m - 1) * 100

        # Volatility
        vol_1y_daily = None
        recent_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        if len(recent_1y) > 50:
            vol_1y_daily = recent_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100

        results.append({
            "Fund": fund,
            "Ret_1Y": ret_1y,
            "Roll_3Y_Mean": r3_mean,
            "Roll_3Y_Median": r3_med,
            "Roll_3Y_Min": r3_min,
            "Roll_3Y_Max": r3_max,
            "Roll_3Y_WinRate": win_3y,
            "Roll_5Y_Mean": r5_mean,
            "Roll_5Y_Median": r5_med,
            "Roll_5Y_Min": r5_min,
            "Roll_5Y_Max": r5_max,
            "Roll_5Y_WinRate": win_5y,
            "Downside_Capture": downside_capture,
            "Upside_Capture": upside_capture,
            "Capture_Ratio": capture_ratio,
            "Sortino": sortino,
            "Sortino_3Y": sortino_3y,
            "Max_DD_Full": max_dd_full,
            "Max_DD_3Y": max_dd_3y,
            "Current_DD": current_dd,
            "Sharpe": sharpe,
            "Calmar": calmar,
            "Info_Ratio": info_ratio,
            "Vol_1Y": vol_1y_daily,
            "Ann_Return": ann_ret * 100,
        })

    df = pd.DataFrame(results)
    df = df.merge(aum_latest, on="Fund", how="left")
    df = df.merge(df_er, on="Fund", how="left")

    # Category average Sortino
    cat_sortino = df["Sortino"].median()
    df["Sortino_vs_Cat"] = df["Sortino"] - cat_sortino

    # Peer Max DD comparison
    cat_dd = df["Max_DD_Full"].median()
    df["DD_vs_Peers"] = df["Max_DD_Full"] - cat_dd  # less negative = better

    return df


# ═══════════════════════════════════════════════════════════════════
# RANKING ENGINE
# ═══════════════════════════════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def compute_composite(df, weights):
    """Build composite score from user-defined weights."""
    d = df.copy()

    # Factor scores (0-100 percentile)
    d["S_Roll3Y"] = pctrank(d["Roll_3Y_Mean"])
    d["S_Roll5Y"] = pctrank(d["Roll_5Y_Mean"])
    d["S_Roll3Y_Win"] = pctrank(d["Roll_3Y_WinRate"])
    d["S_Roll5Y_Win"] = pctrank(d["Roll_5Y_WinRate"])
    d["S_DownCap"] = pctrank(d["Downside_Capture"], asc=False)  # lower = better
    d["S_UpCap"] = pctrank(d["Upside_Capture"])                 # higher = better
    d["S_CaptureRatio"] = pctrank(d["Capture_Ratio"])            # higher = better
    d["S_Sortino"] = pctrank(d["Sortino"])
    d["S_Sortino3Y"] = pctrank(d["Sortino_3Y"])
    d["S_MaxDD"] = pctrank(d["Max_DD_Full"])                     # less negative = better
    d["S_MaxDD3Y"] = pctrank(d["Max_DD_3Y"])
    d["S_Calmar"] = pctrank(d["Calmar"])
    d["S_InfoRatio"] = pctrank(d["Info_Ratio"])
    d["S_ER"] = pctrank(d["ER"], asc=False)                     # lower = better

    w_map = {
        "S_Roll3Y": weights["rolling_3y"],
        "S_Roll5Y": weights["rolling_5y"],
        "S_Roll3Y_Win": weights["win_rate_3y"],
        "S_DownCap": weights["downside_cap"],
        "S_UpCap": weights["upside_cap"],
        "S_Sortino": weights["sortino"],
        "S_MaxDD": weights["max_dd"],
        "S_Calmar": weights["calmar"],
        "S_InfoRatio": weights["info_ratio"],
        "S_ER": weights["expense"],
    }

    def comp(row):
        tw = ts = 0
        for col, wt in w_map.items():
            if wt > 0 and pd.notna(row.get(col)):
                ts += row[col] * wt
                tw += wt
        return ts / tw if tw > 0 else np.nan

    d["Score"] = d.apply(comp, axis=1)
    d["Rank"] = d["Score"].rank(ascending=False, method="min").astype(int)
    d = d.sort_values("Rank").reset_index(drop=True)

    def signal(s):
        if s >= 78: return "Elite"
        if s >= 62: return "Strong"
        if s >= 48: return "Above Avg"
        if s >= 35: return "Average"
        if s >= 22: return "Below Avg"
        return "Weak"

    d["Signal"] = d["Score"].apply(signal)
    return d


# ═══════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#7a8baa", size=12),
    margin=dict(l=55, r=25, t=50, b=45),
    xaxis=dict(gridcolor="rgba(26,39,68,0.7)", zerolinecolor="rgba(26,39,68,0.7)"),
    yaxis=dict(gridcolor="rgba(26,39,68,0.7)", zerolinecolor="rgba(26,39,68,0.7)"),
)

SIG_CLS = {
    "Elite": "sig-strong", "Strong": "sig-good", "Above Avg": "sig-neutral",
    "Average": "sig-weak", "Below Avg": "sig-avoid", "Weak": "sig-avoid",
}


def fmt(v, d=1, suffix=""):
    if pd.isna(v): return '<span style="color:var(--text-dim);">—</span>'
    return f"{v:.{d}f}{suffix}"


def fmt_color(v, d=1, suffix="%", reverse=False):
    if pd.isna(v): return '<span style="color:var(--text-dim);">—</span>'
    good = v < 0 if reverse else v > 0
    c = "var(--green)" if good else "var(--red)"
    return f'<span style="color:{c};">{v:.{d}f}{suffix}</span>'


def short_name(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "")
             .replace("  ", " ").strip())


def rank_html(r):
    if r == 1: return '<span class="rk rk-1">1</span>'
    if r == 2: return '<span class="rk rk-2">2</span>'
    if r == 3: return '<span class="rk rk-3">3</span>'
    return f'<span class="rk rk-n">{int(r)}</span>'


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════
def main():
    # ── Load data ──
    with st.spinner("Loading NAV data..."):
        nav, fund_names, aum_latest, df_er = load_all_data()
    with st.spinner("Computing metrics across all rolling windows..."):
        df_raw = compute_all_metrics(nav, fund_names, aum_latest, df_er)

    # ── Header ──
    st.markdown("# 🔬 SmallCap Fund Ranking Engine")
    st.markdown(
        '<p style="color:var(--text-dim);font-size:14px;margin-top:-10px;">'
        'Multi-factor ranking: Rolling Returns · Capture Ratios · Sortino · Drawdowns · '
        f'{len(df_raw)} funds analysed · Data through {nav["Date"].max().strftime("%d %b %Y")}'
        '</p>', unsafe_allow_html=True
    )

    # ── Sidebar: Weight Configuration ──
    with st.sidebar:
        st.markdown("### ⚙️ Factor Weights")
        st.markdown(
            '<p style="font-size:12px;color:var(--text-dim);">Adjust how much each factor '
            'contributes to the composite score. Weights are normalised automatically.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("**📊 Rolling Returns**")
        w_r3y = st.slider("Rolling 3Y CAGR", 0, 30, 15, key="w1")
        w_r5y = st.slider("Rolling 5Y CAGR", 0, 30, 15, key="w2")
        w_win = st.slider("3Y Win Rate vs Bench", 0, 20, 5, key="w3")

        st.markdown("**🛡️ Capture Ratios**")
        w_dcap = st.slider("Downside Capture", 0, 30, 20, key="w4")
        w_ucap = st.slider("Upside Capture", 0, 20, 10, key="w5")

        st.markdown("**📈 Risk-Adjusted**")
        w_sort = st.slider("Sortino Ratio", 0, 25, 15, key="w6")
        w_dd = st.slider("Max Drawdown", 0, 20, 10, key="w7")
        w_calm = st.slider("Calmar Ratio", 0, 15, 5, key="w8")

        st.markdown("**🔎 Other**")
        w_ir = st.slider("Information Ratio", 0, 15, 3, key="w9")
        w_er = st.slider("Expense Ratio", 0, 15, 2, key="w10")

        total_w = w_r3y + w_r5y + w_win + w_dcap + w_ucap + w_sort + w_dd + w_calm + w_ir + w_er
        if total_w == 0:
            total_w = 1

        weights = {
            "rolling_3y": w_r3y / total_w,
            "rolling_5y": w_r5y / total_w,
            "win_rate_3y": w_win / total_w,
            "downside_cap": w_dcap / total_w,
            "upside_cap": w_ucap / total_w,
            "sortino": w_sort / total_w,
            "max_dd": w_dd / total_w,
            "calmar": w_calm / total_w,
            "info_ratio": w_ir / total_w,
            "expense": w_er / total_w,
        }

        st.markdown("---")
        st.markdown("**Effective Weights**")
        for k, v in weights.items():
            lbl = k.replace("_", " ").title()
            bar_w = int(v * 200)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:12px;'
                f'color:var(--text-dim);margin:2px 0;">'
                f'<span>{lbl}</span><span style="font-family:IBM Plex Mono,monospace;">{v*100:.1f}%</span>'
                f'</div>'
                f'<div style="background:var(--border);height:4px;border-radius:2px;margin-bottom:6px;">'
                f'<div style="background:var(--blue);height:4px;border-radius:2px;width:{bar_w}px;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Compute Rankings ──
    df = compute_composite(df_raw, weights)

    # ── Top Metrics ──
    top = df.iloc[0] if len(df) > 0 else None
    n_strong = len(df[df["Signal"].isin(["Elite", "Strong"])])
    avg_dcap = df["Downside_Capture"].median()
    avg_sortino = df["Sortino"].median()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="m-card green">
            <div class="m-label">Top Ranked Fund</div>
            <div class="m-value" style="font-size:18px;">{short_name(top["Fund"]) if top is not None else "—"}</div>
            <div class="m-sub">Score: {top["Score"]:.1f}/100</div>
        </div>
        <div class="m-card">
            <div class="m-label">Strong / Elite Funds</div>
            <div class="m-value">{n_strong}</div>
            <div class="m-sub">out of {len(df)}</div>
        </div>
        <div class="m-card amber">
            <div class="m-label">Median Downside Capture</div>
            <div class="m-value">{avg_dcap:.0f}%</div>
            <div class="m-sub">{"✓ Below 100" if avg_dcap and avg_dcap < 100 else "⚠ Above 100"}</div>
        </div>
        <div class="m-card purple">
            <div class="m-label">Median Sortino Ratio</div>
            <div class="m-value">{avg_sortino:.2f}</div>
            <div class="m-sub">category average</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Rankings", "📊 Rolling Returns", "🛡️ Capture Ratios",
        "📈 Risk Metrics", "🔎 Fund Deep-Dive"
    ])

    # ════════════════════════════════════════════
    # TAB 1: MAIN RANKINGS TABLE
    # ════════════════════════════════════════════
    with tab1:
        st.markdown("## Composite Rankings")
        st.markdown(
            '<div class="info-box"><p><strong>How it works:</strong> Each fund is scored '
            'on 10 factors across rolling returns, capture ratios, Sortino ratio, and drawdowns. '
            'Factors are converted to percentile ranks (0–100) and combined using your chosen '
            'weights from the sidebar. Adjust weights to reflect your investment philosophy.</p></div>',
            unsafe_allow_html=True,
        )

        html = '<table class="tbl"><thead><tr>'
        headers = ["#", "Fund", "Score", "Signal", "Roll 3Y%", "Roll 5Y%",
                   "Down Cap", "Up Cap", "Sortino", "Max DD%", "Calmar"]
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"

        for _, r in df.iterrows():
            sig_cls = SIG_CLS.get(r["Signal"], "sig-neutral")
            html += "<tr>"
            html += f'<td>{rank_html(r["Rank"])}</td>'
            html += f'<td style="font-family:Space Grotesk,sans-serif;font-weight:500;">{short_name(r["Fund"])}</td>'
            html += f'<td><strong>{r["Score"]:.1f}</strong></td>'
            html += f'<td><span class="sig {sig_cls}">{r["Signal"]}</span></td>'
            html += f'<td>{fmt(r["Roll_3Y_Mean"], 1, "%")}</td>'
            html += f'<td>{fmt(r["Roll_5Y_Mean"], 1, "%")}</td>'
            # downside capture: lower is better, color accordingly
            if pd.notna(r["Downside_Capture"]):
                dc_color = "var(--green)" if r["Downside_Capture"] < 90 else ("var(--amber)" if r["Downside_Capture"] < 100 else "var(--red)")
                html += f'<td><span style="color:{dc_color};">{r["Downside_Capture"]:.0f}%</span></td>'
            else:
                html += '<td>—</td>'
            # upside capture: higher is better
            if pd.notna(r["Upside_Capture"]):
                uc_color = "var(--green)" if r["Upside_Capture"] > 100 else "var(--amber)"
                html += f'<td><span style="color:{uc_color};">{r["Upside_Capture"]:.0f}%</span></td>'
            else:
                html += '<td>—</td>'
            html += f'<td>{fmt_color(r["Sortino"], 2, "")}</td>'
            html += f'<td>{fmt_color(r["Max_DD_Full"], 1, "%")}</td>'
            html += f'<td>{fmt(r["Calmar"], 2, "")}</td>'
            html += "</tr>"

        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

        # Download
        st.markdown("")
        export_df = df[["Rank", "Fund", "Score", "Signal", "Roll_3Y_Mean", "Roll_5Y_Mean",
                        "Roll_3Y_WinRate", "Roll_5Y_WinRate", "Downside_Capture", "Upside_Capture",
                        "Capture_Ratio", "Sortino", "Sortino_3Y", "Max_DD_Full", "Max_DD_3Y",
                        "Calmar", "Info_Ratio", "Sharpe", "AUM", "ER"]].copy()
        export_df.columns = ["Rank", "Fund", "Score", "Signal", "Roll_3Y_CAGR%", "Roll_5Y_CAGR%",
                             "3Y_WinRate%", "5Y_WinRate%", "Downside_Capture%", "Upside_Capture%",
                             "Capture_Ratio", "Sortino", "Sortino_3Y", "Max_DD_Full%", "Max_DD_3Y%",
                             "Calmar", "Info_Ratio", "Sharpe", "AUM_Cr", "Expense_Ratio%"]
        buf = io.BytesIO()
        export_df.round(2).to_excel(buf, index=False, engine="openpyxl")
        st.download_button("📥 Download Full Rankings (.xlsx)", buf.getvalue(),
                           "smallcap_rankings.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ════════════════════════════════════════════
    # TAB 2: ROLLING RETURNS
    # ════════════════════════════════════════════
    with tab2:
        st.markdown("## Rolling Return Analysis")
        st.markdown(
            '<div class="info-box"><p><strong>Why rolling returns?</strong> Point-to-point returns '
            'can be misleading — a fund that started measuring from a market bottom looks great. '
            'Rolling returns compute the CAGR for <strong>every possible</strong> 3-year or 5-year '
            'window, showing you the full range of investor outcomes.</p></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            # Rolling 3Y chart
            chart_df = df.dropna(subset=["Roll_3Y_Mean"]).sort_values("Roll_3Y_Mean", ascending=True).tail(20)
            fig = go.Figure()
            # Min-max range
            fig.add_trace(go.Bar(
                y=[short_name(f) for f in chart_df["Fund"]],
                x=chart_df["Roll_3Y_Max"] - chart_df["Roll_3Y_Min"],
                base=chart_df["Roll_3Y_Min"],
                orientation="h",
                marker=dict(color="rgba(59,130,246,0.12)", line=dict(width=0)),
                name="Range",
                hovertemplate="Min: %{base:.1f}% | Max: %{x:.1f}%<extra></extra>",
            ))
            # Mean dot
            fig.add_trace(go.Scatter(
                y=[short_name(f) for f in chart_df["Fund"]],
                x=chart_df["Roll_3Y_Mean"],
                mode="markers",
                marker=dict(color="#3b82f6", size=10, symbol="diamond"),
                name="Mean CAGR",
                hovertemplate="%{x:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                title="Rolling 3Y CAGR — Range & Average",
                showlegend=False, height=550, **PLOT_THEME,
            )
            fig.update_xaxes(title_text="CAGR (%)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            chart_df = df.dropna(subset=["Roll_5Y_Mean"]).sort_values("Roll_5Y_Mean", ascending=True).tail(20)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[short_name(f) for f in chart_df["Fund"]],
                x=chart_df["Roll_5Y_Max"] - chart_df["Roll_5Y_Min"],
                base=chart_df["Roll_5Y_Min"],
                orientation="h",
                marker=dict(color="rgba(168,85,247,0.12)", line=dict(width=0)),
                name="Range",
            ))
            fig.add_trace(go.Scatter(
                y=[short_name(f) for f in chart_df["Fund"]],
                x=chart_df["Roll_5Y_Mean"],
                mode="markers",
                marker=dict(color="#a855f7", size=10, symbol="diamond"),
                name="Mean CAGR",
            ))
            fig.update_layout(
                title="Rolling 5Y CAGR — Range & Average",
                showlegend=False, height=550, **PLOT_THEME,
            )
            fig.update_xaxes(title_text="CAGR (%)")
            st.plotly_chart(fig, use_container_width=True)

        # Win rate table
        st.markdown("### Benchmark Win Rate")
        st.markdown(
            '<p style="color:var(--text-dim);font-size:13px;">Percentage of rolling windows where '
            'the fund beat the equal-weight benchmark (proxy for Nifty Smallcap 250 TRI).</p>',
            unsafe_allow_html=True,
        )
        wr_df = df.dropna(subset=["Roll_3Y_WinRate"]).sort_values("Roll_3Y_WinRate", ascending=False).head(15)
        wr_html = '<table class="tbl"><thead><tr><th>#</th><th>Fund</th><th>3Y Win Rate</th><th>5Y Win Rate</th><th>Roll 3Y Mean</th><th>Roll 5Y Mean</th></tr></thead><tbody>'
        for i, (_, r) in enumerate(wr_df.iterrows(), 1):
            wr3_color = "var(--green)" if pd.notna(r["Roll_3Y_WinRate"]) and r["Roll_3Y_WinRate"] > 50 else "var(--red)"
            wr5_color = "var(--green)" if pd.notna(r["Roll_5Y_WinRate"]) and r["Roll_5Y_WinRate"] > 50 else "var(--red)"
            wr_html += f'<tr><td>{i}</td><td style="font-family:Space Grotesk,sans-serif;">{short_name(r["Fund"])}</td>'
            wr_html += f'<td><span style="color:{wr3_color};">{fmt(r["Roll_3Y_WinRate"],1,"%")}</span></td>'
            wr_html += f'<td><span style="color:{wr5_color};">{fmt(r["Roll_5Y_WinRate"],1,"%")}</span></td>'
            wr_html += f'<td>{fmt(r["Roll_3Y_Mean"],1,"%")}</td><td>{fmt(r["Roll_5Y_Mean"],1,"%")}</td></tr>'
        wr_html += "</tbody></table>"
        st.markdown(wr_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TAB 3: CAPTURE RATIOS
    # ════════════════════════════════════════════
    with tab3:
        st.markdown("## Downside & Upside Capture Analysis")
        st.markdown(
            '<div class="info-box"><p>'
            '<strong>Downside Capture &lt; 80%</strong> = Fund loses less than 80% of what the '
            'market loses in down months. This is the #1 metric for small-cap survival.<br>'
            '<strong>Upside Capture &gt; 100%</strong> = Fund gains more than the market in up months.<br>'
            '<strong>Capture Ratio</strong> = Upside / Downside. Above 1.2 is excellent — the fund '
            'asymmetrically captures more gains and fewer losses.</p></div>',
            unsafe_allow_html=True,
        )

        # Scatter: Downside vs Upside
        cap_df = df.dropna(subset=["Downside_Capture", "Upside_Capture"])
        fig = go.Figure()

        # Quadrant shading
        fig.add_shape(type="rect", x0=50, x1=90, y0=100, y1=160,
                      fillcolor="rgba(34,197,94,0.06)", line=dict(width=0))
        fig.add_annotation(x=70, y=155, text="IDEAL ZONE", showarrow=False,
                           font=dict(color="#22c55e", size=11))
        fig.add_shape(type="rect", x0=100, x1=150, y0=50, y1=100,
                      fillcolor="rgba(239,68,68,0.06)", line=dict(width=0))
        fig.add_annotation(x=125, y=55, text="DANGER ZONE", showarrow=False,
                           font=dict(color="#ef4444", size=11))

        # Reference lines
        fig.add_hline(y=100, line=dict(color="rgba(122,139,170,0.3)", dash="dash", width=1))
        fig.add_vline(x=80, line=dict(color="rgba(122,139,170,0.3)", dash="dash", width=1))
        fig.add_vline(x=100, line=dict(color="rgba(122,139,170,0.3)", dash="dash", width=1))

        colors = []
        for _, r in cap_df.iterrows():
            if r["Downside_Capture"] < 80 and r["Upside_Capture"] > 100:
                colors.append("#22c55e")
            elif r["Downside_Capture"] > 100:
                colors.append("#ef4444")
            else:
                colors.append("#3b82f6")

        fig.add_trace(go.Scatter(
            x=cap_df["Downside_Capture"],
            y=cap_df["Upside_Capture"],
            mode="markers+text",
            marker=dict(size=12, color=colors, line=dict(width=1, color="rgba(255,255,255,0.2)")),
            text=[short_name(f).split()[0] for f in cap_df["Fund"]],
            textposition="top center",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate="<b>%{customdata}</b><br>Downside: %{x:.0f}%<br>Upside: %{y:.0f}%<extra></extra>",
            customdata=[short_name(f) for f in cap_df["Fund"]],
        ))

        fig.update_layout(
            title="Capture Ratio Map — Downside vs Upside",
            xaxis_title="Downside Capture (%)",
            yaxis_title="Upside Capture (%)",
            height=550,
            **PLOT_THEME,
        )
        fig.update_xaxes(range=[50, 150])
        fig.update_yaxes(range=[50, 160])
        st.plotly_chart(fig, use_container_width=True)

        # Capture ratio bar chart
        cap_sorted = df.dropna(subset=["Capture_Ratio"]).sort_values("Capture_Ratio", ascending=True).tail(20)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=[short_name(f) for f in cap_sorted["Fund"]],
            x=cap_sorted["Capture_Ratio"],
            orientation="h",
            marker=dict(
                color=[("#22c55e" if v > 1.2 else ("#3b82f6" if v > 1.0 else "#ef4444"))
                       for v in cap_sorted["Capture_Ratio"]],
            ),
        ))
        fig2.add_vline(x=1.0, line=dict(color="rgba(122,139,170,0.4)", dash="dash"))
        fig2.add_vline(x=1.2, line=dict(color="rgba(34,197,94,0.4)", dash="dash"))
        fig2.update_layout(title="Capture Ratio (Upside/Downside) — Higher is Better",
                           showlegend=False, height=500, **PLOT_THEME)
        fig2.update_xaxes(title_text="Capture Ratio")
        st.plotly_chart(fig2, use_container_width=True)

    # ════════════════════════════════════════════
    # TAB 4: RISK METRICS
    # ════════════════════════════════════════════
    with tab4:
        st.markdown("## Risk-Adjusted Performance")

        c1, c2 = st.columns(2)

        with c1:
            # Sortino chart
            sort_df = df.dropna(subset=["Sortino"]).sort_values("Sortino", ascending=True).tail(20)
            cat_avg = df["Sortino"].median()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[short_name(f) for f in sort_df["Fund"]],
                x=sort_df["Sortino"],
                orientation="h",
                marker=dict(
                    color=[("#22c55e" if v > cat_avg else "#ef4444") for v in sort_df["Sortino"]],
                ),
            ))
            fig.add_vline(x=cat_avg, line=dict(color="#eab308", dash="dash", width=1.5),
                          annotation_text=f"Category Median: {cat_avg:.2f}",
                          annotation_position="top right",
                          annotation_font=dict(color="#eab308", size=11))
            fig.update_layout(title="Sortino Ratio — Downside-Adjusted Returns",
                              showlegend=False, height=520, **PLOT_THEME)
            fig.update_xaxes(title_text="Sortino Ratio")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Max Drawdown
            dd_df = df.dropna(subset=["Max_DD_Full"]).sort_values("Max_DD_Full", ascending=False).tail(20)
            cat_dd = df["Max_DD_Full"].median()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[short_name(f) for f in dd_df["Fund"]],
                x=dd_df["Max_DD_Full"],
                orientation="h",
                marker=dict(
                    color=[("#22c55e" if v > cat_dd else "#ef4444") for v in dd_df["Max_DD_Full"]],
                ),
            ))
            fig.add_vline(x=cat_dd, line=dict(color="#eab308", dash="dash", width=1.5),
                          annotation_text=f"Median: {cat_dd:.0f}%",
                          annotation_position="top left",
                          annotation_font=dict(color="#eab308", size=11))
            fig.update_layout(title="Max Drawdown (Full History) — Less Negative is Better",
                              showlegend=False, height=520, **PLOT_THEME)
            fig.update_xaxes(title_text="Max Drawdown (%)")
            st.plotly_chart(fig, use_container_width=True)

        # Risk summary table
        st.markdown("### Complete Risk Dashboard")
        risk_sorted = df.sort_values("Rank").head(20)
        rhtml = '<table class="tbl"><thead><tr><th>#</th><th>Fund</th><th>Sortino</th><th>Sortino 3Y</th><th>vs Cat Avg</th><th>Sharpe</th><th>Max DD</th><th>Max DD 3Y</th><th>Calmar</th><th>Info Ratio</th></tr></thead><tbody>'
        for _, r in risk_sorted.iterrows():
            vs_cat = r.get("Sortino_vs_Cat")
            if pd.notna(vs_cat):
                vs_str = f'<span class="pill {"pill-g" if vs_cat > 0 else "pill-r"}">{vs_cat:+.2f}</span>'
            else:
                vs_str = "—"
            rhtml += f'<tr><td>{rank_html(r["Rank"])}</td>'
            rhtml += f'<td style="font-family:Space Grotesk,sans-serif;">{short_name(r["Fund"])}</td>'
            rhtml += f'<td>{fmt_color(r["Sortino"], 2, "")}</td>'
            rhtml += f'<td>{fmt_color(r["Sortino_3Y"], 2, "")}</td>'
            rhtml += f'<td>{vs_str}</td>'
            rhtml += f'<td>{fmt(r["Sharpe"], 2)}</td>'
            rhtml += f'<td>{fmt_color(r["Max_DD_Full"], 1, "%")}</td>'
            rhtml += f'<td>{fmt_color(r["Max_DD_3Y"], 1, "%")}</td>'
            rhtml += f'<td>{fmt(r["Calmar"], 2)}</td>'
            rhtml += f'<td>{fmt_color(r["Info_Ratio"], 2, "")}</td></tr>'
        rhtml += "</tbody></table>"
        st.markdown(rhtml, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TAB 5: FUND DEEP-DIVE
    # ════════════════════════════════════════════
    with tab5:
        st.markdown("## Fund Deep-Dive")

        fund_options = df.sort_values("Rank")["Fund"].tolist()
        selected = st.selectbox("Select a fund to analyse", fund_options,
                                format_func=short_name)

        frow = df[df["Fund"] == selected].iloc[0]

        # Fund header metrics
        sig_cls = SIG_CLS.get(frow["Signal"], "sig-neutral")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;margin:1rem 0;">
            <span class="rk rk-{"1" if frow["Rank"]==1 else ("2" if frow["Rank"]==2 else ("3" if frow["Rank"]==3 else "n"))}"
                  style="width:40px;height:40px;font-size:16px;">{int(frow["Rank"])}</span>
            <div>
                <div style="font-size:20px;font-weight:600;color:var(--text);">{short_name(selected)}</div>
                <span class="sig {sig_cls}">{frow["Signal"]}</span>
                <span style="color:var(--text-dim);font-size:13px;margin-left:12px;">
                    Score: <strong style="color:var(--text);">{frow["Score"]:.1f}</strong>/100
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        dc_val = frow["Downside_Capture"]
        dc_status = "✓ Below 80" if pd.notna(dc_val) and dc_val < 80 else ("⚠ Below 100" if pd.notna(dc_val) and dc_val < 100 else "✗ Above 100")
        uc_val = frow["Upside_Capture"]
        uc_status = "✓ Above 100" if pd.notna(uc_val) and uc_val > 100 else "✗ Below 100"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="m-card">
                <div class="m-label">Rolling 3Y CAGR</div>
                <div class="m-value">{frow["Roll_3Y_Mean"]:.1f}%</div>
                <div class="m-sub">Min {fmt(frow["Roll_3Y_Min"],1)}% · Max {fmt(frow["Roll_3Y_Max"],1)}%</div>
            </div>
            <div class="m-card">
                <div class="m-label">Rolling 5Y CAGR</div>
                <div class="m-value">{fmt(frow["Roll_5Y_Mean"],1)}%</div>
                <div class="m-sub">Min {fmt(frow["Roll_5Y_Min"],1)}% · Max {fmt(frow["Roll_5Y_Max"],1)}%</div>
            </div>
            <div class="m-card {"green" if pd.notna(dc_val) and dc_val < 80 else "red"}">
                <div class="m-label">Downside Capture</div>
                <div class="m-value">{fmt(dc_val,0)}%</div>
                <div class="m-sub">{dc_status}</div>
            </div>
            <div class="m-card {"green" if pd.notna(uc_val) and uc_val > 100 else "amber"}">
                <div class="m-label">Upside Capture</div>
                <div class="m-value">{fmt(uc_val,0)}%</div>
                <div class="m-sub">{uc_status}</div>
            </div>
            <div class="m-card purple">
                <div class="m-label">Sortino Ratio</div>
                <div class="m-value">{fmt(frow["Sortino"],2)}</div>
                <div class="m-sub">vs Cat: {fmt(frow["Sortino_vs_Cat"],2, "")}</div>
            </div>
            <div class="m-card red">
                <div class="m-label">Max Drawdown</div>
                <div class="m-value">{fmt(frow["Max_DD_Full"],1)}%</div>
                <div class="m-sub">3Y: {fmt(frow["Max_DD_3Y"],1)}%</div>
            </div>
            <div class="m-card">
                <div class="m-label">Calmar Ratio</div>
                <div class="m-value">{fmt(frow["Calmar"],2)}</div>
                <div class="m-sub">Ann Return / Max DD</div>
            </div>
            <div class="m-card">
                <div class="m-label">AUM / Expense</div>
                <div class="m-value">₹{fmt(frow["AUM"],0)} Cr</div>
                <div class="m-sub">ER: {fmt(frow["ER"],2)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # NAV chart with drawdown
        st.markdown("### NAV History & Drawdown")
        fund_nav = nav[["Date", selected]].dropna()
        fund_nav = fund_nav.rename(columns={selected: "NAV"})
        fund_nav["Cummax"] = fund_nav["NAV"].cummax()
        fund_nav["DD"] = (fund_nav["NAV"] - fund_nav["Cummax"]) / fund_nav["Cummax"] * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.65, 0.35])

        fig.add_trace(go.Scatter(
            x=fund_nav["Date"], y=fund_nav["NAV"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            line=dict(color="#3b82f6", width=1.5),
            name="NAV", hovertemplate="%{x|%d %b %Y}<br>NAV: ₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=fund_nav["Date"], y=fund_nav["DD"],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#ef4444", width=1),
            name="Drawdown", hovertemplate="%{x|%d %b %Y}<br>DD: %{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            height=480, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk", color="#7a8baa"),
            margin=dict(l=55, r=25, t=30, b=35),
        )
        fig.update_xaxes(gridcolor="rgba(26,39,68,0.5)")
        fig.update_yaxes(gridcolor="rgba(26,39,68,0.5)")
        fig.update_yaxes(title_text="NAV (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart: factor scores
        st.markdown("### Factor Score Profile")
        score_cols = {
            "S_Roll3Y": "Rolling 3Y", "S_Roll5Y": "Rolling 5Y",
            "S_DownCap": "Down Capture", "S_UpCap": "Up Capture",
            "S_Sortino": "Sortino", "S_MaxDD": "Max DD",
            "S_Calmar": "Calmar", "S_InfoRatio": "Info Ratio",
            "S_ER": "Low Cost",
        }
        available = {k: v for k, v in score_cols.items() if pd.notna(frow.get(k))}
        if available:
            vals = [frow[k] for k in available]
            labels = list(available.values())
            vals.append(vals[0])
            labels.append(labels[0])

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill="toself",
                fillcolor="rgba(59,130,246,0.12)",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=6, color="#3b82f6"),
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=10, color="#7a8baa"),
                                    gridcolor="rgba(26,39,68,0.5)"),
                    angularaxis=dict(tickfont=dict(size=11, color="#94a3b8"),
                                     gridcolor="rgba(26,39,68,0.5)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                height=420, **{k: v for k, v in PLOT_THEME.items() if k not in ("xaxis", "yaxis")},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Footer ──
    st.markdown(
        '<div style="text-align:center;padding:2rem 0 1rem;color:var(--text-dim);font-size:12px;">'
        'Built with advanced quantitative methodology · Rolling Returns · Capture Ratios · '
        'Sortino · Drawdown Analysis<br>'
        'Benchmark: Equal-weight proxy of all small-cap funds in universe · Risk-free rate: 6% p.a.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
