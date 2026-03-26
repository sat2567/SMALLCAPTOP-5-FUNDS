"""
SmallCap Fund Ranking App — Quantitative Engine
Dual Layout: Long-Term Deep Quant + Short-Term Momentum
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmallCap Quant Rankings", layout="wide", page_icon="📊")


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner="Loading fund & benchmark data...")
def load_data():
    raw = pd.read_excel("smallcapfinalrank.xlsx")
    fund_names = raw.iloc[1, 1:].dropna().tolist()
    nav = raw.iloc[3:, : len(fund_names) + 1].copy()
    nav.columns = ["Date"] + fund_names
    nav = nav[pd.to_datetime(nav["Date"], errors="coerce").notna()].copy()
    nav["Date"] = pd.to_datetime(nav["Date"])
    nav = nav.sort_values("Date").reset_index(drop=True)
    for f in fund_names:
        nav[f] = pd.to_numeric(nav[f], errors="coerce")

    try:
        bench = pd.read_csv("Nifty_500_TRI_Combined.csv")
        bench["Date"] = pd.to_datetime(bench["Date"])
        nav = pd.merge(nav, bench, on="Date", how="left")
        nav["Benchmark"] = nav["Benchmark"].ffill()
    except Exception:
        st.sidebar.warning("⚠️ Nifty_500_TRI_Combined.csv not found. Using category average as proxy.")
        valid = [f for f in fund_names if nav[f].notna().sum() > 252]
        nav["Benchmark"] = nav[valid].mean(axis=1)

    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund", "AUM"]]

    er_raw = pd.read_excel("smallcap_expense_ratio.xlsx")
    er_recs, cur = [], None
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


# ═══════════════════════════════════════════════
# QUANTITATIVE METRICS ENGINE
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner="Running quantitative analysis...")
def compute_metrics(_nav, fund_names, aum_latest, df_er):
    nav = _nav.copy()
    latest_date = nav["Date"].max()
    monthly = nav.set_index("Date").resample("ME").last()
    monthly_ret = monthly.pct_change().dropna(how="all")
    bench_ret = monthly_ret["Benchmark"]
    rf_m = 0.06 / 12

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 252:
            continue
        fund_monthly = monthly_ret[fund].dropna()
        common = fund_monthly.index.intersection(bench_ret.dropna().index)
        if len(common) < 12:
            continue
        f_ret = fund_monthly.loc[common]
        b_ret_c = bench_ret.loc[common]
        daily_fd = fd.set_index("Date")[fund]

        track_yrs = len(fd) / 252

        # ────── ROLLING CAGR (3Y, 5Y) ──────
        def rolling_cagr(series, years, wm):
            mn = series.resample("ME").last().dropna()
            if len(mn) < wm:
                return None, None, None, None
            cagrs = []
            for i in range(wm, len(mn)):
                s, e = mn.iloc[i - wm], mn.iloc[i]
                if s > 0:
                    cagrs.append(((e / s) ** (1 / years) - 1) * 100)
            if not cagrs:
                return None, None, None, None
            a = np.array(cagrs)
            return np.mean(a), np.median(a), np.min(a), np.max(a)

        r3_mean, r3_med, r3_min, r3_max = rolling_cagr(daily_fd, 3, 36)
        r5_mean, r5_med, r5_min, r5_max = rolling_cagr(daily_fd, 5, 60)

        # ────── BENCHMARK WIN RATE ──────
        def win_rate(fund_s, bench_s, years, wm):
            fm = fund_s.resample("ME").last().dropna()
            bm = bench_s.resample("ME").last().dropna()
            ci = fm.index.intersection(bm.index)
            fm, bm = fm.loc[ci], bm.loc[ci]
            if len(fm) < wm:
                return None
            wins = total = 0
            for i in range(wm, len(fm)):
                fs, fe = fm.iloc[i - wm], fm.iloc[i]
                bs, be = bm.iloc[i - wm], bm.iloc[i]
                if fs > 0 and bs > 0:
                    if (fe / fs) ** (1 / years) > (be / bs) ** (1 / years):
                        wins += 1
                    total += 1
            return wins / total * 100 if total > 0 else None

        bench_daily = nav.set_index("Date")["Benchmark"]
        win_3y = win_rate(daily_fd, bench_daily, 3, 36)

        # ────── CAPTURE RATIOS ──────
        down_m = b_ret_c[b_ret_c < 0]
        up_m = b_ret_c[b_ret_c > 0]
        dc = (f_ret.loc[down_m.index].mean() / down_m.mean()) * 100 if len(down_m) > 3 else None
        uc = (f_ret.loc[up_m.index].mean() / up_m.mean()) * 100 if len(up_m) > 3 else None
        cap_ratio = uc / dc if (uc and dc and dc != 0) else None

        # ────── SORTINO (full + rolling mean/min) ──────
        excess = f_ret - rf_m
        neg = excess[excess < 0]
        ds = np.sqrt(np.mean(neg ** 2)) if len(neg) > 3 else None
        sortino = (f_ret.mean() - rf_m) / ds if ds and ds > 0 else None

        rolling_sortinos = []
        for i in range(36, len(f_ret)):
            w = f_ret.iloc[i - 36 : i]
            ex = w - rf_m
            ng = ex[ex < 0]
            d_s = np.sqrt(np.mean(ng ** 2)) if len(ng) > 3 else None
            if d_s and d_s > 0:
                rolling_sortinos.append((w.mean() - rf_m) / d_s)
        roll_sortino_mean = np.mean(rolling_sortinos) if rolling_sortinos else None
        roll_sortino_min = np.min(rolling_sortinos) if rolling_sortinos else None

        # ────── DRAWDOWN ANALYSIS ──────
        prices = fd[fund].values
        dates = fd["Date"].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd = dd.min() * 100
        current_dd = (prices[-1] - prices.max()) / prices.max() * 100

        # Recovery analysis: avg days to recover from >5% drawdowns
        recovery_days = []
        dd_start = None
        for i in range(len(dd)):
            if dd[i] < -0.05 and dd_start is None:
                dd_start = i
            elif dd[i] >= 0 and dd_start is not None:
                recovery_days.append(i - dd_start)
                dd_start = None
        avg_recovery = np.mean(recovery_days) if recovery_days else None
        max_recovery = np.max(recovery_days) if recovery_days else None
        dd_events = len(recovery_days)

        # ────── ALPHA CONSISTENCY ──────
        # Rolling 12M alpha: % of windows with positive alpha
        alphas_12m = []
        for i in range(12, len(f_ret)):
            fr = f_ret.iloc[i - 12 : i].sum()
            br = b_ret_c.loc[f_ret.iloc[i - 12 : i].index].sum()
            alphas_12m.append(fr - br)
        alpha_hit_rate = np.mean([a > 0 for a in alphas_12m]) * 100 if alphas_12m else None
        avg_alpha_12m = np.mean(alphas_12m) * 100 if alphas_12m else None

        # ────── TAIL RATIO ──────
        # Avg gain in best 10% months / avg loss in worst 10% months
        n = len(f_ret)
        if n >= 20:
            sorted_r = f_ret.sort_values()
            bottom = sorted_r.iloc[: max(1, n // 10)]
            top = sorted_r.iloc[-max(1, n // 10) :]
            tail_ratio = abs(top.mean() / bottom.mean()) if bottom.mean() != 0 else None
        else:
            tail_ratio = None

        # ────── PAIN INDEX ──────
        # Average depth of drawdown across all days (lower = less pain)
        pain_index = np.mean(np.abs(dd)) * 100

        # ────── ULCER INDEX ──────
        # RMS of drawdown — penalises deep prolonged drawdowns more
        ulcer = np.sqrt(np.mean(dd ** 2)) * 100

        # ────── BASIC RATIOS ──────
        ann_ret = f_ret.mean() * 12
        vol_ann = f_ret.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.06) / vol_ann if vol_ann > 0 else None
        calmar = abs(ann_ret * 100 / max_dd) if max_dd != 0 else None
        active = f_ret - b_ret_c.loc[f_ret.index]
        te = active.std() * np.sqrt(12)
        info_ratio = (active.mean() * 12) / te if te > 0 else None

        # ────── MOMENTUM (1Y, 6M) ──────
        latest_nav = fd.iloc[-1][fund]
        ld = fd.iloc[-1]["Date"]

        def get_ret(days):
            mask = (nav["Date"] >= ld - timedelta(days=days + 15)) & (nav["Date"] <= ld - timedelta(days=days - 15))
            ns = nav[mask].dropna(subset=[fund])
            if len(ns) == 0:
                return None
            nv = ns.iloc[(ns["Date"] - (ld - timedelta(days=days))).abs().argsort().iloc[0]][fund]
            return (latest_nav / nv - 1) * 100

        ret_1y = get_ret(365)
        ret_6m = get_ret(180)
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol_1y = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None

        results.append({
            "Fund": fund, "Track_Yrs": track_yrs,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol_1Y": vol_1y,
            "Roll_3Y_Mean": r3_mean, "Roll_3Y_Med": r3_med, "Roll_3Y_Min": r3_min, "Roll_3Y_Max": r3_max,
            "Roll_5Y_Mean": r5_mean, "Roll_5Y_Med": r5_med, "Roll_5Y_Min": r5_min, "Roll_5Y_Max": r5_max,
            "Win_3Y": win_3y,
            "Down_Cap": dc, "Up_Cap": uc, "Cap_Ratio": cap_ratio,
            "Sortino": sortino, "Roll_Sortino_Mean": roll_sortino_mean, "Roll_Sortino_Min": roll_sortino_min,
            "Max_DD": max_dd, "Current_DD": current_dd,
            "Avg_Recovery_Days": avg_recovery, "Max_Recovery_Days": max_recovery, "DD_Events": dd_events,
            "Alpha_Hit_Rate": alpha_hit_rate, "Avg_Alpha_12M": avg_alpha_12m,
            "Tail_Ratio": tail_ratio, "Pain_Index": pain_index, "Ulcer_Index": ulcer,
            "Sharpe": sharpe, "Calmar": calmar, "Info_Ratio": info_ratio, "Ann_Ret": ann_ret * 100,
        })

    df = pd.DataFrame(results)
    df = df.merge(aum_latest, on="Fund", how="left")
    df = df.merge(df_er, on="Fund", how="left")
    return df


# ═══════════════════════════════════════════════
# RANKING ENGINE
# ═══════════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def rank_all(df, w_lt, w_st):
    d = df.copy()

    # ── Long-Term factor scores ──
    d["S_R3"] = pctrank(d["Roll_3Y_Mean"])
    d["S_R5"] = pctrank(d["Roll_5Y_Mean"])
    d["S_W3"] = pctrank(d["Win_3Y"])
    d["S_DC"] = pctrank(d["Down_Cap"], asc=False)
    d["S_UC"] = pctrank(d["Up_Cap"])
    d["S_SO"] = pctrank(d["Sortino"])
    d["S_RSO"] = pctrank(d["Roll_Sortino_Mean"])
    d["S_DD"] = pctrank(d["Max_DD"])
    d["S_CA"] = pctrank(d["Calmar"])
    d["S_IR"] = pctrank(d["Info_Ratio"])
    d["S_AH"] = pctrank(d["Alpha_Hit_Rate"])
    d["S_TR"] = pctrank(d["Tail_Ratio"])
    d["S_PI"] = pctrank(d["Pain_Index"], asc=False)
    d["S_RC"] = pctrank(d["Avg_Recovery_Days"], asc=False)
    d["S_ER"] = pctrank(d["ER"], asc=False)

    def comp(row, wmap, min_yrs=0):
        if row.get("Track_Yrs", 0) < min_yrs:
            return np.nan
        tw = ts = 0
        for c, wt in wmap.items():
            if wt > 0 and pd.notna(row.get(c)):
                ts += row[c] * wt
                tw += wt
        return ts / tw if tw > 0 else np.nan

    d["Score_LT"] = d.apply(lambda r: comp(r, w_lt, min_yrs=3.0), axis=1)
    d["Rank_LT"] = d["Score_LT"].rank(ascending=False, method="min").astype("Int64")

    # ── Short-Term factor scores ──
    d["S_6M"] = pctrank(d["Ret_6M"])
    d["S_1Y"] = pctrank(d["Ret_1Y"])
    d["S_V1"] = pctrank(d["Vol_1Y"], asc=False)

    d["Score_ST"] = d.apply(lambda r: comp(r, w_st), axis=1)
    d["Rank_ST"] = d["Score_ST"].rank(ascending=False, method="min").astype("Int64")

    def sig(x):
        if pd.isna(x): return "N/A"
        if x >= 78: return "Elite"
        if x >= 62: return "Strong"
        if x >= 48: return "Above Avg"
        if x >= 35: return "Average"
        return "Below Avg"

    d["Signal_LT"] = d["Score_LT"].apply(sig)
    d["Signal_ST"] = d["Score_ST"].apply(sig)
    return d


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


# ═══════════════════════════════════════════════
# STYLING HELPERS
# ═══════════════════════════════════════════════
def color_signal(val):
    m = {"Elite": "background-color:#dcfce7;color:#166534;", "Strong": "background-color:#e0f2fe;color:#075985;",
         "Above Avg": "background-color:#fef9c3;color:#854d0e;", "Average": "background-color:#fed7aa;color:#9a3412;",
         "Below Avg": "background-color:#fecaca;color:#991b1b;"}
    return m.get(val, "")

def color_score(val):
    if pd.isna(val): return ""
    if val >= 70: return "background-color:#dcfce7;font-weight:700;"
    if val >= 50: return "background-color:#e0f2fe;"
    if val >= 35: return "background-color:#fef9c3;"
    return "background-color:#fecaca;"

def color_pct(val):
    if pd.isna(val): return ""
    return "color:#16a34a;" if val > 0 else "color:#dc2626;"

def color_dc(val):
    if pd.isna(val): return ""
    if val < 80: return "color:#16a34a;font-weight:600;"
    if val < 100: return "color:#ca8a04;"
    return "color:#dc2626;font-weight:600;"

def color_uc(val):
    if pd.isna(val): return ""
    return "color:#16a34a;font-weight:600;" if val > 100 else "color:#ca8a04;"

def fmt(v, d=1, suf=""):
    return f"{v:.{d}f}{suf}" if pd.notna(v) else "—"


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, df_er = load_data()
    df_raw = compute_metrics(nav, fund_names, aum_latest, df_er)

    st.title("📊 SmallCap Quantitative Rankings")
    st.caption(f"{len(df_raw)} funds · Data through {nav['Date'].max().strftime('%d %b %Y')}")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Long-Term Weights")
        st.caption("Adjust factor importance. Normalised automatically.")

        w_r3 = st.slider("Rolling 3Y CAGR", 0, 25, 12)
        w_r5 = st.slider("Rolling 5Y CAGR", 0, 25, 12)
        w_w3 = st.slider("3Y Win Rate vs Bench", 0, 15, 5)
        w_dc = st.slider("Downside Capture", 0, 25, 14)
        w_uc = st.slider("Upside Capture", 0, 15, 6)
        w_so = st.slider("Sortino (Full)", 0, 20, 8)
        w_rso = st.slider("Rolling 3Y Sortino (Mean)", 0, 20, 6)
        w_dd = st.slider("Max Drawdown", 0, 15, 6)
        w_ca = st.slider("Calmar Ratio", 0, 10, 3)
        w_ah = st.slider("Alpha Hit Rate", 0, 15, 8)
        w_tr = st.slider("Tail Ratio", 0, 10, 4)
        w_pi = st.slider("Pain Index (lower=better)", 0, 10, 4)
        w_rc = st.slider("Recovery Speed", 0, 10, 4)
        w_ir = st.slider("Information Ratio", 0, 10, 3)
        w_er = st.slider("Expense Ratio", 0, 10, 3)

        t = w_r3+w_r5+w_w3+w_dc+w_uc+w_so+w_rso+w_dd+w_ca+w_ah+w_tr+w_pi+w_rc+w_ir+w_er
        if t == 0: t = 1
        w_lt = {k: v/t for k, v in zip(
            ["S_R3","S_R5","S_W3","S_DC","S_UC","S_SO","S_RSO","S_DD","S_CA","S_AH","S_TR","S_PI","S_RC","S_IR","S_ER"],
            [w_r3,w_r5,w_w3,w_dc,w_uc,w_so,w_rso,w_dd,w_ca,w_ah,w_tr,w_pi,w_rc,w_ir,w_er]
        )}

        st.divider()
        st.header("🚀 Momentum Weights")
        sw_6m = st.slider("6M Return", 0, 100, 45)
        sw_1y = st.slider("1Y Return", 0, 100, 40)
        sw_v1 = st.slider("1Y Volatility", 0, 50, 15)
        st2 = sw_6m + sw_1y + sw_v1
        if st2 == 0: st2 = 1
        w_st = {"S_6M": sw_6m/st2, "S_1Y": sw_1y/st2, "S_V1": sw_v1/st2}

    df = rank_all(df_raw, w_lt, w_st)

    tab_rank, tab_deep = st.tabs(["🏆 Rankings", "🔎 Fund Deep-Dive"])

    # ═══════════════════════════════════
    # TAB 1: RANKINGS
    # ═══════════════════════════════════
    with tab_rank:
        board = st.radio("View:", [
            "🏛️ Long-Term Quantitative (funds > 3 yrs)",
            "🚀 Short-Term Momentum (all funds)"
        ], horizontal=True)
        st.divider()

        if "Long-Term" in board:
            dv = df[df["Rank_LT"].notna()].sort_values("Rank_LT").copy()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🥇 Top Fund", short(dv.iloc[0]["Fund"]), f"Score {dv.iloc[0]['Score_LT']:.1f}")
            c2.metric("Median Down Capture", f"{dv['Down_Cap'].median():.0f}%")
            c3.metric("Median Alpha Hit Rate", f"{dv['Alpha_Hit_Rate'].median():.0f}%")
            c4.metric("Funds Ranked", len(dv))

            disp = dv[[
                "Rank_LT", "Fund", "Score_LT", "Signal_LT", "Track_Yrs",
                "Roll_3Y_Mean", "Roll_5Y_Mean", "Win_3Y",
                "Down_Cap", "Up_Cap", "Cap_Ratio",
                "Sortino", "Roll_Sortino_Mean",
                "Alpha_Hit_Rate", "Tail_Ratio",
                "Max_DD", "Pain_Index", "Avg_Recovery_Days",
                "Calmar", "Info_Ratio", "AUM", "ER",
            ]].copy()
            disp.columns = [
                "Rank", "Fund", "Score", "Signal", "Age",
                "Roll 3Y%", "Roll 5Y%", "Win 3Y%",
                "Down Cap", "Up Cap", "Cap Ratio",
                "Sortino", "Roll Sortino",
                "Alpha Hit%", "Tail Ratio",
                "Max DD%", "Pain Idx", "Avg Recovery (days)",
                "Calmar", "Info Ratio", "AUM (Cr)", "Exp Ratio%",
            ]
            disp["Fund"] = disp["Fund"].apply(short)
            disp = disp.round({
                "Score": 1, "Age": 1, "Roll 3Y%": 1, "Roll 5Y%": 1, "Win 3Y%": 1,
                "Down Cap": 0, "Up Cap": 0, "Cap Ratio": 2,
                "Sortino": 2, "Roll Sortino": 2,
                "Alpha Hit%": 1, "Tail Ratio": 2,
                "Max DD%": 1, "Pain Idx": 2, "Avg Recovery (days)": 0,
                "Calmar": 2, "Info Ratio": 2, "AUM (Cr)": 0, "Exp Ratio%": 2,
            })

            styled = (disp.style
                .map(color_signal, subset=["Signal"])
                .map(color_score, subset=["Score"])
                .map(color_dc, subset=["Down Cap"])
                .map(color_uc, subset=["Up Cap"])
                .format(na_rep="—")
                .set_properties(**{"text-align": "center", "font-size": "12.5px"})
                .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
            )
            st.dataframe(styled, use_container_width=True, height=650, hide_index=True)

            with st.expander("📖 What each metric means"):
                st.markdown("""
| Metric | What it tells you |
|---|---|
| **Rolling 3Y / 5Y CAGR** | Average annualised return across every possible 3 or 5 year window — removes timing luck |
| **Win Rate 3Y** | % of rolling 3-year windows where the fund beat the benchmark |
| **Downside Capture** | How much the fund falls when the benchmark falls. Below 80 is excellent |
| **Upside Capture** | How much the fund gains when the benchmark gains. Above 100 means it amplifies rallies |
| **Capture Ratio** | Upside ÷ Downside. Above 1.2 = strong asymmetry in your favour |
| **Sortino** | Return per unit of downside risk. Only penalises bad volatility, not good |
| **Rolling Sortino** | Average Sortino across rolling 3-year windows — tests consistency |
| **Alpha Hit Rate** | % of rolling 12-month windows with positive alpha vs benchmark |
| **Tail Ratio** | Avg gain in best 10% months ÷ Avg loss in worst 10%. Above 1.2 = fat right tail |
| **Max Drawdown** | Worst peak-to-trough fall ever. Tests crisis behaviour |
| **Pain Index** | Average depth of all drawdowns. Lower = smoother ride |
| **Avg Recovery** | Average trading days to recover from >5% drawdowns |
| **Calmar** | Annualised return ÷ Max Drawdown. Higher = better return per unit of crash risk |
| **Information Ratio** | Annualised alpha ÷ tracking error. Consistency of outperformance |
""")

        else:
            dv = df.sort_values("Rank_ST").copy()
            c1, c2, c3 = st.columns(3)
            c1.metric("🔥 Top Momentum", short(dv.iloc[0]["Fund"]), f"Score {dv.iloc[0]['Score_ST']:.1f}")
            c2.metric("Median 1Y Return", f"{dv['Ret_1Y'].median():.1f}%")
            c3.metric("Funds Ranked", len(dv))

            disp = dv[[
                "Rank_ST", "Fund", "Score_ST", "Signal_ST", "Track_Yrs",
                "Ret_6M", "Ret_1Y", "Vol_1Y", "AUM",
            ]].copy()
            disp.columns = ["Rank", "Fund", "Score", "Signal", "Age", "6M Ret%", "1Y Ret%", "Vol 1Y%", "AUM (Cr)"]
            disp["Fund"] = disp["Fund"].apply(short)
            disp = disp.round({"Score": 1, "Age": 1, "6M Ret%": 1, "1Y Ret%": 1, "Vol 1Y%": 1, "AUM (Cr)": 0})

            styled = (disp.style
                .map(color_signal, subset=["Signal"])
                .map(color_score, subset=["Score"])
                .map(color_pct, subset=["6M Ret%", "1Y Ret%"])
                .format(na_rep="—")
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
            )
            st.dataframe(styled, use_container_width=True, height=650, hide_index=True)

        # Download
        buf = io.BytesIO()
        df.round(2).to_excel(buf, index=False, engine="openpyxl")
        st.download_button("📥 Download all data", buf.getvalue(), "smallcap_quant_rankings.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ═══════════════════════════════════
    # TAB 2: FUND DEEP-DIVE
    # ═══════════════════════════════════
    with tab_deep:
        selected = st.selectbox("Pick a fund", df.sort_values("Fund")["Fund"].tolist(), format_func=short)
        r = df[df["Fund"] == selected].iloc[0]

        # ── Header ──
        st.subheader(short(selected))
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Track record:** {r['Track_Yrs']:.1f} years")
        lt_txt = f"{r['Score_LT']:.1f} (#{r['Rank_LT']})" if pd.notna(r["Score_LT"]) else "< 3 years"
        c2.markdown(f"**🏛️ Long-Term Score:** {lt_txt}")
        c3.markdown(f"**🚀 Momentum Score:** {r['Score_ST']:.1f} (#{r['Rank_ST']})")
        st.divider()

        # ── Section 1: Return Profile ──
        st.markdown("#### Return Profile")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("6M Return", fmt(r["Ret_6M"], 1, "%"))
        c2.metric("1Y Return", fmt(r["Ret_1Y"], 1, "%"))
        c3.metric("Rolling 3Y CAGR", fmt(r["Roll_3Y_Mean"], 1, "%"),
                   f"Range: {fmt(r['Roll_3Y_Min'],1)}% to {fmt(r['Roll_3Y_Max'],1)}%" if pd.notna(r["Roll_3Y_Min"]) else None)
        c4.metric("Rolling 5Y CAGR", fmt(r["Roll_5Y_Mean"], 1, "%"),
                   f"Range: {fmt(r['Roll_5Y_Min'],1)}% to {fmt(r['Roll_5Y_Max'],1)}%" if pd.notna(r["Roll_5Y_Min"]) else None)
        c5.metric("3Y Win Rate", fmt(r["Win_3Y"], 1, "%"), "vs benchmark")

        # ── Section 2: Risk & Protection ──
        st.markdown("#### Risk & Downside Protection")
        c1, c2, c3, c4, c5 = st.columns(5)
        dc_v = r["Down_Cap"]
        c1.metric("Downside Capture", fmt(dc_v, 0, "%"),
                   "✓ Below 80" if pd.notna(dc_v) and dc_v < 80 else ("Below 100" if pd.notna(dc_v) and dc_v < 100 else "Above 100 ⚠"))
        c2.metric("Upside Capture", fmt(r["Up_Cap"], 0, "%"),
                   "✓ Above 100" if pd.notna(r["Up_Cap"]) and r["Up_Cap"] > 100 else None)
        c3.metric("Capture Ratio", fmt(r["Cap_Ratio"], 2),
                   "✓ > 1.2" if pd.notna(r["Cap_Ratio"]) and r["Cap_Ratio"] > 1.2 else None)
        c4.metric("Max Drawdown", fmt(r["Max_DD"], 1, "%"))
        c5.metric("Current Drawdown", fmt(r["Current_DD"], 1, "%"))

        # ── Section 3: Consistency ──
        st.markdown("#### Consistency & Risk-Adjusted Quality")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sortino Ratio", fmt(r["Sortino"], 3))
        c2.metric("Rolling Sortino (3Y avg)", fmt(r["Roll_Sortino_Mean"], 3),
                   f"Worst window: {fmt(r['Roll_Sortino_Min'],3)}" if pd.notna(r["Roll_Sortino_Min"]) else None)
        c3.metric("Alpha Hit Rate", fmt(r["Alpha_Hit_Rate"], 1, "%"), "% of 12M windows with +alpha")
        c4.metric("Tail Ratio", fmt(r["Tail_Ratio"], 2),
                   "Good" if pd.notna(r["Tail_Ratio"]) and r["Tail_Ratio"] > 1.2 else "Weak" if pd.notna(r["Tail_Ratio"]) else None)
        c5.metric("Info Ratio", fmt(r["Info_Ratio"], 2))

        # ── Section 4: Drawdown Behaviour ──
        st.markdown("#### Drawdown Behaviour")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pain Index", fmt(r["Pain_Index"], 2), "avg drawdown depth")
        c2.metric("Ulcer Index", fmt(r["Ulcer_Index"], 2) if "Ulcer_Index" in r else "—")
        c3.metric("Avg Recovery", f"{int(r['Avg_Recovery_Days'])} days" if pd.notna(r["Avg_Recovery_Days"]) else "—")
        c4.metric("Max Recovery", f"{int(r['Max_Recovery_Days'])} days" if pd.notna(r["Max_Recovery_Days"]) else "—")
        c5.metric("DD Events (>5%)", f"{int(r['DD_Events'])}" if pd.notna(r["DD_Events"]) else "—")

        # ── Section 5: Cost & Size ──
        st.markdown("#### Cost & Size")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expense Ratio", fmt(r["ER"], 2, "%"))
        c2.metric("AUM", f"₹{r['AUM']:.0f} Cr" if pd.notna(r["AUM"]) else "—")
        c3.metric("Sharpe Ratio", fmt(r["Sharpe"], 2))
        c4.metric("Calmar Ratio", fmt(r["Calmar"], 2))

        st.divider()

        # ── Charts ──
        st.markdown("#### NAV vs Benchmark & Drawdown")

        fn = nav[["Date", selected, "Benchmark"]].dropna(subset=[selected]).copy()
        fn = fn.rename(columns={selected: "NAV"})
        if not fn.empty and fn.iloc[0]["Benchmark"] > 0:
            fn["Bench_Scaled"] = fn["Benchmark"] * (fn.iloc[0]["NAV"] / fn.iloc[0]["Benchmark"])
        else:
            fn["Bench_Scaled"] = fn["Benchmark"]
        fn["Peak"] = fn["NAV"].cummax()
        fn["DD"] = (fn["NAV"] - fn["Peak"]) / fn["Peak"] * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.65, 0.35], subplot_titles=("NAV vs Benchmark (rebased)", "Drawdown %"))
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["NAV"], fill="tozeroy",
                                  fillcolor="rgba(59,130,246,0.06)", line=dict(color="#3b82f6", width=1.5),
                                  name="Fund NAV"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["Bench_Scaled"],
                                  line=dict(color="#94a3b8", width=1.5, dash="dot"),
                                  name="Benchmark"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["DD"], fill="tozeroy",
                                  fillcolor="rgba(239,68,68,0.08)", line=dict(color="#ef4444", width=1),
                                  name="Drawdown"), row=2, col=1)
        fig.update_layout(height=450, margin=dict(l=50, r=20, t=30, b=30), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Capture scatter ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Capture Map (you vs peers)")
            cap_df = df.dropna(subset=["Down_Cap", "Up_Cap"])
            fig2 = go.Figure()
            fig2.add_hline(y=100, line=dict(color="rgba(0,0,0,0.12)", dash="dash", width=1))
            fig2.add_vline(x=80, line=dict(color="rgba(0,0,0,0.12)", dash="dash", width=1))
            fig2.add_trace(go.Scatter(
                x=cap_df["Down_Cap"], y=cap_df["Up_Cap"], mode="markers",
                marker=dict(
                    size=[14 if f == selected else 8 for f in cap_df["Fund"]],
                    color=["#3b82f6" if f == selected else "#d1d5db" for f in cap_df["Fund"]],
                    line=dict(width=[2 if f == selected else 0 for f in cap_df["Fund"]], color="#1d4ed8"),
                ),
                text=[short(f) for f in cap_df["Fund"]],
                hovertemplate="<b>%{text}</b><br>Down: %{x:.0f}%<br>Up: %{y:.0f}%<extra></extra>",
            ))
            fig2.update_layout(xaxis_title="Downside Capture %", yaxis_title="Upside Capture %",
                               height=350, margin=dict(l=50, r=20, t=20, b=40), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown("#### Factor Score Radar")
            score_map = {
                "S_R3": "Roll 3Y", "S_R5": "Roll 5Y", "S_DC": "Low Down Cap",
                "S_UC": "High Up Cap", "S_SO": "Sortino", "S_AH": "Alpha Hit",
                "S_DD": "Low Drawdown", "S_TR": "Tail Ratio", "S_PI": "Low Pain",
                "S_RC": "Fast Recovery", "S_ER": "Low Cost",
            }
            avail = {k: v for k, v in score_map.items() if pd.notna(r.get(k))}
            if avail:
                vals = [r[k] for k in avail]
                labels = list(avail.values())
                vals.append(vals[0])
                labels.append(labels[0])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatterpolar(
                    r=vals, theta=labels, fill="toself",
                    fillcolor="rgba(59,130,246,0.08)", line=dict(color="#3b82f6", width=2),
                    marker=dict(size=5),
                ))
                fig3.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=10)),
                               angularaxis=dict(tickfont=dict(size=11))),
                    height=350, margin=dict(l=50, r=50, t=20, b=20), showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

        # ── Rolling 3Y CAGR range ──
        st.markdown("#### Rolling 3Y CAGR — Where does this fund sit?")
        r3_df = df.dropna(subset=["Roll_3Y_Mean"]).sort_values("Roll_3Y_Mean", ascending=True).tail(15)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            y=[short(f) for f in r3_df["Fund"]],
            x=r3_df["Roll_3Y_Max"] - r3_df["Roll_3Y_Min"],
            base=r3_df["Roll_3Y_Min"], orientation="h",
            marker=dict(color=["rgba(59,130,246,0.15)" if f == selected else "rgba(0,0,0,0.04)" for f in r3_df["Fund"]]),
            hovertemplate="Min: %{base:.1f}% · Max: %{customdata:.1f}%<extra></extra>",
            customdata=r3_df["Roll_3Y_Max"],
        ))
        fig4.add_trace(go.Scatter(
            y=[short(f) for f in r3_df["Fund"]],
            x=r3_df["Roll_3Y_Mean"], mode="markers",
            marker=dict(color=["#3b82f6" if f == selected else "#d1d5db" for f in r3_df["Fund"]], size=9, symbol="diamond"),
        ))
        fig4.update_layout(title="Rolling 3Y CAGR — Min / Mean / Max", showlegend=False,
                           height=400, margin=dict(l=50, r=20, t=40, b=30))
        fig4.update_xaxes(title_text="CAGR (%)")
        st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()
