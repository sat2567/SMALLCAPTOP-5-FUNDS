"""
SmallCap Fund Ranking — Dual-Engine 3-Factor Model
Benchmark: Nifty 500 TRI

ENGINE 1 — ESTABLISHED FUNDS (> 3 years)
  Factor 1: Rolling 5Y CAGR — full-cycle compounding
  Factor 2: Downside Capture — capital protection
  Factor 3: Alpha Hit Rate (12M rolling) — consistency of skill

ENGINE 2 — EMERGING FUNDS (6 months to 3 years)
  Factor 1: Batting Average — % of months beating Nifty 500
  Factor 2: Downside Capture — still the #1 risk metric
  Factor 3: Alpha Hit Rate (6M rolling) — shorter window, same principle

Why a separate engine for newer funds:
- Rolling 5Y is impossible with < 3 years of data
- Rolling 12M alpha needs 24+ months to be meaningful (only 12 windows with 2 years)
- But downside capture, batting average, and 6M alpha windows work with as few as 8-10 months
- The same PRINCIPLES apply (protect downside, beat benchmark consistently)
  but the MEASUREMENT WINDOWS adapt to available data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmallCap 3-Factor", layout="wide", page_icon="📊")


# ═══════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading data...")
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
        bench_name = "Nifty 500 TRI"
    except Exception:
        valid = [f for f in fund_names if nav[f].notna().sum() > 252]
        nav["Benchmark"] = nav[valid].mean(axis=1)
        bench_name = "Category Average (proxy)"

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
    return nav, fund_names, aum_latest, df_er, bench_name


# ═══════════════════════════════════════════
# METRICS ENGINE
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Computing factors for all funds...")
def compute_all(_nav, fund_names, aum_latest, df_er):
    nav = _nav.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    br = mret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 120:  # need at least ~6 months
            continue
        fr = mret[fund].dropna()
        ci = fr.index.intersection(br.dropna().index)
        if len(ci) < 6:
            continue
        fr, brc = fr.loc[ci], br.loc[ci]

        track_yrs = len(fd) / 252
        daily = fd.set_index("Date")[fund]
        months_available = len(fr)

        # ══════════════════════════════════
        # ESTABLISHED FUND FACTORS (>3 yrs)
        # ══════════════════════════════════

        # Factor 1: Rolling 5Y CAGR
        mn = daily.resample("ME").last().dropna()
        cagrs_5y = []
        for i in range(60, len(mn)):
            s, e = mn.iloc[i - 60], mn.iloc[i]
            if s > 0:
                cagrs_5y.append(((e / s) ** (1 / 5) - 1) * 100)
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        r5_med = np.median(cagrs_5y) if cagrs_5y else None
        r5_min = np.min(cagrs_5y) if cagrs_5y else None
        r5_max = np.max(cagrs_5y) if cagrs_5y else None

        # Rolling 3Y for display
        cagrs_3y = []
        for i in range(36, len(mn)):
            s, e = mn.iloc[i - 36], mn.iloc[i]
            if s > 0:
                cagrs_3y.append(((e / s) ** (1 / 3) - 1) * 100)
        r3_mean = np.mean(cagrs_3y) if cagrs_3y else None

        # Factor 2 (shared): Downside Capture
        dm = brc[brc < 0]
        down_cap = (fr.loc[dm.index].mean() / dm.mean()) * 100 if len(dm) >= 3 else None

        # Upside Capture (display)
        um = brc[brc > 0]
        up_cap = (fr.loc[um.index].mean() / um.mean()) * 100 if len(um) >= 3 else None

        # Factor 3 (established): Alpha Hit Rate — 12M rolling
        alphas_12m = []
        for i in range(12, len(fr)):
            alphas_12m.append(fr.iloc[i - 12 : i].sum() - brc.loc[fr.iloc[i - 12 : i].index].sum())
        alpha_hit_12m = np.mean([a > 0 for a in alphas_12m]) * 100 if alphas_12m else None

        # ══════════════════════════════════
        # EMERGING FUND FACTORS (<3 yrs)
        # ══════════════════════════════════

        # Factor 1 (emerging): Batting Average — % of months beating benchmark
        batting_avg = (fr > brc).mean() * 100

        # Factor 3 (emerging): Alpha Hit Rate — 6M rolling (shorter window)
        alphas_6m = []
        for i in range(6, len(fr)):
            alphas_6m.append(fr.iloc[i - 6 : i].sum() - brc.loc[fr.iloc[i - 6 : i].index].sum())
        alpha_hit_6m = np.mean([a > 0 for a in alphas_6m]) * 100 if alphas_6m else None

        # ══════════════════════════════════
        # SUPPORTING METRICS (display only)
        # ══════════════════════════════════
        prices = fd[fund].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd = dd.min() * 100
        current_dd = (prices[-1] - prices.max()) / prices.max() * 100

        rf_m = 0.06 / 12
        excess = fr - rf_m
        neg = excess[excess < 0]
        ds = np.sqrt(np.mean(neg ** 2)) if len(neg) > 3 else None
        sortino = (fr.mean() - rf_m) / ds if ds and ds > 0 else None

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

        results.append({
            "Fund": fund, "Track_Yrs": track_yrs, "Months": months_available,
            # Established factors
            "Roll_5Y": r5_mean, "Roll_5Y_Med": r5_med, "Roll_5Y_Min": r5_min, "Roll_5Y_Max": r5_max,
            "Roll_3Y": r3_mean,
            "Down_Cap": down_cap, "Up_Cap": up_cap,
            "Alpha_Hit_12M": alpha_hit_12m,
            # Emerging factors
            "Batting_Avg": batting_avg,
            "Alpha_Hit_6M": alpha_hit_6m,
            # Supporting
            "Max_DD": max_dd, "Current_DD": current_dd,
            "Sortino": sortino, "Ret_1Y": ret_1y, "Ret_6M": ret_6m,
        })

    df = pd.DataFrame(results)
    df = df.merge(aum_latest, on="Fund", how="left")
    df = df.merge(df_er, on="Fund", how="left")
    return df


# ═══════════════════════════════════════════
# RANKING
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def rank_all(df, w_lt, w_em):
    d = df.copy()

    # ── ENGINE 1: ESTABLISHED (>3 years) ──
    est = d[d["Track_Yrs"] >= 3].copy()
    est["S_R5"] = pctrank(est["Roll_5Y"])
    est["S_DC"] = pctrank(est["Down_Cap"], asc=False)
    est["S_AH"] = pctrank(est["Alpha_Hit_12M"])

    t1 = sum(w_lt)
    if t1 == 0: t1 = 1

    def comp_lt(row):
        tw = ts = 0
        for col, wt in zip(["S_R5", "S_DC", "S_AH"], w_lt):
            if pd.notna(row.get(col)):
                ts += row[col] * (wt / t1)
                tw += wt / t1
        return ts / tw if tw > 0 else np.nan

    est["Score"] = est.apply(comp_lt, axis=1)
    est["Rank"] = est["Score"].rank(ascending=False, method="min").astype(int)
    est["Engine"] = "Established"

    def sig_lt(x):
        if pd.isna(x): return "N/A"
        if x >= 75: return "Elite"
        if x >= 55: return "Strong"
        if x >= 40: return "Average"
        return "Weak"
    est["Signal"] = est["Score"].apply(sig_lt)

    # ── ENGINE 2: EMERGING (<3 years, >=6 months) ──
    em = d[(d["Track_Yrs"] < 3) & (d["Months"] >= 6)].copy()

    if len(em) > 0:
        em["S_BA"] = pctrank(em["Batting_Avg"])
        em["S_DC"] = pctrank(em["Down_Cap"], asc=False)
        em["S_AH6"] = pctrank(em["Alpha_Hit_6M"])

        t2 = sum(w_em)
        if t2 == 0: t2 = 1

        def comp_em(row):
            tw = ts = 0
            for col, wt in zip(["S_BA", "S_DC", "S_AH6"], w_em):
                if pd.notna(row.get(col)):
                    ts += row[col] * (wt / t2)
                    tw += wt / t2
            return ts / tw if tw > 0 else np.nan

        em["Score"] = em.apply(comp_em, axis=1)
        em["Rank"] = em["Score"].rank(ascending=False, method="min").astype(int)
        em["Engine"] = "Emerging"

        def sig_em(x):
            if pd.isna(x): return "N/A"
            if x >= 70: return "Promising"
            if x >= 45: return "Watch"
            return "Early / Weak"
        em["Signal"] = em["Score"].apply(sig_em)

    # Too young (< 6 months)
    too_young = d[d["Months"] < 6].copy() if "Months" in d.columns else pd.DataFrame()
    if len(too_young) > 0:
        too_young["Score"] = np.nan
        too_young["Rank"] = np.nan
        too_young["Signal"] = "Too Young"
        too_young["Engine"] = "Too Young"

    return est, em, too_young


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


def fmt(v, d=1, suf=""):
    return f"{v:.{d}f}{suf}" if pd.notna(v) else "—"


# ═══════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════
def c_sig(val):
    m = {
        "Elite": "background-color:#dcfce7;color:#166534;",
        "Strong": "background-color:#e0f2fe;color:#075985;",
        "Average": "background-color:#fef9c3;color:#854d0e;",
        "Weak": "background-color:#fecaca;color:#991b1b;",
        "Promising": "background-color:#dcfce7;color:#166534;",
        "Watch": "background-color:#fef9c3;color:#854d0e;",
        "Early / Weak": "background-color:#fecaca;color:#991b1b;",
    }
    return m.get(val, "")

def c_score(val):
    if pd.isna(val): return ""
    if val >= 70: return "background-color:#dcfce7;font-weight:700;"
    if val >= 50: return "background-color:#e0f2fe;"
    if val >= 35: return "background-color:#fef9c3;"
    return "background-color:#fecaca;"

def c_dc(val):
    if pd.isna(val): return ""
    if val < 50: return "color:#16a34a;font-weight:700;"
    if val < 80: return "color:#16a34a;"
    if val < 100: return "color:#ca8a04;"
    return "color:#dc2626;font-weight:600;"

def c_hit(val):
    if pd.isna(val): return ""
    if val >= 65: return "color:#16a34a;font-weight:600;"
    if val >= 50: return "color:#ca8a04;"
    return "color:#dc2626;"


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, df_er, bench_name = load_data()
    df_raw = compute_all(nav, fund_names, aum_latest, df_er)

    st.title("📊 SmallCap 3-Factor Ranking")
    st.caption(f"{len(df_raw)} funds · Benchmark: **{bench_name}** · Data through {nav['Date'].max().strftime('%d %b %Y')}")

    # Sidebar
    with st.sidebar:
        st.header("Established Fund Weights")
        st.caption("Funds with 3+ years of track record")
        lt1 = st.slider("Rolling 5Y CAGR", 10, 60, 35, key="lt1")
        lt2 = st.slider("Downside Capture", 10, 60, 35, key="lt2")
        lt3 = st.slider("Alpha Hit Rate (12M)", 10, 60, 30, key="lt3")

        st.divider()
        st.header("Emerging Fund Weights")
        st.caption("Funds with 6 months to 3 years")
        em1 = st.slider("Batting Average", 10, 60, 30, key="em1")
        em2 = st.slider("Downside Capture ", 10, 60, 40, key="em2")
        em3 = st.slider("Alpha Hit Rate (6M)", 10, 60, 30, key="em3")

    est, em, too_young = rank_all(df_raw, [lt1, lt2, lt3], [em1, em2, em3])

    tab_rank, tab_deep = st.tabs(["🏆 Rankings", "🔎 Fund Deep-Dive"])

    # ═══════════════════════════════════
    # TAB 1: RANKINGS
    # ═══════════════════════════════════
    with tab_rank:
        board = st.radio("View:", [
            "🏛️ Established Funds (3+ years)",
            "🚀 Emerging Funds (6 months – 3 years)",
        ], horizontal=True)
        st.divider()

        if "Established" in board:
            c1, c2, c3, c4 = st.columns(4)
            if len(est) > 0:
                top = est.sort_values("Rank").iloc[0]
                c1.metric("🥇 #1 Fund", short(top["Fund"]), f"Score {top['Score']:.1f}")
            c2.metric("Median Down Capture", f"{est['Down_Cap'].median():.0f}%")
            c3.metric("Median Alpha Hit", f"{est['Alpha_Hit_12M'].median():.0f}%")
            c4.metric("Funds Ranked", len(est))

            disp = est.sort_values("Rank")[[
                "Rank", "Fund", "Score", "Signal",
                "Roll_5Y", "Down_Cap", "Alpha_Hit_12M",
                "Roll_3Y", "Up_Cap", "Max_DD", "Sortino", "AUM", "ER",
            ]].copy()
            disp.columns = [
                "Rank", "Fund", "Score", "Signal",
                "Roll 5Y%", "Down Cap%", "Alpha Hit%",
                "Roll 3Y%", "Up Cap%", "Max DD%", "Sortino", "AUM (Cr)", "Exp Ratio%",
            ]
            disp["Fund"] = disp["Fund"].apply(short)
            disp = disp.round({
                "Score": 1, "Roll 5Y%": 1, "Down Cap%": 0, "Alpha Hit%": 1,
                "Roll 3Y%": 1, "Up Cap%": 0, "Max DD%": 1, "Sortino": 2, "AUM (Cr)": 0, "Exp Ratio%": 2,
            })

            styled = (disp.style
                .map(c_sig, subset=["Signal"])
                .map(c_score, subset=["Score"])
                .map(c_dc, subset=["Down Cap%"])
                .map(c_hit, subset=["Alpha Hit%"])
                .format(na_rep="—")
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
            )
            st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

            with st.expander("📖 How Established funds are ranked"):
                st.markdown("""
| Factor | Weight | What it measures |
|---|---|---|
| **Rolling 5Y CAGR** | Configurable | Average annualised return across every possible 5-year window. Removes timing luck. |
| **Downside Capture** | Configurable | Fund's loss as % of benchmark loss in down months. Below 80 = good, below 50 = elite. |
| **Alpha Hit Rate (12M)** | Configurable | % of rolling 12-month windows where the fund beat the Nifty 500 TRI. |

Only funds with **3+ years** of track record are ranked here. This ensures the rolling 5Y CAGR has real data and the Alpha Hit Rate has enough 12-month windows to be statistically meaningful.
""")

        else:  # Emerging
            if len(em) == 0:
                st.info("No emerging funds (6 months – 3 years) found in the dataset.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                top = em.sort_values("Rank").iloc[0]
                c1.metric("🥇 Top Emerging", short(top["Fund"]), f"Score {top['Score']:.1f}")
                c2.metric("Median Down Capture", f"{em['Down_Cap'].median():.0f}%")
                c3.metric("Median Batting Avg", f"{em['Batting_Avg'].median():.0f}%")
                c4.metric("Funds Ranked", len(em))

                disp = em.sort_values("Rank")[[
                    "Rank", "Fund", "Score", "Signal", "Track_Yrs", "Months",
                    "Batting_Avg", "Down_Cap", "Alpha_Hit_6M",
                    "Up_Cap", "Ret_6M", "Ret_1Y", "Max_DD", "AUM", "ER",
                ]].copy()
                disp.columns = [
                    "Rank", "Fund", "Score", "Signal", "Age (Yrs)", "Months",
                    "Batting Avg%", "Down Cap%", "Alpha Hit 6M%",
                    "Up Cap%", "6M Ret%", "1Y Ret%", "Max DD%", "AUM (Cr)", "Exp Ratio%",
                ]
                disp["Fund"] = disp["Fund"].apply(short)
                disp = disp.round({
                    "Score": 1, "Age (Yrs)": 1, "Batting Avg%": 1,
                    "Down Cap%": 0, "Alpha Hit 6M%": 1, "Up Cap%": 0,
                    "6M Ret%": 1, "1Y Ret%": 1, "Max DD%": 1, "AUM (Cr)": 0, "Exp Ratio%": 2,
                })

                styled = (disp.style
                    .map(c_sig, subset=["Signal"])
                    .map(c_score, subset=["Score"])
                    .map(c_dc, subset=["Down Cap%"])
                    .map(c_hit, subset=["Batting Avg%", "Alpha Hit 6M%"])
                    .format(na_rep="—")
                    .set_properties(**{"text-align": "center", "font-size": "13px"})
                    .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
                )
                st.dataframe(styled, use_container_width=True, height=400, hide_index=True)

                with st.expander("📖 How Emerging funds are ranked"):
                    st.markdown("""
| Factor | Weight | What it measures |
|---|---|---|
| **Batting Average** | Configurable | % of months where the fund beat the Nifty 500 TRI. Simple, robust, works with few months. |
| **Downside Capture** | Configurable | Same as established — fund's loss as % of benchmark loss. Works with as few as 3 down months. |
| **Alpha Hit Rate (6M)** | Configurable | % of rolling 6-month windows with positive alpha. Shorter window adapts to limited data. |

**Why different factors?** Rolling 5Y CAGR is impossible with < 3 years of data. Rolling 12M alpha needs 24+ months to produce enough windows. Batting Average and 6M Alpha Hit Rate apply the **same principle** (consistency of outperformance) using **shorter measurement windows** that are valid with 6–36 months of data.

**⚠️ Confidence caveat:** Emerging fund scores have less statistical reliability. A fund with 8 months of data and 60% batting average could easily regress to 40% over the next year. Use these rankings as early signals, not final verdicts.
""")

            if len(too_young) > 0:
                with st.expander(f"📎 {len(too_young)} fund(s) too young to rank (< 6 months)"):
                    st.dataframe(
                        too_young[["Fund", "Track_Yrs", "AUM"]].assign(
                            Fund=too_young["Fund"].apply(short)
                        ).round(1).rename(columns={"Track_Yrs": "Age (Yrs)", "AUM": "AUM (Cr)"}),
                        hide_index=True, use_container_width=True,
                    )

        # Download
        all_funds = pd.concat([est, em, too_young], ignore_index=True) if len(em) > 0 else est
        buf = io.BytesIO()
        all_funds.round(2).to_excel(buf, index=False, engine="openpyxl")
        st.download_button("📥 Download all data", buf.getvalue(), "smallcap_3factor.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ═══════════════════════════════════
    # TAB 2: FUND DEEP-DIVE
    # ═══════════════════════════════════
    with tab_deep:
        all_df = pd.concat([est, em, too_young], ignore_index=True) if len(em) > 0 else est
        fund_list = all_df.sort_values("Rank", na_position="last")["Fund"].tolist()
        selected = st.selectbox("Pick a fund", fund_list, format_func=short)

        r = all_df[all_df["Fund"] == selected].iloc[0]
        engine = r.get("Engine", "")

        st.subheader(short(selected))

        # Header
        rank_txt = f"Rank #{int(r['Rank'])}" if pd.notna(r["Rank"]) else "Not ranked"
        engine_label = "🏛️ Established" if engine == "Established" else ("🚀 Emerging" if engine == "Emerging" else "⏳ Too Young")
        st.markdown(f"**{rank_txt}** · {engine_label} · **{r['Signal']}** · Score: **{fmt(r['Score'], 1)}**/100 · Age: {r['Track_Yrs']:.1f} yrs ({int(r['Months'])} months)")
        st.divider()

        # The 3 Factors (context-aware)
        st.markdown("#### The 3 Factors" + (" *(adapted for shorter track record)*" if engine == "Emerging" else ""))

        c1, c2, c3 = st.columns(3)

        with c1:
            if engine == "Established":
                st.metric("Rolling 5Y CAGR", fmt(r["Roll_5Y"], 1, "%"),
                           f"Range: {fmt(r['Roll_5Y_Min'],1)}% to {fmt(r['Roll_5Y_Max'],1)}%" if pd.notna(r.get("Roll_5Y_Min")) else None)
                if pd.notna(r["Roll_5Y"]):
                    cat_med = est["Roll_5Y"].median()
                    diff = r["Roll_5Y"] - cat_med
                    st.caption(f"Category median: {cat_med:.1f}% · You: {'+' if diff > 0 else ''}{diff:.1f}%")
            else:
                ba = r["Batting_Avg"]
                label = "✅ Strong (>60%)" if pd.notna(ba) and ba >= 60 else ("⚠️ Average" if pd.notna(ba) and ba >= 45 else "❌ Below average")
                st.metric("Batting Average", fmt(ba, 1, "%"), label)
                st.caption(f"Beat Nifty 500 TRI in {ba:.0f}% of months" if pd.notna(ba) else "")

        with c2:
            dc = r["Down_Cap"]
            label = "✅ Elite (<50)" if pd.notna(dc) and dc < 50 else (
                "✅ Good (<80)" if pd.notna(dc) and dc < 80 else (
                "⚠️ Okay (<100)" if pd.notna(dc) and dc < 100 else "❌ Amplifies losses"))
            st.metric("Downside Capture", fmt(dc, 0, "%"), label)
            if pd.notna(dc):
                st.caption(f"When Nifty 500 falls 10%, this fund falls ~{dc/10:.1f}%")

        with c3:
            if engine == "Established":
                ah = r["Alpha_Hit_12M"]
                label = "✅ Consistent (>65%)" if pd.notna(ah) and ah >= 65 else (
                    "⚠️ Average (50-65%)" if pd.notna(ah) and ah >= 50 else "❌ Inconsistent")
                st.metric("Alpha Hit Rate (12M)", fmt(ah, 1, "%"), label)
                st.caption("% of rolling 12-month windows with positive alpha")
            else:
                ah6 = r["Alpha_Hit_6M"]
                label = "✅ Promising (>65%)" if pd.notna(ah6) and ah6 >= 65 else (
                    "⚠️ Mixed" if pd.notna(ah6) and ah6 >= 45 else "❌ Weak")
                st.metric("Alpha Hit Rate (6M)", fmt(ah6, 1, "%"), label)
                st.caption("% of rolling 6-month windows with positive alpha")

        st.divider()

        # Supporting
        st.markdown("#### Supporting Context")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("6M Return", fmt(r.get("Ret_6M"), 1, "%"))
        c2.metric("1Y Return", fmt(r.get("Ret_1Y"), 1, "%"))
        c3.metric("Upside Capture", fmt(r.get("Up_Cap"), 0, "%"))
        c4.metric("Max Drawdown", fmt(r["Max_DD"], 1, "%"))
        c5.metric("Sortino", fmt(r.get("Sortino"), 2))

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Drawdown", fmt(r["Current_DD"], 1, "%"))
        c2.metric("AUM", f"₹{r['AUM']:.0f} Cr" if pd.notna(r.get("AUM")) else "—")
        c3.metric("Expense Ratio", fmt(r.get("ER"), 2, "%"))

        if engine == "Emerging":
            st.info("⚠️ **Limited data warning:** This fund has only "
                    f"{int(r['Months'])} months of data. Rankings are directional, not definitive. "
                    "Re-evaluate after 3 years of track record.")

        st.divider()

        # NAV Chart
        st.markdown("#### NAV vs Nifty 500 TRI")
        fn = nav[["Date", selected, "Benchmark"]].dropna(subset=[selected]).copy()
        fn = fn.rename(columns={selected: "NAV"})
        if not fn.empty and fn.iloc[0]["Benchmark"] > 0:
            fn["Bench"] = fn["Benchmark"] * (fn.iloc[0]["NAV"] / fn.iloc[0]["Benchmark"])
        else:
            fn["Bench"] = fn["Benchmark"]
        fn["Peak"] = fn["NAV"].cummax()
        fn["DD"] = (fn["NAV"] - fn["Peak"]) / fn["Peak"] * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.65, 0.35])
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["NAV"], fill="tozeroy",
                                  fillcolor="rgba(59,130,246,0.06)",
                                  line=dict(color="#3b82f6", width=1.5), name="Fund"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["Bench"],
                                  line=dict(color="#94a3b8", width=1.5, dash="dot"), name="Nifty 500 TRI"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["DD"], fill="tozeroy",
                                  fillcolor="rgba(239,68,68,0.08)",
                                  line=dict(color="#ef4444", width=1), name="Drawdown"), row=2, col=1)
        fig.update_layout(height=420, margin=dict(l=50, r=20, t=20, b=30), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

        # Peer Charts
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Capture Map vs Peers")
            cdf = all_df.dropna(subset=["Down_Cap"])
            if "Batting_Avg" in cdf.columns and engine == "Emerging":
                # For emerging: Down Cap vs Batting Avg
                cdf2 = cdf.dropna(subset=["Batting_Avg"])
                fig2 = go.Figure()
                fig2.add_hline(y=50, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
                fig2.add_vline(x=100, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
                fig2.add_trace(go.Scatter(
                    x=cdf2["Down_Cap"], y=cdf2["Batting_Avg"], mode="markers",
                    marker=dict(
                        size=[16 if f == selected else 9 for f in cdf2["Fund"]],
                        color=["#3b82f6" if f == selected else "#d1d5db" for f in cdf2["Fund"]],
                    ),
                    text=[short(f) for f in cdf2["Fund"]],
                    hovertemplate="<b>%{text}</b><br>Down Cap: %{x:.0f}%<br>Batting: %{y:.1f}%<extra></extra>",
                ))
                fig2.update_layout(xaxis_title="Downside Capture %", yaxis_title="Batting Average %",
                                   height=350, margin=dict(l=50, r=20, t=20, b=50), showlegend=False)
            else:
                # For established: Down Cap vs Alpha Hit
                cdf2 = cdf.dropna(subset=["Alpha_Hit_12M"])
                fig2 = go.Figure()
                fig2.add_hline(y=50, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
                fig2.add_vline(x=80, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
                fig2.add_shape(type="rect", x0=0, x1=80, y0=50, y1=100,
                               fillcolor="rgba(34,197,94,0.05)", line=dict(width=0))
                fig2.add_trace(go.Scatter(
                    x=cdf2["Down_Cap"], y=cdf2["Alpha_Hit_12M"], mode="markers",
                    marker=dict(
                        size=[16 if f == selected else 9 for f in cdf2["Fund"]],
                        color=["#3b82f6" if f == selected else "#d1d5db" for f in cdf2["Fund"]],
                    ),
                    text=[short(f) for f in cdf2["Fund"]],
                    hovertemplate="<b>%{text}</b><br>Down Cap: %{x:.0f}%<br>Alpha Hit: %{y:.1f}%<extra></extra>",
                ))
                fig2.update_layout(xaxis_title="Downside Capture % (lower = better)",
                                   yaxis_title="Alpha Hit Rate %",
                                   height=350, margin=dict(l=50, r=20, t=20, b=50), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            if engine == "Established" and pd.notna(r.get("Roll_5Y")):
                st.markdown("#### Rolling 5Y CAGR vs Peers")
                r5df = est.dropna(subset=["Roll_5Y"]).sort_values("Roll_5Y", ascending=True).tail(15)
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    y=[short(f) for f in r5df["Fund"]],
                    x=r5df["Roll_5Y_Max"] - r5df["Roll_5Y_Min"],
                    base=r5df["Roll_5Y_Min"], orientation="h",
                    marker=dict(color=["rgba(59,130,246,0.15)" if f == selected else "rgba(0,0,0,0.04)"
                                       for f in r5df["Fund"]]),
                ))
                fig3.add_trace(go.Scatter(
                    y=[short(f) for f in r5df["Fund"]],
                    x=r5df["Roll_5Y"], mode="markers",
                    marker=dict(color=["#3b82f6" if f == selected else "#d1d5db" for f in r5df["Fund"]],
                                size=10, symbol="diamond"),
                ))
                fig3.update_layout(showlegend=False, height=350, margin=dict(l=50, r=20, t=20, b=30))
                fig3.update_xaxes(title_text="CAGR %")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.markdown("#### Monthly Returns vs Benchmark")
                fn_m = nav[["Date", selected, "Benchmark"]].dropna(subset=[selected]).copy()
                fn_m = fn_m.set_index("Date").resample("ME").last().pct_change().dropna() * 100
                fn_m = fn_m.tail(min(24, len(fn_m)))
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=fn_m.index, y=fn_m[selected], name="Fund",
                                       marker=dict(color="#3b82f6")))
                fig3.add_trace(go.Scatter(x=fn_m.index, y=fn_m["Benchmark"], name="Nifty 500",
                                           line=dict(color="#94a3b8", width=2)))
                fig3.update_layout(showlegend=True, height=350,
                                   margin=dict(l=50, r=20, t=20, b=30),
                                   legend=dict(orientation="h", y=1.05))
                fig3.update_yaxes(title_text="Monthly Return %")
                st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
