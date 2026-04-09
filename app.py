"""
Dual-Engine — SmallCap + LargeCap
Quant Rankings (Established Compounders / Momentum Efficiency)
Fund Sector Flow & Sector Consensus (SmallCap only, when sector data is present)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import warnings, os

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Fund Quants — SC + LC", layout="wide", page_icon="📊")

# ═══════════════════════════════════════════
# FUND QUALITATIVE PROFILES  (SmallCap)
# ═══════════════════════════════════════════
PROVEN = {
    "Bandhan Small Cap Fund-Reg(G)": {"PE": 16.78, "PB": 2.07, "Stance": "Deep Value",
        "Label": "Deep Value — Buy & Hold",
        "Detail": "PE 16.8x / PB 2.07x — lowest in the group. Finance-heavy at value prices. Minimal sector rotation over 12 months. Conviction-based, low-churn approach. Buys cheap, sits and waits."},
    "Bank of India Small Cap Fund-Reg(G)": {"PE": 32.82, "PB": 4.02, "Stance": "Balanced Momentum",
        "Label": "Cyclical Rotation — Defensive to Offensive",
        "Detail": "Biggest single move: +8pp into banking while cutting healthcare -4pp and finance -4pp. Classic cyclical rotation from defensive to rate-sensitive sectors."},
    "Edelweiss Small Cap Fund-Reg(G)": {"PE": 31.21, "PB": 4.13, "Stance": "Risk-Adjusted Growth",
        "Label": "Diversified All-Weather Portfolio",
        "Detail": "Most balanced allocation — no sector above 13%. Added to banks, autos, chemicals while trimming capital goods. Lowest concentration risk in the group."},
    "Canara Rob Small Cap Fund-Reg(G)": {"PE": 28.33, "PB": 3.64, "Stance": "Quality / Hedged",
        "Label": "Financial Barbell + Defensive Hedge",
        "Detail": "Finance + Banking = 24.3%, highest combined financial exposure. FMCG at 6.1% as defensive ballast. Barbell: high-beta financials for upside, staples for downside protection."},
    "Invesco India Smallcap Fund-Reg(G)": {"PE": 39.47, "PB": 4.24, "Stance": "Ultra-Growth Premium",
        "Label": "Structural Growth — Healthcare + Consumer",
        "Detail": "Healthcare at 21.8% — highest single-sector bet of ANY fund (+6.6pp). Retailing 9.4%. Banking +8pp. Betting on India's structural consumption story. Highest PE (39.5x)."},
    "Mahindra Manulife Small Cap Fund-Reg(G)": {"PE": 29.50, "PB": 3.85, "Stance": "Diversified Growth",
        "Label": "Diversified Multi-Sector",
        "Detail": "Healthcare 14.8%, autos 8.1%, finance 7.9%, iron & steel 6.9%, banking 7%. Broad-based exposure across manufacturing, healthcare, and financials with no extreme concentration."},
}

MOMENTUM_INFO = {
    "TRUSTMF Small Cap Fund-Reg(G)": {"PE": 37.31, "PB": 4.87, "Stance": "Terminal Value Premium",
        "Label": "Concentrated Cyclical — Capex + Credit",
        "Detail": "Capital goods 12.2% + banking 11.6% (+7.7pp). Healthcare trimmed -4pp. Bet on capex cycle + credit growth."},
    "Union Small Cap Fund-Reg(G)": {"PE": 39.27, "PB": 5.24, "Stance": "High-Growth Momentum",
        "Label": "Capex Cycle Maximalist",
        "Detail": "Capital goods at 18.7% — highest of any fund, +7.1pp. Banking +7.2pp, autos +4pp. All-in on India's capex cycle."},
    "DSP Small Cap Fund-Reg(G)": {"PE": 28.39, "PB": 3.52, "Stance": "Concentrated Core",
        "Label": "Domestic Manufacturing Contrarian",
        "Detail": "Autos 16.8% (highest, +5.4pp), chemicals 11.2%, FMCG 10.5%. AVOIDS banking — only fund with no significant banking weight."},
    "Aditya Birla SL Small Cap Fund(G)": {"PE": 27.71, "PB": 4.10, "Stance": "Operating Leverage",
        "Label": "Rate Cut Beneficiary",
        "Detail": "Added +7pp banking and +6pp healthcare while cutting capital goods -5pp. Positioning for operating leverage from lower rates."},
    "Mirae Asset Small Cap Fund-Reg(G)": {"PE": 24.80, "PB": 3.92, "Stance": "GARP",
        "Label": "GARP — Broad Based Builder",
        "Detail": "Added to 6+ sectors simultaneously. Broadest build-up. Lowest PE (24.8x) among momentum funds. Buying growth without overpaying."},
    "Invesco India Smallcap Fund-Reg(G)": {"PE": 39.47, "PB": 4.24, "Stance": "Ultra-Growth Premium",
        "Label": "Structural Growth — Healthcare + Consumer",
        "Detail": "Healthcare 21.8%, Retailing 9.4%, Banking +8pp. Structural consumption conviction."},
    "Mahindra Manulife Small Cap Fund-Reg(G)": {"PE": 29.50, "PB": 3.85, "Stance": "Diversified Growth",
        "Label": "Diversified Multi-Sector",
        "Detail": "Healthcare 14.8%, autos 8.1%, iron & steel 6.9%. Balanced across manufacturing and services."},
}

ALL_INFO = {**PROVEN, **MOMENTUM_INFO}
ALL_QUAL_FUNDS = list(dict.fromkeys(list(PROVEN.keys()) + list(MOMENTUM_INFO.keys())))

SECTOR_READINGS = {
    "Bank": "Strongest crowd trade. Nearly EVERY fund added 5-8pp. Unanimous positioning for rate cuts + credit growth.",
    "Healthcare": "High average but split. Invesco betting huge (21.8%), others steady or trimming. Valuation debate.",
    "Finance": "Core structural holding. Stable across all funds — treated as base allocation, not tactical.",
    "Capital Goods": "Diverging. Union (18.7%) and TRUSTMF heavy; Invesco/Aditya Birla exiting. Capex conviction split.",
    "Automobile & Ancillaries": "Growing consensus on domestic auto + EV supply chain. DSP heaviest (16.8%).",
    "Chemicals": "Selective China+1 additions. DSP and Edelweiss building; others stable.",
    "FMCG": "Defensive hedge. Held by DSP (10.5%) and Canara Rob. Absent from aggressive momentum funds.",
    "Realty": "Housing cycle play. Concentrated in Bandhan and Invesco. Not universal.",
    "Construction Materials": "Small steady positions. No strong conviction either direction.",
    "IT": "Small positions. Selective small-cap IT services exposure.",
    "Retailing": "Invesco-only conviction bet (9.4%). Not consensus.",
    "Consumer Durables": "Small but growing. Several funds adding 2-3pp.",
    "Business Services": "Diverging. Mirae adding (+5pp), others trimming.",
    "Iron & Steel": "Small value positions. Commodity view split.",
    "Textile": "Tiny, fading positions. Not a live theme.",
    "Electricals": "Niche positions. Manufacturing/capex adjacent.",
}

MONTHS = ["Jan_25", "Jun_25", "Sep_25", "Dec_25", "Feb_26"]
MONTH_LABELS = ["Jan 2025", "Jun 2025", "Sep 2025", "Dec 2025", "Feb 2026"]

# ═══════════════════════════════════════════
# LARGECAP — FUNDS FOR SECTOR ANALYSIS
# (from Established Compounder + Momentum rankings)
# ═══════════════════════════════════════════
LC_QUAL_FUNDS = [
    "ICICI Pru Large Cap Fund(G)",
    "Mahindra Manulife Large Cap Fund-Reg(G)",
    "Canara Rob Large Cap Fund-Reg(G)",
    "Mirae Asset Large Cap Fund-Reg(G)",
    "Edelweiss Large Cap Fund-Reg(G)",
    "WOC Large Cap Fund-Reg(G)",
    "Bank of India Large Cap Fund-Reg(G)",
    "Groww Largecap Fund-Reg(G)",
    "Invesco India Largecap Fund-Reg(G)",
    "Bandhan Large Cap Fund-Reg(G)",
    "Nippon India Large Cap Fund(G)",
    "Taurus Large Cap Fund-Reg(G)",
    "SBI Large Cap Fund-Reg(G)",
    "Bajaj Finserv Large Cap Fund-Reg(G)",
    "HSBC Large Cap Fund(G)",
]

LC_MONTHS = ["Apr_25", "May_25", "Jun_25", "Jul_25", "Aug_25", "Sep_25",
             "Oct_25", "Nov_25", "Dec_25", "Jan_26", "Feb_26", "Mar_26"]
LC_MONTH_LABELS = ["Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025", "Sep 2025",
                   "Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026"]


def short_sc(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
              .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


def short_lc(f):
    return (f.replace("Large Cap", "LC").replace("Largecap", "LC").replace("Large cap", "LC")
              .replace("Fund-Reg(G)", "").replace("Fund(G)", "").replace("Fund-Reg(IDCW)", "")
              .replace("Fund(IDCW)", "").strip())


# ═══════════════════════════════════════════
# DATA LOADING — SMALLCAP
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading SmallCap Data...")
def load_sc_data():
    raw = pd.read_excel("smallcapfinalrank.xlsx")
    fund_names = raw.iloc[1, 1:].dropna().tolist()
    nav = raw.iloc[3:, :len(fund_names)+1].copy()
    nav.columns = ["Date"] + fund_names
    nav = nav[pd.to_datetime(nav["Date"], errors="coerce").notna()].copy()
    nav["Date"] = pd.to_datetime(nav["Date"])
    nav = nav.sort_values("Date").reset_index(drop=True)
    for f in fund_names:
        nav[f] = pd.to_numeric(nav[f], errors="coerce")

    try:
        bench = pd.read_csv("Nifty_500_TRI_Combined.csv")
        bench["Date"] = pd.to_datetime(bench["Date"])
        nav = pd.merge(nav, bench[["Date", "Benchmark"]], on="Date", how="left")
        nav["Benchmark"] = nav["Benchmark"].ffill()
        bench_name = "Nifty 500 TRI"
    except Exception:
        try:
            bench = pd.read_csv("BSE SMAALCAP.xlsx - Abakkus Small Cap Fund-Reg(G)1.csv", skiprows=2)
            bench["Date"] = pd.to_datetime(bench["Date"])
            bench = bench.rename(columns={"Close Price": "Benchmark"})
            nav = pd.merge(nav, bench[["Date", "Benchmark"]], on="Date", how="left")
            nav["Benchmark"] = nav["Benchmark"].ffill()
            bench_name = "BSE Small-Cap TRI"
        except Exception:
            valid = [f for f in fund_names if nav[f].notna().sum() > 252]
            nav["Benchmark"] = nav[valid].pct_change().mean(axis=1).add(1).cumprod() * 100
            bench_name = "SmallCap Index (Proxy)"

    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund", "AUM"]]

    return nav, fund_names, aum_latest, bench_name


@st.cache_data(show_spinner="Loading SmallCap Sector Data...")
def load_sc_sectors():
    df_raw = pd.read_excel("SECTORALLCOATIONSMALLCAP.xlsx")
    d = df_raw.iloc[2:].copy()
    d.columns = ["Fund", "Sector", "Feb_26", "Dec_25", "Sep_25", "Jun_25", "Jan_25",
                  "c7", "c8", "c9", "c10", "c11"]
    d = d[["Fund", "Sector", "Feb_26", "Dec_25", "Sep_25", "Jun_25", "Jan_25"]]
    d = d.dropna(subset=["Fund", "Sector"])
    d = d[~d["Fund"].str.contains("Accord", na=False)]
    d = d[d["Sector"] != "Sector"]
    for c in MONTHS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ═══════════════════════════════════════════
# DATA LOADING — LARGECAP
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading LargeCap Data...")
def load_lc_data():
    r1 = pd.read_excel("largecap1.xlsx", header=None)
    r2 = pd.read_excel("largecap2.xlsx", header=None)

    funds1 = r1.iloc[2, 1:].dropna().tolist()
    nav1 = r1.iloc[4:, :len(funds1)+1].copy()
    nav1.columns = ["Date"] + funds1
    nav1 = nav1[pd.to_datetime(nav1["Date"], errors="coerce").notna()].copy()
    nav1["Date"] = pd.to_datetime(nav1["Date"])

    funds2 = r2.iloc[2, 1:].dropna().tolist()
    nav2 = r2.iloc[4:, :len(funds2)+1].copy()
    nav2.columns = ["Date"] + funds2
    nav2 = nav2[pd.to_datetime(nav2["Date"], errors="coerce").notna()].copy()
    nav2["Date"] = pd.to_datetime(nav2["Date"])

    nav = pd.merge(nav1, nav2, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    fund_names = funds1 + funds2
    for f in fund_names:
        nav[f] = pd.to_numeric(nav[f], errors="coerce")

    exclude_kw = ["Long-Short", "Long Short", "DynaSIF", "Qsif"]
    fund_names = [f for f in fund_names if not any(kw.lower() in f.lower() for kw in exclude_kw)]

    valid = [f for f in fund_names if nav[f].notna().sum() > 252]
    nav["Benchmark"] = nav[valid].pct_change().mean(axis=1).add(1).cumprod() * 100
    bench_name = "LargeCap Index (Proxy)"

    return nav, fund_names, bench_name


@st.cache_data(show_spinner="Loading LargeCap Sector Data...")
def load_lc_sectors():
    df_raw = pd.read_excel("sectorflows.xlsx", header=None)
    # Row 3 has headers: Scheme Name, Sector, then 12 monthly dates (Mar-26 → Apr-25)
    dates_raw = df_raw.iloc[3, 2:14].tolist()  # 12 date columns
    d = df_raw.iloc[4:, :14].copy()
    col_names = ["Fund", "Sector"] + LC_MONTHS[::-1][:len(dates_raw)]
    # Dates are Mar-26, Feb-26, Jan-26, Dec-25, Nov-25, Oct-25, Sep-25, Aug-25, Jul-25, Jun-25, May-25, Apr-25
    d.columns = ["Fund", "Sector",
                 "Mar_26", "Feb_26", "Jan_26", "Dec_25", "Nov_25", "Oct_25",
                 "Sep_25", "Aug_25", "Jul_25", "Jun_25", "May_25", "Apr_25"]
    d = d.dropna(subset=["Fund", "Sector"])
    d = d[~d["Fund"].str.contains("Accord", na=False)]
    d = d[d["Sector"] != "Sector"]
    # Filter to only the qualified funds
    d = d[d["Fund"].isin(LC_QUAL_FUNDS)]
    for c in ["Mar_26", "Feb_26", "Jan_26", "Dec_25", "Nov_25", "Oct_25",
              "Sep_25", "Aug_25", "Jul_25", "Jun_25", "May_25", "Apr_25"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


@st.cache_data(show_spinner="Loading PE/PBV Data...")
def load_lc_pe():
    """Parse stacked multi-fund PE file → tidy DataFrame with harmonic mean PE & PBV."""
    df = pd.read_excel("pe.xlsx", header=None)
    records, cur = [], None
    for _, row in df.iterrows():
        val = str(row[0])
        if "Scheme Name:" in val:
            cur = val.replace("Scheme Name:", "").strip()
        elif cur:
            try:
                dt = pd.to_datetime(row[0])
                records.append({"Fund": cur, "Date": dt,
                    "PE_HM": pd.to_numeric(row[2], errors="coerce"),
                    "PBV_HM": pd.to_numeric(row[4], errors="coerce"),
                    "DivYield": pd.to_numeric(row[5], errors="coerce"),
                    "MCAP_Cr": pd.to_numeric(row[6], errors="coerce")})
            except Exception:
                pass
    return pd.DataFrame(records)


@st.cache_data(show_spinner="Loading Turnover Data...")
def load_lc_turnover():
    """Parse stacked multi-fund portfolio ratios → tidy DataFrame with turnover."""
    df = pd.read_excel("portfolio_ratios.xlsx", header=None)
    records, cur = [], None
    for _, row in df.iterrows():
        val = str(row[0])
        if "Scheme Name:" in val:
            cur = val.replace("Scheme Name:", "").strip()
        elif cur:
            try:
                dt = pd.to_datetime(row[0])
                records.append({"Fund": cur, "Date": dt,
                    "Turnover": pd.to_numeric(row[1], errors="coerce")})
            except Exception:
                pass
    return pd.DataFrame(records)


@st.cache_data(show_spinner="Loading Stock Allocations...")
def load_lc_stocks():
    """Parse stock/issuer allocation file → tidy DataFrame."""
    df = pd.read_excel("stockalloacations.xlsx", header=None)
    d = df.iloc[4:, :9].copy()
    d.columns = ["Fund", "Company", "Asset", "Sector",
                  "Mar_26", "Feb_26", "Aug_25", "Jan_25", "Jul_24"]
    d = d.dropna(subset=["Fund", "Company"])
    d = d[d["Fund"].isin(LC_QUAL_FUNDS)]
    for c in ["Mar_26", "Feb_26", "Aug_25", "Jan_25", "Jul_24"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d
def compute_all(nav_in, fund_names):
    nav = nav_in.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    br = mret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 120:
            continue

        fr = mret[fund].dropna() if fund in mret.columns else pd.Series(dtype=float)
        ci = fr.index.intersection(br.dropna().index)
        if len(ci) < 6:
            continue
        fr, brc = fr.loc[ci], br.loc[ci]

        mn = fd.set_index("Date")[fund].resample("ME").last().dropna()
        cagrs_5y = [((mn.iloc[i] / mn.iloc[i-60])**(1/5)-1)*100 for i in range(60, len(mn)) if mn.iloc[i-60] > 0]
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        cagrs_3y = [((mn.iloc[i] / mn.iloc[i-36])**(1/3)-1)*100 for i in range(36, len(mn)) if mn.iloc[i-36] > 0]
        r3_mean = np.mean(cagrs_3y) if cagrs_3y else None

        up_m = brc[brc > 0]
        dn_m = brc[brc < 0]
        up_cap = (fr.loc[up_m.index].mean() / up_m.mean()) * 100 if len(up_m) > 3 else None
        down_cap = (fr.loc[dn_m.index].mean() / dn_m.mean()) * 100 if len(dn_m) > 3 else None

        prices = fd[fund].values
        peaks = np.maximum.accumulate(prices)
        dd_series = (prices - peaks) / peaks
        ulcer = np.sqrt(np.mean(dd_series**2)) * 100
        max_dd = dd_series.min() * 100

        ld = fd.iloc[-1]["Date"]
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None

        def get_ret(days):
            past = fd[fd["Date"] <= ld - timedelta(days=days)]
            if past.empty:
                return None
            return (prices[-1] / past.iloc[-1][fund] - 1) * 100

        ret_6m = get_ret(180)
        ret_1y = get_ret(365)
        mom_6m = ret_6m / np.sqrt(vol) if (vol and vol > 0 and ret_6m is not None) else None
        mom_1y = ret_1y / np.sqrt(vol) if (vol and vol > 0 and ret_1y is not None) else None

        results.append({
            "Fund": fund, "Track_Yrs": len(fd) / 252,
            "Roll_3Y": r3_mean, "Roll_5Y": r5_mean, "Up_Cap": up_cap, "Down_Cap": down_cap,
            "Cap_Ratio": (up_cap / down_cap if (up_cap and down_cap and down_cap != 0) else None),
            "Ulcer_Index": ulcer, "Max_DD": max_dd,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol": vol,
            "Mom_6M_RA": mom_6m, "Mom_1Y_RA": mom_1y,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════
# SHARED RANKING ENGINE
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    if not v.any():
        return s
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def rank_funds(df, w_est, w_mom):
    est = df[df["Track_Yrs"] >= 3].copy()
    if not est.empty:
        est["S_R3"] = pctrank(est["Roll_3Y"])
        est["S_R5"] = pctrank(est["Roll_5Y"])
        est["S_UC"] = pctrank(est["Up_Cap"])
        est["S_DC"] = pctrank(est["Down_Cap"], False)
        est["S_UI"] = pctrank(est["Ulcer_Index"], False)
        est["Score"] = (est["S_R3"].fillna(0)*w_est[0] + est["S_R5"].fillna(0)*w_est[1] +
                        est["S_UC"].fillna(0)*w_est[2] + est["S_DC"].fillna(0)*w_est[3] +
                        est["S_UI"].fillna(0)*w_est[4]) / sum(w_est)
        est["Rank"] = est["Score"].rank(ascending=False, method="min")

    mom = df[df["Mom_6M_RA"].notna()].copy()
    if not mom.empty:
        mom["S_M6"] = pctrank(mom["Mom_6M_RA"])
        mom["S_M1"] = pctrank(mom["Mom_1Y_RA"])
        mom["Score"] = (mom["S_M6"].fillna(0)*w_mom[0] + mom["S_M1"].fillna(0)*w_mom[1]) / sum(w_mom)
        mom["Rank"] = mom["Score"].rank(ascending=False, method="min")

    return est, mom


# ═══════════════════════════════════════════
# TABLE STYLING HELPERS
# ═══════════════════════════════════════════
def c_dc(val):
    if pd.isna(val): return ""
    if val < 50: return "color:#16a34a;font-weight:700;"
    if val < 80: return "color:#16a34a;"
    if val < 100: return "color:#ca8a04;"
    return "color:#dc2626;font-weight:600;"

def c_uc(val):
    if pd.isna(val): return ""
    if val > 120: return "color:#16a34a;font-weight:700;"
    if val > 100: return "color:#16a34a;"
    return "color:#ca8a04;"

def c_cr(val):
    if pd.isna(val): return ""
    if val > 1.3: return "background-color:#dcfce7;color:#166534;font-weight:700;"
    if val > 1.1: return "background-color:#e0f2fe;color:#075985;"
    if val > 1.0: return "color:#ca8a04;"
    return "background-color:#fecaca;color:#991b1b;"

def c_pe(val):
    if pd.isna(val): return ""
    if val < 25: return "background-color:#dcfce7;color:#166534;font-weight:600;"
    if val < 32: return "background-color:#fef9c3;color:#854d0e;"
    return "background-color:#fecaca;color:#991b1b;font-weight:600;"

def c_pb(val):
    if pd.isna(val): return ""
    if val < 3: return "background-color:#dcfce7;color:#166534;font-weight:600;"
    if val < 4.2: return "background-color:#fef9c3;color:#854d0e;"
    return "background-color:#fecaca;color:#991b1b;"

def c_alloc(val):
    if pd.isna(val): return ""
    if val >= 15: return "background-color:#2563eb;color:white;font-weight:700;"
    if val >= 10: return "background-color:#60a5fa;color:white;"
    if val >= 7: return "background-color:#93c5fd;"
    if val >= 4: return "background-color:#bfdbfe;"
    if val >= 2: return "background-color:#dbeafe;"
    return ""

def c_trend(val):
    if "↑↑" in str(val): return "background-color:#166534;color:white;font-weight:700;"
    if "↑" in str(val): return "background-color:#dcfce7;color:#166534;"
    if "↓↓" in str(val): return "background-color:#991b1b;color:white;font-weight:700;"
    if "↓" in str(val): return "background-color:#fecaca;color:#991b1b;"
    return "color:#9ca3af;"


# ═══════════════════════════════════════════
# RENDER QUANT RANKINGS (shared for SC/LC)
# ═══════════════════════════════════════════
def render_quant(est, mom, shortener, has_aum=False):
    view = st.radio("Ranking Engine", [
        "🏛️ Established Compounders (Funds > 3 Yrs)",
        "🏎️ Momentum Efficiency"
    ], horizontal=True, key=f"engine_{shortener.__name__}")

    target = est if "Established" in view else mom

    if target.empty:
        st.warning("No funds qualify for this engine.")
        return

    disp = target.sort_values("Rank").copy()

    if "Established" in view:
        cols = ["Rank", "Fund", "Score", "Roll_3Y", "Roll_5Y", "Up_Cap", "Down_Cap", "Cap_Ratio", "Ulcer_Index", "Max_DD"]
        names = ["Rank", "Fund", "Score", "3Y CAGR", "5Y CAGR", "Up Cap%", "Down Cap%", "Up/Down", "Ulcer Index", "Max DD%"]
        if has_aum and "AUM" in disp.columns:
            cols.append("AUM"); names.append("AUM (Cr)")
        # Add PE/PBV/Turnover if available
        for src, dst in [("PE_HM", "PE (HM)"), ("PBV_HM", "PBV (HM)"), ("Turnover", "Turnover%")]:
            if src in disp.columns:
                cols.append(src); names.append(dst)
    else:
        cols = ["Rank", "Fund", "Score", "Ret_6M", "Ret_1Y", "Vol", "Mom_6M_RA", "Mom_1Y_RA", "Up_Cap", "Down_Cap", "Cap_Ratio"]
        names = ["Rank", "Fund", "Score", "6M Ret%", "1Y Ret%", "Vol%", "6M RA", "1Y RA", "Up Cap%", "Down Cap%", "Up/Down"]
        if has_aum and "AUM" in disp.columns:
            cols.append("AUM"); names.append("AUM (Cr)")
        for src, dst in [("PE_HM", "PE (HM)"), ("Turnover", "Turnover%")]:
            if src in disp.columns:
                cols.append(src); names.append(dst)

    avail = [c for c in cols if c in disp.columns]
    avail_names = [names[cols.index(c)] for c in avail]
    disp = disp[avail].copy()
    disp.columns = avail_names
    disp["Fund"] = disp["Fund"].apply(shortener)

    num_cols = [c for c in avail_names if c not in ("Rank", "Fund", "AUM (Cr)", "Up/Down", "PE (HM)", "PBV (HM)", "Turnover%")]

    fmt = {c: "{:.1f}" for c in num_cols}
    fmt["Rank"] = "{:.0f}"
    fmt["Up/Down"] = "{:.2f}"
    if "AUM (Cr)" in avail_names:
        fmt["AUM (Cr)"] = "{:.0f}"
    if "PE (HM)" in avail_names:
        fmt["PE (HM)"] = "{:.1f}x"
    if "PBV (HM)" in avail_names:
        fmt["PBV (HM)"] = "{:.2f}x"
    if "Turnover%" in avail_names:
        fmt["Turnover%"] = "{:.0f}"

    styled = disp.style.background_gradient(subset=["Score"], cmap="RdYlGn")
    if "Down Cap%" in avail_names:
        styled = styled.map(c_dc, subset=["Down Cap%"])
    if "Up Cap%" in avail_names:
        styled = styled.map(c_uc, subset=["Up Cap%"])
    if "Up/Down" in avail_names:
        styled = styled.map(c_cr, subset=["Up/Down"])
    if "PE (HM)" in avail_names:
        styled = styled.map(c_pe, subset=["PE (HM)"])
    if "PBV (HM)" in avail_names:
        styled = styled.map(c_pb, subset=["PBV (HM)"])
    styled = styled.format(fmt).format(na_rep="—")
    st.dataframe(styled, use_container_width=True, height=700, hide_index=True)


# ═══════════════════════════════════════════
# RENDER SMALLCAP SECTOR TABS
# ═══════════════════════════════════════════
def render_sc_sector_flow(sector_data):
    st.markdown("## Fund Strategy Profiles & Sector Flow")

    # ── Profile table ──
    st.markdown("#### All Fund Profiles at a Glance")
    profile_rows = []
    for fund in ALL_QUAL_FUNDS:
        info_p = ALL_INFO.get(fund, {})
        profile_rows.append({
            "Fund": short_sc(fund), "PE": info_p.get("PE"), "PB": info_p.get("PB"),
            "Valuation Stance": info_p.get("Stance", ""),
        })
    pdf = pd.DataFrame(profile_rows)
    styled_p = (pdf.style.map(c_pe, subset=["PE"]).map(c_pb, subset=["PB"])
        .format({"PE": "{:.1f}x", "PB": "{:.2f}x"}, na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "600"}))
    st.dataframe(styled_p, use_container_width=True, height=420, hide_index=True)
    st.caption("PE color: 🟢 Value (<25x) · 🟡 Growth (25-32x) · 🔴 Premium (>32x)  |  PB color: 🟢 <3x · 🟡 3-4.2x · 🔴 >4.2x")

    with st.expander("ℹ️ What do the Valuation Stances mean?"):
        st.markdown("""
| Stance | Meaning |
|---|---|
| **Deep Value** | Buying stocks at significant discount to intrinsic value. Lowest PE/PB. |
| **GARP** | Growth At Reasonable Price — balances earnings momentum with valuation discipline. |
| **Operating Leverage** | Targeting companies where small revenue increases lead to outsized profit jumps. |
| **Concentrated Core** | High-conviction in 2-3 sector themes. |
| **Quality / Hedged** | High-quality growth + defensive positions for downside cushion. |
| **Risk-Adjusted Growth** | Growth-oriented with strict risk controls, wide diversification. |
| **Balanced Momentum** | Rotates between defensive and cyclical sectors by market regime. |
| **Terminal Value Premium** | Pays premium for long-duration growth. |
| **High-Growth Momentum** | Chasing strongest recent performers. Highest churn. |
| **Ultra-Growth Premium** | Highest conviction growth bets at expensive valuations. |
| **Diversified Growth** | Broad exposure, no extreme bets, moderate PE/PB. |
""")

    st.divider()

    # ── Fund dropdown ──
    st.markdown("#### Sector Flow — Pick a Fund")
    selected = st.selectbox("Select Fund", ALL_QUAL_FUNDS, format_func=short_sc)
    info = ALL_INFO.get(selected, {})
    fd = sector_data[sector_data["Fund"] == selected].copy()

    st.markdown(f"### {short_sc(selected)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("PE Ratio", f"{info.get('PE', '—')}x")
    c2.metric("PB Ratio", f"{info.get('PB', '—')}x")
    c3.metric("Stance", info.get("Stance", "—"))

    fd_sig = fd[fd[MONTHS].max(axis=1) >= 1.5].copy()
    fd_sig = fd_sig.sort_values("Feb_26", ascending=False)

    if len(fd_sig) == 0:
        st.warning("No sector data available.")
        return

    st.markdown("#### Month-by-month breakdown")
    table_rows = []
    for _, row in fd_sig.iterrows():
        jan, feb = row["Jan_25"], row["Feb_26"]
        if pd.notna(jan) and pd.notna(feb):
            diff = feb - jan
            if diff > 3: trend = f"↑↑ +{diff:.1f}pp"
            elif diff > 1: trend = f"↑ +{diff:.1f}pp"
            elif diff < -3: trend = f"↓↓ {diff:.1f}pp"
            elif diff < -1: trend = f"↓ {diff:.1f}pp"
            else: trend = f"→ {diff:+.1f}pp"
        elif pd.notna(feb):
            trend = "New"
        else:
            trend = "—"
        table_rows.append({
            "Sector": row["Sector"],
            "Jan 2025": round(jan, 1) if pd.notna(jan) else None,
            "Jun 2025": round(row["Jun_25"], 1) if pd.notna(row["Jun_25"]) else None,
            "Sep 2025": round(row["Sep_25"], 1) if pd.notna(row["Sep_25"]) else None,
            "Dec 2025": round(row["Dec_25"], 1) if pd.notna(row["Dec_25"]) else None,
            "Feb 2026": round(feb, 1) if pd.notna(feb) else None,
            "12M Trend": trend,
        })
    tdf = pd.DataFrame(table_rows)
    alloc_cols = ["Jan 2025", "Jun 2025", "Sep 2025", "Dec 2025", "Feb 2026"]
    styled = (tdf.style.map(c_alloc, subset=alloc_cols).map(c_trend, subset=["12M Trend"])
        .format(na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Sector"], **{"text-align": "left", "font-weight": "600"})
        .set_properties(subset=["12M Trend"], **{"font-weight": "600"}))
    st.dataframe(styled, use_container_width=True, height=500, hide_index=True)

    st.markdown("#### Biggest moves (Jan 2025 → Feb 2026)")
    moves = []
    for _, row in fd_sig.iterrows():
        if pd.notna(row["Jan_25"]) and pd.notna(row["Feb_26"]):
            diff = row["Feb_26"] - row["Jan_25"]
            if abs(diff) > 1.5:
                moves.append((row["Sector"], diff, row["Feb_26"]))
    moves.sort(key=lambda x: -abs(x[1]))
    if moves:
        cols = st.columns(min(len(moves), 5))
        for i, (sector, diff, current) in enumerate(moves[:5]):
            with cols[i]:
                st.metric(sector, f"{current:.1f}%", f"{'+' if diff > 0 else ''}{diff:.1f}pp")
    else:
        st.caption("No major sector shifts (>1.5pp) detected.")


def render_sc_consensus(sector_data):
    st.markdown("## Sector Consensus")
    st.markdown("Where are all 10 funds converging and diverging?")

    top_sectors = (sector_data[sector_data["Fund"].isin(ALL_QUAL_FUNDS)]
                    .groupby("Sector")["Feb_26"].mean()
                    .sort_values(ascending=False).head(15).index.tolist())

    cons = []
    for sector in top_sectors:
        allocs, changes = [], []
        cnt = 0
        for fund in ALL_QUAL_FUNDS:
            fd = sector_data[(sector_data["Fund"] == fund) & (sector_data["Sector"] == sector)]
            if len(fd) > 0 and pd.notna(fd.iloc[0]["Feb_26"]):
                allocs.append(fd.iloc[0]["Feb_26"]); cnt += 1
                if pd.notna(fd.iloc[0]["Jan_25"]):
                    changes.append(fd.iloc[0]["Feb_26"] - fd.iloc[0]["Jan_25"])
        avg_a = np.mean(allocs) if allocs else 0
        avg_c = np.mean(changes) if changes else 0
        if avg_c > 2: direction = "Strong Addition ↑↑"
        elif avg_c > 0.5: direction = "Adding ↑"
        elif avg_c < -2: direction = "Strong Reduction ↓↓"
        elif avg_c < -0.5: direction = "Trimming ↓"
        else: direction = "Stable →"
        cons.append({"Sector": sector, "Avg Alloc%": round(avg_a, 1), "Funds": cnt,
                      "Avg 12M Change": round(avg_c, 1), "Direction": direction})

    cdf = pd.DataFrame(cons)
    cdf_s = cdf.sort_values("Avg Alloc%", ascending=True)
    bar_colors = []
    for _, r in cdf_s.iterrows():
        ch = r["Avg 12M Change"]
        if ch > 2: bar_colors.append("#22c55e")
        elif ch > 0.5: bar_colors.append("#86efac")
        elif ch < -2: bar_colors.append("#ef4444")
        elif ch < -0.5: bar_colors.append("#fca5a5")
        else: bar_colors.append("#94a3b8")

    fig = go.Figure()
    fig.add_trace(go.Bar(y=cdf_s["Sector"], x=cdf_s["Avg Alloc%"], orientation="h",
        marker=dict(color=bar_colors),
        text=[f"{v:.1f}%" for v in cdf_s["Avg Alloc%"]], textposition="outside"))
    fig.update_layout(height=420, margin=dict(l=10, r=40, t=10, b=30),
                      xaxis_title="Avg allocation %", showlegend=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Funds adding · 🔴 Funds trimming · ⚪ Stable")

    # Cross-fund heatmap
    st.markdown("#### Cross-fund allocation heatmap — Feb 2026")
    heat_sectors = top_sectors[:12]
    heat_data = []
    for sector in heat_sectors:
        row_d = {"Sector": sector}
        for fund in ALL_QUAL_FUNDS:
            fd = sector_data[(sector_data["Fund"] == fund) & (sector_data["Sector"] == sector)]
            val = fd.iloc[0]["Feb_26"] if len(fd) > 0 and pd.notna(fd.iloc[0]["Feb_26"]) else 0
            row_d[short_sc(fund)] = round(val, 1)
        heat_data.append(row_d)

    hdf = pd.DataFrame(heat_data)
    f_short = [short_sc(f) for f in ALL_QUAL_FUNDS]
    z = hdf[f_short].values

    fig2 = go.Figure(data=go.Heatmap(
        z=z, x=f_short, y=hdf["Sector"],
        colorscale=[[0, "#f8fafc"], [0.2, "#dbeafe"], [0.4, "#93c5fd"],
                    [0.6, "#3b82f6"], [0.8, "#1d4ed8"], [1.0, "#1e3a5f"]],
        text=z, texttemplate="%{text:.1f}", textfont=dict(size=11),
        colorbar=dict(title="Alloc %", thickness=15)))
    fig2.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════
# RENDER LARGECAP SECTOR FLOW
# ═══════════════════════════════════════════
def render_lc_sector_flow(sector_data):
    st.markdown("## LargeCap — Fund Sector Flow")
    st.markdown("Month-by-month sector allocation for top-ranked large-cap funds.")

    selected = st.selectbox("Select Fund", LC_QUAL_FUNDS, format_func=short_lc, key="lc_fund_select")
    fd = sector_data[sector_data["Fund"] == selected].copy()

    st.markdown(f"### {short_lc(selected)}")

    # Use the latest 5 months that have data for display
    display_months = ["Apr_25", "Jun_25", "Sep_25", "Dec_25", "Mar_26"]
    display_labels = ["Apr 2025", "Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026"]
    # Check which exist with data
    avail = [m for m in display_months if m in fd.columns and fd[m].notna().any()]
    if not avail:
        # fallback: use all available
        all_m = ["Apr_25", "May_25", "Jun_25", "Jul_25", "Aug_25", "Sep_25",
                 "Oct_25", "Nov_25", "Dec_25", "Jan_26", "Feb_26", "Mar_26"]
        avail = [m for m in all_m if m in fd.columns and fd[m].notna().any()]

    if not avail:
        st.warning("No sector data available for this fund.")
        return

    latest = avail[-1]
    earliest = avail[0]

    fd_sig = fd[fd[avail].max(axis=1) >= 1.5].copy()
    fd_sig = fd_sig.sort_values(latest, ascending=False)

    if len(fd_sig) == 0:
        st.warning("No significant sector allocations found.")
        return

    st.markdown("#### Month-by-month breakdown")
    table_rows = []
    for _, row in fd_sig.iterrows():
        first_val, last_val = row[earliest], row[latest]
        if pd.notna(first_val) and pd.notna(last_val):
            diff = last_val - first_val
            if diff > 3: trend = f"↑↑ +{diff:.1f}pp"
            elif diff > 1: trend = f"↑ +{diff:.1f}pp"
            elif diff < -3: trend = f"↓↓ {diff:.1f}pp"
            elif diff < -1: trend = f"↓ {diff:.1f}pp"
            else: trend = f"→ {diff:+.1f}pp"
        elif pd.notna(last_val):
            trend = "New"
        else:
            trend = "—"

        r = {"Sector": row["Sector"]}
        for m in avail:
            label = m.replace("_", " 20")  # Apr_25 → Apr 2025
            r[label] = round(row[m], 1) if pd.notna(row[m]) else None
        r["Trend"] = trend
        table_rows.append(r)

    tdf = pd.DataFrame(table_rows)
    alloc_cols = [c for c in tdf.columns if c not in ("Sector", "Trend")]
    styled = (tdf.style.map(c_alloc, subset=alloc_cols).map(c_trend, subset=["Trend"])
        .format(na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Sector"], **{"text-align": "left", "font-weight": "600"})
        .set_properties(subset=["Trend"], **{"font-weight": "600"}))
    st.dataframe(styled, use_container_width=True, height=500, hide_index=True)

    # Biggest moves
    st.markdown(f"#### Biggest moves ({earliest.replace('_',' ')} → {latest.replace('_',' ')})")
    moves = []
    for _, row in fd_sig.iterrows():
        if pd.notna(row[earliest]) and pd.notna(row[latest]):
            diff = row[latest] - row[earliest]
            if abs(diff) > 1.5:
                moves.append((row["Sector"], diff, row[latest]))
    moves.sort(key=lambda x: -abs(x[1]))
    if moves:
        cols = st.columns(min(len(moves), 5))
        for i, (sector, diff, current) in enumerate(moves[:5]):
            with cols[i]:
                st.metric(sector, f"{current:.1f}%", f"{'+' if diff > 0 else ''}{diff:.1f}pp")
    else:
        st.caption("No major sector shifts (>1.5pp) detected.")


def render_lc_consensus(sector_data):
    st.markdown("## LargeCap — Sector Consensus")
    st.markdown(f"Where are the {len(LC_QUAL_FUNDS)} selected funds converging and diverging?")

    # Use latest month with data
    all_m = ["Apr_25", "May_25", "Jun_25", "Jul_25", "Aug_25", "Sep_25",
             "Oct_25", "Nov_25", "Dec_25", "Jan_26", "Feb_26", "Mar_26"]
    avail = [m for m in all_m if m in sector_data.columns and sector_data[m].notna().any()]
    if not avail:
        st.warning("No sector data available."); return
    latest = avail[-1]
    earliest = avail[0]

    top_sectors = (sector_data.groupby("Sector")[latest].mean()
                    .sort_values(ascending=False).head(15).index.tolist())

    cons = []
    for sector in top_sectors:
        allocs, changes = [], []
        cnt = 0
        for fund in LC_QUAL_FUNDS:
            fd = sector_data[(sector_data["Fund"] == fund) & (sector_data["Sector"] == sector)]
            if len(fd) > 0 and pd.notna(fd.iloc[0][latest]):
                allocs.append(fd.iloc[0][latest]); cnt += 1
                if pd.notna(fd.iloc[0][earliest]):
                    changes.append(fd.iloc[0][latest] - fd.iloc[0][earliest])
        avg_a = np.mean(allocs) if allocs else 0
        avg_c = np.mean(changes) if changes else 0
        if avg_c > 2: direction = "Strong Addition ↑↑"
        elif avg_c > 0.5: direction = "Adding ↑"
        elif avg_c < -2: direction = "Strong Reduction ↓↓"
        elif avg_c < -0.5: direction = "Trimming ↓"
        else: direction = "Stable →"
        cons.append({"Sector": sector, "Avg Alloc%": round(avg_a, 1), "Funds": cnt,
                      "Avg Change": round(avg_c, 1), "Direction": direction})

    cdf = pd.DataFrame(cons)
    cdf_s = cdf.sort_values("Avg Alloc%", ascending=True)
    bar_colors = []
    for _, r in cdf_s.iterrows():
        ch = r["Avg Change"]
        if ch > 2: bar_colors.append("#22c55e")
        elif ch > 0.5: bar_colors.append("#86efac")
        elif ch < -2: bar_colors.append("#ef4444")
        elif ch < -0.5: bar_colors.append("#fca5a5")
        else: bar_colors.append("#94a3b8")

    fig = go.Figure()
    fig.add_trace(go.Bar(y=cdf_s["Sector"], x=cdf_s["Avg Alloc%"], orientation="h",
        marker=dict(color=bar_colors),
        text=[f"{v:.1f}%" for v in cdf_s["Avg Alloc%"]], textposition="outside"))
    fig.update_layout(height=420, margin=dict(l=10, r=40, t=10, b=30),
                      xaxis_title="Avg allocation %", showlegend=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Funds adding · 🔴 Funds trimming · ⚪ Stable")

    # Cross-fund heatmap
    st.markdown(f"#### Cross-fund allocation heatmap — {latest.replace('_',' ')}")
    heat_sectors = top_sectors[:12]
    heat_data = []
    for sector in heat_sectors:
        row_d = {"Sector": sector}
        for fund in LC_QUAL_FUNDS:
            fd = sector_data[(sector_data["Fund"] == fund) & (sector_data["Sector"] == sector)]
            val = fd.iloc[0][latest] if len(fd) > 0 and pd.notna(fd.iloc[0][latest]) else 0
            row_d[short_lc(fund)] = round(val, 1)
        heat_data.append(row_d)

    hdf = pd.DataFrame(heat_data)
    f_short = [short_lc(f) for f in LC_QUAL_FUNDS]
    z = hdf[f_short].values

    fig2 = go.Figure(data=go.Heatmap(
        z=z, x=f_short, y=hdf["Sector"],
        colorscale=[[0, "#f8fafc"], [0.2, "#dbeafe"], [0.4, "#93c5fd"],
                    [0.6, "#3b82f6"], [0.8, "#1d4ed8"], [1.0, "#1e3a5f"]],
        text=z, texttemplate="%{text:.1f}", textfont=dict(size=11),
        colorbar=dict(title="Alloc %", thickness=15)))
    fig2.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════
# RENDER LARGECAP PE / VALUATION TIMELINE
# ═══════════════════════════════════════════
def render_lc_valuations(pe_data, turnover_data):
    st.markdown("## LargeCap — PE & Valuation Monitor")
    st.markdown("Month-by-month PE (Harmonic Mean), PBV, Dividend Yield & Turnover for selected funds.")

    selected = st.selectbox("Select Fund", LC_QUAL_FUNDS, format_func=short_lc, key="lc_val_fund")

    fd_pe = pe_data[pe_data["Fund"] == selected].sort_values("Date", ascending=False).head(12)
    fd_tr = turnover_data[turnover_data["Fund"] == selected].sort_values("Date", ascending=False).head(12)

    if fd_pe.empty:
        st.warning("No PE data available for this fund.")
        return

    st.markdown(f"### {short_lc(selected)}")

    # Latest metrics
    latest = fd_pe.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PE (HM)", f"{latest['PE_HM']:.1f}x" if pd.notna(latest['PE_HM']) else "—")
    c2.metric("PBV (HM)", f"{latest['PBV_HM']:.2f}x" if pd.notna(latest['PBV_HM']) else "—")
    c3.metric("Div Yield", f"{latest['DivYield']:.2f}%" if pd.notna(latest['DivYield']) else "—")
    if not fd_tr.empty:
        c4.metric("Turnover", f"{fd_tr.iloc[0]['Turnover']:.0f}%")
    else:
        c4.metric("Turnover", "—")

    # Monthly table
    st.markdown("#### Monthly trend")
    rows = []
    for _, r in fd_pe.iterrows():
        row = {"Month": r["Date"].strftime("%b %Y"),
               "PE (HM)": round(r["PE_HM"], 1) if pd.notna(r["PE_HM"]) else None,
               "PBV (HM)": round(r["PBV_HM"], 2) if pd.notna(r["PBV_HM"]) else None,
               "Div Yield%": round(r["DivYield"], 2) if pd.notna(r["DivYield"]) else None,
               "MCAP (Cr)": round(r["MCAP_Cr"], 0) if pd.notna(r["MCAP_Cr"]) else None}
        # Find matching turnover
        tr_match = fd_tr[fd_tr["Date"] == r["Date"]]
        row["Turnover%"] = round(tr_match.iloc[0]["Turnover"], 0) if len(tr_match) > 0 and pd.notna(tr_match.iloc[0]["Turnover"]) else None
        rows.append(row)

    tdf = pd.DataFrame(rows)
    styled = (tdf.style
        .map(c_pe, subset=["PE (HM)"])
        .format({"PE (HM)": "{:.1f}x", "PBV (HM)": "{:.2f}x", "Div Yield%": "{:.2f}",
                 "MCAP (Cr)": "{:,.0f}", "Turnover%": "{:.0f}"}, na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Month"], **{"text-align": "left", "font-weight": "600"}))
    st.dataframe(styled, use_container_width=True, height=450, hide_index=True)

    # PE chart
    chart_df = fd_pe.sort_values("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df["PE_HM"], mode="lines+markers",
        name="PE (HM)", line=dict(color="#3b82f6", width=2)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="PE (Harmonic Mean)", xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Cross-fund PE comparison
    st.markdown("#### Cross-fund PE comparison (latest)")
    comp = []
    for fund in LC_QUAL_FUNDS:
        fd = pe_data[pe_data["Fund"] == fund].sort_values("Date", ascending=False)
        if len(fd) > 0 and pd.notna(fd.iloc[0]["PE_HM"]):
            tr_f = turnover_data[turnover_data["Fund"] == fund].sort_values("Date", ascending=False)
            comp.append({
                "Fund": short_lc(fund),
                "PE (HM)": round(fd.iloc[0]["PE_HM"], 1),
                "PBV (HM)": round(fd.iloc[0]["PBV_HM"], 2) if pd.notna(fd.iloc[0]["PBV_HM"]) else None,
                "Div Yield%": round(fd.iloc[0]["DivYield"], 2) if pd.notna(fd.iloc[0]["DivYield"]) else None,
                "Turnover%": round(tr_f.iloc[0]["Turnover"], 0) if len(tr_f) > 0 and pd.notna(tr_f.iloc[0]["Turnover"]) else None,
            })
    comp_df = pd.DataFrame(comp).sort_values("PE (HM)")
    styled_c = (comp_df.style
        .map(c_pe, subset=["PE (HM)"])
        .map(c_pb, subset=["PBV (HM)"])
        .format({"PE (HM)": "{:.1f}x", "PBV (HM)": "{:.2f}x", "Div Yield%": "{:.2f}", "Turnover%": "{:.0f}"}, na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "600"}))
    st.dataframe(styled_c, use_container_width=True, height=550, hide_index=True)
    st.caption("PE color: 🟢 Value (<25x) · 🟡 Growth (25-32x) · 🔴 Premium (>32x)")


# ═══════════════════════════════════════════
# RENDER LARGECAP STOCK ALLOCATION
# ═══════════════════════════════════════════
def render_lc_stocks(stock_data):
    st.markdown("## LargeCap — Stock Allocation")

    funds_with_data = [f for f in LC_QUAL_FUNDS if f in stock_data["Fund"].unique()]
    if not funds_with_data:
        st.warning("No stock data available."); return

    selected = st.selectbox("Select Fund", funds_with_data, format_func=short_lc, key="lc_stock_fund")
    fd = stock_data[stock_data["Fund"] == selected].copy()

    st.markdown(f"### {short_lc(selected)} — Top Holdings")

    # Determine latest column with data
    alloc_cols = ["Mar_26", "Feb_26", "Aug_25", "Jan_25", "Jul_24"]
    latest_col = None
    for c in alloc_cols:
        if fd[c].notna().any():
            latest_col = c; break

    if latest_col is None:
        st.warning("No allocation data."); return

    fd = fd.sort_values(latest_col, ascending=False)

    # Display table
    disp_cols_map = {"Mar_26": "Mar 2026", "Feb_26": "Feb 2026", "Aug_25": "Aug 2025",
                     "Jan_25": "Jan 2025", "Jul_24": "Jul 2024"}
    avail_alloc = [c for c in alloc_cols if fd[c].notna().any()]

    tdf = fd[["Company", "Sector"] + avail_alloc].copy()
    tdf.columns = ["Company", "Sector"] + [disp_cols_map[c] for c in avail_alloc]

    # Add trend
    if len(avail_alloc) >= 2:
        first_c, last_c = avail_alloc[-1], avail_alloc[0]
        changes = fd[last_c] - fd[first_c]
        trends = []
        for ch in changes:
            if pd.isna(ch): trends.append("—")
            elif ch > 1: trends.append(f"↑ +{ch:.1f}pp")
            elif ch < -1: trends.append(f"↓ {ch:.1f}pp")
            else: trends.append(f"→ {ch:+.1f}pp")
        tdf["Trend"] = trends

    # Color the allocation columns
    alloc_display = [disp_cols_map[c] for c in avail_alloc]

    def c_stock(val):
        if pd.isna(val): return ""
        if val >= 8: return "background-color:#1d4ed8;color:white;font-weight:700;"
        if val >= 5: return "background-color:#3b82f6;color:white;"
        if val >= 3: return "background-color:#93c5fd;"
        if val >= 1.5: return "background-color:#dbeafe;"
        return ""

    styled = (tdf.style
        .map(c_stock, subset=alloc_display)
        .format({c: "{:.2f}" for c in alloc_display}, na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Company"], **{"text-align": "left", "font-weight": "600"})
        .set_properties(subset=["Sector"], **{"text-align": "left"}))
    if "Trend" in tdf.columns:
        styled = styled.map(c_trend, subset=["Trend"])
    st.dataframe(styled, use_container_width=True, height=700, hide_index=True)

    # Top 10 bar chart
    top10 = fd.head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top10["Company"].iloc[::-1],
        x=top10[latest_col].iloc[::-1],
        orientation="h",
        marker=dict(color="#3b82f6"),
        text=[f"{v:.1f}%" for v in top10[latest_col].iloc[::-1]],
        textposition="outside"))
    fig.update_layout(height=350, margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Allocation %", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Concentration metrics
    st.markdown("#### Concentration")
    top5 = fd.head(5)[latest_col].sum()
    top10_sum = fd.head(10)[latest_col].sum()
    top20 = fd.head(20)[latest_col].sum()
    total_stocks = len(fd[fd[latest_col].notna()])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 5", f"{top5:.1f}%")
    c2.metric("Top 10", f"{top10_sum:.1f}%")
    c3.metric("Top 20", f"{top20:.1f}%")
    c4.metric("Total Stocks", f"{total_stocks}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    # ── Sidebar weights (shared) ──
    with st.sidebar:
        st.subheader("🏛️ Compounder Weights")
        w_cagr3 = st.slider("3Y Rolling CAGR", 0, 100, 20)
        w_cagr5 = st.slider("5Y Rolling CAGR", 0, 100, 25)
        w_up = st.slider("Upside Capture", 0, 100, 15)
        w_down = st.slider("Downside Capture", 0, 100, 25)
        w_ulcer = st.slider("Ulcer Index", 0, 100, 15)
        st.divider()
        st.subheader("🏎️ Momentum Weights")
        m6 = st.slider("6M Dampened RA", 0, 100, 50)
        m1 = st.slider("1Y Dampened RA", 0, 100, 50)

    w_est = [w_cagr3, w_cagr5, w_up, w_down, w_ulcer]
    w_mom = [m6, m1]

    # ── Load whatever data is available ──
    has_sc = os.path.exists("smallcapfinalrank.xlsx")
    has_lc = os.path.exists("largecap1.xlsx") and os.path.exists("largecap2.xlsx")

    if has_sc and has_lc:
        st.title("📊 Dual-Engine — SmallCap + LargeCap")
    elif has_sc:
        st.title("🚀 SmallCap Dual-Engine")
    elif has_lc:
        st.title("🏦 LargeCap Dual-Engine")
    else:
        st.error("No data files found.")
        return

    # ── Load optional LC enrichment data ──
    lc_pe_data, lc_tr_data, lc_stock_data = None, None, None
    has_lc_pe = os.path.exists("pe.xlsx")
    has_lc_tr = os.path.exists("portfolio_ratios.xlsx")
    has_lc_stocks = os.path.exists("stockalloacations.xlsx")
    if has_lc_pe:
        lc_pe_data = load_lc_pe()
    if has_lc_tr:
        lc_tr_data = load_lc_turnover()
    if has_lc_stocks:
        lc_stock_data = load_lc_stocks()

    # ── Build top-level tabs ──
    tab_labels = []
    if has_sc:
        tab_labels.append("🚀 SmallCap Rankings")
    if has_lc:
        tab_labels.append("🏦 LargeCap Rankings")

    has_sc_sectors = False
    if has_sc:
        try:
            sc_sector_data = load_sc_sectors()
            has_sc_sectors = True
        except Exception:
            pass
        if has_sc_sectors:
            tab_labels.append("🔬 SC Sector Flow")
            tab_labels.append("🔎 SC Sector Consensus")

    has_lc_sectors = False
    if os.path.exists("sectorflows.xlsx"):
        try:
            lc_sector_data = load_lc_sectors()
            has_lc_sectors = len(lc_sector_data) > 0
        except Exception:
            pass
        if has_lc_sectors:
            tab_labels.append("🔬 LC Sector Flow")
            tab_labels.append("🔎 LC Sector Consensus")

    if lc_pe_data is not None:
        tab_labels.append("📈 LC Valuations")
    if lc_stock_data is not None and len(lc_stock_data) > 0:
        tab_labels.append("🏗️ LC Stock Holdings")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── SmallCap Rankings ──
    if has_sc:
        with tabs[tab_idx]:
            sc_nav, sc_funds, sc_aum, sc_bench = load_sc_data()
            sc_raw = compute_all(sc_nav, sc_funds)
            sc_raw = sc_raw.merge(sc_aum, on="Fund", how="left")
            sc_est, sc_mom = rank_funds(sc_raw, w_est, w_mom)
            st.caption(f"Benchmark: {sc_bench}  ·  {len(sc_funds)} funds  ·  Data through {sc_nav['Date'].max().strftime('%d %b %Y')}")
            render_quant(sc_est, sc_mom, short_sc, has_aum=True)
        tab_idx += 1

    # ── LargeCap Rankings (with PE & Turnover merged in) ──
    if has_lc:
        with tabs[tab_idx]:
            lc_nav, lc_funds, lc_bench = load_lc_data()
            lc_raw = compute_all(lc_nav, lc_funds)
            # Merge latest PE & Turnover into ranking data
            if lc_pe_data is not None:
                pe_latest = (lc_pe_data.sort_values("Date", ascending=False)
                    .groupby("Fund").first().reset_index()[["Fund", "PE_HM", "PBV_HM"]])
                lc_raw = lc_raw.merge(pe_latest, on="Fund", how="left")
            if lc_tr_data is not None:
                tr_latest = (lc_tr_data.sort_values("Date", ascending=False)
                    .groupby("Fund").first().reset_index()[["Fund", "Turnover"]])
                lc_raw = lc_raw.merge(tr_latest, on="Fund", how="left")
            lc_est, lc_mom = rank_funds(lc_raw, w_est, w_mom)
            st.caption(f"Benchmark: {lc_bench}  ·  {len(lc_funds)} funds  ·  Data through {lc_nav['Date'].max().strftime('%d %b %Y')}")
            render_quant(lc_est, lc_mom, short_lc, has_aum=False)
        tab_idx += 1

    # ── SC Sector Tabs ──
    if has_sc_sectors:
        with tabs[tab_idx]:
            render_sc_sector_flow(sc_sector_data)
        tab_idx += 1
        with tabs[tab_idx]:
            render_sc_consensus(sc_sector_data)
        tab_idx += 1

    # ── LC Sector Tabs ──
    if has_lc_sectors:
        with tabs[tab_idx]:
            render_lc_sector_flow(lc_sector_data)
        tab_idx += 1
        with tabs[tab_idx]:
            render_lc_consensus(lc_sector_data)
        tab_idx += 1

    # ── LC Valuations ──
    if lc_pe_data is not None:
        with tabs[tab_idx]:
            render_lc_valuations(lc_pe_data, lc_tr_data if lc_tr_data is not None else pd.DataFrame())
        tab_idx += 1

    # ── LC Stock Holdings ──
    if lc_stock_data is not None and len(lc_stock_data) > 0:
        with tabs[tab_idx]:
            render_lc_stocks(lc_stock_data)


if __name__ == "__main__":
    main()
