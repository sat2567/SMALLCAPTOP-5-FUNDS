"""
Dual-Engine — SmallCap + LargeCap
Quant Rankings · Sector Flow · Sector Consensus · PE Monitor · Stock Holdings
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
# SMALLCAP QUALITATIVE PROFILES
# ═══════════════════════════════════════════
PROVEN = {
    "Bandhan Small Cap Fund-Reg(G)": {"PE": 16.78, "PB": 2.07, "Stance": "Deep Value"},
    "Bank of India Small Cap Fund-Reg(G)": {"PE": 32.82, "PB": 4.02, "Stance": "Balanced Momentum"},
    "Edelweiss Small Cap Fund-Reg(G)": {"PE": 31.21, "PB": 4.13, "Stance": "Risk-Adjusted Growth"},
    "Canara Rob Small Cap Fund-Reg(G)": {"PE": 28.33, "PB": 3.64, "Stance": "Quality / Hedged"},
    "Invesco India Smallcap Fund-Reg(G)": {"PE": 39.47, "PB": 4.24, "Stance": "Ultra-Growth Premium"},
    "Mahindra Manulife Small Cap Fund-Reg(G)": {"PE": 29.50, "PB": 3.85, "Stance": "Diversified Growth"},
}
MOMENTUM_INFO = {
    "TRUSTMF Small Cap Fund-Reg(G)": {"PE": 37.31, "PB": 4.87, "Stance": "Terminal Value Premium"},
    "Union Small Cap Fund-Reg(G)": {"PE": 39.27, "PB": 5.24, "Stance": "High-Growth Momentum"},
    "DSP Small Cap Fund-Reg(G)": {"PE": 28.39, "PB": 3.52, "Stance": "Concentrated Core"},
    "Aditya Birla SL Small Cap Fund(G)": {"PE": 27.71, "PB": 4.10, "Stance": "Operating Leverage"},
    "Mirae Asset Small Cap Fund-Reg(G)": {"PE": 24.80, "PB": 3.92, "Stance": "GARP"},
    "Invesco India Smallcap Fund-Reg(G)": {"PE": 39.47, "PB": 4.24, "Stance": "Ultra-Growth Premium"},
    "Mahindra Manulife Small Cap Fund-Reg(G)": {"PE": 29.50, "PB": 3.85, "Stance": "Diversified Growth"},
}
ALL_INFO = {**PROVEN, **MOMENTUM_INFO}
ALL_QUAL_FUNDS = list(dict.fromkeys(list(PROVEN.keys()) + list(MOMENTUM_INFO.keys())))
MONTHS = ["Jan_25", "Jun_25", "Sep_25", "Dec_25", "Feb_26"]

# ═══════════════════════════════════════════
# LARGECAP — SELECTED FUNDS
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

def short_sc(f):
    return (f.replace("Small Cap","SC").replace("Smallcap","SC")
              .replace("Fund-Reg(G)","").replace("Fund(G)","").strip())

def short_lc(f):
    return (f.replace("Large Cap","LC").replace("Largecap","LC")
              .replace("Fund-Reg(G)","").replace("Fund(G)","")
              .replace("Fund-Reg(IDCW)","").replace("Fund(IDCW)","").strip())

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
    for f in fund_names: nav[f] = pd.to_numeric(nav[f], errors="coerce")
    try:
        bench = pd.read_csv("Nifty_500_TRI_Combined.csv")
        bench["Date"] = pd.to_datetime(bench["Date"])
        nav = pd.merge(nav, bench[["Date","Benchmark"]], on="Date", how="left")
        nav["Benchmark"] = nav["Benchmark"].ffill()
        bn = "Nifty 500 TRI"
    except Exception:
        valid = [f for f in fund_names if nav[f].notna().sum()>252]
        nav["Benchmark"] = nav[valid].pct_change().mean(axis=1).add(1).cumprod()*100
        bn = "SmallCap Index (Proxy)"
    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund","Month_End","AUM","AAUM","Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund","AUM"]]
    return nav, fund_names, aum_latest, bn

@st.cache_data(show_spinner="Loading SmallCap Sector Data...")
def load_sc_sectors():
    d = pd.read_excel("SECTORALLCOATIONSMALLCAP.xlsx").iloc[2:].copy()
    d.columns = ["Fund","Sector","Feb_26","Dec_25","Sep_25","Jun_25","Jan_25","c7","c8","c9","c10","c11"]
    d = d[["Fund","Sector","Feb_26","Dec_25","Sep_25","Jun_25","Jan_25"]]
    d = d.dropna(subset=["Fund","Sector"])
    d = d[~d["Fund"].str.contains("Accord", na=False)]
    d = d[d["Sector"]!="Sector"]
    for c in MONTHS: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ═══════════════════════════════════════════
# DATA LOADING — LARGECAP
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading LargeCap NAV Data...")
def load_lc_data():
    r1 = pd.read_excel("largecap1.xlsx", header=None)
    r2 = pd.read_excel("largecap2.xlsx", header=None)
    funds1 = r1.iloc[2,1:].dropna().tolist()
    nav1 = r1.iloc[4:,:len(funds1)+1].copy(); nav1.columns=["Date"]+funds1
    nav1 = nav1[pd.to_datetime(nav1["Date"],errors="coerce").notna()].copy(); nav1["Date"]=pd.to_datetime(nav1["Date"])
    funds2 = r2.iloc[2,1:].dropna().tolist()
    nav2 = r2.iloc[4:,:len(funds2)+1].copy(); nav2.columns=["Date"]+funds2
    nav2 = nav2[pd.to_datetime(nav2["Date"],errors="coerce").notna()].copy(); nav2["Date"]=pd.to_datetime(nav2["Date"])
    nav = pd.merge(nav1, nav2, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    fns = funds1+funds2
    for f in fns: nav[f] = pd.to_numeric(nav[f], errors="coerce")
    ex = ["Long-Short","Long Short","DynaSIF","Qsif"]
    fns = [f for f in fns if not any(k.lower() in f.lower() for k in ex)]
    valid = [f for f in fns if nav[f].notna().sum()>252]
    nav["Benchmark"] = nav[valid].pct_change().mean(axis=1).add(1).cumprod()*100
    return nav, fns, "LargeCap Index (Proxy)"

@st.cache_data(show_spinner="Loading LargeCap Sector Data...")
def load_lc_sectors():
    d = pd.read_excel("sectorflows.xlsx", header=None).iloc[4:,:14].copy()
    d.columns = ["Fund","Sector","Mar_26","Feb_26","Jan_26","Dec_25","Nov_25","Oct_25","Sep_25","Aug_25","Jul_25","Jun_25","May_25","Apr_25"]
    d = d.dropna(subset=["Fund","Sector"])
    d = d[~d["Fund"].str.contains("Accord", na=False)]
    d = d[d["Sector"]!="Sector"]
    d = d[d["Fund"].isin(LC_QUAL_FUNDS)]
    for c in d.columns[2:]: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def _parse_stacked(path, value_cols):
    df = pd.read_excel(path, header=None)
    recs, cur = [], None
    for _, row in df.iterrows():
        v = str(row[0])
        if "Scheme Name:" in v:
            cur = v.replace("Scheme Name:","").strip()
        elif cur:
            try:
                dt = pd.to_datetime(row[0])
                rec = {"Fund": cur, "Date": dt}
                for ci, cn in value_cols:
                    rec[cn] = pd.to_numeric(row[ci], errors="coerce")
                recs.append(rec)
            except Exception: pass
    return pd.DataFrame(recs)

@st.cache_data(show_spinner="Loading PE / PBV Data...")
def load_lc_pe():
    return _parse_stacked("pe.xlsx", [(2,"PE_HM"),(4,"PBV_HM"),(5,"DivYield"),(6,"MCAP_Cr")])

@st.cache_data(show_spinner="Loading Turnover Data...")
def load_lc_turnover():
    return _parse_stacked("portfolio_ratios.xlsx", [(1,"Turnover"),(2,"Liquidity"),(3,"Liquidity_Avg")])

@st.cache_data(show_spinner="Loading Stock Allocations...")
def load_lc_stocks():
    df = pd.read_excel("stockalloacations.xlsx", header=None)
    d = df.iloc[4:,:9].copy()
    d.columns = ["Fund","Company","Asset","Sector","Mar_26","Feb_26","Aug_25","Jan_25","Jul_24"]
    d = d.dropna(subset=["Fund","Company"])
    d = d[d["Fund"].isin(LC_QUAL_FUNDS)]
    for c in ["Mar_26","Feb_26","Aug_25","Jan_25","Jul_24"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ═══════════════════════════════════════════
# SHARED QUANT ENGINE
# ═══════════════════════════════════════════
def compute_all(nav_in, fund_names):
    nav = nav_in.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    br = mret["Benchmark"]
    results = []
    for fund in fund_names:
        fd = nav[["Date",fund]].dropna()
        if len(fd)<120: continue
        fr = mret[fund].dropna() if fund in mret.columns else pd.Series(dtype=float)
        ci = fr.index.intersection(br.dropna().index)
        if len(ci)<6: continue
        fr, brc = fr.loc[ci], br.loc[ci]
        mn = fd.set_index("Date")[fund].resample("ME").last().dropna()
        c5 = [((mn.iloc[i]/mn.iloc[i-60])**(1/5)-1)*100 for i in range(60,len(mn)) if mn.iloc[i-60]>0]
        c3 = [((mn.iloc[i]/mn.iloc[i-36])**(1/3)-1)*100 for i in range(36,len(mn)) if mn.iloc[i-36]>0]
        up_m = brc[brc>0]; dn_m = brc[brc<0]
        uc = (fr.loc[up_m.index].mean()/up_m.mean())*100 if len(up_m)>3 else None
        dc = (fr.loc[dn_m.index].mean()/dn_m.mean())*100 if len(dn_m)>3 else None
        prices = fd[fund].values; peaks = np.maximum.accumulate(prices)
        dd = (prices-peaks)/peaks; ulcer = np.sqrt(np.mean(dd**2))*100
        ld = fd.iloc[-1]["Date"]
        r1y = fd[fd["Date"]>=ld-timedelta(days=365)]
        vol = r1y[fund].pct_change().dropna().std()*np.sqrt(252)*100 if len(r1y)>50 else None
        def gr(days):
            p = fd[fd["Date"]<=ld-timedelta(days=days)]
            return (prices[-1]/p.iloc[-1][fund]-1)*100 if not p.empty else None
        r6, r1 = gr(180), gr(365)
        m6 = r6/np.sqrt(vol) if (vol and vol>0 and r6 is not None) else None
        m1y = r1/np.sqrt(vol) if (vol and vol>0 and r1 is not None) else None
        results.append({"Fund":fund,"Track_Yrs":len(fd)/252,
            "Roll_3Y":np.mean(c3) if c3 else None,"Roll_5Y":np.mean(c5) if c5 else None,
            "Up_Cap":uc,"Down_Cap":dc,"Cap_Ratio":(uc/dc if (uc and dc and dc!=0) else None),
            "Ulcer_Index":ulcer,"Max_DD":dd.min()*100,
            "Ret_6M":r6,"Ret_1Y":r1,"Vol":vol,"Mom_6M_RA":m6,"Mom_1Y_RA":m1y})
    return pd.DataFrame(results)

def pctrank(s, asc=True):
    v = s.notna()
    if not v.any(): return s
    r = s[v].rank(ascending=asc, pct=True)*100
    out = pd.Series(np.nan, index=s.index); out[v] = r; return out

def rank_funds(df, w_est, w_mom):
    est = df[df["Track_Yrs"]>=3].copy()
    if not est.empty:
        est["S_R3"]=pctrank(est["Roll_3Y"]); est["S_R5"]=pctrank(est["Roll_5Y"])
        est["S_UC"]=pctrank(est["Up_Cap"]); est["S_DC"]=pctrank(est["Down_Cap"],False)
        est["S_UI"]=pctrank(est["Ulcer_Index"],False)
        est["Score"]=(est["S_R3"].fillna(0)*w_est[0]+est["S_R5"].fillna(0)*w_est[1]+
            est["S_UC"].fillna(0)*w_est[2]+est["S_DC"].fillna(0)*w_est[3]+
            est["S_UI"].fillna(0)*w_est[4])/sum(w_est)
        est["Rank"]=est["Score"].rank(ascending=False, method="min")
    mom = df[df["Mom_6M_RA"].notna()].copy()
    if not mom.empty:
        mom["S_M6"]=pctrank(mom["Mom_6M_RA"]); mom["S_M1"]=pctrank(mom["Mom_1Y_RA"])
        mom["Score"]=(mom["S_M6"].fillna(0)*w_mom[0]+mom["S_M1"].fillna(0)*w_mom[1])/sum(w_mom)
        mom["Rank"]=mom["Score"].rank(ascending=False, method="min")
    return est, mom

# ═══════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════
def c_dc(v):
    if pd.isna(v): return ""
    if v<50: return "color:#16a34a;font-weight:700;"
    if v<80: return "color:#16a34a;"
    if v<100: return "color:#ca8a04;"
    return "color:#dc2626;font-weight:600;"
def c_uc(v):
    if pd.isna(v): return ""
    if v>120: return "color:#16a34a;font-weight:700;"
    if v>100: return "color:#16a34a;"
    return "color:#ca8a04;"
def c_cr(v):
    if pd.isna(v): return ""
    if v>1.3: return "background-color:#dcfce7;color:#166534;font-weight:700;"
    if v>1.1: return "background-color:#e0f2fe;color:#075985;"
    if v>1.0: return "color:#ca8a04;"
    return "background-color:#fecaca;color:#991b1b;"
def c_pe(v):
    if pd.isna(v): return ""
    if v<25: return "background-color:#dcfce7;color:#166534;font-weight:600;"
    if v<32: return "background-color:#fef9c3;color:#854d0e;"
    return "background-color:#fecaca;color:#991b1b;font-weight:600;"
def c_pb(v):
    if pd.isna(v): return ""
    if v<3: return "background-color:#dcfce7;color:#166534;font-weight:600;"
    if v<4.2: return "background-color:#fef9c3;color:#854d0e;"
    return "background-color:#fecaca;color:#991b1b;"
def c_alloc(v):
    if pd.isna(v): return ""
    if v>=15: return "background-color:#2563eb;color:white;font-weight:700;"
    if v>=10: return "background-color:#60a5fa;color:white;"
    if v>=7: return "background-color:#93c5fd;"
    if v>=4: return "background-color:#bfdbfe;"
    if v>=2: return "background-color:#dbeafe;"
    return ""
def c_trend(v):
    s = str(v)
    if "↑↑" in s: return "background-color:#166534;color:white;font-weight:700;"
    if "↑" in s: return "background-color:#dcfce7;color:#166534;"
    if "↓↓" in s: return "background-color:#991b1b;color:white;font-weight:700;"
    if "↓" in s: return "background-color:#fecaca;color:#991b1b;"
    return "color:#9ca3af;"
def c_stock(v):
    if pd.isna(v): return ""
    if v>=8: return "background-color:#1d4ed8;color:white;font-weight:700;"
    if v>=5: return "background-color:#3b82f6;color:white;"
    if v>=3: return "background-color:#93c5fd;"
    if v>=1.5: return "background-color:#dbeafe;"
    return ""
def c_turn(v):
    if pd.isna(v): return ""
    if v<=30: return "background-color:#dcfce7;color:#166534;"
    if v<=60: return "background-color:#fef9c3;color:#854d0e;"
    if v<=100: return "background-color:#fed7aa;color:#9a3412;"
    return "background-color:#fecaca;color:#991b1b;font-weight:600;"

# ═══════════════════════════════════════════
# RENDER QUANT RANKINGS
# ═══════════════════════════════════════════
def render_quant(est, mom, shortener, has_aum=False):
    view = st.radio("Ranking Engine",[
        "🏛️ Established Compounders (Funds > 3 Yrs)","🏎️ Momentum Efficiency"
    ], horizontal=True, key=f"engine_{shortener.__name__}")
    target = est if "Established" in view else mom
    if target.empty: st.warning("No funds qualify."); return
    disp = target.sort_values("Rank").copy()
    if "Established" in view:
        cols =["Rank","Fund","Score","Roll_3Y","Roll_5Y","Up_Cap","Down_Cap","Cap_Ratio","Ulcer_Index","Max_DD"]
        names=["Rank","Fund","Score","3Y CAGR","5Y CAGR","Up Cap%","Down Cap%","Up/Down","Ulcer Index","Max DD%"]
    else:
        cols =["Rank","Fund","Score","Ret_6M","Ret_1Y","Vol","Mom_6M_RA","Mom_1Y_RA","Up_Cap","Down_Cap","Cap_Ratio"]
        names=["Rank","Fund","Score","6M Ret%","1Y Ret%","Vol%","6M RA","1Y RA","Up Cap%","Down Cap%","Up/Down"]
    if has_aum and "AUM" in disp.columns: cols.append("AUM"); names.append("AUM (Cr)")
    for src,dst in [("PE_HM","PE (HM)"),("Turnover","Turnover%"),("Liquidity","Liquidity (Days)")]:
        if src in disp.columns: cols.append(src); names.append(dst)
    avail = [c for c in cols if c in disp.columns]
    avail_n = [names[cols.index(c)] for c in avail]
    disp = disp[avail].copy(); disp.columns = avail_n
    disp["Fund"] = disp["Fund"].apply(shortener)
    num_c = [c for c in avail_n if c not in ("Rank","Fund","AUM (Cr)","Up/Down","PE (HM)","Turnover%","Liquidity (Days)")]
    fmt = {c:"{:.1f}" for c in num_c}
    fmt["Rank"]="{:.0f}"
    if "Up/Down" in avail_n: fmt["Up/Down"]="{:.2f}"
    if "AUM (Cr)" in avail_n: fmt["AUM (Cr)"]="{:.0f}"
    if "PE (HM)" in avail_n: fmt["PE (HM)"]="{:.1f}x"
    if "Turnover%" in avail_n: fmt["Turnover%"]="{:.0f}"
    if "Liquidity (Days)" in avail_n: fmt["Liquidity (Days)"]="{:.2f}"
    styled = disp.style.background_gradient(subset=["Score"], cmap="RdYlGn")
    if "Down Cap%" in avail_n: styled=styled.map(c_dc, subset=["Down Cap%"])
    if "Up Cap%" in avail_n: styled=styled.map(c_uc, subset=["Up Cap%"])
    if "Up/Down" in avail_n: styled=styled.map(c_cr, subset=["Up/Down"])
    if "PE (HM)" in avail_n: styled=styled.map(c_pe, subset=["PE (HM)"])
    if "Turnover%" in avail_n: styled=styled.map(c_turn, subset=["Turnover%"])
    styled = styled.format(fmt).format(na_rep="—")
    st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
    if "PE (HM)" in avail_n:
        st.caption("PE: 🟢 <25x · 🟡 25-32x · 🔴 >32x  |  Turnover: 🟢 ≤30 · 🟡 ≤60 · 🟠 ≤100 · 🔴 >100")

# ═══════════════════════════════════════════
# SMALLCAP SECTOR TABS
# ═══════════════════════════════════════════
def render_sc_sector_flow(sector_data):
    st.markdown("## Fund Strategy Profiles & Sector Flow")
    st.markdown("#### All Fund Profiles")
    rows = [{"Fund":short_sc(f),"PE":ALL_INFO.get(f,{}).get("PE"),"PB":ALL_INFO.get(f,{}).get("PB"),
             "Stance":ALL_INFO.get(f,{}).get("Stance","")} for f in ALL_QUAL_FUNDS]
    pdf = pd.DataFrame(rows)
    st.dataframe(pdf.style.map(c_pe, subset=["PE"]).map(c_pb, subset=["PB"])
        .format({"PE":"{:.1f}x","PB":"{:.2f}x"}, na_rep="—")
        .set_properties(**{"text-align":"center","font-size":"13px"})
        .set_properties(subset=["Fund"],**{"text-align":"left","font-weight":"600"}),
        use_container_width=True, height=420, hide_index=True)
    st.divider()
    st.markdown("#### Sector Flow — Pick a Fund")
    sel = st.selectbox("Select Fund", ALL_QUAL_FUNDS, format_func=short_sc)
    info = ALL_INFO.get(sel, {})
    fd = sector_data[sector_data["Fund"]==sel].copy()
    st.markdown(f"### {short_sc(sel)}")
    c1,c2,c3=st.columns(3)
    c1.metric("PE",f"{info.get('PE','—')}x"); c2.metric("PB",f"{info.get('PB','—')}x"); c3.metric("Stance",info.get("Stance","—"))
    fd_sig = fd[fd[MONTHS].max(axis=1)>=1.5].sort_values("Feb_26", ascending=False)
    if len(fd_sig)==0: st.warning("No sector data."); return
    trows = []
    for _,r in fd_sig.iterrows():
        j,f2 = r["Jan_25"],r["Feb_26"]
        if pd.notna(j) and pd.notna(f2):
            d=f2-j
            if d>3: tr=f"↑↑ +{d:.1f}pp"
            elif d>1: tr=f"↑ +{d:.1f}pp"
            elif d<-3: tr=f"↓↓ {d:.1f}pp"
            elif d<-1: tr=f"↓ {d:.1f}pp"
            else: tr=f"→ {d:+.1f}pp"
        elif pd.notna(f2): tr="New"
        else: tr="—"
        trows.append({"Sector":r["Sector"],
            "Jan 2025":round(j,1) if pd.notna(j) else None,
            "Jun 2025":round(r["Jun_25"],1) if pd.notna(r["Jun_25"]) else None,
            "Sep 2025":round(r["Sep_25"],1) if pd.notna(r["Sep_25"]) else None,
            "Dec 2025":round(r["Dec_25"],1) if pd.notna(r["Dec_25"]) else None,
            "Feb 2026":round(f2,1) if pd.notna(f2) else None,"12M Trend":tr})
    tdf=pd.DataFrame(trows)
    ac=["Jan 2025","Jun 2025","Sep 2025","Dec 2025","Feb 2026"]
    st.dataframe(tdf.style.map(c_alloc,subset=ac).map(c_trend,subset=["12M Trend"])
        .format(na_rep="—").set_properties(**{"text-align":"center","font-size":"13px"})
        .set_properties(subset=["Sector"],**{"text-align":"left","font-weight":"600"}),
        use_container_width=True, height=500, hide_index=True)

def render_sc_consensus(sd):
    st.markdown("## Sector Consensus — SmallCap")
    tops = sd[sd["Fund"].isin(ALL_QUAL_FUNDS)].groupby("Sector")["Feb_26"].mean().sort_values(ascending=False).head(15).index.tolist()
    cons=[]
    for s in tops:
        al,ch,cn=[],[],0
        for f in ALL_QUAL_FUNDS:
            fd=sd[(sd["Fund"]==f)&(sd["Sector"]==s)]
            if len(fd)>0 and pd.notna(fd.iloc[0]["Feb_26"]):
                al.append(fd.iloc[0]["Feb_26"]); cn+=1
                if pd.notna(fd.iloc[0]["Jan_25"]): ch.append(fd.iloc[0]["Feb_26"]-fd.iloc[0]["Jan_25"])
        aa=np.mean(al) if al else 0; ac=np.mean(ch) if ch else 0
        d="Strong Addition ↑↑" if ac>2 else "Adding ↑" if ac>0.5 else "Strong Reduction ↓↓" if ac<-2 else "Trimming ↓" if ac<-0.5 else "Stable →"
        cons.append({"Sector":s,"Avg Alloc%":round(aa,1),"Funds":cn,"Avg Change":round(ac,1),"Direction":d})
    cdf=pd.DataFrame(cons).sort_values("Avg Alloc%",ascending=True)
    bc=["#22c55e" if r["Avg Change"]>2 else "#86efac" if r["Avg Change"]>0.5 else "#ef4444" if r["Avg Change"]<-2 else "#fca5a5" if r["Avg Change"]<-0.5 else "#94a3b8" for _,r in cdf.iterrows()]
    fig=go.Figure(); fig.add_trace(go.Bar(y=cdf["Sector"],x=cdf["Avg Alloc%"],orientation="h",marker=dict(color=bc),text=[f"{v:.1f}%" for v in cdf["Avg Alloc%"]],textposition="outside"))
    fig.update_layout(height=420,margin=dict(l=10,r=40,t=10,b=30),xaxis_title="Avg allocation %",showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Cross-fund heatmap — Feb 2026")
    hs=tops[:12]; hd=[]
    for s in hs:
        rd={"Sector":s}
        for f in ALL_QUAL_FUNDS:
            fd=sd[(sd["Fund"]==f)&(sd["Sector"]==s)]
            rd[short_sc(f)]=round(fd.iloc[0]["Feb_26"],1) if len(fd)>0 and pd.notna(fd.iloc[0]["Feb_26"]) else 0
        hd.append(rd)
    hdf=pd.DataFrame(hd); fs=[short_sc(f) for f in ALL_QUAL_FUNDS]; z=hdf[fs].values
    fig2=go.Figure(data=go.Heatmap(z=z,x=fs,y=hdf["Sector"],colorscale=[[0,"#f8fafc"],[0.2,"#dbeafe"],[0.4,"#93c5fd"],[0.6,"#3b82f6"],[0.8,"#1d4ed8"],[1.0,"#1e3a5f"]],text=z,texttemplate="%{text:.1f}",textfont=dict(size=11),colorbar=dict(title="Alloc%",thickness=15)))
    fig2.update_layout(height=440,margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(tickangle=-45,tickfont=dict(size=11)),yaxis=dict(tickfont=dict(size=11),autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════
# LARGECAP SECTOR TABS
# ═══════════════════════════════════════════
def _lc_months(fd):
    ms=["Apr_25","May_25","Jun_25","Jul_25","Aug_25","Sep_25","Oct_25","Nov_25","Dec_25","Jan_26","Feb_26","Mar_26"]
    return [m for m in ms if m in fd.columns and fd[m].notna().any()]

def render_lc_sector_flow(sd):
    st.markdown("## LargeCap — Fund Sector Flow")
    sel=st.selectbox("Select Fund",LC_QUAL_FUNDS,format_func=short_lc,key="lcsf")
    fd=sd[sd["Fund"]==sel].copy(); st.markdown(f"### {short_lc(sel)}")
    av=_lc_months(fd)
    if not av: st.warning("No sector data."); return
    lt,ea=av[-1],av[0]
    fs=fd[fd[av].max(axis=1)>=1.5].sort_values(lt,ascending=False)
    if len(fs)==0: st.warning("No significant sectors."); return
    trows=[]
    for _,r in fs.iterrows():
        fv,lv=r[ea],r[lt]
        if pd.notna(fv) and pd.notna(lv):
            d=lv-fv
            if d>3:tr=f"↑↑ +{d:.1f}pp"
            elif d>1:tr=f"↑ +{d:.1f}pp"
            elif d<-3:tr=f"↓↓ {d:.1f}pp"
            elif d<-1:tr=f"↓ {d:.1f}pp"
            else:tr=f"→ {d:+.1f}pp"
        elif pd.notna(lv):tr="New"
        else:tr="—"
        row={"Sector":r["Sector"]}
        for m in av: row[m.replace("_"," 20")]=round(r[m],1) if pd.notna(r[m]) else None
        row["Trend"]=tr; trows.append(row)
    tdf=pd.DataFrame(trows); ad=[c for c in tdf.columns if c not in ("Sector","Trend")]
    st.dataframe(tdf.style.map(c_alloc,subset=ad).map(c_trend,subset=["Trend"]).format(na_rep="—")
        .set_properties(**{"text-align":"center","font-size":"13px"})
        .set_properties(subset=["Sector"],**{"text-align":"left","font-weight":"600"}),
        use_container_width=True,height=500,hide_index=True)
    st.markdown(f"#### Biggest moves ({ea.replace('_',' ')} → {lt.replace('_',' ')})")
    mv=[(r["Sector"],r[lt]-r[ea],r[lt]) for _,r in fs.iterrows() if pd.notna(r[ea]) and pd.notna(r[lt]) and abs(r[lt]-r[ea])>1.5]
    mv.sort(key=lambda x:-abs(x[1]))
    if mv:
        cols=st.columns(min(len(mv),5))
        for i,(s,d,c) in enumerate(mv[:5]):
            with cols[i]: st.metric(s,f"{c:.1f}%",f"{'+' if d>0 else ''}{d:.1f}pp")

def render_lc_consensus(sd):
    st.markdown("## LargeCap — Sector Consensus")
    av=_lc_months(sd)
    if not av: st.warning("No data."); return
    lt,ea=av[-1],av[0]
    tops=sd.groupby("Sector")[lt].mean().sort_values(ascending=False).head(15).index.tolist()
    cons=[]
    for s in tops:
        al,ch,cn=[],[],0
        for f in LC_QUAL_FUNDS:
            fd=sd[(sd["Fund"]==f)&(sd["Sector"]==s)]
            if len(fd)>0 and pd.notna(fd.iloc[0][lt]):
                al.append(fd.iloc[0][lt]);cn+=1
                if pd.notna(fd.iloc[0][ea]):ch.append(fd.iloc[0][lt]-fd.iloc[0][ea])
        aa=np.mean(al) if al else 0; ac=np.mean(ch) if ch else 0
        d="Strong Addition ↑↑" if ac>2 else "Adding ↑" if ac>0.5 else "Strong Reduction ↓↓" if ac<-2 else "Trimming ↓" if ac<-0.5 else "Stable →"
        cons.append({"Sector":s,"Avg Alloc%":round(aa,1),"Funds":cn,"Avg Change":round(ac,1),"Direction":d})
    cdf=pd.DataFrame(cons).sort_values("Avg Alloc%",ascending=True)
    bc=["#22c55e" if r["Avg Change"]>2 else "#86efac" if r["Avg Change"]>0.5 else "#ef4444" if r["Avg Change"]<-2 else "#fca5a5" if r["Avg Change"]<-0.5 else "#94a3b8" for _,r in cdf.iterrows()]
    fig=go.Figure(); fig.add_trace(go.Bar(y=cdf["Sector"],x=cdf["Avg Alloc%"],orientation="h",marker=dict(color=bc),text=[f"{v:.1f}%" for v in cdf["Avg Alloc%"]],textposition="outside"))
    fig.update_layout(height=420,margin=dict(l=10,r=40,t=10,b=30),xaxis_title="Avg allocation %",showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"#### Cross-fund heatmap — {lt.replace('_',' ')}")
    hs=tops[:12];hd=[]
    for s in hs:
        rd={"Sector":s}
        for f in LC_QUAL_FUNDS:
            fd=sd[(sd["Fund"]==f)&(sd["Sector"]==s)]
            rd[short_lc(f)]=round(fd.iloc[0][lt],1) if len(fd)>0 and pd.notna(fd.iloc[0][lt]) else 0
        hd.append(rd)
    hdf=pd.DataFrame(hd);fs=[short_lc(f) for f in LC_QUAL_FUNDS];z=hdf[fs].values
    fig2=go.Figure(data=go.Heatmap(z=z,x=fs,y=hdf["Sector"],colorscale=[[0,"#f8fafc"],[0.2,"#dbeafe"],[0.4,"#93c5fd"],[0.6,"#3b82f6"],[0.8,"#1d4ed8"],[1.0,"#1e3a5f"]],text=z,texttemplate="%{text:.1f}",textfont=dict(size=11),colorbar=dict(title="Alloc%",thickness=15)))
    fig2.update_layout(height=440,margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(tickangle=-45,tickfont=dict(size=11)),yaxis=dict(tickfont=dict(size=11),autorange="reversed"))
    st.plotly_chart(fig2,use_container_width=True)

# ═══════════════════════════════════════════
# LARGECAP PE & VALUATION MONITOR
# ═══════════════════════════════════════════
def render_lc_valuations(pe_data, tr_data):
    st.markdown("## LargeCap — PE & Valuation Monitor")
    st.markdown("Harmonic Mean PE, Dividend Yield, Turnover & Liquidity for the **15 selected funds**.")

    # Cross-fund snapshot
    st.markdown("#### All Selected Funds — Latest Snapshot")
    comp=[]
    for f in LC_QUAL_FUNDS:
        fp=pe_data[pe_data["Fund"]==f].sort_values("Date",ascending=False)
        if len(fp)==0: continue
        r=fp.iloc[0]
        ft=tr_data[tr_data["Fund"]==f].sort_values("Date",ascending=False) if len(tr_data)>0 else pd.DataFrame()
        comp.append({"Fund":short_lc(f),
            "PE (HM)":round(r["PE_HM"],1) if pd.notna(r["PE_HM"]) else None,
            "Div Yield%":round(r["DivYield"],2) if pd.notna(r["DivYield"]) else None,
            "Wt Avg MCAP (Cr)":round(r["MCAP_Cr"],0) if pd.notna(r["MCAP_Cr"]) else None,
            "Turnover%":round(ft.iloc[0]["Turnover"],0) if len(ft)>0 and pd.notna(ft.iloc[0]["Turnover"]) else None,
            "Liquidity (Days)":round(ft.iloc[0]["Liquidity"],2) if len(ft)>0 and pd.notna(ft.iloc[0]["Liquidity"]) else None})
    cdf=pd.DataFrame(comp).sort_values("PE (HM)")
    st.dataframe(cdf.style.map(c_pe,subset=["PE (HM)"]).map(c_turn,subset=["Turnover%"])
        .format({"PE (HM)":"{:.1f}x","Div Yield%":"{:.2f}","Wt Avg MCAP (Cr)":"{:,.0f}","Turnover%":"{:.0f}","Liquidity (Days)":"{:.2f}"},na_rep="—")
        .set_properties(**{"text-align":"center","font-size":"13px"})
        .set_properties(subset=["Fund"],**{"text-align":"left","font-weight":"600"}),
        use_container_width=True,height=560,hide_index=True)
    st.caption("PE: 🟢 <25x · 🟡 25-32x · 🔴 >32x  |  Turnover: 🟢 ≤30 · 🟡 ≤60 · 🟠 ≤100 · 🔴 >100")

    st.divider()

    # Per-fund monthly drill-down
    st.markdown("#### Fund Monthly Trend")
    sel=st.selectbox("Select Fund",LC_QUAL_FUNDS,format_func=short_lc,key="lcval")
    fp=pe_data[pe_data["Fund"]==sel].sort_values("Date",ascending=False).head(12)
    ft=tr_data[tr_data["Fund"]==sel].sort_values("Date",ascending=False).head(12) if len(tr_data)>0 else pd.DataFrame()
    if fp.empty: st.warning("No PE data."); return
    la=fp.iloc[0]
    la_tr=ft.iloc[0] if len(ft)>0 else pd.Series()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("PE (HM)",f"{la['PE_HM']:.1f}x" if pd.notna(la['PE_HM']) else "—")
    c2.metric("Div Yield",f"{la['DivYield']:.2f}%" if pd.notna(la['DivYield']) else "—")
    c3.metric("Turnover",f"{la_tr['Turnover']:.0f}%" if len(la_tr)>0 and pd.notna(la_tr.get('Turnover')) else "—")
    c4.metric("Liquidity",f"{la_tr['Liquidity']:.2f} days" if len(la_tr)>0 and pd.notna(la_tr.get('Liquidity')) else "—")
    rows=[]
    for _,r in fp.iterrows():
        row={"Month":r["Date"].strftime("%b %Y"),
             "PE (HM)":round(r["PE_HM"],1) if pd.notna(r["PE_HM"]) else None,
             "Div Yield%":round(r["DivYield"],2) if pd.notna(r["DivYield"]) else None,
             "MCAP (Cr)":round(r["MCAP_Cr"],0) if pd.notna(r["MCAP_Cr"]) else None}
        tm=ft[ft["Date"]==r["Date"]] if len(ft)>0 else pd.DataFrame()
        row["Turnover%"]=round(tm.iloc[0]["Turnover"],0) if len(tm)>0 and pd.notna(tm.iloc[0]["Turnover"]) else None
        row["Liquidity (Days)"]=round(tm.iloc[0]["Liquidity"],2) if len(tm)>0 and pd.notna(tm.iloc[0]["Liquidity"]) else None
        rows.append(row)
    tdf=pd.DataFrame(rows)
    st.dataframe(tdf.style.map(c_pe,subset=["PE (HM)"])
        .format({"PE (HM)":"{:.1f}x","Div Yield%":"{:.2f}","MCAP (Cr)":"{:,.0f}","Turnover%":"{:.0f}","Liquidity (Days)":"{:.2f}"},na_rep="—")
        .set_properties(**{"text-align":"center","font-size":"13px"})
        .set_properties(subset=["Month"],**{"text-align":"left","font-weight":"600"}),
        use_container_width=True,height=450,hide_index=True)
    cd=fp.sort_values("Date")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=cd["Date"],y=cd["PE_HM"],mode="lines+markers",line=dict(color="#3b82f6",width=2)))
    fig.update_layout(height=280,margin=dict(l=10,r=10,t=30,b=10),yaxis_title="PE (HM)",showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════
# LARGECAP STOCK HOLDINGS
# ═══════════════════════════════════════════
def render_lc_stocks(stock_data):
    st.markdown("## LargeCap — Stock Holdings")
    st.markdown("Stock-level allocation for the **selected funds** across time snapshots.")
    fwd=[f for f in LC_QUAL_FUNDS if f in stock_data["Fund"].unique()]
    if not fwd: st.warning("No stock data."); return
    sel=st.selectbox("Select Fund",fwd,format_func=short_lc,key="lcstk")
    fd=stock_data[stock_data["Fund"]==sel].copy()
    st.markdown(f"### {short_lc(sel)}")
    acs=["Mar_26","Feb_26","Aug_25","Jan_25","Jul_24"]
    lc=next((c for c in acs if fd[c].notna().any()),None)
    if lc is None: st.warning("No data."); return
    fd=fd.sort_values(lc,ascending=False)
    dm={"Mar_26":"Mar 2026","Feb_26":"Feb 2026","Aug_25":"Aug 2025","Jan_25":"Jan 2025","Jul_24":"Jul 2024"}
    aa=[c for c in acs if fd[c].notna().any()]
    tdf=fd[["Company","Sector"]+aa].copy(); tdf.columns=["Company","Sector"]+[dm[c] for c in aa]
    if len(aa)>=2:
        fc,llc=aa[-1],aa[0]
        tdf["Trend"]=[
            "—" if pd.isna(r[llc]) or pd.isna(r[fc])
            else f"↑ +{r[llc]-r[fc]:.1f}pp" if r[llc]-r[fc]>1
            else f"↓ {r[llc]-r[fc]:.1f}pp" if r[llc]-r[fc]<-1
            else f"→ {r[llc]-r[fc]:+.1f}pp"
            for _,r in fd.iterrows()]
    ad=[dm[c] for c in aa]
    styled=tdf.style.map(c_stock,subset=ad).format({c:"{:.2f}" for c in ad},na_rep="—")
    styled=styled.set_properties(**{"text-align":"center","font-size":"13px"})
    styled=styled.set_properties(subset=["Company"],**{"text-align":"left","font-weight":"600"})
    styled=styled.set_properties(subset=["Sector"],**{"text-align":"left"})
    if "Trend" in tdf.columns: styled=styled.map(c_trend,subset=["Trend"])
    st.dataframe(styled,use_container_width=True,height=700,hide_index=True)
    # Top 10 bar
    t10=fd.head(10)
    fig=go.Figure(); fig.add_trace(go.Bar(y=t10["Company"].iloc[::-1],x=t10[lc].iloc[::-1],orientation="h",marker=dict(color="#3b82f6"),text=[f"{v:.1f}%" for v in t10[lc].iloc[::-1]],textposition="outside"))
    fig.update_layout(height=350,margin=dict(l=10,r=40,t=10,b=10),xaxis_title="Allocation %",showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
    # Concentration
    st.markdown("#### Concentration")
    tot=len(fd[fd[lc].notna()])
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Top 5",f"{fd.head(5)[lc].sum():.1f}%")
    c2.metric("Top 10",f"{fd.head(10)[lc].sum():.1f}%")
    c3.metric("Top 20",f"{fd.head(20)[lc].sum():.1f}%")
    c4.metric("Total Stocks",f"{tot}")

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    with st.sidebar:
        st.subheader("🏛️ Compounder Weights")
        w1=st.slider("3Y Rolling CAGR",0,100,20); w2=st.slider("5Y Rolling CAGR",0,100,25)
        w3=st.slider("Upside Capture",0,100,15); w4=st.slider("Downside Capture",0,100,25)
        w5=st.slider("Ulcer Index",0,100,15)
        st.divider()
        st.subheader("🏎️ Momentum Weights")
        m6=st.slider("6M Dampened RA",0,100,50); m1=st.slider("1Y Dampened RA",0,100,50)
    we=[w1,w2,w3,w4,w5]; wm=[m6,m1]

    has_sc=os.path.exists("smallcapfinalrank.xlsx")
    has_lc=os.path.exists("largecap1.xlsx") and os.path.exists("largecap2.xlsx")
    if has_sc and has_lc: st.title("📊 Dual-Engine — SmallCap + LargeCap")
    elif has_sc: st.title("🚀 SmallCap Dual-Engine")
    elif has_lc: st.title("🏦 LargeCap Dual-Engine")
    else: st.error("No data files found."); return

    # Load enrichment data
    pe=load_lc_pe() if os.path.exists("pe.xlsx") else pd.DataFrame()
    tr=load_lc_turnover() if os.path.exists("portfolio_ratios.xlsx") else pd.DataFrame()
    stk=load_lc_stocks() if os.path.exists("stockalloacations.xlsx") else pd.DataFrame()
    scs=pd.DataFrame(); lcs=pd.DataFrame()
    if has_sc:
        try: scs=load_sc_sectors()
        except: pass
    if os.path.exists("sectorflows.xlsx"):
        try: lcs=load_lc_sectors()
        except: pass

    # Build tabs
    tl=[]
    if has_sc: tl.append("🚀 SC Rankings")
    if has_lc: tl.append("🏦 LC Rankings")
    if len(scs)>0: tl+= ["🔬 SC Sector Flow","🔎 SC Consensus"]
    if len(lcs)>0: tl+= ["🔬 LC Sector Flow","🔎 LC Consensus"]
    if len(pe)>0: tl.append("📈 LC Valuations")
    if len(stk)>0: tl.append("🏗️ LC Stock Holdings")

    tabs=st.tabs(tl); t=0

    if has_sc:
        with tabs[t]:
            sn,sf,sa,sb=load_sc_data()
            sr=compute_all(sn,sf).merge(sa,on="Fund",how="left")
            se,sm=rank_funds(sr,we,wm)
            st.caption(f"Benchmark: {sb}  ·  {len(sf)} funds  ·  Data through {sn['Date'].max().strftime('%d %b %Y')}")
            render_quant(se,sm,short_sc,has_aum=True)
        t+=1

    if has_lc:
        with tabs[t]:
            ln,lf,lb=load_lc_data()
            lr=compute_all(ln,lf)
            # MERGE PE + TURNOVER INTO RANKING DATA
            if len(pe)>0:
                pl=pe.sort_values("Date",ascending=False).groupby("Fund").first().reset_index()[["Fund","PE_HM"]]
                lr=lr.merge(pl,on="Fund",how="left")
            if len(tr)>0:
                trl=tr.sort_values("Date",ascending=False).groupby("Fund").first().reset_index()[["Fund","Turnover","Liquidity"]]
                lr=lr.merge(trl,on="Fund",how="left")
            le,lm=rank_funds(lr,we,wm)
            st.caption(f"Benchmark: {lb}  ·  {len(lf)} funds  ·  Data through {ln['Date'].max().strftime('%d %b %Y')}")
            render_quant(le,lm,short_lc,has_aum=False)
        t+=1

    if len(scs)>0:
        with tabs[t]: render_sc_sector_flow(scs)
        t+=1
        with tabs[t]: render_sc_consensus(scs)
        t+=1

    if len(lcs)>0:
        with tabs[t]: render_lc_sector_flow(lcs)
        t+=1
        with tabs[t]: render_lc_consensus(lcs)
        t+=1

    if len(pe)>0:
        with tabs[t]: render_lc_valuations(pe,tr)
        t+=1

    if len(stk)>0:
        with tabs[t]: render_lc_stocks(stk)

if __name__=="__main__":
    main()
