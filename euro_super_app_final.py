from pathlib import Path
import io, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from euro_super_engine_final import *

st.set_page_config(page_title="Euro Super Monte Carlo", page_icon="📊", layout="wide")
st.title("Euro Super Monte Carlo")
st.caption("Portfolio construction, currency hedging, historical risk estimation and 10-year Monte Carlo projections")
BASE=Path(__file__).resolve().parent

@st.cache_data(show_spinner=False)
def macro_data(refresh=False): return fetch_official_macro(refresh=refresh)

# ---------- 1. Simulation setup ----------
st.header("1. Simulation setup")
st.write("Define the projection period and the starting size of Euro Super. Monetary amounts below are in **billions of euros (€bn)**.")
c1,c2,c3,c4=st.columns(4)
with c1:
    sims=st.number_input("Number of Monte Carlo simulations",1_000,250_000,10_000,1_000,help="Number of simulated future portfolio paths. More simulations improve stability but take longer.")
with c2:
    start=st.number_input("Projection start year",2027,2100,2027,1)
with c3:
    horizon=st.number_input("Projection horizon (years)",1,40,10,1)
with c4:
    seed=st.number_input("Random seed",0,999999,42,1,help="Keeps results reproducible. Using the same seed and assumptions gives the same simulation draws.")

c1,c2,c3=st.columns(3)
with c1: initial=st.number_input("Initial FDC reserve (€bn)",0.,500.,29.38,.10,format="%.2f")
with c2: c0=st.number_input("First-year contributions (€bn)",0.,100.,2.90,.10,format="%.2f")
with c3:
    fee_pct=st.number_input("Annual management fee (%)",0.,5.,0.20,.05,format="%.2f",help="Applied each year to assets under management. 0.20 means 0.20%, not 20%.")
    fee=fee_pct/100

# ---------- 2. Portfolio universe ----------
st.header("2. Portfolio universe")
st.write("All **13 historical proxies** remain available. Select any subset. You can use 10 assets, add an 11th or more, replace a proxy, or remove one entirely.")
display_catalog=CATALOG[["Asset Class","Quote Currency","Economic FX"]].rename(columns={"Asset Class":"Asset class","Quote Currency":"Trading currency","Economic FX":"Main currency exposure"})
st.dataframe(display_catalog,use_container_width=True)
selected=st.multiselect("Assets included in the portfolio",list(CATALOG.index),default=DEFAULT_SELECTED,help="The allocation tables below will contain only the assets selected here.")
if not selected: st.warning("Select at least one asset to continue."); st.stop()

# ---------- 3. Historical sample and FX ----------
st.header("3. Historical data and currency hedging")
native_all=load_native_returns(BASE,selected)
avail_start,avail_end=available_history_window(native_all)
left,right=st.columns(2)
with left:
    hist_start=st.date_input("Historical analysis start date",value=avail_start.to_timestamp().date(),min_value=avail_start.to_timestamp().date(),max_value=avail_end.to_timestamp().date(),help="Volatility and correlations are estimated only over the selected historical window.")
with right:
    hist_end=st.date_input("Historical analysis end date",value=avail_end.to_timestamp().date(),min_value=avail_start.to_timestamp().date(),max_value=avail_end.to_timestamp().date())
if hist_start>hist_end: st.error("The historical start date must be before the end date."); st.stop()
start_p=pd.Period(hist_start,freq="M"); end_p=pd.Period(hist_end,freq="M")
native=native_all.loc[(native_all.index>=start_p)&(native_all.index<=end_p)]

refresh_macro=st.button("Refresh official currency and short-rate data", help="Normally unnecessary. The application stores cleaned monthly ECB and Federal Reserve data locally after the first successful download, so later runs can work offline.")
if refresh_macro:
    macro_data.clear()
macro,fetch_errors=macro_data(refresh=refresh_macro)
status=macro.get("_status",[])
if status:
    with st.expander("Currency and interest-rate data sources currently used"):
        st.write("The model first uses its local cached copies. A refresh downloads the official series again and replaces those local copies.")
        for item in status: st.write("• "+item)
if fetch_errors:
    st.warning("A source could not be refreshed. If a local cached copy exists, the model continues to use it. Details: "+" | ".join(fetch_errors))

st.subheader("Currency hedge settings")
st.write("For assets with a supported US-dollar exposure, choose what percentage of the position is protected against movements of the US dollar versus the euro. **0% = no currency hedge; 100% = fully hedged in the model.**")
hedge_default={p:(1.0 if p=="XBAE" else 0.0) for p in selected}
fxdf=pd.DataFrame(index=selected)
fxdf["Trading currency"]=[CATALOG.loc[p,"Quote Currency"] for p in selected]
fxdf["Currency exposure"]=[CATALOG.loc[p,"Economic FX"] for p in selected]
fxdf["Currency hedge (%)"]=[100*hedge_default[p] if bool(CATALOG.loc[p,"USD_conversion_supported"]) else 0.0 for p in selected]
fxdf=st.data_editor(fxdf,use_container_width=True,disabled=["Trading currency","Currency exposure"],column_config={"Currency hedge (%)":st.column_config.NumberColumn(min_value=0.,max_value=100.,step=5.,format="%.0f%%",help="Percentage of the asset position hedged against USD/EUR currency movements.")})
hedges=(fxdf["Currency hedge (%)"].astype(float)/100).to_dict()

carry_label=st.radio("How should the historical cost or benefit of currency hedging be estimated?",["Use historical EUR and USD short-term interest rates","Use one manual annual assumption"],horizontal=True,help="Currency hedging has a financing cost or benefit. The model approximates it using the difference between euro and US short-term interest rates.")
carry_mode="historical_rates" if carry_label.startswith("Use historical") else "manual"
manual_carry_pct=0.0
if carry_mode=="manual":
    manual_carry_pct=st.number_input("Manual annual hedging carry: euro rate minus US rate (%)",-20.,20.,0.,.25,format="%.2f",help="Example: -2.00% means the euro short-term rate is assumed to be 2 percentage points below the US short-term rate, creating an approximate 2% annual hedging cost before implementation frictions.")
manual_carry=manual_carry_pct/100

# Explain and visibly show that historical-rate selection changes the model.
if carry_mode=="historical_rates":
    cs=historical_carry_series(macro,native.index,manual_carry)
    usable=cs.dropna()
    if len(usable):
        x1,x2,x3=st.columns(3)
        x1.metric("Average historical annual hedging carry",f"{usable.mean():.2%}")
        x2.metric("Lowest annual hedging carry",f"{usable.min():.2%}")
        x3.metric("Highest annual hedging carry",f"{usable.max():.2%}")
        with st.expander("See the historical hedging carry used by the model"):
            carry_view=pd.DataFrame({"Month":usable.index.astype(str),"Annual EUR rate minus USD rate":usable.values})
            st.dataframe(carry_view,use_container_width=True,hide_index=True,column_config={"Annual EUR rate minus USD rate":st.column_config.NumberColumn(format="%.2%")})
    else: st.warning("Historical short-rate data are unavailable, so the manual fallback value will be used where needed.")
else:
    st.info(f"The model will use a constant annual hedging carry of {manual_carry:.2%} for the hedged portion of supported USD assets.")

with st.expander("How currency hedging is calculated"):
    st.markdown(r"""
For an unhedged USD asset, the return for a euro investor is calculated as:

\[1+R_{EUR}=(1+R_{USD})\times(FX_{t-1}/FX_t)\]

where the ECB exchange rate is US dollars per euro. For the hedged portion, the historical approximation is:

\[R_{hedged}\approx R_{USD}+(r_{EUR}-r_{USD})/12\]

The final historical return is the weighted combination of the hedged and unhedged portions. XBAE is already EUR-hedged by fund design, so the model does not hedge it a second time. For VEA, this is only an approximation because its underlying holdings span several currencies.
""")

eurret,fxnotes=apply_fx_hedging(native,hedges,macro,carry_mode,manual_carry)
vol,corr,hist=stats(eurret)
if hist.empty: st.error("No common monthly return observations remain for all selected assets in this date range. Choose a wider date range or fewer assets."); st.stop()
m1,m2,m3=st.columns(3); m1.metric("Common usable start",str(hist.index.min())); m2.metric("Common usable end",str(hist.index.max())); m3.metric("Common monthly observations",len(hist))
for n in sorted(set(fxnotes)): st.info(n)

# ---------- 4. Expected returns ----------
st.header("4. Forward-looking expected returns")
st.write("These are the annual expected returns used as the centre of the Monte Carlo simulation. Defaults follow the current project mapping to J.P. Morgan 2026 long-term capital market assumptions and can be edited.")
mu_df=pd.DataFrame({"Expected annual return (%)":[100*DEFAULT_CMA.get(p,.05) for p in selected]},index=selected)
mu_df=st.data_editor(mu_df,use_container_width=True,column_config={"Expected annual return (%)":st.column_config.NumberColumn(min_value=-50.,max_value=50.,step=.10,format="%.2f%%")})
mu=(mu_df["Expected annual return (%)"].astype(float)/100).to_dict()

# ---------- 5. Allocation ----------
st.header("5. Strategic asset allocation")
st.write("Enter portfolio weights as percentages. Each period must total **exactly 100%** before the simulation can run.")
a_pct=pd.DataFrame(0.,index=selected,columns=["Years 1 to 5 (%)","Years 6 to 8 (%)","Year 9 onward (%)"])
map_cols={"Years 1-5":"Years 1 to 5 (%)","Years 6-8":"Years 6 to 8 (%)","Years 9+":"Year 9 onward (%)"}
for oldcol,d in DEFAULT_ALLOC.items():
    for p,v in d.items():
        if p in a_pct.index: a_pct.loc[p,map_cols[oldcol]]=100*v
a_pct=st.data_editor(a_pct,use_container_width=True,column_config={c:st.column_config.NumberColumn(min_value=0.,max_value=100.,step=.5,format="%.1f%%") for c in a_pct.columns})

sums=a_pct.sum()
st.subheader("Allocation check")
cols=st.columns(3)
for col,box in zip(a_pct.columns,cols):
    total=float(sums[col]); delta=total-100
    box.metric(col.replace(" (%)",""),f"{total:.1f}%",delta=f"{delta:+.1f} percentage points" if abs(delta)>.001 else "Ready")
    box.progress(min(max(total/100,0),1))
valid=np.allclose(sums.values,100,atol=1e-6)
if valid: st.success("All three allocation periods total 100%. The portfolio is ready to simulate.")
else: st.error("At least one allocation period does not total 100%. Adjust the percentages above before running the simulation.")
a=a_pct/100; a.columns=["Years 1-5","Years 6-8","Years 9+"]

# ---------- 6. Risk ----------
st.header("6. Volatility and correlation assumptions")
risk_label=st.radio("Risk estimates used in the simulation",["Use historical volatility and correlations after EUR currency adjustment","Edit volatility and correlations manually"],horizontal=True)
if risk_label.startswith("Edit"):
    vdf=(vol*100).rename("Annualized volatility (%)").to_frame()
    vdf=st.data_editor(vdf,use_container_width=True,column_config={"Annualized volatility (%)":st.column_config.NumberColumn(min_value=0.,max_value=100.,step=.10,format="%.2f%%")})
    vol_use=vdf.iloc[:,0].astype(float)/100
    corr_use=st.data_editor(corr,use_container_width=True,column_config={c:st.column_config.NumberColumn(min_value=-1.,max_value=1.,step=.01,format="%.2f") for c in corr.columns}).astype(float)
else:
    vol_use=vol; corr_use=corr
    with st.expander("View historical volatility and correlation estimates"):
        st.dataframe((vol*100).rename("Annualized volatility (%)").to_frame(),use_container_width=True,column_config={"Annualized volatility (%)":st.column_config.NumberColumn(format="%.2f%%")})
        st.dataframe(corr,use_container_width=True)

# ---------- 7. Cash flows ----------
st.header("7. Contributions, inflation and withdrawals")
st.write("Edit the annual macroeconomic and cash-flow assumptions. Contribution and withdrawal amounts are in **billions of euros (€bn)**.")
years=list(range(int(start),int(start+horizon)))
inf=[.021,.019,.023]+[.020]*max(0,len(years)-3); cg=[.0363,.0394,.0445]+[.030]*max(0,len(years)-3)
macrodf=pd.DataFrame({"Year":years,"Inflation (%)":[100*x for x in inf[:len(years)]],"Annual contribution growth (%)":[100*x for x in cg[:len(years)]],"Withdrawal from portfolio (€bn)":[0.]*len(years)})
macrodf=st.data_editor(macrodf,use_container_width=True,disabled=["Year"],column_config={"Inflation (%)":st.column_config.NumberColumn(step=.1,format="%.2f%%"),"Annual contribution growth (%)":st.column_config.NumberColumn(step=.1,format="%.2f%%"),"Withdrawal from portfolio (€bn)":st.column_config.NumberColumn(min_value=0.,step=.1,format="%.2f")})

# ---------- 8. Scenarios ----------
st.header("8. Stress scenarios")
st.write("These settings describe how each scenario changes the central assumptions. They are **stress-test assumptions, not forecasts**.")
sc=pd.DataFrame({
    "Change in expected annual returns (percentage points)":[0.,1.5,-2.0],
    "Change in volatility (%)":[0.,-10.,25.],
    "Change in annual contributions (%)":[0.,10.,-10.]
},index=["Base case","Favorable case","Adverse case"])
sc=st.data_editor(sc,use_container_width=True,num_rows="dynamic",column_config={
    "Change in expected annual returns (percentage points)":st.column_config.NumberColumn(step=.25,format="%+.2f"),
    "Change in volatility (%)":st.column_config.NumberColumn(step=5.,format="%+.0f%%",help="Example: +25% means every asset's volatility is multiplied by 1.25."),
    "Change in annual contributions (%)":st.column_config.NumberColumn(step=5.,format="%+.0f%%",help="Example: -10% means contributions are multiplied by 0.90.")})
with st.expander("How to read these three scenario settings"):
    st.markdown("**Change in expected annual returns:** adds or subtracts percentage points from every asset's expected return.  **Change in volatility:** changes the size of market fluctuations relative to the baseline.  **Change in annual contributions:** changes the amount contributed to Euro Super relative to the baseline.")

# ---------- 9. Run ----------
st.header("9. Run the Monte Carlo simulation")
runbtn=st.button("Run Monte Carlo simulation",type="primary",disabled=not valid,use_container_width=True)
if runbtn:
    inflation=dict(zip(macrodf.Year.astype(int),(macrodf["Inflation (%)"].astype(float)/100)))
    growth=dict(zip(macrodf.Year.astype(int),(macrodf["Annual contribution growth (%)"].astype(float)/100)))
    withdrawals={int(y):float(w) for y,w in zip(macrodf.Year,macrodf["Withdrawal from portfolio (€bn)"]) if float(w)!=0}
    cfg=Config(int(start),int(horizon),int(sims),int(seed),float(initial),float(c0),float(fee),inflation,growth,withdrawals)
    results={}
    with st.spinner("Running simulations..."):
        for name,row in sc.iterrows():
            shift=float(row["Change in expected annual returns (percentage points)"])/100
            vol_mult=1+float(row["Change in volatility (%)"])/100
            contrib_mult=1+float(row["Change in annual contributions (%)"])/100
            results[name]=run(cfg,selected,mu,vol_use,corr_use,a,shift,vol_mult,contrib_mult)
    st.success("Simulation complete")

    # ---------- 10. Results dashboard ----------
    st.header("10. Monte Carlo results")
    st.caption("All monetary values are in billions of euros (€bn). Percentages are probabilities or drawdowns across the simulated paths.")

    summary=pd.DataFrame({k:v["summary"] for k,v in results.items()}).T
    base_name="Base case" if "Base case" in results else list(results)[0]
    base=results[base_name]
    bs=base["summary"]

    st.subheader("Key results: base case")
    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric("Median assets at horizon",f"€{bs['Median terminal AUM (€bn)']:.1f}bn")
    k2.metric("5th percentile",f"€{bs['P05 terminal AUM (€bn)']:.1f}bn",help="Only 5% of simulated terminal outcomes are below this value.")
    k3.metric("95th percentile",f"€{bs['P95 terminal AUM (€bn)']:.1f}bn",help="95% of simulated terminal outcomes are below this value.")
    k4.metric("Cumulative contributions",f"€{bs['Total contributions (€bn)']:.1f}bn")
    k5.metric("Median investment gain",f"€{bs['Median investment gain (€bn)']:.1f}bn",help="Median terminal assets minus initial reserve and net external contributions.")

    st.subheader("Range of possible portfolio outcomes")
    st.write("The centre line is the median simulation. The darker band contains the middle 50% of outcomes (25th to 75th percentile), while the wider band contains 90% of outcomes (5th to 95th percentile).")
    fan=base["annual"][["Year","P05 AUM (€bn)","P25 AUM (€bn)","Median AUM (€bn)","P75 AUM (€bn)","P95 AUM (€bn)"]].copy()
    fan["Year"]=fan["Year"].astype(str)
    outer=alt.Chart(fan).mark_area(opacity=.18).encode(x=alt.X("Year:O",title="Year"),y=alt.Y("P05 AUM (€bn):Q",title="Assets under management (€bn)"),y2="P95 AUM (€bn):Q",tooltip=["Year:O",alt.Tooltip("P05 AUM (€bn):Q",format=".1f"),alt.Tooltip("P95 AUM (€bn):Q",format=".1f")])
    inner=alt.Chart(fan).mark_area(opacity=.35).encode(x="Year:O",y="P25 AUM (€bn):Q",y2="P75 AUM (€bn):Q")
    median=alt.Chart(fan).mark_line(strokeWidth=3).encode(x="Year:O",y=alt.Y("Median AUM (€bn):Q"),tooltip=["Year:O",alt.Tooltip("Median AUM (€bn):Q",format=".1f")])
    st.altair_chart((outer+inner+median).properties(height=430),use_container_width=True)
    st.caption("Wide band: 5th–95th percentile | Inner band: 25th–75th percentile | Line: median")

    st.subheader("Scenario comparison")
    scenario_rows=[]
    for name,res in results.items():
        for _,r in res["annual"].iterrows(): scenario_rows.append({"Year":str(int(r["Year"])),"Scenario":name,"Median assets (€bn)":float(r["Median AUM (€bn)"])})
    scen_df=pd.DataFrame(scenario_rows)
    scen_chart=alt.Chart(scen_df).mark_line(point=True,strokeWidth=2.5).encode(x=alt.X("Year:O",title="Year"),y=alt.Y("Median assets (€bn):Q",title="Median assets under management (€bn)"),color=alt.Color("Scenario:N",title="Scenario"),tooltip=["Year:O","Scenario:N",alt.Tooltip("Median assets (€bn):Q",format=".1f")])
    st.altair_chart(scen_chart.properties(height=380),use_container_width=True)

    pick=st.selectbox("Scenario to inspect in detail",list(results),index=list(results).index(base_name) if base_name in results else 0)
    chosen=results[pick]; cs=chosen["summary"]

    st.subheader(f"Terminal outcome distribution: {pick}")
    st.write("Each bar counts how many Monte Carlo paths finish inside an asset-value range. A taller bar means that terminal range occurred more often in the simulation.")
    term_df=pd.DataFrame({"Terminal assets (€bn)":chosen["terminal"]})
    hist_chart=alt.Chart(term_df).mark_bar().encode(
        x=alt.X("Terminal assets (€bn):Q",bin=alt.Bin(maxbins=30),title="Terminal assets under management (€bn)"),
        y=alt.Y("count():Q",title="Number of simulations"),
        tooltip=[alt.Tooltip("count():Q",title="Simulations")]
    ).properties(height=380)
    st.altair_chart(hist_chart,use_container_width=True)

    st.subheader("Downside risk")
    d1,d2,d3,d4=st.columns(4)
    d1.metric("Probability below initial reserve",f"{cs['Prob terminal < initial']:.1%}",help=f"Probability that terminal assets are below the initial €{initial:.2f}bn reserve.")
    d2.metric("Probability below contributed capital",f"{cs['Prob terminal < initial+net contributions']:.1%}",help="Probability that terminal assets are below the initial reserve plus net cumulative contributions. This means investment growth did not fully preserve the externally supplied capital.")
    d3.metric("Median maximum drawdown",f"{cs['Median max drawdown']:.1%}",help="Median of the worst peak-to-trough loss experienced along each simulated path.")
    d4.metric("Severe-path maximum drawdown",f"{cs['P95 worst max drawdown']:.1%}",help="A severe downside drawdown threshold: only about 5% of simulated paths experience an even worse maximum drawdown.")

    st.subheader("Where terminal assets come from")
    source_df=pd.DataFrame({"Component":["Initial FDC reserve","Cumulative contributions","Median investment gain","Withdrawals"],"Amount (€bn)":[float(initial),float(cs["Total contributions (€bn)"]),float(cs["Median investment gain (€bn)"]),-float(cs["Total withdrawals (€bn)"])]})
    source_chart=alt.Chart(source_df).mark_bar().encode(x=alt.X("Component:N",sort=None,title=None),y=alt.Y("Amount (€bn):Q",title="Contribution to terminal assets (€bn)"),tooltip=["Component:N",alt.Tooltip("Amount (€bn):Q",format=".1f")])
    st.altair_chart(source_chart.properties(height=330),use_container_width=True)

    with st.expander("Detailed annual results"):
        detail=chosen["annual"].copy()
        pct_cols=["Prob below initial","Median max drawdown"]
        st.dataframe(detail,use_container_width=True,hide_index=True,column_config={c:st.column_config.NumberColumn(format="%.1%%") for c in pct_cols})

    with st.expander("Compare terminal statistics across all scenarios"):
        readable=summary[["Median terminal AUM (€bn)","P05 terminal AUM (€bn)","P95 terminal AUM (€bn)","Median investment gain (€bn)","Prob terminal < initial","Prob terminal < initial+net contributions","Median max drawdown"]].copy()
        st.dataframe(readable,use_container_width=True,column_config={"Prob terminal < initial":st.column_config.NumberColumn(format="%.1%%"),"Prob terminal < initial+net contributions":st.column_config.NumberColumn(format="%.1%%"),"Median max drawdown":st.column_config.NumberColumn(format="%.1%%")})

    st.info("PAYGO adequacy and liquidity-floor probabilities become meaningful once post-2037 withdrawals and a liquidity-floor rule are entered. The current dashboard does not invent those thresholds.")

    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("selected_proxies.csv",CATALOG.loc[selected].to_csv()); z.writestr("currency_hedging_policy.csv",fxdf.to_csv()); z.writestr("expected_returns.csv",mu_df.to_csv()); z.writestr("allocations.csv",a_pct.to_csv()); z.writestr("historical_volatility.csv",vol_use.to_csv()); z.writestr("historical_correlation.csv",corr_use.to_csv()); z.writestr("macro_cash_flows.csv",macrodf.to_csv(index=False)); z.writestr("stress_scenarios.csv",sc.to_csv())
        for k,v in results.items(): z.writestr(f"{k}_annual_results.csv",v["annual"].to_csv(index=False)); z.writestr(f"{k}_summary.csv",v["summary"].to_csv(header=False))
    st.download_button("Download all model outputs",bio.getvalue(),"euro_super_outputs.zip","application/zip",use_container_width=True)

with st.expander("Important methodology notes"):
    st.markdown("""
- All 13 proxies remain available, but only selected assets enter the portfolio and risk matrix.
- Expected returns are forward-looking assumptions; historical returns estimate volatility and correlations.
- XBAE is already EUR-hedged and is not hedged a second time.
- A fund trading in EUR is not automatically economically hedged to EUR.
- VEA's USD hedge is an approximation rather than a full hedge of every underlying currency.
- Listed real estate, infrastructure, private-equity and venture-capital proxies are historical modelling proxies and do not perfectly reproduce institutional private-market assets.
""")
