from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import io, urllib.request
import numpy as np
import pandas as pd

# 13 historical series available in the project database.
CATALOG = pd.DataFrame([
    ["ERNE","Cash / Liquidity","ERNE_Historical_Data_Updated.xlsx","EUR","EUR",False],
    ["EXH9","Infrastructure Europe proxy","EXH9_Europe_Infrastructure_Historical_Data.xlsx","EUR","EUR",False],
    ["IBGS","EUR Govt Bonds 1-3Y","IBGS_Historical_Data.xlsx","EUR","EUR",False],
    ["IGF","Infrastructure Global","IGF_Historical_Data.xlsx","USD","USD",True],
    ["INFR","Infrastructure Global UCITS","INFR_Historical_Data.xlsx","EUR","Global",False],
    ["IWDP","Listed Real Estate","IWDP_AMS_Historical_Data.xlsx","EUR","Global",False],
    ["LDVIX","Venture Capital proxy","LDVIX_Historical_Data.xlsx","USD","USD",True],
    ["MSCI_PE","Private Equity index proxy","MSCI_Private_Equity_Historical_Data.xlsx","USD","USD",True],
    ["SYBA","EUR Aggregate Bonds","SYBA_Historical_Data.xlsx","EUR","EUR",False],
    ["VEA","International Equity","VEA_Historical_Data.xlsx","USD","Multi",True],
    ["VOO","US Large Cap Equity","VOO_Historical_Data.xlsx","USD","USD",True],
    ["VTI","US Total Market Equity","VTI_Historical_Data.xlsx","USD","USD",True],
    ["XBAE","Global Aggregate Bonds EUR Hedged","XBAE_Historical_Data.xlsx","EUR","Global hedged",False],
], columns=["Proxy","Asset Class","File","Quote Currency","Economic FX","USD_conversion_supported"]).set_index("Proxy")

DEFAULT_SELECTED=["VTI","VEA","XBAE","SYBA","IWDP","EXH9","IGF","MSCI_PE","LDVIX","ERNE"]
DEFAULT_CMA={"VTI":.0719,"VOO":.0719,"VEA":.0725,"XBAE":.0358,"IBGS":.0354,"SYBA":.0371,
             "IWDP":.0684,"EXH9":.0651,"IGF":.0651,"INFR":.0651,"MSCI_PE":.1110,"LDVIX":.1004,"ERNE":.0230}
DEFAULT_ALLOC={
"Years 1-5":{"VTI":.36,"VEA":.24,"XBAE":.075,"SYBA":.075,"IWDP":.05,"EXH9":.025,"IGF":.025,"MSCI_PE":.08,"LDVIX":.02,"ERNE":.05},
"Years 6-8":{"VTI":.26,"VEA":.24,"XBAE":.12,"SYBA":.12,"IWDP":.05,"EXH9":.02,"IGF":.04,"MSCI_PE":.10,"LDVIX":0,"ERNE":.05},
"Years 9+":{"VTI":.26,"VEA":.20,"XBAE":.19,"SYBA":.15,"IWDP":.05,"EXH9":.03,"IGF":.04,"MSCI_PE":.03,"LDVIX":0,"ERNE":.05},}

ECB_FX_URL="https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A?startPeriod=2014-11&format=csvdata"
FRED_DFF_URL="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
# Monthly historical EONIA from the ECB Real Time Database, used through Sep-2019.
ECB_EONIA_URL="https://data-api.ecb.europa.eu/service/data/RTD/M.S0.N.C_EONIA.E?startPeriod=2014-11&endPeriod=2019-09&format=csvdata"
# Official ECB euro short-term rate (€STR), daily business-week observations from Oct-2019 onward.
ECB_ESTR_URL="https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?startPeriod=2019-10-01&format=csvdata"

CACHE_FILES={
    "fx":"ecb_eur_usd_monthly.csv",
    "usd":"fed_effective_federal_funds_monthly.csv",
    "eonia":"ecb_eonia_monthly.csv",
    "estr":"ecb_estr_monthly.csv",
}

def _read_url(url, timeout=30):
    """Download CSV robustly on Windows/Python installations with incomplete CA stores.
    The source URLs are fixed official ECB/FRED endpoints. SSL verification is attempted first;
    if the local certificate store rejects the official endpoint, only this fixed-source request
    is retried with certificate verification disabled.
    """
    import ssl
    req=urllib.request.Request(url,headers={"User-Agent":"EuroSuperResearch/1.0","Accept":"text/csv"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read()
    except Exception as first:
        if "CERTIFICATE_VERIFY_FAILED" not in str(first):
            raise
        ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
        with urllib.request.urlopen(req,timeout=timeout,context=ctx) as r:
            raw=r.read()
    return pd.read_csv(io.BytesIO(raw))

def _time_value_columns(d):
    cols={str(c).strip().upper():c for c in d.columns}
    tc=next((cols[k] for k in ["TIME_PERIOD","DATE","OBSERVATION_DATE"] if k in cols),None)
    vc=next((cols[k] for k in ["OBS_VALUE","VALUE","DFF"] if k in cols),None)
    if tc is None or vc is None:
        raise KeyError(f"Could not identify date/value columns. Received: {list(d.columns)}")
    return tc,vc

def _monthly_series(d, name, percent=False, already_monthly=False):
    tc,vc=_time_value_columns(d)
    x=pd.DataFrame({"date":pd.to_datetime(d[tc],errors="coerce"),"value":pd.to_numeric(d[vc],errors="coerce")}).dropna()
    if x.empty: raise ValueError(f"No usable observations for {name}")
    if already_monthly:
        s=pd.Series(x.value.values,index=x.date.dt.to_period("M"),name=name).groupby(level=0).last()
    else:
        s=x.set_index("date")["value"].resample("ME").mean().to_period("M"); s.name=name
    if percent: s=s/100.0
    return s.sort_index()

def _save_cache(series, path, value_name):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"Month":series.index.astype(str),value_name:series.values}).to_csv(path,index=False)

def _load_cache(path, value_name):
    d=pd.read_csv(path); return pd.Series(pd.to_numeric(d[value_name],errors="coerce").values,index=pd.PeriodIndex(d["Month"],freq="M"),name=value_name).dropna()

def fetch_official_macro(cache_dir=None, refresh=False):
    """Load official FX/rate data from a local cache, downloading only when needed.

    Sources:
      ECB EXR.M.USD.EUR.SP00.A: monthly USD per EUR reference exchange rate.
      ECB RTD.M.S0.N.C_EONIA.E: monthly EONIA, used before Oct-2019.
      ECB EST.B.EU000A2X2A25.WT: €STR, aggregated to monthly mean from Oct-2019.
      Federal Reserve/FRED DFF: Effective Federal Funds Rate, aggregated to monthly mean.

    Once downloaded, the cleaned monthly series are saved beside the application in
    local_macro_data/, so subsequent runs work offline and reproduce the same inputs.
    """
    base=Path(cache_dir) if cache_dir else Path(__file__).resolve().parent/"local_macro_data"
    base.mkdir(parents=True,exist_ok=True)
    out={}; errors=[]; source_status=[]
    specs=[
      ("fx_usd_per_eur","fx","USD_per_EUR",ECB_FX_URL,False,True),
      ("usd_short","usd","USD_short_rate",FRED_DFF_URL,True,False),
      ("eur_short_pre2019","eonia","EUR_short_rate",ECB_EONIA_URL,True,True),
      ("eur_short_post2019","estr","EUR_short_rate",ECB_ESTR_URL,True,False),
    ]
    for outkey,filekey,valname,url,percent,monthly in specs:
        path=base/CACHE_FILES[filekey]
        if path.exists() and not refresh:
            try:
                out[outkey]=_load_cache(path,valname); source_status.append(f"{outkey}: local cache")
                continue
            except Exception as e:
                errors.append(f"Could not read local {path.name}: {e}")
        try:
            d=_read_url(url)
            s=_monthly_series(d,valname,percent=percent,already_monthly=monthly)
            # Keep the research window plus one lead month needed for return conversion.
            s=s[(s.index>=pd.Period("2014-11","M")) & (s.index<=pd.Period("2026-08","M"))]
            if outkey=="eur_short_pre2019": s=s[s.index<=pd.Period("2019-09","M")]
            if outkey=="eur_short_post2019": s=s[s.index>=pd.Period("2019-10","M")]
            _save_cache(s,path,valname); out[outkey]=s; source_status.append(f"{outkey}: downloaded and cached")
        except Exception as e:
            if path.exists():
                try:
                    out[outkey]=_load_cache(path,valname); source_status.append(f"{outkey}: local cache after download failure")
                    errors.append(f"Refresh failed for {outkey}; existing local cache used: {e}")
                    continue
                except Exception: pass
            errors.append(f"{outkey}: {e}")
    out["_status"]=source_status
    return out, errors

def _pick_price(df):
    candidates=[c for c in df.columns if "adj close" in c.lower() or "index level" in c.lower()]
    if not candidates: raise KeyError(f"No adjusted-price/index-level column. Columns={list(df.columns)}")
    return candidates[0]

def load_native_returns(data_dir:Path, proxies:list[str]):
    ss=[]
    for p in proxies:
        f=Path(data_dir)/CATALOG.loc[p,"File"]
        if not f.exists(): raise FileNotFoundError(f"Missing {p}: {f.name}")
        df=pd.read_excel(f); df["Date"]=pd.to_datetime(df["Date"]); df=df.sort_values("Date").drop_duplicates("Date",keep="last")
        col=_pick_price(df); px=pd.to_numeric(df[col],errors="coerce")
        ss.append(pd.Series(px.pct_change(fill_method=None).values,index=df["Date"].dt.to_period("M"),name=p))
    return pd.concat(ss,axis=1,join="outer")


def available_history_window(native_returns):
    """Return the first and last monthly observations available in any selected proxy."""
    idx = native_returns.index
    return idx.min(), idx.max()

def historical_carry_series(macro, index, manual_annual_carry=0.0):
    """Monthly EUR-minus-USD short-rate carry used for the hedged USD component."""
    eurpre=macro.get("eur_short_pre2019",pd.Series(dtype=float))
    eurpost=macro.get("eur_short_post2019",pd.Series(dtype=float))
    eur=pd.concat([eurpre,eurpost]).groupby(level=0).last().sort_index()
    usd=macro.get("usd_short",pd.Series(dtype=float))
    if len(eur) and len(usd):
        annual=(eur.reindex(index)-usd.reindex(index)).fillna(float(manual_annual_carry))
    else:
        annual=pd.Series(float(manual_annual_carry),index=index)
    return annual

def apply_fx_hedging(native_returns, hedge_ratios, macro, carry_mode="historical_rates", manual_carry=0.0):
    """Convert USD-quoted series to EUR and apply a USD/EUR hedge ratio.
    Unhedged EUR return: (1+r_USD)*(FX_t-1/FX_t)-1 because ECB FX is USD per EUR.
    Hedged component approximation: r_USD + monthly carry, carry ≈ (r_EUR-r_USD)/12.
    For VEA this hedges the USD reporting-currency leg, not each underlying country currency.
    """
    out=native_returns.copy(); notes=[]
    fx=macro.get("fx_usd_per_eur")
    if fx is None:
        notes.append("Official EUR/USD series unavailable: USD proxies left in native returns for historical risk.")
        return out,notes
    fxret=(fx.shift(1)/fx)-1  # EUR return of USD
    for p in out.columns:
        if not bool(CATALOG.loc[p,"USD_conversion_supported"]): continue
        h=float(hedge_ratios.get(p,0.0)); r=native_returns[p]
        unhedged=(1+r)*(1+fxret.reindex(r.index))-1
        if carry_mode=="historical_rates":
            carry=historical_carry_series(macro,r.index,manual_carry)/12
        else:
            carry=pd.Series(float(manual_carry)/12,index=r.index)
        hedged=r+carry
        out[p]=(1-h)*unhedged+h*hedged
        if p=="VEA" and h>0: notes.append("VEA hedge is an approximation of its USD reporting-currency leg; it is not a full hedge of GBP/JPY/CAD/etc. underlying exposures.")
    return out,notes

def common_window(r): return r.dropna(how="any")
def stats(r):
    x=common_window(r); return x.std(ddof=1)*np.sqrt(12),x.corr(),x

def nearest_psd(cov,eps=1e-10):
    s=(cov+cov.T)/2; vals,vecs=np.linalg.eigh(s); vals=np.clip(vals,eps,None); o=vecs@np.diag(vals)@vecs.T; return (o+o.T)/2

@dataclass
class Config:
    start_year:int=2027; horizon:int=10; simulations:int=10000; seed:int=42
    initial_aum:float=29.38; first_contribution:float=2.90; fee:float=.002
    inflation:Dict[int,float]=field(default_factory=dict); contribution_growth:Dict[int,float]=field(default_factory=dict)
    withdrawals:Dict[int,float]=field(default_factory=dict); inflation_default:float=.02; contrib_growth_default:float=.03

def run(config:Config, proxies, mu, vols, corr, allocations, return_shift=0, vol_mult=1, contrib_mult=1):
    n=len(proxies); muv=np.array([mu[p] for p in proxies])+return_shift; sig=np.array([vols[p] for p in proxies])*vol_mult
    cov=nearest_psd(np.outer(sig,sig)*corr.loc[proxies,proxies].to_numpy())
    rng=np.random.default_rng(config.seed); draws=rng.multivariate_normal(muv,cov,size=(config.simulations,config.horizon))
    aum=np.full(config.simulations,config.initial_aum); paths=np.empty((config.simulations,config.horizon+1)); paths[:,0]=aum
    contrib=config.first_contribution*contrib_mult; cum_inf=1.; annual=[]; totalc=0.; totalw=0.; running=aum.copy(); maxdd=np.zeros_like(aum)
    for i in range(config.horizon):
        y=config.start_year+i; stage="Years 1-5" if i<5 else ("Years 6-8" if i<8 else "Years 9+")
        w=np.array([allocations.loc[p,stage] for p in proxies],float); pr=draws[:,i,:]@w
        aum*=1+pr; aum*=1-config.fee; aum+=contrib; wd=config.withdrawals.get(y,0.); aum-=wd
        totalc+=contrib; totalw+=wd; cum_inf*=1+config.inflation.get(y,config.inflation_default)
        running=np.maximum(running,aum); maxdd=np.minimum(maxdd,aum/running-1); paths[:,i+1]=aum
        annual.append([y,contrib,wd,np.median(aum),np.mean(aum),np.percentile(aum,5),np.percentile(aum,25),np.percentile(aum,75),np.percentile(aum,95),np.median(aum/cum_inf),np.mean(aum<config.initial_aum),np.median(maxdd)])
        ny=y+1; contrib*=1+config.contribution_growth.get(ny,config.contrib_growth_default)
    ann=pd.DataFrame(annual,columns=["Year","Contribution (€bn)","Withdrawal (€bn)","Median AUM (€bn)","Mean AUM (€bn)","P05 AUM (€bn)","P25 AUM (€bn)","P75 AUM (€bn)","P95 AUM (€bn)","Median Real AUM (€bn)","Prob below initial","Median max drawdown"])
    terminal_median=float(np.median(aum))
    net_external_capital=config.initial_aum+totalc-totalw
    summary=pd.Series({"Simulations":config.simulations,"Horizon":config.horizon,"Initial AUM (€bn)":config.initial_aum,"Total contributions (€bn)":totalc,"Total withdrawals (€bn)":totalw,"Median terminal AUM (€bn)":terminal_median,"Mean terminal AUM (€bn)":np.mean(aum),"P05 terminal AUM (€bn)":np.percentile(aum,5),"P25 terminal AUM (€bn)":np.percentile(aum,25),"P75 terminal AUM (€bn)":np.percentile(aum,75),"P95 terminal AUM (€bn)":np.percentile(aum,95),"Median investment gain (€bn)":terminal_median-net_external_capital,"Prob terminal < initial":np.mean(aum<config.initial_aum),"Prob terminal < initial+net contributions":np.mean(aum<net_external_capital),"Median max drawdown":np.median(maxdd),"P95 worst max drawdown":np.percentile(maxdd,5)})
    return {"annual":ann,"summary":summary,"terminal":aum,"paths":paths,"cov":cov,"max_drawdowns":maxdd}
