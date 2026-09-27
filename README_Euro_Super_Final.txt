EURO SUPER MONTE CARLO - FINAL FLEXIBLE VERSION

RUN
1. Install Python 3.11+
2. pip install -r requirements_euro_super_final.txt
3. Put the 13 supplied Excel files in the same folder as the two Python scripts.
4. streamlit run euro_super_app_final.py

WHAT IS INCLUDED
- All 13 project proxies: ERNE, EXH9, IBGS, IGF, INFR, IWDP, LDVIX, MSCI PE, SYBA, VEA, VOO, VTI, XBAE.
- Any 1-13 proxies can be selected. Portfolio weights must equal 100% in each allocation stage.
- Three editable allocation stages: years 1-5, 6-8, 9+.
- Editable CMA expected returns, risk matrix, macro cash flows, fees, withdrawals and stress scenarios.
- Historical FX normalization for USD-priced proxies using official ECB EUR/USD.
- Per-proxy FX hedge ratio from 0%-100% where USD conversion is supported.
- Historical hedge carry approximated by EUR short rate minus USD short rate.
- 10,000 simulations default, editable to 250,000; reproducible random seed.
- AUM percentiles, real AUM, drawdown, terminal distribution, probability metrics and CSV ZIP export.

FX FORMULAS
ECB publishes USD per EUR (X_t).
EUR return of one USD over a month = X_(t-1)/X_t - 1.
Unhedged EUR asset return = (1+r_USD)*(X_(t-1)/X_t)-1.
Approx. hedged return = r_USD + (r_EUR_short-r_USD_short)/12.
Partially hedged return = (1-h)*R_unhedged + h*R_hedged.

IMPORTANT MODELING CHOICE
The default JPM expected returns used in this project are already EUR-term capital-market assumptions. Therefore FX hedging changes the historical volatility/correlation estimate but does NOT automatically add carry to forward expected-return drift. This prevents double counting FX. If the team later switches to local-currency CMAs, the forward-drift treatment should be changed accordingly.

OFFICIAL EXTERNAL DATA SOURCES
- ECB EXR dataset / EUR foreign exchange reference rates: https://data.ecb.europa.eu/data/datasets/exr/data-information
- ECB exchange-rate methodology: https://data.ecb.europa.eu/methodology/exchange-rates
- ECB EONIA historical monthly series: https://data.ecb.europa.eu/data/datasets/FM/FM.M.U2.EUR.4F.MM.EONIA.HSTA
- ECB euro short-term rate (€STR): https://data.ecb.europa.eu/data/datasets/est/data-information
- Federal Reserve Bank of New York / Effective Federal Funds Rate via FRED: https://fred.stlouisfed.org/series/EFFR
- New York Fed SOFR methodology/data (reference only): https://www.newyorkfed.org/markets/reference-rates/sofr

PROJECT-SUPPLIED FORWARD-LOOKING SOURCES
- J.P. Morgan 2026 Long-Term Capital Market Assumptions, EUR page used as primary CMA mapping.
- BlackRock Capital Market Assumptions, August 2026 workbook, used as cross-check.
- PGIM 2026 Capital Market Assumptions, used as cross-check/dispersion source.

LIMITATIONS TO DISCLOSE
- EXH9 is a European utilities ETF used as a historical proxy for European infrastructure, not a pure private-infrastructure index.
- IWDP is listed real estate, while strategic institutional real estate may be private/core.
- MSCI PE and LDVIX are proxies, not direct private-market NAV histories.
- VEA is multi-currency underneath. Hedging its USD quote does not reproduce a full hedge of all underlying currencies.
- A quote currency is not the same thing as economic currency exposure.
