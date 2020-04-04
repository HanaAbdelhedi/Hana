import pandas as pd
import numpy as np 
import scipy
import scipy.stats

def drawdown(return_series: pd.Series):
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks= wealth_index.cummax()
    drawdowns= (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "wealth":wealth_index,
        "peaks": previous_peaks,
        "drawdown":drawdowns       
    })



def get_ffme_returns():
    me_m=pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                 header=0, index_col=0, parse_dates=True, 
                 na_values=-99.99
                )
    rets=me_m [['Lo 10', 'Hi 10']]
    rets.columns=['SmallCaps', 'LargeCaps']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets



def get_hifi_returns():
    """"
    Load and format the EDHEC hedge Fund Index returns
    """
    hfi=pd.read_csv("data/edhec-hedgefundindices.csv",
                    header=0, index_col=0, parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def get_ind_size():
    ind=pd.read_csv("data/ind30_m_size.csv",
                header=0, index_col=0, parse_dates=True)

    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    ind=pd.read_csv("data/ind30_m_nfirms.csv",
                header=0, index_col=0, parse_dates=True)

    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind



def get_ind_returns():
    ind=pd.read_csv("data/ind30_m_vw_rets.csv",
                header=0, index_col=0, parse_dates=True)/100

    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind



def semideviation(r):
    is_negative=r<0
    return r[is_negative].std(ddof=0)

def skewness(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    statistics,p_value=scipy.stats.jarque_bera(r)
    return p_value>level


def var_historic(r, level=5):
    """
    Value at risk using historic data- 
    The percentage chance that you will loose a certain amount of a given asset
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) # we changed the sign here
    else:
        raise TypeError("Expected to be a series or a data frame")
        
        

from scipy.stats import norm
def var_gaussian( r, level=5, modified=False):
    z= norm.ppf(level/100) 
    
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z=(z+
           (z**2-1)*s/6+
           (z**3-3*z)*(k-3)/24 -
           (2*z**3-5*z)*(s**2)/36
          )
        
    return (-(r.mean()+z*r.std(ddof=0)))



def cvar_historic(r, level=5):
    """
    CvAR is average of all those returns that are worse than VaR
    """
    if isinstance(r, pd.Series):
        is_beyond= r<= -var_historic(r, level=level) # give us a mask and added - sign in the var historic 
        return - r[is_beyond].mean() # we apply that mask on r and reconvert to a positive value 
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
        
        
def annualized_vol(r,periods_per_year):
    return r.std()*(periods_per_year**0.5)

def annualized_rets(r, periods_per_year):
    coumpounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return coumpounded_growth**(periods_per_year/n_periods)-1

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualized_rets(excess_ret, periods_per_year)
    ann_vol=annualized_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol
        
def portfolio_return(weights, returns):
    return weights.T@ returns

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5



def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plot 2 asset efficient Frontier 
    """
    if er.shape[0]!=2 or er.shape[0]!=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights=[np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets=[portfolio_return(w, er)for w in weights]
    vols=[portfolio_vol(w, cov)for w in weights]
    ef=pd.DataFrame({
        "Return":rets, "Volatility":vols
    })
    return ef.plot.line(x="Volatility", y="Return", style=style)


from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    from target retunr to weight
    """
    n=er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds=((0.0, 1.0),)*n
    
    return_is_target = {
        'type':'eq',
        'args':(er,),
        'fun': lambda weights, er:target_return - portfolio_return(weights, er)
       
    }
    weights_sum_to_1={
        'type':'eq',
        'fun':lambda weights:np.sum(weights) -1
    
    }
    weights=minimize(portfolio_vol, init_guess,
                     args=(cov,), method="SLSQP",
                     options={'disp': False},
                     constraints=(return_is_target, weights_sum_to_1),
                     bounds=bounds     
                    )
    return weights.x



def optimal_weights(n_points, er, cov):
    """
    generate a list of weights to run the optimizer on to optimize the vol
    """
    target_rs=np.linspace(er.min(), er.max(), n_points)
    weights=[minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate, er, cov):
    """
    Rteurns the weights of the portfolio that gives you the maximum sharpe ratio 
    given the riskfree rate and expected retunr and covariance matrix
    """
    n=er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds=((0.0, 1.0),)*n
    
    
    weights_sum_to_1={
        'type':'eq',
        'fun':lambda weights:np.sum(weights) -1
    
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio given weights
        """
        r=portfolio_return(weights, er)
        vol=portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
        
        
        
    
    weights=minimize(neg_sharpe_ratio, init_guess,
                     args=(riskfree_rate, er, cov,), method="SLSQP",
                     options={'disp': False},
                     constraints=(weights_sum_to_1),
                     bounds=bounds     
                    )
    return weights.x

def gmv(cov):
    """
    Returns weighst of the global minimum vol portfolio
    given the covariance matrix
    """
    n=cov.shape[0]
    return msr(0, np.repeat(1, n), cov)
    
def plot_ef(n_points, er, cov, style=".-", show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    weights= optimal_weights(n_points, er, cov)
    rets=[portfolio_return(w, er)for w in weights]
    vols=[portfolio_vol(w, cov)for w in weights]
    ef=pd.DataFrame({
        "Return":rets, "Volatility":vols
    })
    ax=ef.plot.line(x="Volatility", y="Return", style=style)
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n, n)
        r_ew=portfolio_return(w_ew, er)
        vol_ew=portfolio_vol(w_ew,cov)
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=12)
        
    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv, er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)
        
        
    if show_cml: 
        ax.set_xlim(left=0)
        rf=0.1
        w_msr=msr(riskfree_rate, er, cov)
        r_msr=portfolio_return(w_msr, er)
        vol_msr=portfolio_vol(w_msr, cov)
        #Add Capital Market Line
        cml_x=[0, vol_msr]
        cml_y=[riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)
    return ax

def get_tmi_rets():
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_mktcap=ind_nfirms*ind_size
    total_mktcap=ind_mktcap.sum(axis="columns")
    ind_capweight=ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return=(ind_capweight*ind_return).sum(axis="columns")
    total_market_index=drawdown(total_market_return).wealth
    tmi_rets=total_market_return
    return tmi_rets

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy , given a set of returns for the risky assets.
    Returns a dictionary containing: asset value histor, risk budget history, risky weight history
    """
    #Set up the CPPI Parameters
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=start*floor
    peak=start
    
    if isinstance(risky_r, pd.Series):
        risky_r=pd.DataFrame(risky_r, columns=["R"])
        
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=riskfree_rate/12
  
    account_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)
    for step in range(n_steps):
        if drawdown is not None:
            peak=np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        
        cushion=(account_value-floor_value)/account_value #my risk budget
        risky_w=cushion*m # how much money to allocate to the risky assets # no more than 100 because then we will leverage and we don't want that
        risky_w=np.minimum(risky_w,1) # not above 100
        risky_w=np.maximum(risky_w,0)# not bellow 0
        safe_w=1-risky_w 
        risky_allocation= account_value*risky_w
        safe_allocation=account_value*safe_w
        #update the account value
        account_value= risky_allocation*(1+risky_r.iloc[step])+ safe_allocation*(1+safe_r.iloc[step]) #iloc index when we have an integer
        #save the values so I can look at the history and plot it etc
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
    risky_wealth = start*(1+risky_r).cumprod() 
    backtest_result={
        "Wealth":account_history,
        "Risky Wealth":risky_wealth,
        "Risky Budget":cushion_history,
        "Risky Allocation":risky_w_history,
        "m":m,
        "start":start,
        "floor":floor,
        "risky_r":risky_r,
        "safe_r":safe_r     
        
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregate summary stats for the returns in the columns of r
    """
    ann_r=r.aggregate(annualized_rets, periods_per_year=12)
    ann_vol=r.aggregate(annualized_vol, periods_per_year=12)
    ann_sr=r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd=r.aggregate(lambda r:drawdown(r).drawdown.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5=r.aggregate(var_gaussian, modified=True)
    hist_cvar5=r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return":ann_r,
        "Annualized Vol":ann_vol,
        "Skewness":skew,
        "Kurtosis":kurt,
        "Cornish-fisher Var(5%)": cf_var5,
        "Historic CVaR(5%)": hist_cvar5,
        "Sharpe Ratio":ann_sr,
        "Max Drawdown":dd
               
    })



    
def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, prices=True, s_0=100.0):
    """
    Evolution of a stock price using a Geometric Brownian Motion Model
    """
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)
    rets_plus_1=np.random.normal(loc=(1+mu*dt), scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0]=1

    if prices: 
        return s_0*pd.DataFrame(rets_plus_1).cumprod()
    else:
        return pd.DataFrame(rets_plus_1-1) 
    
    
def discount(t,r):
   
    discounts=pd.DataFrame([(r+1)**-i for i in t])
    discounts.index=t 
    return discounts
   


def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index)and amounts
    r can be scalar, or a Series or DataFramewith the number of rows matching the num of rows in flows
    """
    dates=flows.index
    discounts=discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()



def  funding_ratio(assets, liabilities, r):
    """
    Compute the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets,r)/pv(liabilities,r)


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12): 
    """
    Returns a series of cash flow generated by a bond,
    indexed by a coupon number
    """
    n_coupons= round(maturity*coupons_per_year)
    coupon_amt=principal*coupon_rate/coupons_per_year
    coupon_times=np.arange(1, n_coupons+1)
    cash_flows=pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1]+= principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Compute the price of a bond that pays regular coupons until maturity 
    at which time the principal and the final coupon is returned
    This is not designed to be effective, rather, it is to illustrate the underlying principal behind bond pricing
    If discount rate is a DataFrame , then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the dicount_rate DataFrame is assumed to be the coupon number 
    """
     
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates=discount_rate.index
        prices=pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.iloc[t]= bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else:
        
        # base case... single time period 
        if maturity <=0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows=bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
         

def maucaulay_duration(flows, discount_rate):
    """
    Compute the mauculay duration of a sequence cash flow
    """
    discounted_flows=discount(flows.index, discount_rate)*flows
    weights=discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate): #chash flow of target, short bond, long bond
    """
    Returns the weight W in cf_s that, along with(1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t=maucaulay_duration(cf_t, discount_rate)
    d_s=maucaulay_duration(cf_s, discount_rate)
    d_l=maucaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s) #return the weight in the short bond

def inst_to_ann(r):
    """
    Convert short rate to annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert annualized to a short rate
    """
    return np.log1p(r)


import math 
def cir (n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate a random interest rate evolution over time using the CIR model 
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0=b
    r_0=ann_to_inst(r_0)
    dt=1/steps_per_year
    num_steps=int(n_years*steps_per_year)+1 # because n_years might be a float
    
    
    shock=np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates=np.empty_like(shock)
    rates[0]=r_0
    
    #For the Price Generation 
    h=math.sqrt(a**2+ 2*sigma**2)
    prices=np.empty_like(shock)
    ###
    
    def price(ttm, r):
        _A=((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B=(2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1) )
        _P=_A*np.exp(-_B*r)
        return _P
    prices[0]= price(n_years, r_0)
                 
            
    for step in range(1, num_steps):
        r_t=rates[step-1]
        d_r_t=(a*(b-r_t)*dt)+sigma*np.sqrt(r_t)*shock[step]
        rates[step]= abs(r_t+d_r_t)
        #generate prices at time t as well
        prices[step]=price(n_years-step*dt, rates[step])
            
    rates=pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ##ForPrices
    prices=pd.DataFrame(data=prices, index=range(num_steps))
    ##
    return rates, prices


def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Compute the total retunr of a Bond based on monthly bond prices and coupon payments 
    Assumes that dividends (coupons) are paid out at the end of the period(e.g. end of 3 months for quartely div)
    and that dividends are reinvested in the bond
    """
    coupons=pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max=monthly_prices.index.max()
    pay_date=np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date]=principal*coupon_rate/coupons_per_year
    total_returns=(monthly_prices+coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

    
    

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Returns T*N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape==r2.shape:
        raise ValueError(" r1 and r2 need to be the same shape")
    weights=allocator(r1,r2, **kwargs)# allocator will take r1 and r2 and a bunch of other things
    if not weights.shape==r1.shape:
        raise ValueError("Allocator returned weights that don't match r1")
    r_mix=weights*r1+(1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Return a t*N DataFrame of PSP weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def terminal value(rets):
    """
    Rturns the final value at the end of the return period for each scenario
    """
    return (rets+1).prod()

def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    terminal_wealth=(rets+1).prod()
    breach=terminal_wealth<foor #how many times did I end up bellow my floor
    reach=terminal_wealth>=cap
    p_breach=breach.mean() if breach.sum()>0 else np.nan
    p_reach=breach.mean if reach.sum()>0 else np.nan
    e_short =(floor-terminal_wealth[breach]).mean() if breach.sum>0 else np.nan
    e_surplus=(cap-terminal_wealth[reach]).mean()if reach.sum>0 else np.nan
    sum_stats=pd.DataFrame.from_dict({
        "mean":terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach":p_breach,
        "e_short":e_short,
        "p_reach":p_reach,
        "e_surplus":e_surplus     
    }, orient ="index", columns=[name])
    return sum_stats

def glidpath allocator( r1, r2, start_glide=1, end_glide=0):
    n_points= r1.shape[0]
    n_col=r1.shape[1]
    path= pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths.index=r1.index
    paths.columns=r1.columns
    return paths
