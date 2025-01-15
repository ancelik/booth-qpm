'''

This function download and clean all the data needed to run the PortfolioCrossSection script.

Chicago Booth course on Quantitative Portfolio Management
by Ralph S.J. Koijen and Sangmin S. Oh.

2023-09-19 : Initial Code

'''

# Import Packages
import pandas as pd
import numpy as np
import wrds
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import qpm_download

def cross_section_compact(_SAMPLE_START, _SAMPLE_END, _STRATEGY_NAME, signal_variables):
    
    # Establish connection with wrds
    db = wrds.Connection()
    
    ###############################################
    ## Step 1. Import Fundamentals from Compustat
    ###############################################
    print('Step 1. Import Fundamentals from Compustat')
    
    # Create list of variables to download
    variables_string = ', '.join(f'a.{vvv}' for vvv in signal_variables)
    
    # Define your SQL statement for Compustat data
    if len(signal_variables) > 0:
        sql_statement = f"""
        SELECT a.gvkey, a.datadate, a.at, a.ni, a.prcc_c, {variables_string}
        FROM COMP.FUNDA as a
        WHERE a.consol = 'C' AND a.popsrc = 'D' AND a.datafmt = 'STD' AND a.curcd = 'USD'
        AND a.indfmt = 'INDL' AND a.datadate >= '{_SAMPLE_START}' AND a.datadate <= '{_SAMPLE_END}'
        """
    else:
        sql_statement = f"""
        SELECT a.gvkey, a.datadate, a.at, a.ni, a.prcc_c
        FROM COMP.FUNDA as a
        WHERE a.consol = 'C' AND a.popsrc = 'D' AND a.datafmt = 'STD' AND a.curcd = 'USD'
        AND a.indfmt = 'INDL' AND a.datadate >= '{_SAMPLE_START}' AND a.datadate <= '{_SAMPLE_END}'
        """
    
    # Perform the query
    df_Compustat = db.raw_sql(sql_statement)
    
    # Require minimum information
    df_Compustat = df_Compustat[(df_Compustat['at'].notna()) & 
                                (df_Compustat['ni'].notna()) & 
                                (df_Compustat['prcc_c'].notna())]
    df_Compustat.drop(['ni', 'prcc_c'],axis=1, inplace=True)
    if (_STRATEGY_NAME != 'AssetGrowth') & (_STRATEGY_NAME != 'Quality'):
        df_Compustat.drop('at', axis=1, inplace=True)
    
    # Define your SQL statement for link dataset
    sql_statement = """
    SELECT a.gvkey, b.lpermno, b.linkdt, b.linkenddt
    FROM comp.names as a
    INNER JOIN crsp.ccmxpf_lnkhist as b
    ON a.gvkey = b.gvkey
    WHERE b.linktype in ('LC', 'LU')
    AND b.linkprim in ('P', 'C')
    ORDER BY a.gvkey
    """
    
    # Perform the query
    df_Link = db.raw_sql(sql_statement)
    
    # Rename variables
    df_Link = df_Link.rename(columns={'lpermno':'permno','linkdt':'StartDate','linkenddt':'EndDate'})
    
    # Reformat dates
    df_Compustat['datadate'] = pd.to_datetime(df_Compustat['datadate'])
    df_Link['StartDate'] = pd.to_datetime(df_Link['StartDate'])
    df_Link['EndDate'] = pd.to_datetime(df_Link['EndDate'])
    df_Link.loc[df_Link['EndDate'].isna(), 'EndDate'] = '2024-12-31'
    
    # Merge
    df_Compustat = pd.merge(df_Compustat, df_Link, on='gvkey', how='inner')
    
    print('Done')
    ###############################################
    ## Step 2. Adjust Fundamentals from Compustat
    ###############################################
    print('Step 2. Adjust Fundamentals from Compustat')
    
    # Restrict to observations with valid date
    df_Compustat = df_Compustat[(df_Compustat['StartDate'] <= df_Compustat['datadate']) & 
                                (df_Compustat['EndDate'] >= df_Compustat['datadate'])]
    df_Compustat = df_Compustat.drop(['StartDate','EndDate'], axis=1)
    
    # Convert gvkey to numeric
    df_Compustat['gvkey'] = pd.to_numeric(df_Compustat['gvkey'])
    
    # Assume 6-month reporting lag
    df_Compustat['ym'] = pd.PeriodIndex(df_Compustat.datadate, freq='M') + 6
    
    # Convert to monthly frequency
    df_Compustat['temp'] = 12
    df_Compustat = df_Compustat.loc[df_Compustat.index.repeat(df_Compustat['temp'])].reset_index(drop=True)
    df_Compustat.drop('temp', axis=1, inplace=True)
    
    # Roll over 
    df_Compustat['temp'] = 1
    df_Compustat = df_Compustat.sort_values(by=['gvkey','ym'])
    df_Compustat['cumcount'] = df_Compustat.groupby(['gvkey','ym'])['temp'].transform('cumsum')
    df_Compustat['ym'] = df_Compustat['ym'] + df_Compustat['cumcount'] - 1
    df_Compustat.drop(['temp','cumcount'], axis=1, inplace=True)

    # Last edits
    df_Compustat = df_Compustat.sort_values(by=['gvkey','ym','datadate'])
    df_Compustat = df_Compustat.groupby(['gvkey','ym']).last().reset_index()
    df_Compustat = df_Compustat.sort_values(by=['permno','ym','datadate'])
    df_Compustat = df_Compustat.groupby(['permno','ym']).last().reset_index()
    df_Compustat.drop('datadate', axis=1, inplace=True)
    
    print('Done')
    ###############################################
    ## Step 3. Import Returns and Factors
    ###############################################
    print('Step 3. Import Returns and Factors')
    
    # Define your SQL statement for monthly data
    sql_statement = """
    SELECT a.permno, b.ticker, a.date, a.ret, a.vol, 
           a.shrout, a.prc, b.shrcd, b.exchcd, c.dlstcd, c.dlret
    FROM crsp_m_stock.msf as a
    LEFT JOIN crsp_m_stock.msenames as b
    ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
    LEFT JOIN crsp_m_stock.msedelist as c
    ON a.permno=c.permno AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
    WHERE a.date >= '{}' AND a.date <= '{}'
    """
    
    # Perform the query
    df_CRSP = db.raw_sql(sql_statement.format(_SAMPLE_START, _SAMPLE_END))
    
    # Reformat date
    df_CRSP['date'] = pd.to_datetime(df_CRSP['date'])
    df_CRSP['ym'] = pd.PeriodIndex(df_CRSP.date, freq='M')
    df_CRSP = df_CRSP.drop('date', axis=1).reset_index(drop=True)
    
    # Adjust returns for delisting
    df_CRSP.loc[( df_CRSP['dlret'].isna() ) & ( (df_CRSP['dlstcd']==500) | ( (df_CRSP['dlstcd']>=520) & (df_CRSP['dlstcd']<=584) ) ) & ( (df_CRSP['exchcd']==1) | (df_CRSP['exchcd']==2) ), 'dlret'] = -0.35
    df_CRSP.loc[( df_CRSP['dlret'].isna() ) & ( (df_CRSP['dlstcd']==500) | ( (df_CRSP['dlstcd']>=520) & (df_CRSP['dlstcd']<=584) ) ) & (df_CRSP['exchcd']==3), 'dlret'] = -0.55
    df_CRSP.loc[df_CRSP['dlret']<-1, 'dlret'] = -1.0
    df_CRSP.loc[df_CRSP['dlret'].isna(), 'dlret'] = 0.0
    df_CRSP['ret'] = df_CRSP['ret'] + df_CRSP['dlret']
    df_CRSP.loc[(df_CRSP['ret'].isna()) & (df_CRSP['dlret']!=0.0), 'ret'] = df_CRSP['dlret']
    df_CRSP.drop(['dlret','dlstcd'], axis=1, inplace=True)
    
    # Convert units and construct market cap
    df_CRSP['shrout'] = df_CRSP['shrout']/1000
    df_CRSP['vol'] = df_CRSP['vol']/10000
    df_CRSP['me'] = df_CRSP['shrout']*abs(df_CRSP['prc'])
    
    # Retain only common shares traded on NASDAQ, NYSE and AMEX
    df_CRSP = df_CRSP[( (df_CRSP['shrcd'] == 10) | (df_CRSP['shrcd'] == 11) | (df_CRSP['shrcd'] == 12) ) & 
                      ( (df_CRSP['exchcd'] == 1) | (df_CRSP['exchcd'] == 2) | (df_CRSP['exchcd'] == 3) )]
    
    # Define your SQL statement for Fama-French factors
    sql_statement = """
    SELECT date, mktrf, smb, hml, rf, umd, rmw, cma
    FROM ff.fivefactors_monthly
    """
    
    # Perform the query
    df_FF = db.raw_sql(sql_statement)
    
    # Reformat date
    df_FF['date'] = pd.to_datetime(df_FF['date'])
    df_FF['ym'] = pd.PeriodIndex(df_FF.date, freq='M')
    df_FF.drop('date', axis=1, inplace=True)
    
    # Define your SQL statement for Market returns from CRSP
    sql_statement = """
    SELECT date, vwretd
    FROM crsp_m_stock.msi
    """
    
    # Perform the query
    df_Mkt = db.raw_sql(sql_statement)
    
    # Reformat date
    df_Mkt['date'] = pd.to_datetime(df_Mkt['date'])
    df_Mkt['ym'] = pd.PeriodIndex(df_Mkt.date, freq='M')
    df_Mkt.drop('date', axis=1, inplace=True)
    
    print('Done')
    ###############################################
    ## Step 4. Merge Datasets
    ###############################################
    print('Step 4. Merge Datasets')
    
    # Merge CRSP and Compustat
    df_full = pd.merge(df_CRSP, df_Compustat, on=['permno','ym'], how='inner', validate='1:1')

    # Merge master dataset with factors
    df_full = pd.merge(df_full, df_FF, on='ym', how='inner', validate='m:1')
    df_full = pd.merge(df_full, df_Mkt, on='ym', how='inner', validate='m:1')
    
    print('Done')
    ###############################################
    ## Step 5. Compute Rolling Beta if Necessary and Last Edits
    ###############################################
    print('Step 5. Compute Rolling Beta and Last Edits')
    
    if _STRATEGY_NAME == 'Quality':
        df_full = qpm_download.rolling_betas(df_full)

    # Rename and construct last variables depending on the strategy
    df_full = df_full.rename(columns={'ym':'ldate'})
    df_full = df_full.rename(columns={'ret':'daret'})
    if _STRATEGY_NAME == 'Value':
        df_full = df_full.rename(columns={'ceq':'be'})
    elif _STRATEGY_NAME == 'Quality':
        df_full['profitA'] = (df_full['revt']-df_full['cogs'])/df_full['at']
    
    # Reformat date
    df_full['ldate'] = df_full['ldate'].dt.to_timestamp()

    print('Done')
    ###############################################
    ## Step 6. Lag Market Cap
    ###############################################
    print('Step 6. Lag Market Cap')
    
    df_full.drop_duplicates(subset = ['permno', 'ldate'], keep = 'first', inplace = True)
    df_full.sort_values(by = ['permno', 'ldate'], inplace = True)

    df_full['ldate_lag'] = df_full.groupby(['permno'])['ldate'].shift(1)
    df_full['screen'] = (df_full['ldate_lag'] == df_full['ldate'] - pd.DateOffset(months=1)).astype(int).replace(0, np.nan)

    df_full['me_lagged'] = df_full.groupby(['permno'])['me'].shift(1).multiply(df_full['screen'])
    df_full.drop(['ldate_lag','screen'], axis=1, inplace=True)
    
    # Save Fama-French Data
    df_full[['ldate', 'rf', 'mktrf', 'smb', 'hml', 'umd', 'rmw', 'cma']].drop_duplicates().to_parquet('FFData.parquet')
    
    print('Done')
    
    return df_full

def cross_section(_SAMPLE_START, _SAMPLE_END):
    
    # Establish connection with wrds
    db = wrds.Connection()
    
    ###############################################
    ## Step 1. Import Fundamentals from Compustat
    ###############################################
    print('Step 1. Import Fundamentals from Compustat')
    
    # Define your SQL statement for Compustat data
    sql_statement = """
    SELECT a.gvkey, a.datadate, a.conm, a.fyear, a.at, 
           a.prcc_c, a.ni, a.ceq, a.revt, a.cogs
    FROM COMP.FUNDA as a
    WHERE a.consol = 'C' AND a.popsrc = 'D' AND a.datafmt = 'STD' AND a.curcd = 'USD'
    AND a.indfmt = 'INDL' AND a.datadate >= '{}' AND a.datadate <= '{}'
    """
    
    # Perform the query
    df_Compustat = db.raw_sql(sql_statement.format(_SAMPLE_START, _SAMPLE_END))
    
    # Require minimum information
    df_Compustat = df_Compustat[(df_Compustat['at'].notna()) & 
                                (df_Compustat['ni'].notna()) & 
                                (df_Compustat['prcc_c'].notna())]
    
    # Define your SQL statement for link dataset
    sql_statement = """
    SELECT a.gvkey, b.lpermno, b.linkdt, b.linkenddt
    FROM comp.names as a
    INNER JOIN crsp.ccmxpf_lnkhist as b
    ON a.gvkey = b.gvkey
    WHERE b.linktype in ('LC', 'LU')
    AND b.linkprim in ('P', 'C')
    ORDER BY a.gvkey
    """
    
    # Perform the query
    df_Link = db.raw_sql(sql_statement)
    
    # Rename variables
    df_Link = df_Link.rename(columns={'lpermno':'permno','linkdt':'StartDate','linkenddt':'EndDate'})
    
    # Reformat dates
    df_Compustat['datadate'] = pd.to_datetime(df_Compustat['datadate'])
    df_Link['StartDate'] = pd.to_datetime(df_Link['StartDate'])
    df_Link['EndDate'] = pd.to_datetime(df_Link['EndDate'])
    df_Link.loc[df_Link['EndDate'].isna(), 'EndDate'] = '2024-12-31'
    
    # Merge
    df_Compustat = pd.merge(df_Compustat, df_Link, on='gvkey', how='inner')
    
    print('Done')
    ###############################################
    ## Step 2. Adjust Fundamentals from Compustat
    ###############################################
    print('Step 2. Adjust Fundamentals from Compustat')
        
    # Restrict to observations with valid date
    df_Compustat = df_Compustat[(df_Compustat['StartDate'] <= df_Compustat['datadate']) & 
                                (df_Compustat['EndDate'] >= df_Compustat['datadate'])]
    df_Compustat = df_Compustat.drop(['StartDate','EndDate'], axis=1)
    
    # Convert gvkey to numeric
    df_Compustat['gvkey'] = pd.to_numeric(df_Compustat['gvkey'])
    
    # Assume 6-month reporting lag
    df_Compustat['ym'] = pd.PeriodIndex(df_Compustat.datadate, freq='M') + 6
    
    # Convert to monthly frequency
    df_Compustat['temp'] = 12
    df_Compustat = df_Compustat.loc[df_Compustat.index.repeat(df_Compustat['temp'])].reset_index(drop=True)
    df_Compustat.drop('temp', axis=1, inplace=True)
    
    # Roll over 
    df_Compustat['temp'] = 1
    df_Compustat = df_Compustat.sort_values(by=['gvkey','ym'])
    df_Compustat['cumcount'] = df_Compustat.groupby(['gvkey','ym'])['temp'].transform('cumsum')
    df_Compustat['ym'] = df_Compustat['ym'] + df_Compustat['cumcount'] - 1
    df_Compustat.drop(['temp','cumcount'], axis=1, inplace=True)

    # Last edits
    df_Compustat = df_Compustat.sort_values(by=['gvkey','ym','datadate'])
    df_Compustat = df_Compustat.groupby(['gvkey','ym']).last().reset_index()
    df_Compustat = df_Compustat.sort_values(by=['permno','ym','datadate'])
    df_Compustat = df_Compustat.groupby(['permno','ym']).last().reset_index()
    
    print('Done')
    ###############################################
    ## Step 3. Import Fundamentals from Trucost
    ###############################################
    print('Step 3. Import Fundamentals from Trucost')


    # Define your SQL statement for Trucost ESG scores
    sql_statement = """
    SELECT scoredate, scorevalue, institutionid, aspectname
    FROM TRUCOST.WRDS_ESG
    WHERE aspectname in ('Environmental Dimension', 'S&P Global ESG Score',
                         'Economic Governance Dimension', 'Social Dimension')
    AND csascoretypename = 'Modeled'
    """

    # Perform the query
    df_Scores = db.raw_sql(sql_statement)

    # Require minimum information
    df_Scores = df_Scores[(df_Scores['scoredate'].notna()) & 
                          (df_Scores['scorevalue'].notna())]

    # Reformat date
    df_Scores['scoredate'] = pd.to_datetime(df_Scores['scoredate'])

    # Define your SQL statement for Trucost carbon intensity
    sql_statement = """
    SELECT institutionid, periodenddate, di_319407
    FROM TRUCOST.WRDS_ENVIRONMENT
    """

    # Perform the query
    df_CI = db.raw_sql(sql_statement)

    # Require minimum information
    df_CI = df_CI[(df_CI['periodenddate'].notna()) & 
                  (df_CI['di_319407'].notna())]

    # Reformat date
    df_CI['periodenddate'] = pd.to_datetime(df_CI['periodenddate'])

    # Define your SQL statement for firms' identifiers
    sql_statement = """
    SELECT gvkey, institutionid
    FROM trucost.wrds_companies
    """
    # Perform the query
    df_ID = db.raw_sql(sql_statement)

    # Drop duplicates
    df_ID = df_ID.drop_duplicates(subset = ['institutionid'], keep = False)

    print('Done')
    ###############################################
    ## Step 4. Adjust Fundamentals from Trucost
    ###############################################
    print('Step 4. Adjust Fundamentals from Trucost')

    # Rename scores
    df_Scores.loc[df_Scores['aspectname'] == 'S&P Global ESG Score', 'aspectname'] = 'ESG_score'
    df_Scores.loc[df_Scores['aspectname'] == 'Environmental Dimension', 'aspectname'] = 'E_score'
    df_Scores.loc[df_Scores['aspectname'] == 'Social Dimension', 'aspectname'] = 'S_score'
    df_Scores.loc[df_Scores['aspectname'] == 'Economic Governance Dimension', 'aspectname'] = 'G_score'

    # Reshape dataset
    df_Scores = df_Scores.pivot(index=['scoredate','institutionid'], columns='aspectname', values='scorevalue')
    df_Scores = df_Scores.reset_index()

    # Merge with firms' identifiers
    df_Scores = pd.merge(df_Scores, df_ID, on='institutionid', how='outer', validate='m:1')
    df_Scores = pd.merge(df_Scores, df_Link, on='gvkey', how='inner')

    # Restrict to observations with valid date
    df_Scores = df_Scores[(df_Scores['StartDate'] <= df_Scores['scoredate']) & 
                          (df_Scores['EndDate'] >= df_Scores['scoredate'])]
    df_Scores = df_Scores.drop(['StartDate','EndDate'], axis=1)

    # Keep variables of interest
    df_Scores = df_Scores[['permno','scoredate','ESG_score','E_score','S_score','G_score']]
    df_Scores = df_Scores.sort_values(by=['permno','scoredate'])

    # Convert to monthly
    df_Scores.set_index('scoredate', inplace=True)
    df_Scores = df_Scores.groupby('permno').resample('M').ffill()
    df_Scores = df_Scores.drop('permno', axis=1).reset_index()

    # Minor adjustments
    df_Scores['ldate'] = pd.PeriodIndex(df_Scores.scoredate, freq='M')
    df_Scores = df_Scores[['permno','ldate','ESG_score','E_score','S_score','G_score']]
    df_Scores = df_Scores.sort_values(by=['permno','ldate'])
    df_Scores = df_Scores.groupby(['permno','ldate']).last().reset_index()

    # Rename carbon intensity
    df_CI = df_CI.rename(columns={'di_319407':'carbon_intensity'})

    # Merge with firms' identifiers
    df_CI = pd.merge(df_CI, df_ID, on='institutionid', how='outer', validate='m:1')
    df_CI = pd.merge(df_CI, df_Link, on='gvkey', how='inner')

    # Restrict to observations with valid date
    df_CI = df_CI[(df_CI['StartDate'] <= df_CI['periodenddate']) & 
                  (df_CI['EndDate'] >= df_CI['periodenddate'])]
    df_CI = df_CI.drop(['StartDate','EndDate'], axis=1)

    # Keep variables of interest
    df_CI = df_CI[['permno','periodenddate','carbon_intensity']]
    df_CI = df_CI.sort_values(by=['permno','periodenddate'])
    df_CI = df_CI.groupby(['permno','periodenddate']).last().reset_index()

    # Convert to monthly
    df_CI.set_index('periodenddate', inplace=True)
    df_CI = df_CI.groupby('permno').resample('M').ffill()
    df_CI = df_CI.drop('permno', axis=1).reset_index()

    # Minor adjustments
    df_CI['ldate'] = pd.PeriodIndex(df_CI.periodenddate, freq='M')
    df_CI = df_CI[['permno','ldate','carbon_intensity']]
    df_CI = df_CI.sort_values(by=['permno','ldate'])
    df_CI = df_CI.groupby(['permno','ldate']).last().reset_index()

    # Merge
    df_ESG = pd.merge(df_CI, df_Scores, on=['permno','ldate'], how='outer')
    df_ESG = df_ESG[['permno','ldate','ESG_score','E_score',
                     'S_score','G_score','carbon_intensity']]

    print('Done')
    ###############################################
    ## Step 5. Import Returns and Factors
    ###############################################
    print('Step 5. Import Returns and Factors')
        
    # Define your SQL statement for monthly data
    sql_statement = """
    SELECT a.permno, b.ticker, a.date, a.ret, a.retx, a.vol, 
           a.shrout, a.prc, b.shrcd, b.exchcd, b.comnam, c.dlstcd, c.dlret
    FROM crsp_m_stock.msf as a
    LEFT JOIN crsp_m_stock.msenames as b
    ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
    LEFT JOIN crsp_m_stock.msedelist as c
    ON a.permno=c.permno AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
    WHERE a.date >= '{}' AND a.date <= '{}'
    """
    
    # Perform the query
    df_CRSP = db.raw_sql(sql_statement.format(_SAMPLE_START, _SAMPLE_END))
    
    # Reformat date
    df_CRSP['date'] = pd.to_datetime(df_CRSP['date'])
    df_CRSP['ym'] = pd.PeriodIndex(df_CRSP.date, freq='M')
    df_CRSP = df_CRSP.drop('date', axis=1).reset_index(drop=True)
    
    # Adjust returns for delisting
    df_CRSP.loc[( df_CRSP['dlret'].isna() ) & ( (df_CRSP['dlstcd']==500) | ( (df_CRSP['dlstcd']>=520) & (df_CRSP['dlstcd']<=584) ) ) & ( (df_CRSP['exchcd']==1) | (df_CRSP['exchcd']==2) ), 'dlret'] = -0.35
    df_CRSP.loc[( df_CRSP['dlret'].isna() ) & ( (df_CRSP['dlstcd']==500) | ( (df_CRSP['dlstcd']>=520) & (df_CRSP['dlstcd']<=584) ) ) & (df_CRSP['exchcd']==3), 'dlret'] = -0.55
    df_CRSP.loc[df_CRSP['dlret']<-1, 'dlret'] = -1.0
    df_CRSP.loc[df_CRSP['dlret'].isna(), 'dlret'] = 0.0
    df_CRSP['ret'] = df_CRSP['ret'] + df_CRSP['dlret']
    df_CRSP.loc[(df_CRSP['ret'].isna()) & (df_CRSP['dlret']!=0.0), 'ret'] = df_CRSP['dlret']
    
    # Convert units and construct market cap
    df_CRSP['shrout'] = df_CRSP['shrout']/1000
    df_CRSP['vol'] = df_CRSP['vol']/10000
    df_CRSP['mve_c'] = df_CRSP['shrout']*abs(df_CRSP['prc'])
    
    # Retain only common shares traded on NASDAQ, NYSE and AMEX
    df_CRSP = df_CRSP[( (df_CRSP['shrcd'] == 10) | (df_CRSP['shrcd'] == 11) | (df_CRSP['shrcd'] == 12) ) & 
                      ( (df_CRSP['exchcd'] == 1) | (df_CRSP['exchcd'] == 2) | (df_CRSP['exchcd'] == 3) )]
    
    # Define your SQL statement for Fama-French factors
    sql_statement = """
    SELECT date, mktrf, smb, hml, rf, umd, rmw, cma
    FROM ff.fivefactors_monthly
    """
    
    # Perform the query
    df_FF = db.raw_sql(sql_statement)
    
    # Reformat date
    df_FF['date'] = pd.to_datetime(df_FF['date'])
    df_FF['ym'] = pd.PeriodIndex(df_FF.date, freq='M')
    df_FF.drop('date', axis=1, inplace=True)
    
    # Define your SQL statement for Market returns from CRSP
    sql_statement = """
    SELECT date, vwretd
    FROM crsp_m_stock.msi
    """
    
    # Perform the query
    df_Mkt = db.raw_sql(sql_statement)
    
    # Reformat date
    df_Mkt['date'] = pd.to_datetime(df_Mkt['date'])
    df_Mkt['ym'] = pd.PeriodIndex(df_Mkt.date, freq='M')
    df_Mkt.drop('date', axis=1, inplace=True)
    
    print('Done')
    ###############################################
    ## Step 6. Merge Datasets
    ###############################################
    print('Step 6. Merge Datasets')
    
    # Merge CRSP and Compustat
    df_full = pd.merge(df_CRSP, df_Compustat, on=['permno','ym'], how='inner', validate='1:1')

    # Merge master dataset with factors
    df_full = pd.merge(df_full, df_FF, on='ym', how='inner', validate='m:1')
    df_full = pd.merge(df_full, df_Mkt, on='ym', how='inner', validate='m:1')
    
    print('Done')
    ###############################################
    ## Step 7. Compute Rolling Beta and Last Edits
    ###############################################
    print('Step 7. Compute Rolling Beta and Last Edits')

    df_full = qpm_download.rolling_betas(df_full)

    # Rename and construct last variables
    df_full = df_full.rename(columns={'ym':'ldate'})
    df_full = df_full.rename(columns={'mve_c':'me'})
    df_full = df_full.rename(columns={'ret':'daret'})
    df_full = df_full.rename(columns={'ceq':'be'})
    df_full['profitA'] = (df_full['revt']-df_full['cogs'])/df_full['at']

    # Merge master dataset with ESG data
    df_full = pd.merge(df_full, df_ESG, on=['permno','ldate'], how='left', validate='1:1')
    
    # Reformat date
    df_full['ldate'] = df_full['ldate'].dt.to_timestamp()
    
    # Restrict to variables of interest
    df_full = df_full[['permno','ticker','conm','retx','vwretd','mktrf','smb','hml','rf','umd','rmw',
                       'cma','ldate','me','be','daret','vol','shrout','prc','shrcd','exchcd',
                       'revt','cogs','at','beta','profitA','ESG_score','E_score','S_score','G_score','carbon_intensity']]

    print('Done')
    ###############################################
    ## Step 8. Lag Market Cap
    ###############################################
    print('Step 8. Lag Market Cap')
    
    df_full.drop_duplicates(subset = ['permno', 'ldate'], keep = 'first', inplace = True)
    df_full.sort_values(by = ['permno', 'ldate'], inplace = True)

    df_full['ldate_lag'] = df_full.groupby(['permno'])['ldate'].shift(1)
    df_full['screen'] = (df_full['ldate_lag'] == df_full['ldate'] - pd.DateOffset(months=1)).astype(int).replace(0, np.nan)

    df_full['me_lagged'] = df_full.groupby(['permno'])['me'].shift(1).multiply(df_full['screen'])
    
    # Save Fama-French Data
    df_full[['ldate', 'rf', 'mktrf', 'smb', 'hml', 'umd', 'rmw', 'cma']].drop_duplicates().to_parquet('FFData.parquet')
    
    print('Done')
    
    return df_full

def time_series(_SAMPLE_START, _SAMPLE_END):
    
    # Establish connection with wrds
    db = wrds.Connection()
    
    ###############################################
    ## Step 1. Import Daily Data
    ###############################################
    print('Step 1. Import Daily Data')
    
    # Define your SQL statement for daily data
    sql_statement = """
    SELECT a.permno, b.ticker, a.date, a.ret                           
    FROM crsp_m_stock.dsf as a
    LEFT JOIN crsp_m_stock.dsenames as b
    ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
    WHERE a.date >= '{}' AND a.date <= '{}' AND b.shrcd between 73 and 73 AND (b.ticker = 'SPY' OR b.ticker = 'XLF')
    """
    
    # Perform the query
    #df_ETF_daily = db.raw_sql(sql_statement.format('2003-01-01', '2023-07-31'))
    df_ETF_daily = db.raw_sql(sql_statement.format('2003-01-01', _SAMPLE_END))
    
    # Construct monthly date
    df_ETF_daily['date'] = pd.to_datetime(df_ETF_daily['date'])
    df_ETF_daily['ym'] = df_ETF_daily['date'].apply(lambda x: x.replace(day=1))

    # Restrict only to variables of interest and rename
    df_ETF_daily = df_ETF_daily[['date','ym','permno','ret']].drop_duplicates()
    df_ETF_daily = df_ETF_daily.rename(columns={'ret':'retd'})
    
    print('Done')
    ###############################################
    ## Step 2. Import Monthly Data
    ###############################################
    print('Step 2. Import Monthly Data')
    
    # Define your SQL statement for monthly data
    sql_statement = """
    SELECT a.permno, b.ticker, a.date, a.ret, b.shrcd
    FROM crsp_m_stock.msf as a
    LEFT JOIN crsp_m_stock.msenames as b
    ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
    WHERE a.date >= '{}' AND a.date <= '{}' AND b.shrcd between 73 and 73 AND (b.ticker = 'SPY' OR b.ticker = 'XLF')
    """
    
    # Perform the query
    df_ETF_monthly = db.raw_sql(sql_statement.format('2003-01-01', _SAMPLE_END))
    
    # Construct monthly date
    df_ETF_monthly['ym'] = pd.to_datetime(df_ETF_monthly['date']).apply(lambda x: x.replace(day=1))
        
    # Restrict only to variables of interest and rename
    df_ETF_monthly = df_ETF_monthly[['ym','permno','ticker','ret']].drop_duplicates()
    df_ETF_monthly = df_ETF_monthly.rename(columns={'ret':'retM'})
    
    print('Done')
    ###############################################
    ## Step 3. Import Fama-French Factors
    ###############################################
    print('Step 3. Import Fama-French Factors')
    
    # Define your SQL statement for FF factors
    sql_statement = """
    SELECT date, mktrf, rf
    FROM ff.fivefactors_monthly
    """
    
    # Perform the query
    df_FF = db.raw_sql(sql_statement)
    
    # Construct monthly date
    df_FF['ym'] = pd.to_datetime(df_FF['date']).apply(lambda x: x.replace(day=1))
    
    # Restrict only to variables of interest and rename
    df_FF = df_FF[['ym','mktrf','rf']]
    
    # Merge the various datasets together
    df_ETF_raw = pd.merge(left = df_ETF_daily, right = df_ETF_monthly, on =['ym','permno'], how = 'inner', validate = 'm:1')
    df_ETF_raw = pd.merge(left = df_ETF_raw, right = df_FF, on ='ym', how = 'inner', validate = 'm:1')
    
    return df_ETF_raw

def etfs(_SAMPLE_START, _SAMPLE_END):
    
    # Establish connection with wrds
    db = wrds.Connection()
    
    ###############################################
    ## Step 1. Import Daily Data
    ###############################################
    print('Step 1. Import Daily Data')
    
    # Define your SQL statement for daily data
    sql_statement = """
    SELECT a.permno, b.ticker, a.date, a.ret                           
    FROM crsp_m_stock.dsf as a
    LEFT JOIN crsp_m_stock.dsenames as b
    ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
    WHERE a.date >= '{}' AND a.date <= '{}' AND b.shrcd between 73 and 73 AND (b.ticker = 'IYF' OR b.ticker = 'IYK' OR b.ticker = 'IYW' OR b.ticker = 'IYZ' OR b.ticker = 'IYE')
    """
    
    # Perform the query
    #df_ETF_daily = db.raw_sql(sql_statement.format('2003-01-01', '2023-07-31'))
    df_ETF_daily = db.raw_sql(sql_statement.format('2003-01-01', _SAMPLE_END))
    
    # Construct monthly date
    df_ETF_daily['date'] = pd.to_datetime(df_ETF_daily['date'])
    df_ETF_daily['ym'] = df_ETF_daily['date'].apply(lambda x: x.replace(day=1))

    # Restrict only to variables of interest and rename
    df_ETF_daily = df_ETF_daily[['date','ym','permno','ret']].drop_duplicates()
    df_ETF_daily = df_ETF_daily.rename(columns={'ret':'retd'})
    
    print('Done')
    ###############################################
    ## Step 2. Import Monthly Data
    ###############################################
    print('Step 2. Import Monthly Data')
    
    # Define your SQL statement for monthly data
    sql_statement = """
    SELECT a.permno, b.ticker, a.date, a.ret, b.shrcd
    FROM crsp_m_stock.msf as a
    LEFT JOIN crsp_m_stock.msenames as b
    ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
    WHERE a.date >= '{}' AND a.date <= '{}' AND b.shrcd between 73 and 73 AND (b.ticker = 'IYF' OR b.ticker = 'IYK' OR b.ticker = 'IYW' OR b.ticker = 'IYZ' OR b.ticker = 'IYE')
    """

    # Perform the query
    df_ETF_monthly = db.raw_sql(sql_statement.format('2003-01-01', _SAMPLE_END))
    
    # Construct monthly date
    df_ETF_monthly['ym'] = pd.to_datetime(df_ETF_monthly['date']).apply(lambda x: x.replace(day=1))
        
    # Restrict only to variables of interest and rename
    df_ETF_monthly = df_ETF_monthly[['ym','permno','ticker','ret']].drop_duplicates()
    df_ETF_monthly = df_ETF_monthly.rename(columns={'ret':'retM'})
    
    print('Done')
    ###############################################
    ## Step 3. Import Fama-French Factors
    ###############################################
    print('Step 3. Import Fama-French Factors')
    
    # Define your SQL statement for FF factors
    sql_statement = """
    SELECT date, mktrf, rf
    FROM ff.fivefactors_monthly
    """
    
    # Perform the query
    df_FF = db.raw_sql(sql_statement)
    
    # Construct monthly date
    df_FF['ym'] = pd.to_datetime(df_FF['date']).apply(lambda x: x.replace(day=1))
    
    # Restrict only to variables of interest and rename
    df_FF = df_FF[['ym','mktrf','rf']]
    
    # Merge the various datasets together
    df_ETF_raw = pd.merge(left = df_ETF_daily, right = df_ETF_monthly, on =['ym','permno'], how = 'inner', validate = 'm:1')
    df_ETF_raw = pd.merge(left = df_ETF_raw, right = df_FF, on ='ym', how = 'inner', validate = 'm:1')
    
    return df_ETF_raw

def FFdaily(_SAMPLE_START, _SAMPLE_END):
    
    # Establish connection with wrds
    db = wrds.Connection()
    
    # Define your SQL statement for FF factors
    sql_statement = """
    SELECT date, mktrf, smb, hml, rf, umd, rmw, cma
    FROM ff.fivefactors_daily
    """
    
    # Perform the query
    df_FF = db.raw_sql(sql_statement.format(_SAMPLE_START, _SAMPLE_END))
    
    # Construct monthly date
    df_FF['date'] = pd.to_datetime(df_FF['date'])
    df_FF['ym'] = pd.PeriodIndex(df_FF['date'], freq='M')
    
    return df_FF
                  
def rolling_betas(df):
    
    # Compute excess returns
    df['retrf'] = df['ret'] - df['rf']
    df['vwmktrf'] = df['vwretd'] - df['rf']
    df = df.sort_values(by=['permno', 'ym']).reset_index(drop=True)

    # Initialize an empty list to hold the betas
    betas = []

    # Loop through each group to perform the rolling regression
    for name, group in df.groupby('permno'):
        nobs = len(group)
        if nobs < 20:
            betas.extend([np.nan]*nobs)
        elif (nobs >= 20) & (nobs < 60):
            endog = group['retrf']
            exog = sm.add_constant(group['vwmktrf'])
            rol_model = RollingOLS(endog=endog, exog=exog, window=nobs, min_nobs=20, expanding=True)
            rol_res = rol_model.fit(params_only=True)
            betas.extend(rol_res.params['vwmktrf'].to_list())
        else:
            endog = group['retrf']
            exog = sm.add_constant(group['vwmktrf'])
            rol_model = RollingOLS(endog=endog, exog=exog, window=60, min_nobs=20, expanding=True)
            rol_res = rol_model.fit(params_only=True)
            betas.extend(rol_res.params['vwmktrf'].to_list())

    # Attach the beta values to the original DataFrame
    df['beta'] = betas
    df.drop(['retrf','vwmktrf'], axis=1, inplace=True)
            
    return df