'''
	--------------------------------------------------------------------
	qpm.py

	This code contains imports and helper functions
	to be used with the Jupyter notebooks for 

	Chicago Booth course on Quantitative Portfolio Management
	by Ralph S.J. Koijen and Sangmin S. Oh.

	2021-12-18 : Initial Code

	--------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------
		PRELIMINARIES
--------------------------------------------------------------------
'''

#------------------------------------------------#
#  Import Packages

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
import matplotlib.dates as mdates

import pdb, os, time

from cycler import cycler
from matplotlib import rcParams
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from scipy.stats.mstats import winsorize

DataFrame = pd.DataFrame
Series = pd.Series

pd.options.mode.chained_assignment = None

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



#------------------------------------------------#
#  Prettify

rcParams['font.serif'] = ['Bitstream Vera Sans']
rcParams['figure.figsize'] = 12, 4
rcParams['font.size'] = 13
rcParams['grid.linewidth'] = 0.3
rcParams['grid.color'] = '0.7'
rcParams['axes.prop_cycle'] = cycler('color', ['maroon', 'gray', 'orange', 'green']) + cycler('linestyle', ['-', '--', ':', '-.'])
rcParams['axes.grid'] = True
rcParams['axes.grid.axis'] = 'y'
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['legend.frameon'] = False

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')


'''
--------------------------------------------------------------------
		MAIN FUNCTIONS
--------------------------------------------------------------------
'''

def return_signal(_STRATEGY_NAME):

    if (_STRATEGY_NAME == 'Size') | (_STRATEGY_NAME == 'Momentum') | (_STRATEGY_NAME == 'STreversal') | (_STRATEGY_NAME == 'Seasonal') | (_STRATEGY_NAME == 'AssetGrowth'):
        
        signal_variables = []
    
    elif _STRATEGY_NAME == 'Value':

        signal_variables = ['be']

    elif _STRATEGY_NAME == 'ESG':

        signal_variables = ['carbon_intensity']

    elif _STRATEGY_NAME == 'Quality':

        signal_variables = ['revt','cogs','beta']

    else:

        raise Exception('Please provide a valid _STRATEGY_NAME..')

    return signal_variables

def list_variables(data_dir, file_name):


	file_type = file_name.split('.')[-1]

	if file_type == 'dta':
		header = pd.io.stata.StataReader('%s/%s' %(data_dir, file_name)).variable_labels()
		header = list(header.keys())
	elif file_type == 'csv':
		header = pd.read_csv('%s/%s' %(data_dir, file_name), nrows = 1)
		header = list(header.columns)
	elif file_type == 'parquet':
		header = pd.read_parquet('%s/%s' %(data_dir, file_name))
		header = list(header.columns)
	else:
		raise Exception('Please provide a valid file_type: .dta or .csv')

	## Print output of the headers based on the type of variables
	list_dic = {'Identifiers' : ['permno', 'ticker', 'comnam', 'conm', 'gvkey', 'cusip', 'lpermco'],
				'Prices and Returns' : ['ret', 'retx', 'prc', 'vwretd', 'ewretd', 'prcc_c', 'prcc_f'],
				'Fama-French' : ['mktrf', 'smb', 'hml', 'rf', 'umd', 'rmw', 'cma'],
				'Fundamentals' : ['aco', 'act', 'ajex', 'am', 'ao', 'ap', 'at', 'be', 'capx', 'che', 'cogs', 
								  'csho', 'cshrc', 'dcpstk', 'dcvt', 'dlc', 'dlcch', 'dltis', 'dltr', 'dltt', 
								  'dm', 'dp', 'drc', 'drlt', 'dv', 'dvc', 'dvp', 'dvpa', 'dvpd', 'dvpsx_c', 'dvt',
								  'ebit', 'ebitda', 'emp', 'epspi', 'epspx', 'fatb', 'fatl', 'ffo', 'fincf', 'fopt',
								  'gdwl', 'gdwlia', 'gdwlip', 'gwo', 'ib', 'ibcom', 'intan', 'invt', 'ivao', 'ivncf',
								  'ivst', 'lco', 'lct', 'lo', 'lt', 'mib', 'msa', 'ni', 'nopi', 'oancf', 'ob', 'oiadp',
								  'oibdp', 'pi', 'ppenb', 'ppegt', 'ppenls', 'ppent', 'prstkc', 'prstkcc', 'pstk', 
								  'pstkl', 'pstkrv', 're', 'rect', 'recta', 'revt', 'sale', 'scstkc', 'seq', 'spi', 
								  'sstk', 'tstkp', 'txdb', 'txdi', 'txditc', 'txfo', 'txfed', 'txp', 'txt','wcap', 'wcapch', 
								  'xacc', 'xad', 'xint', 'xrd', 'xpp', 'xsga', 'cnum', 'dr', 'dc', 'xint0', 'xsga0', 'xad0'],
               'Others' : ['vol','me','ESG_score', 'E_score', 'S_score', 'G_score', 'carbon_intensity']}

	not_others = []
	for item in list_dic.keys():
		print(item + ':')
		print('%s\n' %([x for x in header if x in list_dic[item]]))

def load_data(data_dir, file_name, variable_list = []):

	file_type = file_name.split('.')[-1]

	## List of variables
	basic_list = ['permno', 'daret', 'retx', 'vol', 'shrout', 'prc', 'shrcd', 'exchcd', 'ticker', 'ldate', 'conm', 'me', 'be', 'revt', 'cogs', 'at', 'hml', 'smb', 'mktrf', 'rf', 'umd', 'cma', 'rmw']
	# aux_list = ['ldate_lag', 'ldate_lag12', 'screen', 'me_lagged', 'screen12']
	final_list = list(set(basic_list + variable_list))

	#------------------------------------------------#
	#  Load Raw Data

	print('> Loading Raw Data...')

	if file_type == 'dta':
		if variable_list != []:
			df_full = pd.read_stata('%s/%s' %(data_dir, file_name), columns = final_list)
		else:
			df_full = pd.read_stata('%s/%s' %(data_dir, file_name))
	elif file_type == 'parquet':
		if variable_list != []:
			df_full = pd.read_parquet('%s/%s' %(data_dir, file_name), columns = final_list)
		else:
			df_full = pd.read_parquet('%s/%s' %(data_dir, file_name))    
	elif file_type == 'csv':
		if variable_list != []:
			df_full = pd.read_csv('%s/%s' %(data_dir, file_name), usecols = final_list)
		else:
			df_full = pd.read_csv('%s/%s' %(data_dir, file_name))

		## Additional adjustments regarding data type
		if 'date' in df_full.columns:
			df_full['date'] = df_full['date'].map(lambda x : datetime.strptime(x, '%d%b%Y').strftime('%Y-%m-%d'))
			df_full['date'] = pd.to_datetime(df_full['date'])
		if 'ldate' in df_full.columns:
			df_full['ldate'] = df_full['ldate'].map(lambda x : datetime.strptime(x, '%Ym%m').strftime('%Y-%m-01'))
			df_full['ldate'] = df_full['ldate'].map(lambda x : datetime.strptime(x, '%Y-%m-%d'))
		if 'ym' in df_full.columns:
			df_full['ym'] = df_full['ym'].map(lambda x : datetime.strptime(x, '%Ym%m').strftime('%Y-%m-01'))
			df_full['ym'] = pd.to_datetime(df_full['ym'])
	else:
		raise Exception('Please provide a valid file_type: .dta or .csv')

	## Rename Key Variables
	print('> Renaming key variables...')
	if 'me' not in df_full.columns:
		df_full['me'] = df_full['mve_c']
	if 'daret' not in df_full.columns:
		df_full['daret'] = df_full['ret']
	if 'be' not in df_full.columns:
		df_full['be'] = df_full['ceq']
	if 'profitA' not in df_full.columns:
		df_full['profitA'] = (df_full['revt'] - df_full['cogs']) / df_full['at']

	## Drop Duplicates
	print('> Dropping duplicates...')
	df_full.drop_duplicates(subset = ['permno', 'ldate'], keep = 'first', inplace = True)

	#------------------------------------------------#
	#  Auxiliary Variables

	print('> Creating Auxiliary Variables...')

	df_full.sort_values(by = ['permno', 'ldate'], inplace = True)

	df_full['ldate_lag'] = df_full.groupby(['permno'])['ldate'].shift(1)
	df_full['screen'] = (df_full['ldate_lag'] == df_full['ldate'] - pd.DateOffset(months=1)).astype(int).replace(0, np.nan)

	df_full['ldate_lag12'] = df_full.groupby(['permno'])['ldate'].shift(12)
	df_full['screen12'] = (df_full['ldate_lag12'] == df_full['ldate'] - pd.DateOffset(months=12)).astype(int).replace(0, np.nan)

	df_full['me_lagged'] = df_full.groupby(['permno'])['me'].shift(1).multiply(df_full['screen'])

	## Save Fama-French Data
	df_full[['ldate', 'rf', 'mktrf', 'smb', 'hml', 'umd', 'rmw', 'cma']].drop_duplicates().to_parquet('FFData.parquet')

	return df_full

def load_data_etf(data_dir, file_name):

	file_type = file_name.split('.')[-1]

	#------------------------------------------------#
	#  Load Raw Data

	print('> Loading Raw Data...')

	if file_type == 'dta':
		df_full = pd.read_stata('%s/%s' %(data_dir, file_name))
	elif file_type == 'csv':
		df_full = pd.read_csv('%s/%s' %(data_dir, file_name))
		df_full['date'] = df_full['date'].map(lambda x : datetime.strptime(x, '%d%b%Y').strftime('%Y-%m-%d'))
		df_full['ym'] = df_full['ym'].map(lambda x : datetime.strptime(x, '%Ym%m').strftime('%Y-%m-01'))

		df_full['date'] = pd.to_datetime(df_full['date'])
		df_full['ym'] = pd.to_datetime(df_full['ym'])

	else:
		raise Exception('Please provide a valid file_type: .dta or .csv')

	return df_full

def select_sample(df_input, sample_start, sample_end, remove_micro_caps):

	print('> Selecting Sample for Given Criteria...')

	df = df_input[(df_input['ldate'] <= sample_end) & (df_input['ldate'] >= sample_start)]

	## Drop Stocks with Missing Returns
	df = df.dropna(subset = ['daret'])

	## Drop Stocks with Missing Signal Values
	df = df.dropna(subset = ['signal'])

	## Deal with Micro Caps
	if remove_micro_caps:

		thresholds = df[df['exchcd'] == 1].groupby('ldate')['me_lagged'].quantile(0.2, interpolation = 'lower')
		df['cutoff'] = df['ldate'].map(lambda x : thresholds[x])
		df = df[df['me_lagged'] >= df['cutoff']]

	return df

def create_lag(df, var_name, lag):

	df['ldate_lag_temp'] = df.groupby(['permno'])['ldate'].shift(lag)
	df['screen_temp'] = (df['ldate_lag_temp'] == df['ldate'] - pd.DateOffset(months = lag)).astype(int).replace(0, np.nan)
	return_col = df.groupby(['permno'])[var_name].shift(lag).multiply(df['screen_temp'])
	del df['screen_temp']
	del df['ldate_lag_temp']

	return return_col

def compute_rolling_by_permno(df, var_name, window_size, min_obs, stat_type):

	## Use data only that we need
	sub_df = df.reset_index()[['permno', 'ldate', 'index'] + [var_name]]

	## Create a full panel
	temp_df = sub_df.set_index(['permno', 'ldate']).unstack().stack(dropna = False).reset_index()

	## Compute rolling stat
	if stat_type == 'mean':
		temp_df['temp'] = temp_df.groupby('permno')['Investment'].transform(lambda x : x.rolling(window = window_size, min_periods = min_obs).mean())
	elif stat_type in ['std', 'vol']:
		temp_df['temp'] = temp_df.groupby('permno')['Investment'].transform(lambda x : x.rolling(window = window_size, min_periods = min_obs).std())
	else:
		raise Exception('UNIMPLEMENTED stat_type: %s' %(stat_type))

	return temp_df.dropna(subset = ['index']).set_index('index').sort_index()['temp']



def rank(df, var_name):

	return df.groupby('ldate')[var_name].rank(ascending = True)
	

def create_portfolios(df, sort_frequency, num_port):

	#------------------------------------------------#
	#  Sort Portfolios

	print('> Sorting stocks into %d portfolios at frequency: %s...' %(num_port, sort_frequency))

	if sort_frequency == 'Monthly':

		## Group stocks into portfolios using NYSE breakpoints and monthly signals
		
		df['portfolio'] = df[df['exchcd'] == 1].groupby('ldate')['signal'].transform(lambda x : pd.qcut(x, num_port, labels = range(1, num_port + 1)))
		
		## Fill in the values for Non-NYSE Stocks
		
		df.sort_values(['ldate', 'signal'], inplace = True)
		df['portfolio'] = df.groupby('ldate')['portfolio'].ffill()
		df['portfolio'].fillna(1, inplace = True)

	elif sort_frequency == 'June':

		## Group stocks into portfolios using NYSE breakpoints and monthly signals (note: signal is lagged)
		
		df['lmonth'] = df['ldate'].map(lambda x : x.month)
		df['portfolio'] = df[(df['exchcd'] == 1) & (df['lmonth'] == 7)].groupby('ldate')['signal'].transform(lambda x : pd.qcut(x, num_port, labels = range(1, num_port + 1)))

		## Fill in the values for Non-NYSE Stocks
		
		df.sort_values(['ldate', 'signal'], inplace = True)
		df['temp'] = np.where(df['lmonth'] == 7, df.groupby('ldate')['portfolio'].ffill(), np.nan)
		df['portfolio'].fillna(df['temp'], inplace = True)
		df['temp'] = np.where(df['lmonth'] == 7, 1, np.nan)
		df['portfolio'].fillna(df['temp'], inplace = True)

		## Assign stocks to the same portfolio for July to May
		
		df.sort_values(['permno', 'ldate'], inplace = True)
		df['temp'] = np.where(df['lmonth'] != 7, df.groupby('permno')['portfolio'].ffill(), np.nan)
		df['portfolio'].fillna(df['temp'], inplace = True)

		df.drop(columns = ['temp'], inplace = True)

	else:
	
		raise Exception('Please provide a valid _SORT_FREQUENCY type. It should either be Monthly or June.')

	#------------------------------------------------#
	#  Compute Returns for Different Weighting Schemes

	print('> Computing returns using various weights...')

	## Rank-weighted strategy long-only
	# print(df)
	
	df['signal_rank'] = df.groupby('ldate')['signal'].rank() # <-- this is the right one
	df['Tsignal_rank'] = df.groupby('ldate')['signal_rank'].transform(sum)
	df['weight'] = df['signal_rank'] / df['Tsignal_rank']
	df['wgtd_daret'] = df['weight'] * df['daret']

	df_rets = {}

	df_rets['retP_rank_longonly'] = df.groupby('ldate')['wgtd_daret'].sum()

	## Rank-weighted strategy long-short strategy
	
	df['NStocks'] = df.groupby('ldate')['signal'].transform('count')
	df['weight'] = 4*(df['weight'] - 1/df['NStocks'])
	df['wgtd_daret'] = df['weight'] * df['daret']

	df_rets['retP_rank_longshort'] = df.groupby('ldate')['wgtd_daret'].sum()
	df_rets = DataFrame(df_rets)
	df_rets.head()

	## Value-weighted returns for each of the portfolio
	
	df['Tme'] = df.groupby(['ldate', 'portfolio'])['me_lagged'].transform('sum')
	df['weight'] = df['me_lagged'] / df['Tme']

	for por_num in range(1, num_port + 1):

		sub_df = df[df['portfolio'] == por_num].copy(deep = True)
		sub_df['wgtd_daret'] = sub_df['weight'] * sub_df['daret']

		df_rets['retP_vw_P%d' %(por_num)] = sub_df.groupby('ldate')['wgtd_daret'].sum()

	df_rets['retF_vw'] = df_rets['retP_vw_P%d' %(num_port)] - df_rets['retP_vw_P1']
	# common_index = df_rets['retF_vw'].index

	return df, df_rets.reset_index()

def analyze_strategy(df_strategy, analysis_type):

	#------------------------------------------------#
	#  Prepare Data

	print('> Merging strategy returns with Fama and French factor returns...')

	df_ff = pd.read_parquet('FFData.parquet').sort_values(['ldate']).rename(columns = {'ldate' : 'ym'})

	## Strategy Returns
	df_strategy = df_strategy.rename(columns = {'ldate' : 'ym'})
	df_strategy['ym'] = pd.to_datetime(df_strategy['ym'])

	## Merge Data
	df = pd.merge(df_strategy, df_ff, on = ['ym'], validate = 'many_to_one', indicator = True)

	if analysis_type == 'Performance':

		#------------------------------------------------#
		#  Average Portfolio Return

		select_cols = [x for x in df.columns if 'retP_vw_P' in x]

		plot_series = df[select_cols].mean()
		plot_series.index = plot_series.index.map(lambda x : x.replace('retP_vw_P', ''))

		fig = plt.figure(figsize = (12, 6))
		plt.grid(axis = 'y', zorder = 0)
		plt.bar(plot_series.index, plot_series.values * 100, color = 'maroon', width = 0.5, zorder = 5)
		plt.title('Average Portfolio Return')
		plt.xlabel('Portfolio Quantile'); plt.ylabel('Portfolio Return (%)')
		plt.show()
		plt.close()

		#------------------------------------------------#
		#  Cumulative Returns for the Long-Only Portfolio

		df.sort_values(['ym'], inplace = True)
		df['CLNmkt'] = np.log(1 + df['mktrf'] + df['rf']).cumsum()
		df['CLNretP'] = np.log(1 + df['retP_rank_longonly']).cumsum()
		plot_df = df[['CLNmkt', 'CLNretP', 'ym']].set_index('ym')
		fig = plt.figure(figsize = (12, 6))
		plot_df['CLNretP'].plot(ax = plt.gca(), color = 'maroon', label = 'Portfolio')
		plot_df['CLNmkt'].plot(ax = plt.gca(), color = 'gray', label = 'Market')
		plt.xlabel('Date'); plt.ylabel('Cumulative Return')
		plt.title('Cumulative Returns for the Long-Only Strategy')
		plt.legend()
		plt.show()
		plt.close()

		#------------------------------------------------#
		#  Cumulative Returns for the Long-Short Rank-Based Portfolio

		df.sort_values(['ym'], inplace = True)
		df['CLNrf'] = np.log(1 + df['rf']).cumsum()
		df['CLNretP'] = np.log(1 + df['retP_rank_longshort'] + df['rf']).cumsum()
		plot_df = df[['CLNrf', 'CLNretP', 'ym']].set_index('ym')
		fig = plt.figure(figsize = (12, 6))
		plot_df['CLNretP'].plot(ax = plt.gca(), color = 'maroon', label = 'Portfolio')
		plot_df['CLNrf'].plot(ax = plt.gca(), color = 'gray', label = 'Risk-free Benchmark')
		plt.xlabel('Date'); plt.ylabel('Cumulative Return')
		plt.title('Cumulative Returns for the Long-Short Rank-Based Strategy')
		plt.legend()
		plt.show()
		plt.close()
        
		#------------------------------------------------#
		#  Cumulative Returns for the Long-Short Portfolio-Based Portfolio

		df.sort_values(['ym'], inplace = True)
		df['CLNretF_vw'] = np.log(1 + df['retF_vw'] + df['rf']).cumsum()
		plot_df = df[['CLNrf', 'CLNretF_vw', 'ym']].set_index('ym')
		fig = plt.figure(figsize = (12, 6))
		plot_df['CLNretF_vw'].plot(ax = plt.gca(), color = 'maroon', label = 'Portfolio')
		plot_df['CLNrf'].plot(ax = plt.gca(), color = 'gray', label = 'Risk-free Benchmark')
		plt.xlabel('Date'); plt.ylabel('Cumulative Return')
		plt.title('Cumulative Returns for the Long-Short Portfolio-Based Strategy')
		plt.legend()
		plt.show()
		plt.close()

	elif analysis_type == 'Summary':

		select_cols = ['retP_rank_longonly', 'retP_rank_longshort', 'retF_vw', 'mktrf', 'smb', 'hml']
		print(df[select_cols].describe().T)

	elif analysis_type == 'Factor Regression':

		## Table 1: 3 FF Model
		print('')
		print('---------------------------------------------------------------')
		print('> Running Factor Regressions: Table 1  - 3 Fama-French Factors')
		print('---------------------------------------------------------------')

		import statsmodels.api as sm

		df['retP_rank_longonly_e'] = df['retP_rank_longonly'] - df['rf']

		reg_df = df[['retF_vw', 'mktrf', 'retP_rank_longonly_e', 'retP_rank_longshort', 'smb', 'hml']].dropna()
		m1 = sm.OLS(reg_df['retF_vw'], sm.add_constant(reg_df['mktrf'])).fit()
		m2 = sm.OLS(reg_df['retP_rank_longonly_e'], sm.add_constant(reg_df['mktrf'])).fit()
		m3 = sm.OLS(reg_df['retP_rank_longshort'], sm.add_constant(reg_df['mktrf'])).fit()
		m4 = sm.OLS(reg_df['retF_vw'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml']])).fit()
		m5 = sm.OLS(reg_df['retP_rank_longonly_e'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml']])).fit()
		m6 = sm.OLS(reg_df['retP_rank_longshort'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml']])).fit()

		from statsmodels.iolib.summary2 import summary_col
		summary = summary_col([m1, m2, m3, m4, m5, m6], regressor_order = ['const', 'mktrf', 'hml', 'smb'],
							  stars = True, drop_omitted = True,
							  model_names=['(1)', '(2)', '(3)', '(4)', '(5)', '(6)'],
							  info_dict = {'N':lambda x: "{0:d}".format(int(x.nobs)),
										   'R2':lambda x: "{:.2f}".format(x.rsquared)})
		print(summary)
		print('(1): Long-Short Value Weights ~ CAPM Model')
		print('(2): Long-Only Rank Weights ~ CAPM Model')
		print('(3): Long-Short Rank Weights ~ CAPM Model')
		print('(4): Long-Short Value Weights ~ 3-Factor Fama French Model')
		print('(5): Long-Only Rank Weights ~ 3-Factor Fama French Model')
		print('(6): Long-Short Rank Weights ~ 3-Factor Fama French Model')

		summary_IR, cnt = {}, 1
		for model in [m1, m2, m3, m4, m5, m6]:
			summary_IR[cnt] = {}
			summary_IR[cnt]['Alpha'] = model.params['const'] * 12
			summary_IR[cnt]['Std(resid)'] = np.std(model.resid) * np.sqrt(12)
			summary_IR[cnt]['Information Ratio'] = summary_IR[cnt]['Alpha'] / summary_IR[cnt]['Std(resid)']

			cnt += 1

		summary_IR = DataFrame(summary_IR)
		print('Annualized Information Ratios:')
		print(summary_IR.round(3))

		## Table 2: 3 FF Model
		print('\n\n')
		print('---------------------------------------------------------------')
		print('> Running Factor Regressions: Table 2  - 5 Fama-French Factors + Momentum')
		print('---------------------------------------------------------------')

		import statsmodels.api as sm

		df['retP_rank_longonly_e'] = df['retP_rank_longonly'] - df['rf']

		reg_df = df[['retF_vw', 'mktrf', 'retP_rank_longonly_e', 'retP_rank_longshort', 'smb', 'hml', 'rmw', 'cma', 'umd']].dropna()
		m1 = sm.OLS(reg_df['retF_vw'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml', 'rmw', 'cma']])).fit()
		m2 = sm.OLS(reg_df['retP_rank_longonly_e'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml', 'rmw', 'cma']])).fit()
		m3 = sm.OLS(reg_df['retP_rank_longshort'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml', 'rmw', 'cma']])).fit()
		m4 = sm.OLS(reg_df['retF_vw'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']])).fit()
		m5 = sm.OLS(reg_df['retP_rank_longonly_e'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']])).fit()
		m6 = sm.OLS(reg_df['retP_rank_longshort'], sm.add_constant(reg_df[['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']])).fit()

		from statsmodels.iolib.summary2 import summary_col
		summary = summary_col([m1, m2, m3, m4, m5, m6], regressor_order = ['const', 'mktrf', 'hml', 'smb', 'rmw', 'cma', 'umd'],
							  stars = True, drop_omitted = True,
							  model_names=['(1)', '(2)', '(3)', '(4)', '(5)', '(6)'],
							  info_dict = {'N':lambda x: "{0:d}".format(int(x.nobs)),
										   'R2':lambda x: "{:.2f}".format(x.rsquared)})
		print(summary)
		print('(1): Long-Short Value Weights ~ 5-Factor Fama French Model')
		print('(2): Long-Only Rank Weights ~ 5-Factor Fama French Model')
		print('(3): Long-Short Rank Weights ~ 5-Factor Fama French Model')
		print('(4): Long-Short Value Weights ~ 6-Factor Fama French Model')
		print('(5): Long-Only Rank Weights ~ 6-Factor Fama French Model')
		print('(6): Long-Short Rank Weights ~ 6-Factor Fama French Model')

		summary_IR, cnt = {}, 1
		for model in [m1, m2, m3, m4, m5, m6]:
			summary_IR[cnt] = {}
			summary_IR[cnt]['Alpha'] = model.params['const'] * 12
			summary_IR[cnt]['Std(resid)'] = np.std(model.resid) * np.sqrt(12)
			summary_IR[cnt]['Information Ratio'] = summary_IR[cnt]['Alpha'] / summary_IR[cnt]['Std(resid)']

			cnt += 1

		summary_IR = DataFrame(summary_IR)
		print('Annualized Information Ratios:')
		print(summary_IR.round(3))

	else:

		raise Exception('Please provide a valid analysis type.')


'''
--------------------------------------------------------------------
		FUNCTIONS FOR PRELIMINARY ANAYLSIS
--------------------------------------------------------------------
'''

def plot_cumulative_returns_etf(df, var_list):

	plot_df = df.copy()
	# plot_df = df.set_index('ym')
	for var in var_list:
		plot_df['LN%s' %(var)] = np.log(1 + plot_df['%s' %(var)])
		plot_df['CLN%s' %(var)] = plot_df['LN%s' %(var)].cumsum()

	fig, ax = plt.subplots(1, 1)
	for var in var_list:
		ax.plot(plot_df.index, plot_df['CLN%s' %(var)], label = var)
	plt.xlabel('Date'); plt.ylabel('Cumulative Return')
	plt.legend()
	fig.tight_layout()
	plt.show()
	plt.close()

def plot_variables(df, variable_list, permno_list, start_date, end_date):

	df_plot = df[(df['ldate'] >= start_date) & (df['ldate'] <= end_date) & (df['permno'].isin(permno_list))]
	df_plot = df_plot[variable_list + ['permno', 'ldate']].set_index('ldate')

	for variable in variable_list:
		fig, ax = plt.subplots()
		for permno in permno_list:
			plt.plot(df_plot[df_plot['permno'] == permno][variable], label = permno)
			plt.legend(bbox_to_anchor = [1, 1], loc = 'upper left')
			plt.title('Variable: %s' %(variable))
		plt.show()

def plot_variables(df, variable_list, id_type, id_list, start_date, end_date):

	## Retrieve permno
	if id_type == 'ticker':
		permno_list = []
		for ticker in id_list:
			sub_df = df[df['ticker'] == ticker]
			permno = sub_df[['ticker', 'permno', 'ldate']].sort_values(['ldate'], ascending = False).dropna().iloc[0]['permno']
			permno_list.append(permno)
	elif id_type == 'permno':
		permno_list = id_list
	elif id_type != 'permno':
		raise Exception('Please provide a valid id type: permno or ticker.')

	df_plot = df[(df['ldate'] >= start_date) & (df['ldate'] <= end_date) & (df['permno'].isin(permno_list))]
	df_plot = df_plot[variable_list + ['permno', 'ticker', 'ldate']].set_index('ldate')


	for variable in variable_list:

		fig, ax = plt.subplots()
		for i in range(0, len(id_list)):
			permno = permno_list[i]
			label = id_list[i]
			plt.plot(df_plot[df_plot['permno'] == permno][variable], label = label)
			plt.legend(bbox_to_anchor = [1, 1], loc = 'upper left')
			plt.title('Variable: %s' %(variable))
		plt.show()
