import os
import warnings
import time

import config
import pandas as pd
from io import StringIO

import github

warnings.filterwarnings('once')


def load(country, states, latest=False):
    if country == 'India':
        load_india(states)
    elif country == 'US' or country == 'USA':
        load_us(states, latest)
    else:
        raise ValueError(f'Loading state data for {country} is not supported.')


def load_india(states):
    src_state_wise_cases = os.path.join(config.base_data_dir, config.india_states_cases_offline)

    try:
        df_states = pd.read_csv(src_state_wise_cases)
    except FileNotFoundError:
        print('Error!!! Missing Input Data File(s)!')
        return

    for target_state in states:
        print(f'Processing data for state: {target_state} ' + ' *' * 10)
        df_states.Date = pd.to_datetime(df_states.Date)
        df_state = df_states[['Date', 'Status', target_state]].copy()
        df_state = pd.pivot_table(df_state, values=target_state, columns='Status', index='Date')
        df_state.index = pd.to_datetime(df_state.index)
        df_state[['Total_Confirmed', 'Total_Deceased', 'Total_Recovered']] \
            = df_state[['Confirmed', 'Deceased', 'Recovered']].cumsum(axis=0, skipna=True)
        df_state.to_csv(os.path.join(config.base_data_dir, config.state_covid19_cases.format('IND', target_state)))


def load_us(states, latest=False):
    us_covid19_cases_path = os.path.join(config.base_data_dir, config.us_covid19_cases)
 
    #df_us = pd.read_csv(us_covid19_cases_path)
    import sys
    sys.path.append('src/')
    from delphi_epidata import Epidata
    
    start_date = 20200401
    
    from datetime import datetime
    stop_date = int(datetime.today().strftime('%Y%m%d'))
 
    for target_state in states:
        print(f'Processing data for state: {target_state} ' + ' *' * 10)
        print('Start date = ', start_date, ' End date = ', stop_date)
       
        res_incidence = Epidata.covidcast('jhu-csse', 'confirmed_7dav_incidence_num', 'day', 'state', \
                        [start_date, Epidata.range(start_date, stop_date)], target_state)
        res_death = Epidata.covidcast('jhu-csse', 'deaths_7dav_incidence_num', 'day', 'state', \
                        [start_date, Epidata.range(start_date, stop_date)], target_state)
        
        df_state = pd.DataFrame(columns=['Confirmed', 'Deceased', 'Recovered'])
        if len(res_incidence) > 0 and len(res_death) > 0:
            df_jhu_7day = pd.DataFrame(res_incidence['epidata'])
            df_jhu_7day_deaths = pd.DataFrame(res_death['epidata'])

            df_state['Date'] = pd.to_datetime(df_jhu_7day['time_value'], format='%Y%m%d')
            df_state['Confirmed'] = df_jhu_7day['value']
            df_state['Deceased'] = df_jhu_7day_deaths['value']
            df_state['Recovered'].fillna(value=0, inplace=True)
            
            # ensures sorting with respect to date
            df_state.index = pd.to_datetime(df_state.Date)
            df_state[['Total_Confirmed', 'Total_Deceased', 'Total_Recovered']] \
                = df_state[['Confirmed', 'Deceased', 'Recovered']].cumsum(axis=0, skipna=True)
            df_state.to_csv(os.path.join(config.base_data_dir, f'Cases_USA_{target_state}.csv'), index=False)
        else:
            print(' *** Error: Can not import data from Delphi database. Check src/state_data_loader.py')
            exit()


   
        
# load population data for usa
def load_us_population(states, latest=False):
    population = float('Nan')
    
    # here read the population data
    us_population_path = os.path.join(config.base_data_dir, config.usa_populations_data_offline)
    
    # TODO: Add download if not in the repo
    df_population = pd.read_csv(us_population_path)
    #print(df_population.columns)
    
    #states = list(set(df_population['Name']))
    population = list()
    for target_state in states:
        df_state = df_population.loc[df_population['NAME'] == target_state]
        population.append(int(df_state['POPESTIMATE2019']))
        
    
    
    return population
    


        
# def load_us_old(states, latest=False):
#     us_covid19_cases_path = os.path.join(config.base_data_dir, config.us_covid19_cases)
    
 
#     if not os.path.isfile(us_covid19_cases_path) or latest:
#         g = github.Github()
#         while True:
#             try:
#                 repo = g.get_repo("CSSEGISandData/COVID-19")
#                 files = repo.get_contents("csse_covid_19_data/csse_covid_19_daily_reports_us")
#                 break
#             except github.RateLimitExceededException as e:
#                 print(e)
#                 time.sleep(10)

#         columns = ['Province_State', 'Confirmed', 'Deaths', 'Recovered']

#         df_list = []

#         for f in files:
#             if f.name.endswith(".csv"):
#                 str_io = StringIO(f.decoded_content.decode("utf-8"))
#                 time.sleep(1)

#                 df = pd.read_csv(str_io, usecols=columns)
#                 df['Date'] = f.name.split('.')[0]
#                 df_list.append(df)
                
#         df_concat = pd.concat(df_list, axis=0, ignore_index=True)
#         df_concat.rename(columns={'Deaths': 'Deceased'}, inplace=True)
        
#         df_concat = df_concat.sort_values(by=['Date'])
#         df_concat.to_csv(os.path.join(config.base_data_dir, config.us_covid19_cases), index=False)

#     df_us = pd.read_csv(us_covid19_cases_path)

#     for target_state in states:
#         df_state = df_us.loc[df_us['Province_State'] == target_state].drop('Province_State', axis=1)

#         df_state.index = pd.to_datetime(df_state.Date)
#         df_state['Recovered'].fillna(value=0, inplace=True)

#         df_state[['Total_Confirmed', 'Total_Deceased', 'Total_Recovered']] \
#             = df_state[['Confirmed', 'Deceased', 'Recovered']].cumsum(axis=0)

#         df_state = df_state.sort_values(by=['Date'])
#         df_state.to_csv(os.path.join(config.base_data_dir, f'Cases_USA_{target_state}.csv'), index=False)

        

# def load_us_all(states, latest=False):
#     us_covid19_cases_path = os.path.join(config.base_data_dir, config.us_covid19_cases)
#     #print('SAHIKA:', us_covid19_cases_path)

#     if not os.path.isfile(us_covid19_cases_path) or latest:
#         g = github.Github()
#         while True:
#             try:
#                 repo = g.get_repo("CSSEGISandData/COVID-19")
#                 files = repo.get_contents("csse_covid_19_data/csse_covid_19_daily_reports_us")
#                 break
#             except github.RateLimitExceededException as e:
#                 print(e)
#                 time.sleep(10)

#         columns = ['Province_State', 'Confirmed', 'Deaths', 'Recovered']

#         df_list = []

#         for f in files:
#             if f.name.endswith(".csv"):
#                 str_io = StringIO(f.decoded_content.decode("utf-8"))
#                 time.sleep(1)

#                 df = pd.read_csv(str_io, usecols=columns)
#                 df['Date'] = f.name.split('.')[0]
#                 df_list.append(df)

#         df_concat = pd.concat(df_list, axis=0, ignore_index=True)
#         df_concat.rename(columns={'Deaths': 'Deceased'}, inplace=True)
#         df_concat.to_csv(os.path.join(config.base_data_dir, config.us_covid19_cases), index=False)

#     df_us = pd.read_csv(us_covid19_cases_path)
    
#     print(df_us[:5])
        
#     states = list(set(df_us['Province_State']))
#     states.sort()
          
#     for target_state in states:
#         print(target_state)
#         df_state = df_us.loc[df_us['Province_State'] == target_state].drop('Province_State', axis=1)
#         df_state = df_state.sort_values(by=['Date'])
        
#         df_state.index = pd.to_datetime(df_state.Date)
#         df_state['Recovered'].fillna(value=0, inplace=True)
#         df_state['Confirmed'].fillna(value=0, inplace=True)
#         df_state = df_state.rename(columns={'Confirmed': 'Total_Confirmed', 
#                            'Recovered': 'Total_Recovered',
#                           'Deceased': 'Total_Deceased'})
        
#         df_state['Confirmed'] = np.r_[0.0, np.diff(np.array(df_state['Total_Confirmed']))]
#         df_state['Recovered'] = np.r_[0.0, np.diff(np.array(df_state['Total_Recovered']))]
#         df_state['Deceased'] = np.r_[0.0, np.diff(np.array(df_state['Total_Deceased']))]

#         df_state.to_csv(os.path.join(config.base_data_dir, f'Cases_USA_{target_state}.csv'), index=False)

