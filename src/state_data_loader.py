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
    elif country == 'US':
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
        df_state.to_csv(os.path.join(config.base_data_dir, config.state_covid19_cases.format(target_state)))


def load_us(states, latest=False):
    us_covid19_cases_path = os.path.join(config.base_data_dir, config.us_covid19_cases)

    if not os.path.isfile(us_covid19_cases_path) or latest:
        g = github.Github()
        while True:
            try:
                repo = g.get_repo("CSSEGISandData/COVID-19")
                files = repo.get_contents("csse_covid_19_data/csse_covid_19_daily_reports_us")
                break
            except github.RateLimitExceededException as e:
                print(e)
                time.sleep(10)

        columns = ['Province_State', 'Confirmed', 'Deaths', 'Recovered']

        df_list = []

        for f in files:
            if f.name.endswith(".csv"):
                str_io = StringIO(f.decoded_content.decode("utf-8"))
                time.sleep(1)

                df = pd.read_csv(str_io, usecols=columns)
                df['Date'] = f.name.split('.')[0]
                df_list.append(df)

        df_concat = pd.concat(df_list, axis=0, ignore_index=True)
        df_concat.rename(columns={'Deaths': 'Deceased'}, inplace=True)
        df_concat.to_csv(os.path.join(config.base_data_dir, config.us_covid19_cases), index=False)

    df_us = pd.read_csv(us_covid19_cases_path)

    for target_state in states:
        df_state = df_us.loc[df_us['Province_State'] == target_state].drop('Province_State', axis=1)

        df_state.index = pd.to_datetime(df_state.Date)
        df_state['Recovered'].fillna(value=0, inplace=True)

        df_state[['Total_Confirmed', 'Total_Deceased', 'Total_Recovered']] \
            = df_state[['Confirmed', 'Deceased', 'Recovered']].cumsum(axis=0)

        df_state.to_csv(os.path.join(config.base_data_dir, f'Cases_US_{target_state}.csv'), index=False)
