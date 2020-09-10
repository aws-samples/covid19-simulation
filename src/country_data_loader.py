import config
import os
import pandas as pd
import numpy as np
from datetime import date, datetime, time, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

CONF_CASES_THRESHOLD = 25000

# src_confirmed = config.confirmed_cases_global_online
# src_recovered = config.recovered_cases_global_online
# src_dead = config.deceased_cases_global_online

src_confirmed = os.path.join(config.base_data_dir, config.confirmed_cases_global_offline)
src_recovered = os.path.join(config.base_data_dir, config.recovered_cases_global_offline)
src_dead = os.path.join(config.base_data_dir, config.deceased_cases_global_offline)

def convert (df_src, col):
    df = df_src.copy()
    df = df.rename(columns={'Country/Region':'Country'})
    df.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True, errors='ignore')
    df = df.melt(id_vars=["Country"], 
            var_name="Date", 
            value_name=col)
    

    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df_refined = pd.DataFrame(columns=['Country', col])
    all_countries = df['Country'].unique()
    for country in all_countries:
        dfc = df.loc[df['Country']==country]
        dfc_grouped = dfc.groupby('Date', as_index=False)[col].sum()
        dfc_grouped['Country'] = country
        dfc_grouped['Date_Copy'] = dfc_grouped['Date']
        dfc_grouped['merge_col'] = dfc_grouped['Country'] + '-' + dfc_grouped['Date'].astype(str)
        df_refined = pd.concat([df_refined, dfc_grouped], axis=0)
    
    df_refined = df_refined.set_index('Date')
    
    return df_refined


def load ():
    try:
        df_confirmed = pd.read_csv(src_confirmed)
        df_recovered = pd.read_csv(src_recovered)
        df_dead = pd.read_csv(src_dead)
    except FileNotFoundError:
        print ('Error!!! Missing Input Data File(s)!')
        return

    df_confirmed_updated = convert (df_confirmed, 'Confirmed')
    df_recovered_updated = convert (df_recovered, 'Recovered')
    df_dead_updated = convert (df_dead, 'Dead')

    df_combined = df_confirmed_updated.merge(df_recovered_updated, on=['merge_col'], how='inner')
    df_combined = df_combined.merge(df_dead_updated, on=['merge_col'], how='inner')
    df_combined.drop(['Country_x', 'Country_y', 'Date_Copy_x', 'Date_Copy_y', 'merge_col'], axis=1, inplace=True)
    df_combined = df_combined.rename(columns={'Date_Copy':'Date', 
                                              'Confirmed':'Total_Confirmed', 
                                              'Recovered':'Total_Recovered', 
                                              'Dead':'Total_Deceased'})
    df_combined['Confirmed'] = df_combined['Total_Confirmed'].diff()
    df_combined['Recovered'] = df_combined['Total_Recovered'].diff()
    df_combined['Deceased'] = df_combined['Total_Deceased'].diff()
    df_combined = df_combined[['Date', 'Country', 'Confirmed', 'Recovered', 'Deceased', 'Total_Confirmed', 'Total_Recovered', 'Total_Deceased']].set_index('Date')

    df_combined.loc[df_combined['Country']=='India'].tail(10)

    all_countries = df_combined['Country'].unique()
    for country in all_countries:
        dfc = df_combined.loc[df_combined['Country'] == country]
        if dfc['Total_Confirmed'].max() >= CONF_CASES_THRESHOLD:
            print (f'Storing data for country: {country} ' + ' *' * 10)
            dfc = dfc.iloc[1:]
            dfc.to_csv(os.path.join(config.base_data_dir, config.country_covid19_cases.format(country)))
        

# if __name__ == '__main__':
#     load ()