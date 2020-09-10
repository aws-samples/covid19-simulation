import os

import pandas as pd

import config
import utils


def load(country, counties, latest=False):
    """
    :param country: str
    :param counties: [(state, county)]
    :param latest: bool
    :return: None
    """

    if country == 'USA':
        load_us(counties, latest)
    else:
        raise ValueError(f'Loading county data for {country} is not supported.')


def load_us(counties, latest=False):
    """
    :param counties: [(state, county)]
    :param latest: bool
    :return: None
    """

    us_counties_path = os.path.join(config.base_data_dir, config.us_counties_cases_offline)

    if not os.path.isfile(us_counties_path) or latest:
        df = utils.download_latest_data(config.us_counties_cases_online)
        df.drop('fips', axis=1, inplace=True)

        # df_concat = pd.concat(df, axis=0, ignore_index=True)
        df.rename(columns={'cases': 'Confirmed', 'deaths': 'Deceased', 'date': 'Date'}, inplace=True)
        df.to_csv(us_counties_path, index=False)

    df = pd.read_csv(us_counties_path)

    for state, county in counties:
        df_county = df.loc[df.county == county]
        df_county = df_county.loc[df_county.state == state].drop(['state', 'county'], axis=1)

        df_county.index = pd.to_datetime(df_county.Date)

        df_county[['Total_Confirmed', 'Total_Deceased']] = df_county[['Confirmed', 'Deceased']].cumsum(axis=0)

        df_county.to_csv(os.path.join(config.base_data_dir, f'Cases_USA_{state}_{county}.csv'), index=False)
