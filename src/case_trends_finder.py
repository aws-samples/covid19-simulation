import config
import os
import pandas as pd
import numpy as np
import itertools
import datetime
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (12,4)

import warnings
warnings.filterwarnings("ignore")


class GeoTransmissionStats:
    def __init__(self, country_code, n_population, wave1_case_rates, wave1_rel_case_rates, wave1_start_dt, wave1_peak_dt, 
                 wave2_case_rates, wave2_rel_case_rates, wave2_start_dt, wave2_peak_dt):
        self.country_code = country_code
        self.n_population = n_population
        self.w1_case_rates = wave1_case_rates
        self.w1_rel_case_rates = wave1_rel_case_rates
        
        if wave1_start_dt is None:
            return

        self.w1_start_dt = wave1_start_dt
        self.w1_peak_dt = wave1_peak_dt
        self.w1_days_to_peak = None if wave1_peak_dt is None else (wave1_peak_dt - wave1_start_dt).days
        
        self.w1_mean_case_rate = np.mean(wave1_case_rates)
        self.w1_median_case_rate = np.median(wave1_case_rates)
        self.w1_max_case_rate = np.max(wave1_case_rates)
        self.w1_10_pctl_case_rate = np.percentile(wave1_case_rates, 10)
        self.w1_25_pctl_case_rate = np.percentile(wave1_case_rates, 25)
        self.w1_case_rate_std_dev = np.std(wave1_case_rates)
        self.w1_mean_rel_case_rate = np.mean(wave1_rel_case_rates)
        self.w1_rel_case_rate_std_dev = np.std(wave1_rel_case_rates)
        
        self.w2_case_rates = wave2_case_rates
        self.w2_rel_case_rates = wave2_rel_case_rates
        
        self.w2_start_dt = wave2_start_dt
        self.w2_peak_dt = wave2_peak_dt
        self.w2_days_to_peak = None
        if wave2_start_dt is not None and wave2_peak_dt is not None and wave2_peak_dt > wave2_start_dt:
            self.w2_days_to_peak = (wave2_peak_dt - wave2_start_dt).days
        
        self.days_between_disease_waves = None
        if self.w2_peak_dt is not None:
            self.days_between_disease_waves = (self.w2_peak_dt - self.w1_peak_dt).days
        elif self.w2_start_dt is not None and self.w1_peak_dt is not None:
            self.days_between_disease_waves = (self.w2_start_dt - self.w1_start_dt).days + (self.w1_peak_dt - self.w1_start_dt).days
            
        self.w2_mean_case_rate = None if wave2_case_rates is None else np.mean(wave2_case_rates)
        self.w2_mean_rel_case_rate = None if wave2_rel_case_rates is None else np.mean(wave2_rel_case_rates)
        


class geo_transmission_analyzer:
    
    ### Primary data source
    DATA_SRC = os.path.join(config.base_data_dir, config.oxcgrt_intervention_data_offline)
    ### Case count threshold to consider countries
    CONF_CASES_THRESHOLD = config.min_country_conf_case_threshold
    COUNTRY_POPULATION_SRC = os.path.join(config.base_data_dir, config.country_populations_data)

    def __init__(self, src_population, country_level_projection=False):
        self.target_country = None
        self.df = None
        self.country_level_projection = country_level_projection
        self.country_dict = None
        self.country_populations = None
        self.src_population = src_population
        self.period = config.period #time-span length in days

    # Get confirmed covid19 cases data for countries
    def get_conf_cases (self, start_date=None):
        data_extended = pd.read_csv(self.__class__.DATA_SRC)
        data_extended = data_extended.loc[(data_extended['RegionCode'] == '') | (data_extended['RegionCode'].isnull())]
        data_extended['Date'] = pd.to_datetime(data_extended['Date'], format='%Y%m%d')

        if start_date is not None:
            data_extended = data_extended.loc[data_extended['Date'] >= start_date]

        df_populations = pd.read_csv(self.__class__.COUNTRY_POPULATION_SRC)
        #might choose population from later years as well based on the population data used for simulation
        df_populations['2011 [YR2011]'] = df_populations['2011 [YR2011]'].astype(int)

        selected_countries = data_extended.loc[
            (data_extended['ConfirmedCases'] > self.__class__.CONF_CASES_THRESHOLD), 'CountryCode'].unique()
        
        data_extended = data_extended.loc[data_extended['CountryCode'].isin(selected_countries)]
        df = data_extended[
            ['Date', 'CountryCode', 'CountryName', 'ConfirmedCases', 'ConfirmedDeaths', 'StringencyIndex']]
        # df = df.fillna(method='ffill')
        df = df.fillna(0)
        
        ### Remove last few entries for each country - due to possible data error / deplayed reporting
        for c in df['CountryCode'].unique().tolist():
            df.loc[df['CountryCode'] == c] = df.loc[df['CountryCode'] == c].iloc[:-2]

        return df


    def get_population_data(self, countries):
        df_populations = pd.read_csv(self.__class__.COUNTRY_POPULATION_SRC)
        df_populations['2011 [YR2011]'] = df_populations['2011 [YR2011]'].astype(int)
        country_dict = dict()
        country_populations = dict()
        for cc in countries['CountryCode']:
            if cc in df_populations['Country Code'].tolist():
                country_dict[cc] = countries.loc[countries['CountryCode'] == cc, 'CountryName'].tolist()[0]
                country_populations[cc] = df_populations.loc[df_populations['Country Code'] == cc, '2011 [YR2011]'].iloc[0]
        return country_dict, country_populations
    
    
    # Measure relative rate of change
    def measure_rel_changes (self, conf_cases):
        period = self.period
        periodic_changes = list()
        #print (conf_cases.tolist())
        for i in range(int(len(conf_cases) / period) + 1):
            end_index = i * period + period
            end_index = end_index if end_index <= len(conf_cases) else len(conf_cases)
            if (end_index - i * period) < np.ceil(period / 2):
                continue
            conf_cases_subset = conf_cases[i * period:end_index]

            mean_change = np.mean(np.diff(conf_cases_subset)) if period > 1 else conf_cases_subset[0]
            periodic_changes.append(mean_change)
            #print (conf_cases_subset.tolist(), mean_change)
        periodic_changes = np.array(periodic_changes)
        
        periodic_changes[(periodic_changes == -np.inf) | (periodic_changes == np.inf) | 
                             (periodic_changes == np.nan) | (np.isnan(periodic_changes))] = 0
        periodic_changes[periodic_changes < 0] = 0
        
        periodic_changes_rel = periodic_changes[1:] / periodic_changes[:-1]
        periodic_changes_rel[(periodic_changes_rel == -np.inf) | (periodic_changes_rel == np.inf) | 
                             (periodic_changes_rel == np.nan) | (np.isnan(periodic_changes_rel))] = 0

        return periodic_changes, periodic_changes_rel  
    
    # Optional function to plot disease waves, peaks etc.
    def plot_waves (self, country, df_country, periodic_changes, periodic_changes_smoothed_src, rising_idx, flattenning_idx, 
                    second_rising_idx, second_flattenning_idx):
        try:
            offset = len(df_country) % len(periodic_changes_smoothed_src)
            offset = 0
            if rising_idx > -1:
                start = rising_idx * config.period + offset
                if flattenning_idx > -1:
                    end = flattenning_idx * config.period - offset
                    print ('\n* * * {}: Time to reach wave-1 peak: {} days | Start Dt:{} | End Dt:{} | Offset: {}'.format(
                            self.country_dict[country], 
                            end - start,
                            df_country['Date'].iloc[start].date(),
                            df_country['Date'].iloc[end].date(), 
                            offset)
                        )
                else:
                    print ('\n* * * {}: Time so far: {} days | Start Dt:{}'.format(
                            self.country_dict[country], 
                            len(df_country['ConfirmedCases']) - start,
                            df_country['Date'].iloc[start].date())
                        )

            ### Plotting peaks/troughs against changes
            plt.plot(np.arange(len(periodic_changes)), periodic_changes)
            plt.plot(np.arange(len(periodic_changes_smoothed_src)), periodic_changes_smoothed_src)
            if flattenning_idx > -1:
                plt.plot(flattenning_idx, periodic_changes[flattenning_idx], 'o')
            if rising_idx > -1:
                plt.plot(rising_idx, periodic_changes[rising_idx], 'X')
            if second_rising_idx > -1:
                plt.plot(second_rising_idx, periodic_changes[second_rising_idx], 'o')
            if second_flattenning_idx > -1:
                plt.plot(second_flattenning_idx, periodic_changes[second_flattenning_idx], 'X')            
            plt.show();

        except:
            print ('Error for country: ' + country)
    
    # Detect disease waves and respective starts & peaks
    def get_disease_wave_details (self, df_country, country=None):
        
        periodic_changes, periodic_changes_rel = self.measure_rel_changes(df_country['ConfirmedCases'])

        def consecutive(data, stepsize=1, min_length=10):
            splits = np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)
            splits = [split for split in splits if len(split) >= min_length]
            splits = splits if len(splits) > 0 else list()
            return splits

        #Smoothen the rates using savgol_filter before finding the peak
        window_len = int(len(periodic_changes)/2)-1 if int(len(periodic_changes)/2)%2==0 else int(len(periodic_changes)/2)
        poly_smoothing_degree = 6
        periodic_changes_smoothed_src = savgol_filter(periodic_changes, window_len, poly_smoothing_degree)
        periodic_changes_smoothed_src[periodic_changes_smoothed_src < 0] = 0
        ### To ensure that small upward oscillations are not picked as start of a wave
        periodic_changes_smoothed_cutoff = np.percentile(periodic_changes_smoothed_src, 95) * 0.02 
        periodic_changes_smoothed_src[periodic_changes_smoothed_src < periodic_changes_smoothed_cutoff] = 0

        periodic_changes_smoothed = np.diff(periodic_changes_smoothed_src)

        rising_diffs_idx = np.where(periodic_changes_smoothed > 0.5)[0]
        consecutive_rises = consecutive(rising_diffs_idx)
        rising_diffs_idx = consecutive_rises[0] if len(consecutive_rises) > 0 else list()
        
        # Capture start of 1st wave
        rising_idx = -1
        if len(rising_diffs_idx) > 0:
            rising_idx = rising_diffs_idx[1] if len(rising_diffs_idx) > 1 else rising_diffs_idx[0]
            rising_idx = min (len(periodic_changes_smoothed)-1, rising_idx + 1)

        declining_diffs_idx = np.where(periodic_changes_smoothed <= 0)[0]
        consecutive_drops = consecutive(declining_diffs_idx)
        
        # Capture peak of 1st wave
        flattenning_idx = -1
        for k in range(len(consecutive_drops)):
            if consecutive_drops[k][0] > rising_idx:
                flattenning_idx = consecutive_drops[k][1] if len(consecutive_drops[k]) > 1 else consecutive_drops[k][0]
                break

        if rising_idx > flattenning_idx > -1:
            flattenning_idx, rising_idx = -1, -1
        
        # Capture start of 2nd wave
        second_rising_idx = -1
        if len(consecutive_rises) > 1 and flattenning_idx > -1 \
                    and rising_idx < flattenning_idx < consecutive_rises[1][0] \
                            and periodic_changes[consecutive_rises[1][0]] < 0.9 * periodic_changes[flattenning_idx]:
            second_rising_idx = consecutive_rises[1][0]
        
        # Capture peak of 2nd wave
        second_flattenning_idx = -1
        for k in range(len(consecutive_drops)):
            if consecutive_drops[k][0] > flattenning_idx and second_rising_idx > -1 and second_rising_idx < consecutive_drops[k][0]:
                second_flattenning_idx = consecutive_drops[k][0]
                break
        
        offset = len(df_country) % len(periodic_changes_smoothed)
        
        if country is not None:
            self.plot_waves (country, df_country, periodic_changes, periodic_changes_smoothed_src, 
                             rising_idx, flattenning_idx, second_rising_idx, second_flattenning_idx)
        
        return rising_idx, flattenning_idx, second_rising_idx, second_flattenning_idx, offset
    
    
    def get_date (self, dates, index):
        if index > -1:
            return dates[index]
        else:
            return None

    
    # Get various stats on disease transmission patterns (rate, start dates, peaks, etc.)
    def get_disease_transmission_stats (self, df_country, n_population):
        w1_rising_idx, w1_flattenning_idx, w2_rising_idx, w2_flattenning_idx, offset = self.get_disease_wave_details (df_country)
        w1_start_idx = w1_rising_idx * config.period + offset
        w1_end_idx = w1_flattenning_idx * config.period - offset
                
        if w1_start_idx >= 0:
            if w1_end_idx > 0:
                w1_case_rates, w1_rel_case_rates = self.measure_rel_changes(
                    df_country['ConfirmedCases'].iloc[w1_start_idx:w1_end_idx+1])
            else:
                w1_case_rates, w1_rel_case_rates = self.measure_rel_changes(
                    df_country['ConfirmedCases'].iloc[w1_start_idx:])
            
            w2_start_idx, w2_end_idx = -1, -1
            w2_case_rates, w2_rel_case_rates = None, None
            if w2_rising_idx >= 0:
                w2_start_idx = w2_rising_idx * config.period + offset
                w2_end_idx = w2_flattenning_idx * config.period - offset
                if w2_end_idx > 0:
                    w2_case_rates, w2_rel_case_rates = self.measure_rel_changes(
                        df_country['ConfirmedCases'].iloc[w2_start_idx:w2_end_idx+1])
                else:
                    w2_case_rates, w2_rel_case_rates = self.measure_rel_changes(
                        df_country['ConfirmedCases'].iloc[w2_start_idx:])

            w1_start_dt = self.get_date (df_country['Date'], w1_start_idx) if w1_start_idx > -1 else None
            w1_peak_dt = self.get_date (df_country['Date'], w1_end_idx) if w1_end_idx > -1 else None
            w2_start_dt = self.get_date (df_country['Date'], w2_start_idx) if w2_start_idx > -1 else None
            w2_peak_dt = self.get_date (df_country['Date'], w2_end_idx) if w2_end_idx > -1 else None
            
            geoTransmissionStats =  GeoTransmissionStats(df_country, n_population, w1_case_rates, w1_rel_case_rates, 
                                                   w1_start_dt, w1_peak_dt, w2_case_rates, w2_rel_case_rates, w2_start_dt, w2_peak_dt)
            return geoTransmissionStats
        else:
            return None
        

