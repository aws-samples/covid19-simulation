import config
import os
import pandas as pd
import numpy as np
import itertools
import datetime
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter


import warnings
warnings.filterwarnings("ignore")


class case_trends_match_finder:
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
        self.match_spans = config.match_spans #number of periods/time-spans for rate change calculation & comparison
        self.period = config.period #time-span length in days

    # Get confirmed covid19 cases data for countries
    def get_conf_cases(self, start_date=None):
        data_extended = pd.read_csv(self.__class__.DATA_SRC)
        data_extended['Date'] = pd.to_datetime(data_extended['Date'], format='%Y%m%d')

        if start_date is not None:
            data_extended = data_extended.loc[data_extended['Date'] >= start_date]

        df_populations = pd.read_csv(self.__class__.COUNTRY_POPULATION_SRC)
        df_populations['2011 [YR2011]'] = df_populations['2011 [YR2011]'].astype(int)

        selected_countries = data_extended.loc[
            (data_extended['ConfirmedCases'] > self.__class__.CONF_CASES_THRESHOLD), 'CountryCode'].unique()
        
        data_extended = data_extended.loc[data_extended['CountryCode'].isin(selected_countries)]
        df = data_extended[
            ['Date', 'CountryCode', 'CountryName', 'ConfirmedCases', 'ConfirmedDeaths', 'StringencyIndex']]
        # df = df.fillna(method='ffill')
        df = df.fillna(0)

        ### Remove last 2 entries for each country - due to possible data error / deplayed reporting
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
    
    
    # Measure relative rate of change, i.e. rate of rate of change
    def measure_rel_changes(self, conf_cases):
        period = self.period
        periodic_changes = list()
        for i in range(int(len(conf_cases) / period) + 1):
            end_index = i * period + period
            end_index = end_index if end_index <= len(conf_cases) else len(conf_cases)
            if (end_index - i * period) < np.ceil(period / 2):
                continue
            conf_cases_subset = conf_cases[i * period:end_index]

            mean_change = np.mean(np.diff(conf_cases_subset)) if period > 1 else conf_cases_subset[0]
            periodic_changes.append(mean_change)
        periodic_changes = np.array(periodic_changes)
        
        periodic_changes[(periodic_changes == -np.inf) | (periodic_changes == np.inf) | 
                             (periodic_changes == np.nan) | (np.isnan(periodic_changes))] = 0
        periodic_changes[periodic_changes < 0] = 0
        
        periodic_changes_rel = periodic_changes[1:] / periodic_changes[:-1]
        periodic_changes_rel[(periodic_changes_rel == -np.inf) | (periodic_changes_rel == np.inf) | 
                             (periodic_changes_rel == np.nan) | (np.isnan(periodic_changes_rel))] = 0

        return periodic_changes, periodic_changes_rel   
    
    
    # Calculate case-rate-change, rate of case-rate-change and time index of reaching peak (if at all)
    def get_periodic_relative_changes(self):
        period = self.period
        periodic_changes_all = dict()
        periodic_relative_changes_all = dict()
        flattenning_idx_all = dict()
        for country in self.country_dict:
            df_country = self.df.loc[self.df['CountryCode'] == country]
            periodic_changes, periodic_changes_rel = self.measure_rel_changes(df_country['ConfirmedCases'])
            periodic_changes_all[country] = periodic_changes
            periodic_relative_changes_all[country] = periodic_changes_rel
            
            def consecutive(data, stepsize=1, min_length=10):
                splits = np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)
                splits = [split for split in splits if len(split) >= min_length]
                splits = splits[0] if len(splits) > 0 else list()
                return splits
            
            #Smoothen the rates using savgol_filter before finding the peak
            window_len = int(len(periodic_changes)/2)-1 if int(len(periodic_changes)/2)%2==0 else int(len(periodic_changes)/2)
            periodic_changes_smoothed = savgol_filter(periodic_changes, window_len, 10)
            
            periodic_changes_smoothed = np.diff(periodic_changes_smoothed)
            declining_diffs_idx = np.where(periodic_changes_smoothed <= 0)[0]
            
            declining_diffs_idx = consecutive(declining_diffs_idx)
            
            flattenning_idx = -1
            if len(declining_diffs_idx) > 0:
                flattenning_idx = declining_diffs_idx[1] if len(declining_diffs_idx) > 1 else declining_diffs_idx[0]
            flattenning_idx_all[country] = flattenning_idx
            
        return periodic_changes_all, periodic_relative_changes_all, flattenning_idx_all


    # Get countries sorted by similarity in case-rate trend
    def get_top_matches(self, src_loc_changes_rel, periodic_relative_changes, flattenning_idx):
        match_spans = self.match_spans
        matches = list()
        weights = 1 / np.arange(1, match_spans + 1)[::-1]
        
        #print (src_loc_changes_rel)
        for country in periodic_relative_changes:
            for i in range(len(periodic_relative_changes[country])):
                #if country != self.target_country and i + match_spans <= len(periodic_relative_changes[country]):
                if i + match_spans <= len(periodic_relative_changes[country]):
                    
                    dist = mean_squared_error(
                        src_loc_changes_rel, periodic_relative_changes[country][i:i + match_spans], weights)
                    
                    time_to_peak = flattenning_idx[country] - i - 1
                    time_to_peak = -1 if time_to_peak < 0 else time_to_peak * self.period
                    population_ratio = -1
                    population_diff = -1
                    if self.src_population > 0 and country in self.country_populations:
                        population_ratio = self.country_populations[country] / self.src_population
                        population_diff = abs(self.country_populations[country] - self.src_population)
                    matches.append(
                        [country, self.country_dict[country], i + 1, dist, time_to_peak, population_ratio, population_diff])
        df_dist = pd.DataFrame(matches,
                               columns=['country', 'country_name', 'index', 'dist', 'time_to_peak', 'population_ratio',
                                        'population_diff'])
                
        df_dist = df_dist.sort_values(by='dist').drop_duplicates(subset=['country'], keep='first')
        #print (df_dist)
        
        return df_dist


    # Calculate various potential deviations of case rates (min, max, mean, median etc.) from countries that exhibited matching
    # case rate trends (rate of rate of change) ats some point in time
    def measure_bounds(self, df_matches, periodic_relative_changes):
        matching_countries = list()
        higher_rates = list()
        lower_rates = list()
        time_to_peaks = list()
        match_sims = list()
        peaked_match_sims = list()
        matching_relative_change_rates = list()
        for i, row in df_matches.iterrows():
            country, index, match_sim, time_to_peak = row['country'], row['index'], row['combined_similarity'], row[
                'time_to_peak']
            if time_to_peak >= 0:
                matching_countries.append(country)
                relative_change_rates = periodic_relative_changes[country][index:]
                higher_rates.append(np.max(relative_change_rates) / np.max(relative_change_rates[:self.match_spans]))
                lower_rates.append(np.min(relative_change_rates[relative_change_rates>0]) / np.median(relative_change_rates[:self.match_spans]))
                match_sims.append(match_sim)
                time_to_peaks.append(time_to_peak)
                peaked_match_sims.append(match_sim)
                matching_relative_change_rates.append(relative_change_rates)
       
        relevant_countries_count = len(match_sims)
        print (f"Countries with matching trend and peak followed by decline # [{relevant_countries_count}] - [{matching_countries}]")

        higher_rates, lower_rates, match_sims, peaked_match_sims = np.array(higher_rates), np.array(
            lower_rates), np.array(match_sims), np.array(peaked_match_sims)
        
        # Detect higher & lower bounds and time-to-peak as weighted averages
        higher_bound = np.sum(higher_rates * (peaked_match_sims / np.sum(peaked_match_sims)))
        lower_bound = np.sum(lower_rates * (peaked_match_sims / np.sum(peaked_match_sims)))
        
        time_to_peak = np.sum(time_to_peaks * (peaked_match_sims / np.sum(peaked_match_sims)))
        
        # Calculate mean relative rates of case rate changes upto the time_to_peak
        mean_relative_change_rates = [0 for i in range(int(time_to_peak))]
        for i in range(int(time_to_peak)):
            rates_found = 0
            for j in range(len(matching_relative_change_rates)):
                rate_changes_comp = matching_relative_change_rates[j]
                if len(rate_changes_comp) > i:
                    mean_relative_change_rates[i] += rate_changes_comp[i]
                    rates_found += 1
            if rates_found > 0:
                mean_relative_change_rates[i] /= rates_found
            
        
        return higher_bound, lower_bound, time_to_peak, mean_relative_change_rates, relevant_countries_count


    # Normalize data within a given range (min_val , max_val)
    def normalize(self, data, min_val=0, max_val=1):
        if len(data) > 1:
            ### Min-MAx Normalization
            data_min, data_max = np.min(data), np.max(data)
            data_norm = (data - data_min) / (data_max - data_min)

            ### Normalize within a given bound (min_val, max_val)
            data_norm_bounded = data_norm * (max_val - min_val) + min_val
        else:
            data_norm_bounded = 1
        return data_norm_bounded


    # Get various bounds / relative-rates and avg-time-to-peak for countries that had similar case-rate-change trends
    def get_bounds(self, df_loc, target_country):
                
        self.target_country = target_country
        #self.df = self.get_conf_cases(start_date=df_loc['Date'].iloc[0])
        self.df = self.get_conf_cases(start_date=datetime.datetime.strptime(config.overall_start_date, "%d%m%Y"))

        selected_countries = self.df[['CountryCode', 'CountryName']].drop_duplicates(keep='first').dropna()
        self.country_dict, self.country_populations = self.get_population_data(selected_countries)

        # df.loc[df['CountryCode']=='IND', 'ConfirmedCases'].plot();
        # conf_cases = self.df.loc[self.df['CountryCode']=='IND', 'ConfirmedCases']

        if self.country_level_projection:
            conf_cases = self.df.loc[self.df['CountryCode']==target_country, 'ConfirmedCases']
            conf_cases = conf_cases.iloc[:len(df_loc)+1]
        else:
            conf_cases = df_loc['Total_Confirmed']
            
        _, src_loc_changes_rel = self.measure_rel_changes(conf_cases)
        src_loc_changes_rel = src_loc_changes_rel[-self.match_spans:]

        periodic_changes_all, periodic_relative_changes_all, flattenning_idx_all = self.get_periodic_relative_changes()
        df_matches = self.get_top_matches(src_loc_changes_rel, periodic_relative_changes_all, flattenning_idx_all)
        
        df_matches_selected = df_matches.loc[df_matches['dist'] <= config.sim_dist_threshold]
        df_matches_selected = df_matches.loc[df_matches['time_to_peak'] > -1]
        
        df_matches_selected['trend_similarity'] = 1 - df_matches_selected['dist'] / df_matches_selected['dist'].max()
        df_matches_selected['trend_similarity'] = self.normalize(df_matches_selected['trend_similarity'], min_val=0.1,
                                                                 max_val=1)

        df_matches_selected['population_similarity'] = 1 - df_matches_selected['population_diff'] / df_matches_selected[
            'population_diff'].max()
        df_matches_selected['population_similarity'] = self.normalize(df_matches_selected['population_similarity'],
                                                                      min_val=0.1, max_val=1)

        ### Best strategy to combine ?
        df_matches_selected['combined_similarity'] = config.trend_sim_weightage * df_matches_selected['trend_similarity'] + config.population_sim_weightage * df_matches_selected['population_similarity']
        
        #print('Top matching countries that had a similar case rate trend:')
        higher_bound, lower_bound, avg_time_to_peaks, mean_relative_change_rates, relevant_countries_count = self.measure_bounds (df_matches_selected, periodic_relative_changes_all)
        # print (f'Higher Bound: {higher_bound}, Lower Bound: {lower_bound}')

        #self.get_weighted_rates_of_change(df_loc, periodic_relative_changes_all, df_matches_selected, avg_time_to_peaks)

        return higher_bound, lower_bound, avg_time_to_peaks, mean_relative_change_rates, relevant_countries_count
