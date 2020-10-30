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


class LocationCaseStats:
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
        self.w1_10_pctl_rel_case_rate = np.percentile(wave1_rel_case_rates, 10)
        self.w1_25_pctl_rel_case_rate = np.percentile(wave1_rel_case_rates, 25)
        self.w1_50_pctl_rel_case_rate = np.percentile(wave1_rel_case_rates, 50)
        self.w1_75_pctl_rel_case_rate = np.percentile(wave1_rel_case_rates, 75)
        self.w1_90_pctl_rel_case_rate = np.percentile(wave1_rel_case_rates, 90)
        
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
            #plt.plot(np.arange(len(periodic_changes_rel)), periodic_changes_rel)
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

#             ### Plotting Interventions
#             intervention_scores_loc = os.path.join(config.base_data_dir, config.intervention_scores_loc)
#             intv_scores = pd.read_csv(intervention_scores_loc)
#             country_intv_scores = intv_scores.loc[intv_scores['CountryCode'] == country]
#             country_intv_scores['Date'] = pd.to_datetime(country_intv_scores['Date'])
#             df_country = pd.merge(df_country, country_intv_scores[['Date', 'aggr_weighted_intv_norm']], how='left', on=['Date'])
#             temp = df_country['aggr_weighted_intv_norm'] * 100
#             plt.plot(np.arange(len(df_country['ConfirmedCases'])), temp)

#             plt.plot(np.arange(len(df_country['ConfirmedCases'])), df_country['StringencyIndex'])
#             plt.show()
        except:
            print ('Error for country: ' + country)
    
    
    def get_disease_wave_details (self, df_country, country=None):
        periodic_changes, periodic_changes_rel = self.measure_rel_changes(df_country['ConfirmedCases'])
        #periodic_changes_all[country] = periodic_changes
        #periodic_relative_changes_all[country] = periodic_changes_rel

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
        
        rising_idx = -1
        if len(rising_diffs_idx) > 0:
            rising_idx = rising_diffs_idx[1] if len(rising_diffs_idx) > 1 else rising_diffs_idx[0]
            #rising_idx = min (len(periodic_changes_smoothed)-1, rising_idx + poly_smoothing_degree)
            rising_idx = min (len(periodic_changes_smoothed)-1, rising_idx + 1)

        declining_diffs_idx = np.where(periodic_changes_smoothed <= 0)[0]
        consecutive_drops = consecutive(declining_diffs_idx)
        
        flattenning_idx = -1
        for k in range(len(consecutive_drops)):
            if consecutive_drops[k][0] > rising_idx:
                flattenning_idx = consecutive_drops[k][1] if len(consecutive_drops[k]) > 1 else consecutive_drops[k][0]
                break

        if rising_idx > flattenning_idx > -1:
            flattenning_idx, rising_idx = -1, -1

        second_rising_idx = -1
        if len(consecutive_rises) > 1 and flattenning_idx > -1 \
                    and rising_idx < flattenning_idx < consecutive_rises[1][0] \
                            and periodic_changes[consecutive_rises[1][0]] < 0.9 * periodic_changes[flattenning_idx]:
            second_rising_idx = consecutive_rises[1][0]

        second_flattenning_idx = -1
        for k in range(len(consecutive_drops)):
            if consecutive_drops[k][0] > flattenning_idx and second_rising_idx > -1 and second_rising_idx < consecutive_drops[k][0]:
                second_flattenning_idx = consecutive_drops[k][0]
                break
        
        offset = len(df_country) % len(periodic_changes_smoothed)
        
        if country is not None:
            self.plot_waves (country, df_country, periodic_changes, periodic_changes_smoothed_src, rising_idx, flattenning_idx, second_rising_idx, second_flattenning_idx)
        
        return rising_idx, flattenning_idx, second_rising_idx, second_flattenning_idx, offset
    
    
    def get_date (self, dates, index):
        if index > -1:
            return dates[index]
        else:
            return None

        
    def get_disease_details (self, df_country, n_population):
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
            
            locationCaseStats =  LocationCaseStats(df_country, n_population, w1_case_rates, w1_rel_case_rates, 
                                                   w1_start_dt, w1_peak_dt, w2_case_rates, w2_rel_case_rates, w2_start_dt, w2_peak_dt)
            return locationCaseStats
        
    
    def load_all_disease_details (self, country):
        country_disease_stats = dict()
        for country in self.country_dict:
            df_country = self.df.loc[self.df['CountryCode'] == country]
            country_disease_stats[country] = get_disease_details (df_country, self.country_populations[country])
        return country_disease_stats
    
    
    # Calculate case-rate-change, rate of case-rate-change and time index of reaching peak (if at all)
    def get_periodic_relative_changes(self, target_country):
        period = self.period
        periodic_changes_all = dict()
        periodic_relative_changes_all = dict()
        flattenning_idx_all = dict()
        for country in self.country_dict:
            df_country = self.df.loc[self.df['CountryCode'] == country]
            rising_idx, flattenning_idx, second_rising_idx, second_flattenning_idx, offset = self.get_disease_wave_details (df_country, country=country)
            
            flattenning_idx_all[country] = flattenning_idx
            start_idx = rising_idx * config.period + offset
            end_idx = flattenning_idx * config.period - offset
            
            if country == target_country:
                print ('*** rising_idx: {}  |  flattenning_idx: {}'.format(start_idx, end_idx))
            
            if start_idx >= 0:
                if end_idx > 0:
                    periodic_changes, periodic_changes_rel = self.measure_rel_changes(
                        df_country['ConfirmedCases'].iloc[start_idx:end_idx+1])
                else:
                    periodic_changes, periodic_changes_rel = self.measure_rel_changes(
                        df_country['ConfirmedCases'].iloc[start_idx:])
                periodic_changes_all[country] = periodic_changes
                periodic_relative_changes_all[country] = periodic_changes_rel
                
            
        return periodic_changes_all, periodic_relative_changes_all, flattenning_idx_all
    
    
    # Get countries sorted by similarity in case-rate trend
    # Possible area of improvement:
    # The infection rates tend to be low during the early stage of its spread as well while reaching the 
    # plateau, for all countries. 
    # Hence, depending upon the simulation starting point, if the target geography's infection-rate changes 
    # are captured from a very early or a very late stage of the spread, there will be too many matching coutries, 
    # resulting into unreliable estimation of the infection-rate and time to reach the wave-1 peak.
    # This might be improved with more validations and dynamic forward and backward sliding-windows to understand the 
    # best time-span for estimating the simulation parameters for a geography.    
    def get_top_matches (self, src_loc_changes_rel, periodic_relative_changes, flattenning_idx):
        match_spans = self.match_spans
        matches = list()
        weights = 1 / np.arange(1, match_spans + 1)[::-1]
        
        for country in periodic_relative_changes:
            for i in range(len(periodic_relative_changes[country])):
                if i + match_spans <= len(periodic_relative_changes[country]):
                    
                    dist = mean_squared_error(
                        src_loc_changes_rel, periodic_relative_changes[country][i:i + match_spans], weights)
                    
                    if flattenning_idx[country] < 0:
                        continue
                    
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
    def measure_bounds (self, df_matches, periodic_relative_changes):
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
    def get_bounds (self, df_loc, target_country):
                
        self.target_country = target_country
        self.df = self.get_conf_cases(start_date=datetime.datetime.strptime(config.overall_start_date, "%d%m%Y"))

        selected_countries = self.df[['CountryCode', 'CountryName']].drop_duplicates(keep='first').dropna()
        self.country_dict, self.country_populations = self.get_population_data(selected_countries)

        if self.country_level_projection:
            conf_cases = self.df.loc[self.df['CountryCode']==target_country, 'ConfirmedCases']
            conf_cases = conf_cases.iloc[:len(df_loc)+1]
        else:
            conf_cases = df_loc['Total_Confirmed']
            
        _, src_loc_changes_rel = self.measure_rel_changes(conf_cases)
        src_loc_changes_rel = src_loc_changes_rel[-self.match_spans:]

        periodic_changes_all, periodic_relative_changes_all, flattenning_idx_all = self.get_periodic_relative_changes(target_country)
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
        
        higher_bound, lower_bound, avg_time_to_peaks, mean_relative_change_rates, relevant_countries_count = self.measure_bounds (df_matches_selected, periodic_relative_changes_all)

        #self.get_weighted_rates_of_change(df_loc, periodic_relative_changes_all, df_matches_selected, avg_time_to_peaks)
        
        if flattenning_idx_all[target_country] > -1:
            avg_time_to_peaks = flattenning_idx_all[target_country]
        
        return higher_bound, lower_bound, avg_time_to_peaks, mean_relative_change_rates, relevant_countries_count
