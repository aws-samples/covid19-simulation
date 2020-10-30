import os

import inspect

import config
from case_trends_finder import geo_transmission_analyzer
from simulation import Simulation

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

import pickle

import skopt
import skopt.plots
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime

import gc
import warnings

warnings.filterwarnings("ignore")


class SimulationData:
    def __init__(self):
        self.country_code = None
        self.state_name = None
        self.state_data_orig = None
        self.state_data = None
        self.country_level_projection = None
        self.n_population = None
        self.state_population = None
        self.actual_testing_capacity = None
        self.case_rate = None
        self.adjusted_case_rate = None
        self.scaling_factor = None
        self.wave1_weeks = None
        self.min_initial_infection = None
        self.transmission_prob = None
        self.transmission_control = None
        self.transmission_control_range = None
        self.wave1_weeks_range = None
        self.intervention_scores = None
        self.expected_rates = None
        self.avg_time_to_peaks = None
        self.mean_relative_change_rates = None
        self.intervention_influence_pctg = None
        self.fitment_days = None
        self.test_days = None
        self.projection_days = None
        self.future_projection_days = None
        self.wave1_start_date = None
        self.wave1_peak_detected = None
        self.days_between_disease_waves = None
        self.weeks_between_disease_waves = None
        self.wave2_peak_factor = None
        self.wave2_spread_factor = None
        
    def to_csv(self, csv_name):
        attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        with open(csv_name, 'w+') as f:
            for a in attributes:
                if not(a[0].startswith('__') and a[0].endswith('__')):
                    f.write("%s,%s\n"%(a[0], a[1]))


# Derive incidence rate and fractions of population infected from recent case frequency data
def get_incidence_rate(state_data, rate, population, fitment_days):
    # normalized case rate w.r.t. population
    rate = rate / population / float(config.infected_and_symptomatic_in_population)
    
    avg_active_cases_x_days_back = state_data.iloc[-fitment_days-3 : -fitment_days+2]['Total_Active'].mean() 
    avg_daily_cases_x_days_back = state_data.iloc[-fitment_days-3 : -fitment_days+2]['Confirmed'].mean()

    # approx fraction of active infected population x days back
    active_case_population_fraction_x_days_back = avg_active_cases_x_days_back / population / float(config.infected_and_symptomatic_in_population)
    # approx fraction of total infected population x days back
    daily_case_population_fraction_x_days_back = avg_daily_cases_x_days_back / population / float(config.infected_and_symptomatic_in_population)
    
    #print ("get_incidence_rate", fitment_days, rate, avg_active_cases_x_days_back, avg_daily_cases_x_days_back)
    return rate, active_case_population_fraction_x_days_back, daily_case_population_fraction_x_days_back
    

# Run simulation:
# - for fitment_days with trial params (during training)
# - for (fitment_days + projection_days) with learned params (during testing / projection)
def simulate(sim_data, learning_phase=False):
    testing_capacity = sim_data.actual_testing_capacity * (sim_data.n_population / sim_data.state_population)
                           
    derived_case_rate, active_case_population_fraction_x_days_back, daily_case_population_fraction_x_days_back \
        = get_incidence_rate(sim_data.state_data, sim_data.adjusted_case_rate, sim_data.state_population,
                             sim_data.fitment_days)
    
    derived_case_rate *= sim_data.scaling_factor
            
    Simulation.set_config(time_between_consecutive_pcr_tests=14,
                          attrition_rate=0.05,
                          initial_antibody_immunity_in_population=0.20,
                          add_ab=False)

    if learning_phase:
        n_days = sim_data.fitment_days
    else:
        #n_days = sim_data.fitment_days + sim_data.projection_days
        n_days = sim_data.projection_days
    
    weeks_between_waves = config.gap_weeks_between_disease_waves_default if sim_data.days_between_disease_waves is None else int(round(sim_data.days_between_disease_waves / 7))
    
    simulator = Simulation(sim_data.n_population,
                           n_days,
                           sim_data.wave1_weeks,
                           weeks_between_waves,
                           derived_case_rate,
                           active_case_population_fraction_x_days_back,
                           daily_case_population_fraction_x_days_back,
                           int(testing_capacity),
                           transmission_control=sim_data.transmission_control,
                           transmission_prob=sim_data.transmission_prob,
                           intervention_influence_pctg=sim_data.intervention_influence_pctg, 
                           wave2_peak_factor = sim_data.wave2_peak_factor,
                           wave2_spread_factor = sim_data.wave2_spread_factor,
                           log_results=False
                          )
    
    # Run the simulation to project the spread of infection
    results = simulator.run(learning_phase, n_days=n_days, n_population=sim_data.n_population,
                            intervention_scores=sim_data.intervention_scores)
    
    daily_stats = []
    for dict in results[1]:
        daily_stats.append([dict['Daily New Infection'], dict['Infected working in FC and not in quarantine'],
                            dict['Sent To Quarantine']])
    df_results = pd.DataFrame(daily_stats, columns=['new_cases', 'open_infectious', 'quarantined'])
    
    # Using rolling avg of simulation outcome to smoothen the projection
    df_results = df_results.rolling(10, min_periods=1).mean()
        
    # Scaling the projection for the state's population
    df_results = df_results * (sim_data.state_population / sim_data.n_population)

    df_results['total_cases'] = df_results['new_cases'].cumsum(axis=0, skipna=True)
    # Accommodate the prior (before the fitment period stat date) total confirmed cases into the projected numbers
    df_results['total_cases'] += sim_data.state_data['Total_Confirmed'].iloc[-sim_data.fitment_days]
    
    start_date = sim_data.wave1_start_date
    dates = pd.date_range(start_date, periods=len(daily_stats), freq='D')
    df_results['date'] = dates

    df_results.index = df_results['date']
    
    if sim_data.scaling_factor > 1:
        cols = ['new_cases', 'open_infectious', 'quarantined', 'total_cases']
        df_results[cols] /= sim_data.scaling_factor
        df_results[cols] = df_results[cols].astype(int)
        
    return df_results


# Measure fitment error during parameters learning process
def measure_diff(params, sim_datax, optimize_wave1_weeks):
    sim_data = pickle.loads(pickle.dumps(sim_datax))
        
    if optimize_wave1_weeks:
        sim_data.transmission_control, sim_data.wave1_weeks = params
    else:
        sim_data.transmission_control = params[0]
        sim_data.wave1_weeks = np.median(sim_data.wave1_weeks_range)
    
    # Optimizing transmission_control and wave1_weeks separately would result in better accuracy, though cost more time
    df_results = simulate(sim_data, learning_phase=True)

    projected_cases = df_results['total_cases']
    actual_cases = sim_data.state_data.loc[-sim_data.fitment_days:, 'Total_Confirmed']

    if sim_data.scaling_factor > 1:
        actual_cases /= sim_data.scaling_factor

    comparison_span = min(config.fitment_period_max, sim_data.fitment_days)  # Days to compare model performance for
    weights = 1 / np.arange(1, comparison_span + 1)[::-1]  # More weights to recent cases
    # Measure error using MSLE / RMSE / MSE
    # error = mean_squared_log_error(actual_cases[-comparison_span:], projected_cases[-comparison_span:], weights)
    # error = sqrt(mean_squared_error(actual_cases[-comparison_span:], projected_cases[-comparison_span:], weights))
    #error = mean_squared_error(actual_cases[-comparison_span:], projected_cases[-comparison_span:], weights)
    
    error = mean_squared_error(actual_cases[-comparison_span:], projected_cases[-comparison_span:])
    
    del sim_data
        
    return error


# Learn best parameters for simulation (transmission prob, wave1_weeks) via random / Bayesian search techniques
def fit_and_project(sim_data, n_calls=40, n_jobs=8):
    param_space = [skopt.space.Real(sim_data.transmission_control_range[0], sim_data.transmission_control_range[1],
                                    name='transmission_control', prior='log-uniform')]
    
    optimize_wave1_weeks = True if not sim_data.wave1_peak_detected else False
    if optimize_wave1_weeks:
        param_space.append(skopt.space.Integer(sim_data.wave1_weeks_range[0], sim_data.wave1_weeks_range[1],
                                               name='wave1_weeks'))
        
    def objective(params):
        return measure_diff(params, sim_data, optimize_wave1_weeks)

    def monitor(res):
        print(len(res.func_vals), sep='', end=',')
    
    print (param_space)
    
    print('\n' + '*' * 100)
    print('Learning Iterations # ', sep='', end='')
    measurements = skopt.gp_minimize(objective, param_space, callback=[monitor], n_calls=n_calls, n_jobs=n_jobs)
    
    best_score = measurements.fun
    best_params = measurements.x
    print('\n' + '*' * 100)

    # Best parameters
    print('Lowest Error Observed: {}'.format(best_score))
    print('Best Param(s): {}'.format(best_params))

    return measurements


# Learn simulation parameters (transmission prob, wave1_weeks)
def learn_parameters(sim_data, n_calls=40, n_jobs=8, params_export_path=None):
    opt_results = fit_and_project(sim_data, n_calls=n_calls, n_jobs=n_jobs)
    n_best = 5
    error_scores = opt_results.func_vals
    best_score_indices = np.argsort(opt_results.func_vals)[:n_best]
    print('\n\nBest {} Param(s):'.format(n_best))

    top_scores = list()
    print('- ' * 50)
    for i in best_score_indices:
        print('Params: {}   |   Error: {}'.format(opt_results.x_iters[i], error_scores[i]))
        tranmission_prob = opt_results.x_iters[i][0]
        wave1_weeks = opt_results.x_iters[i][1] if not sim_data.wave1_peak_detected else int(np.mean(sim_data.wave1_weeks_range))
        top_scores.append([error_scores[i], wave1_weeks, tranmission_prob, sim_data.fitment_days, sim_data.test_days])
    print('- ' * 50)
    df_best_params = pd.DataFrame(top_scores, columns=['error', 'wave1_weeks', 'tranmission_prob', 'fitment_days',
                                                       'test_days'])

    if params_export_path is not None:
        print('Writing simulation params at : {}'.format(params_export_path))
        if not os.path.exists(params_export_path.rsplit('/', 1)[0]):
            print('Creating {}'.format(params_export_path.rsplit('/', 1)[0]))
            os.mkdir(params_export_path.rsplit('/', 1)[0])
        df_best_params.to_csv(params_export_path)

    return df_best_params['wave1_weeks'].iloc[0], df_best_params['tranmission_prob'].iloc[0],\
           df_best_params['fitment_days'].iloc[0], df_best_params['test_days'].iloc[0]



def plot_all(sim_data, simulation_titles, intervention_scores_list):
    ylim1, ylim2 = -1, -1
    for i, intervention_scores in enumerate(intervention_scores_list):
        print('\n')
        print(simulation_titles[i] if simulation_titles is not None else 'Simulation # {}'.format(i))
        projection_file_name = config.country_simulation_results_path\
            .format(sim_data.country_code, str(i+1)) if sim_data.country_level_projection \
            else config.state_simulation_results_path.format(sim_data.state_name, str(i+1))
        df_results = pd.read_csv(os.path.join(config.base_output_dir, projection_file_name))
        df_results.index = pd.to_datetime(df_results['date'])
            
        # plot daily confirmed projections
        ylim1_tmp, ylim2_tmp = plot_projection(df_results, sim_data, ylim1, ylim2)
        ylim1 = max(ylim1_tmp, ylim1)
        ylim2 = max(ylim2_tmp, ylim2)
        

# Plot projection results against available actual numbers
def plot_projection(df_results, sim_data, ylim1, ylim2):
    df_loc = sim_data.state_data_orig.copy()
    df_loc['Date'] = pd.to_datetime(df_loc['Date'])
    df_loc.index = df_loc['Date']

    if sim_data.scaling_factor > 1:
        target_cols = ['Confirmed', 'Deceased', 'Recovered', 'Total_Confirmed', 'Total_Deceased', 'Total_Recovered',
                       'Total_Active']
        df_loc[target_cols] /= sim_data.scaling_factor
        df_loc[target_cols] = df_loc[target_cols].astype(int)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    df_loc['Confirmed'].plot(title='Daily Confirmed Cases Projection', label='daily confirmed', ax=ax[0])
    df_loc['Total_Confirmed'].plot(title='Total Confirmed Cases Projection', label='total confirmed', ax=ax[1])
    
    df_results['new_cases'].plot(label='projection', ax=ax[0], color='darkorange')
    df_results['total_cases'].plot(label='projection', ax=ax[1], color='darkorange')
        
    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")
    
    ax[0].set_ylim(top=max(ylim1, ax[0].get_ylim()[1]))
    ax[1].set_ylim(top=max(ylim2, ax[1].get_ylim()[1]))
    
    fig.tight_layout()
    plt.grid()
    plt.show()
    
    return ax[0].get_ylim()[1], ax[1].get_ylim()[1]
    
    
# Determine sample population size for intervention to ensure atleast N number of infections to start with
# This process also determines to what extent the given population size needs to be scaled up (scaling_factor)
def size_projection_population(state_data, case_rate, state_population, fitment_days, min_init_infections):
    n_population_max = config.n_population_max
    n_population = config.n_population
    scaling_factor = 1

    #abs_case_rate = get_rate_of_changes(state_data, days_to_consider=fitment_days)
    incidence_rate, _, _ = get_incidence_rate(state_data, case_rate, state_population, fitment_days)
    # Ensuring that minimum rate yields at least N cases while simulating
    rate_multiple = min_init_infections / incidence_rate
    if n_population < rate_multiple:
        n_population = int(np.ceil(rate_multiple))
        if n_population > n_population_max:
            scaling_factor = n_population / n_population_max
            n_population = n_population_max
    print('Case Rate: {}, Incidence Rate: {}, Projection Population: {}, Scaling Factor: {}'.format(case_rate, incidence_rate, n_population, scaling_factor))
    return n_population, scaling_factor


# Scale daily infection case data and augment respective aggregated, normalized intervention scores
def extend_infection_data(country_code, state_data, scaling_factor, intervention_scores_loc):
    # Read country-wise daily intervention scores (aggregated between 0 to 1) - by intervention_scorer.ipynb
    intv_scores = pd.read_csv(intervention_scores_loc)
    
    country_intv_scores = intv_scores.loc[intv_scores['CountryCode'] == country_code]
    country_intv_scores['Date'] = pd.to_datetime(country_intv_scores['Date'])
    df_state = pd.merge(state_data, country_intv_scores[['Date', 'aggr_weighted_intv_norm']], how='left', on=['Date'])
    
    df_state['aggr_weighted_intv_norm'].fillna(method='ffill', inplace=True)
    # Fill 0 scores with last non-zero score. 0 scores might occur when intervention_scorer.ipynb is run on older data
    df_state['aggr_weighted_intv_norm'].replace(to_replace=0, method='ffill', inplace=True)
    
    if scaling_factor > 1:
        target_cols = ['Confirmed', 'Deceased', 'Recovered', 'Total_Confirmed', 'Total_Deceased', 'Total_Recovered',
                       'Total_Active']
        df_state[target_cols] *= scaling_factor
        df_state[target_cols] = df_state[target_cols].astype(int)

    return df_state


# Load stored params (transmission prob, wave1_weeks)
def get_parameters(params_export_path):
    df_best_params = pd.read_csv(params_export_path)
    return df_best_params['wave1_weeks'].iloc[0], df_best_params['tranmission_prob'].iloc[0], \
           df_best_params['fitment_days'].iloc[0], df_best_params['test_days'].iloc[0]


# Run simulations for different rates (projected, high, low) for each of the the given intervention setups
def run_simulations(sim_data, intervention_scores_list, simulation_titles=None):
    for i, intervention_scores in enumerate(intervention_scores_list):
        sim_data_copy = pickle.loads(pickle.dumps(sim_data))
        sim_data_copy.intervention_scores = intervention_scores
        
        df_results = simulate(sim_data_copy)
        
        projection_file_name = config.country_simulation_results_path\
            .format(sim_data.country_code, str(i+1)) if sim_data.country_level_projection \
            else config.state_simulation_results_path.format(sim_data.state_name, str(i+1))
        df_results.to_csv(os.path.join(config.base_output_dir, projection_file_name))

        del sim_data_copy
                
    sim_data_file_name = config.country_simulation_data_path.format(sim_data.country_code) \
        if sim_data.country_level_projection else config.state_simulation_data_path.format(sim_data.state_name)
    sim_data_file = open(os.path.join(config.base_output_dir, sim_data_file_name), 'wb')
    pickle.dump(sim_data, sim_data_file) 
    
    gc.collect()
    
    # Plotting projections
    if not config.sagemaker_run:
        sim_data_loaded = pickle.load(open(os.path.join(config.base_output_dir, sim_data_file_name), 'rb')) 
        plot_all(sim_data_loaded, simulation_titles, intervention_scores_list)


# Prepare for projection by learning related parameters (e.g. transmission prob, wave1_weeks, higher bound, 
# lower bound, etc.) to run simulation
def prep_projection(country_code, target_state, sim_data, learn_params=True):
    
    intervention_scores_loc = os.path.join(config.base_data_dir, config.intervention_scores_loc)
    try:
        pd.read_csv(intervention_scores_loc)
    except:
        print('Error: File Missing: {}!'.format(intervention_scores_loc))
        print('Load the latest intervention scores by running interventions_scorer.ipynb first and then run this '
              'simulation.')
        return None, 1
    
    if sim_data.country_level_projection:
        state_cases = os.path.join(config.base_data_dir, config.country_covid19_cases.format(target_state))
        params_export_path = os.path.join(config.base_output_dir,
                                          config.country_covid19_params_export_path.format(target_state))
    else:
        state_cases = os.path.join(config.base_data_dir, config.state_covid19_cases.format(country_code, target_state))
        params_export_path = os.path.join(config.base_output_dir,
                                          config.state_covid19_params_export_path.format(country_code, target_state))

    try:
        df_state = pd.read_csv(state_cases)
    except:
        print('Error: Data File Missing: {}!'.format(state_cases))
        return None, 1
    
    df_state['Total_Active'] = df_state['Total_Confirmed'] - df_state['Total_Deceased'] - df_state['Total_Recovered']
    df_state['Date'] = pd.to_datetime(df_state['Date'])
    
    df_state['ConfirmedCases'] = df_state['Total_Confirmed']
    
    # Get stats on this geography's disease transmission patterns (rate, start dates, peaks, etc.)
    ctmf = geo_transmission_analyzer (sim_data.n_population)
    geoTransmissionStats = ctmf.get_disease_transmission_stats (df_state, sim_data.n_population)
    
    try:
        assert geoTransmissionStats is not None and geoTransmissionStats.w1_start_dt is not None, "Error: Starting point of the 1st wave of the infection not found!"
    except AssertionError as err_msg:
        print (err_msg)
        return None, 1
    
    w1_start_dt = geoTransmissionStats.w1_start_dt
    w1_peak_dt = geoTransmissionStats.w1_peak_dt
    last_recorded_date = df_state['Date'].max()
    fitment_end_date = df_state.iloc[:-sim_data.test_days]['Date'].max()
    
    print ('Data Availability Timeline: {} to {}'.format(df_state['Date'].iloc[0], df_state['Date'].iloc[-1]))
    print ('w1_start_dt: {} | w1_peak_dt: {} | w2_start_dt: {} | days_between_disease_waves: {}'.format(w1_start_dt, w1_peak_dt, geoTransmissionStats.w2_start_dt, geoTransmissionStats.days_between_disease_waves))
    
    sim_data.wave1_start_date = w1_start_dt
    # Cases where 1st wave has already peaked
    if w1_peak_dt is not None:
        sim_data.case_rate = geoTransmissionStats.w1_mean_case_rate
        sim_data.init_case_rate = geoTransmissionStats.w1_10_pctl_case_rate
        #sim_data.init_case_rate = geoTransmissionStats.w1_mean_case_rate
        
        expected_fitment_days = 2 * (w1_peak_dt - w1_start_dt).days
        fitment_end_dt = min (w1_start_dt + timedelta(days=expected_fitment_days), last_recorded_date)
        sim_data.fitment_days = (fitment_end_dt - w1_start_dt).days
        #sim_data.test_days = sim_data.test_days + (last_recorded_date - fitment_end_dt).days
        sim_data.test_days = 1 + (last_recorded_date - fitment_end_dt).days
        if sim_data.weeks_between_disease_waves is not None:
            sim_data.days_between_disease_waves = 7 * sim_data.weeks_between_disease_waves
        elif geoTransmissionStats.days_between_disease_waves is not None:
            sim_data.days_between_disease_waves = geoTransmissionStats.days_between_disease_waves
            sim_data.weeks_between_disease_waves = int (np.round(geoTransmissionStats.days_between_disease_waves / 7))        
    else:
        sim_data.case_rate = geoTransmissionStats.w1_mean_case_rate #get_rate_of_changes(df_state, days_to_consider=14)
        sim_data.init_case_rate = geoTransmissionStats.w1_25_pctl_case_rate #sim_data.case_rate
        sim_data.fitment_days = (last_recorded_date - w1_start_dt).days - 1
        sim_data.test_days = 1
        
    sim_data.projection_days = sim_data.fitment_days + sim_data.test_days + sim_data.future_projection_days
    
    print ('Case Rt: {} [{} %]  |  Init Case Rt: {} [{} %]'.format(sim_data.case_rate, 
                                                                   100 * sim_data.case_rate / sim_data.state_population, 
                                                                   sim_data.init_case_rate, 
                                                                   100 * sim_data.init_case_rate / sim_data.state_population))
        
    df_state_src = df_state.copy()
    df_state = df_state if sim_data.test_days <= 0 else df_state.iloc[:-sim_data.test_days]
        
    min_init_infections = sim_data.min_initial_infection
    if w1_peak_dt is None and sim_data.fitment_days / 7 > 4:
        min_init_infections *= 2
    if w1_peak_dt is not None and (sim_data.fitment_days/2) / 7 > 4:
        min_init_infections *= 2
        
    # Determine scaling factor and scaled-down population size to simulate for a small population size (e.g. 3000)
    n_population, scaling_factor = size_projection_population(df_state, sim_data.init_case_rate, sim_data.state_population,
                                                              sim_data.fitment_days, min_init_infections)
    sim_data.n_population = n_population
    sim_data.scaling_factor = scaling_factor
    
    # Transform data - df_state: with scaled down population, df_state_src: with actual population
    df_state = extend_infection_data(country_code, df_state, scaling_factor, intervention_scores_loc)
    df_state_src = extend_infection_data(country_code, df_state_src, scaling_factor, intervention_scores_loc)

    sim_data.state_data = df_state
    sim_data.state_data_orig = df_state_src
        
    if w1_peak_dt is not None:
        sim_data.avg_time_to_peaks = geoTransmissionStats.w1_days_to_peak
        sim_data.mean_relative_change_rates = sim_data.case_rate
        sim_data.adjusted_case_rate = sim_data.case_rate
        sim_data.wave1_peak_detected = True
    else:
        # Get case spread stats from other countries having similar case growth pattern
        sim_data.avg_time_to_peaks = (last_recorded_date - w1_start_dt).days + 3*7 #avg_time_to_peaks
        sim_data.adjusted_case_rate = sim_data.case_rate
        sim_data.wave1_peak_detected = False

    #print('Case Rate (scaled): {}'.format(sim_data.adjusted_case_rate))

    # Optimization parameters setup
    time_to_peak_weeks = int(np.round(sim_data.avg_time_to_peaks / 7))
    
    if sim_data.wave1_peak_detected:
        wave1_weeks_low, wave1_weeks_high = time_to_peak_weeks - 1, time_to_peak_weeks + 1
    elif sim_data.avg_time_to_peaks > 0: #and not config.use_default_wave1_weeks_range:
        wave1_weeks_low, wave1_weeks_high = time_to_peak_weeks - 2, time_to_peak_weeks + 2
    else:
        wave1_weeks_low, wave1_weeks_high = config.wave1_weeks_default_range_low, config.wave1_weeks_default_range_high
    
    wave1_weeks_range = (wave1_weeks_low, wave1_weeks_high)
    transmission_control_range = (config.transmission_control_range_min, config.transmission_control_range_max)
    n_calls = config.optimization_trials_low if (wave1_weeks_high - wave1_weeks_low) <= 6 \
        else config.optimization_trials_high
    
    print('Optimization params config: tranmission_control_range: {} | wave1_weeks_range: {} | n_calls: {}'
          .format(transmission_control_range, wave1_weeks_range, n_calls))

    sim_data.wave1_weeks_range = wave1_weeks_range
    sim_data.transmission_control_range = transmission_control_range

    sim_data.intervention_scores = sim_data.state_data['aggr_weighted_intv_norm'].iloc[-sim_data.fitment_days:].tolist()
    
    # Learn (optimize) best simulation parameters by running simulation iteratively on a parameter space for
    # minimum error
    if learn_params:
        wave1_weeks, transmission_control, _, _ = learn_parameters(sim_data, n_calls=n_calls,
                                                                n_jobs=config.optimization_jobs,
                                                                params_export_path=params_export_path)
    else:
        wave1_weeks, transmission_control, learning_fitment_days, learning_test_days = get_parameters(params_export_path)
        wave1_weeks = int(np.round((wave1_weeks*7 + (sim_data.test_days-learning_test_days)) / 7))
    print('*** DISTRIBUTION PARAMS: wave1_weeks: {}, transmission_control: {}'.format(wave1_weeks, transmission_control))

    sim_data.wave1_weeks = wave1_weeks
    sim_data.transmission_control = transmission_control
    
    gc.collect()
    
    return sim_data, 0


# Sample method outlining how to run the simulations with different intervention setups
def project(sim_data):
    print('Fitment Days: {}  |  Test Days: {}  |  Projection days: {}'.format(sim_data.fitment_days,
                                                                              sim_data.test_days,
                                                                              sim_data.projection_days))
    
    print('-' * 111)
    training_intervention_data = sim_data.state_data['aggr_weighted_intv_norm'].iloc[-sim_data.fitment_days:]
    training_intervention_mean = training_intervention_data.mean()
    print('Mean aggregated Training Intervention: {} (i.e. {} %)'.format(training_intervention_mean,
                                                                         int(training_intervention_mean*100)))

    simulation_titles, intervention_scores_list = list(), list()
    
    test_intervention_mean = 0
    if sim_data.test_days > 0:
        tmp = sim_data.state_data_orig.loc[-sim_data.test_days:]
        test_intervention_mean = tmp.loc[tmp['aggr_weighted_intv_norm'] > 0, 'aggr_weighted_intv_norm'].mean()
        print('Mean aggregated Testing Intervention: {} (i.e. {} %)'.format(test_intervention_mean,
                                                                            int(test_intervention_mean*100)))
    #training_intervention = training_intervention_data.tolist() + training_intervention_data.tolist()
    eff_projection_days = int(sim_data.projection_days) - len(training_intervention_data.tolist())
    
    training_intervention = training_intervention_data.tolist()
    test_intervention = sim_data.state_data_orig['aggr_weighted_intv_norm'].iloc[-sim_data.test_days:].tolist()
    intvs = training_intervention + test_intervention
    
    remaining_projection_period = sim_data.projection_days - len(training_intervention) - len(test_intervention)
    remaining_projection_intervention = [np.mean(intvs[-15:]) for _ in range(remaining_projection_period)]
    intvs += remaining_projection_intervention
    
    if len(remaining_projection_intervention) > 0:
        print('Mean aggregated Future Projection Intervention (assumed from recent past): {} (i.e. {} %)'.format(np.mean(remaining_projection_intervention),
                                                                        int(np.mean(remaining_projection_intervention)*100)))
    # Simulation assuming recorded intervention level during projection period
    simulation_titles.append('Simulation 1: using recorded aggregated intervention:')
    #intervention_scores_list.append(training_intervention + [test_intervention_mean for _ in range(eff_projection_days)])
    intervention_scores_list.append(intvs)
    
    # Simulation assuming 50% intervention level during projection period
    simulation_titles.append('Simulation 2: using 50% end-to-end aggregated interventions:')
    intervention_scores_list.append([0.5 for _ in range(len(training_intervention))] + [0.5 for _ in range(eff_projection_days)])
    
    # Simulation assuming 90% intervention level during projection period
    simulation_titles.append('Simulation 3: using 90% end-to-end aggregated interventions:')
    intervention_scores_list.append([0.9 for _ in range(len(training_intervention))] + [0.9 for _ in range(eff_projection_days)])
    
    
    # Add more intervention scenarios here as per requirement...
    
    run_simulations(sim_data, intervention_scores_list, simulation_titles=simulation_titles)


# orchestration method to be invoked from simulation notebooks / apis
def run(country_code, 
        state, 
        state_population, 
        actual_testing_capacity, 
        future_projection_days,
        country_level_projection=False, 
        min_initial_infection = config.min_initial_infection, 
        transmission_prob=config.transmission_prob_default, 
        intervention_influence_pctg=config.intervention_influence_pctg_default, 
        wave2_peak_factor=config.wave2_peak_factor_default, 
        wave2_spread_factor = config.wave2_spread_factor_default, 
        weeks_between_disease_waves = None):
    
    start_time = datetime.now()
    
    sim_data = SimulationData()
    sim_data.country_code = country_code
    sim_data.state_name = state
    sim_data.country_level_projection = country_level_projection
    sim_data.state_population = state_population
    sim_data.actual_testing_capacity = actual_testing_capacity
    sim_data.test_days = 1
    sim_data.future_projection_days = future_projection_days
    
    sim_data.min_initial_infection = min_initial_infection
    sim_data.transmission_prob = transmission_prob
    sim_data.intervention_influence_pctg = intervention_influence_pctg
    sim_data.weeks_between_disease_waves = weeks_between_disease_waves
    sim_data.wave2_peak_factor = wave2_peak_factor
    sim_data.wave2_spread_factor = wave2_spread_factor
    
    sim_data, status_code = prep_projection(country_code, state, sim_data, learn_params=True)
    
    if status_code == 0:
        project(sim_data)
        print(f'Total Time Taken: {datetime.now() - start_time}'.format())
