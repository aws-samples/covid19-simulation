from pathlib import Path

# Simulation Parameters
add_ab = False
batching = True
ab_cost = 28
ab_cost_cheap = 28
pcr_cost = 77
attrition_rate = 0.05
attrition_frequency = 14
time_to_wait_for_antibody_test = 0
time_to_wait_for_pcr_test_results = 2
time_for_incubation = 2
time_to_stop_symptoms = 8
time_self_isolation = 14
ab_false_positive_rate = 1.0
pcr_false_negative_rate = 1.0
pcr_batch_size = 1
init_test_capacity = 1000
init_ab_test_capacity = 100
ab_tests_sent_per_day = 0
time_between_consecutive_pcr_tests = 14
pcr_test_result_positive_probability = 0.01
initial_antibody_immunity_in_population = 0
infected_and_symptomatic_in_population = 0.14
symptom_covid_vs_flu = 0
awaiting_on_pcr_test_result = 0


# Infra parameters
base_data_dir = 'data/input'
base_output_dir = 'data/output'

sagemaker_run = False
base_data_dir_sagemaker = '/opt/ml/processing/input'
base_output_dir_sagemaker = '/opt/ml/processing/out'

oxcgrt_intervention_data_online = 'https://oxcgrtportal.azurewebsites.net/api/CSVDownload'
oxcgrt_intervention_data_offline = 'OxCGRT_Download_Full.csv'
intervention_impacts_loc = 'countries_intervention_impacts.csv'
intervention_scores_loc = 'countries_aggr_intervention_scores.csv'

confirmed_cases_global_online = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
                                'csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
recovered_cases_global_online = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
                                'csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
deceased_cases_global_online = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
                               'csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
confirmed_cases_global_offline = 'time_series_covid19_confirmed_global.csv'
recovered_cases_global_offline = 'time_series_covid19_recovered_global.csv'
deceased_cases_global_offline = 'time_series_covid19_deaths_global.csv'

country_populations_data = 'country_population_WorldBank.csv'

india_states_cases_online = 'https://api.covid19india.org/csv/latest/state_wise_daily.csv'
india_states_cases_offline = 'india_state_wise_daily.csv'

us_counties_cases_online = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
us_counties_cases_offline = 'us-counties.csv'

country_covid19_cases = 'countries/Cases_{}.csv'
country_covid19_params_export_path = 'countries/params_{}.csv'
country_simulation_results_path = 'countries/projections_{}_{}_{}.csv'
country_simulation_data_path = 'countries/sim_data_{}.pkl'
state_covid19_cases = 'Cases_{}.csv'
us_covid19_cases = 'covid_19_daily_reports_us.csv'
state_covid19_params_export_path = 'params_{}.csv'
state_simulation_results_path = 'projections_{}_{}_{}.csv'
state_simulation_data_path = 'sim_data_{}.pkl'

county_cases_filename = Path(__file__).parent / '../data/us-counties_4_26.txt'
county_population_filename = Path(__file__).parent / '../data/county-population.csv'


# Optimization related parameters

# Consider a country for comparison if its total case count exceeds this threshold
min_country_conf_case_threshold = 25000
# Number of time intervals to use while comparing case rate changes between two countries
match_spans = 5
# Number of days in each comparison interval of match_spans
period = 3
# Start Date (DDMMYYYY) for considering infection data
overall_start_date = '15012020'
# Maximum distance (error) threshold for selecting countries with matching case rate trend
sim_dist_threshold = 0.2
# Weightage of similarity trend match while selecting reference countries
trend_sim_weightage = 0.25
# Weightage of population proximity while selecting reference countries
population_sim_weightage = 0.75

# Minimum number of reference countries required to enable case rate adjustment based on reference countries' trends
min_relevant_countries_count = 5
# Enable/Disable reference country based case rate adjustment
enable_case_rate_adjustment = True
# Maximum scaling factor of case rate based on trends of reference countries
max_rate_adjustment_factor = 3.0

# Percentage/fraction of influence of the interventions on disease transmission
intervention_influence_pctg = 0.75

# Whether to use the default range (1, 5) for wave1_weeks optimization
use_default_wave1_weeks_range = False

# Whether to optimize the number of weeks to reach the peak of infection Wave-1. If False, the number derived from
# the reference countries or median of default range will be used
optimize_wave1_weeks = True

# Maximum population size for simulation (higher number will increase the duration of simulation cycles)
n_population_max = 3000
# Preferred population size for simulation
n_population = 1500
# Minimum existing infection count while initiating the simulation (actual infection rate would be scaled up/down to
# ensure this minimum count)
min_initial_infection = 5
# Max number of days to fit while optimizing the simulation parameters
fitment_period_max = 21

# Default higher and lower bound factors
higher_bound_min = 1.2
lower_bound_min = 0.2

# Min and max range for transmission probability optimization
transmission_prob_range_min = 0.00005
transmission_prob_range_max = 0.005

# Default min and max number of weeks to reach wave-1 peak for optimization
wave1_weeks_default_range_low = 1
wave1_weeks_default_range_high = 5

# Optimization trials (runs) min and max, and job count
optimization_trials_low = 100
optimization_trials_high = 150
optimization_jobs = 8
