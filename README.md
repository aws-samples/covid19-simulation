# COVID-19 Simulation

## Overview

This software simulates COVID-19 case projections at various regional granularity level. The output is the projection of the total confirmed cases over a specific timeline for a target state or a country, for a given degree of intervention. We have provided a few sample simulations at state and country levels in covid19_simulator.ipynb notebook. 

In terms of the working methodology, this solution first tries to understand the approximate time to peak of the daily COVID-19 cases for the target entity (state/country), expected case rates and associated higher & lower bounds. It determines the ranges for these parameters from countries that have exhibited similar case rate growth trends in the past. Next, it selects the best (optimal) parameters using optimization techniques on a simulation model. The simulation model is subsequently re-invoked with the optimized parameters and intervention scores to generate a case-count projection for a configurable time frame. Refer to the section **Intervention Impact Scoring** for details on how to score the relative effectiveness of different interventions for a country and to the section **Simulation Orchestration** for more information on running the simulations.

The core simulation model is a bottom-up, stochastic approach, in which each individual is simulated separately with variability in the disease progression. Subsequently, individuals are aggregated to generate population level statistics. The variability in the disease model accounts for current uncertainties in the disease progression. This simulation approach incorporates insights from medical journal publications elaborating recurring waves of Influenza pandemics, notably https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0060343. It assumes two waves of infection surges following Gaussian distributions and uses the distributions while generating the infection case-count projections in a probabilistic manner. In absence of any historical data on COVID-19 pandemic, data on Influenza Pandemics has been considered as a similar, relatable baseline in this data-driven solution. 
Additionally, the simulation also incorporates various crucial factors, like transmission probability, testing efficiency, intervention impacts, attrition, etc. to come up with more realistic projections.

Though our sample notebooks demonstrate simulations at country and state levels, the approach is generic enough to be applied at other granularity levels, such as, county, district, city, etc.

## Key Components

The code components can be grouped into two categories: Python modules and Notebooks. Notebooks in turn have two variations: Jupyter notebooks and Amazon SageMaker enabled notebooks. As some of the operations, such as optimizations, supervised model training, etc. are quite resource intensive in nature, the SageMaker notebooks offer an option of  pushing the relevant processing to AWS and execute as SageMaker managed processes. Refer to the section **Running SageMaker Notebooks** for details on how to setup your environment for running the SageMaker notebooks.

### **Intervention Impact Scoring**

**interventions_scorer.ipynb (SageMaker version: interventions_scorer_sagemaker.ipynb)**
Scores COVID-19 interventions (using **intervention_effectiveness_scorer.py**) for various countries using a combination of Supervised ML technique on the research data from OXCGRT (https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker). However, the code can work with any intervention data following a schema similar to that of the OXCGRT feed.
The 3 ways to score the intervention effectiveness are as follows:

1. score_stringency_idx: Fits the daily intervention scores in a regression model with OXCGRT provided StringencyIndex as the dependent variable. Subsequently, it extracts the intervention impact scores as model's feature importances.
2. score_conf_cases: Fits the daily intervention scores in a regression model with the changes in case-count (moving average) as the dependent variable. Feature Importance scores are used as intervention impact inducators. 
3. score_intv_effects: Measures the changes in total case count by turning off the interventions one by one. Scores the interventions in proportion to the respective changes resulting from it being turned off. 

Finally, these 3 scores can be combined using a configurable weighted average. Though these approaches would be affected by the correlations among the interventions, as a whole, it can offer an approximate relative importance view of the interventions.
These intervention impacts or weights can be used to come up with a single aggregated intervention score for a country on a daily basis. The intervention data used are at country level only. Country level data might not be a good approximation for states, as states have varying degrees of intervention and violation. Hence, while trying to project for individual states, we recommend collecting similar data at that respective state level. 

### Simulation Orchestration

**covid19_simulator.ipynb (SageMaker version: covid19_simulator_sagemker.ipynb)**
Notebook to run the simulation for a target location (country, state, etc.). This notebook invokes simulation_orchestrator.py with necessary parameters. The equivalent SageMaker notebook (covid19_simulator_sagemker.ipynb) creates a docker bundle of the code and registers with Amazon Elastic Container Registry (Amazon ECR), transfers the data to Amazon S3 and finally runs the code as a SageMaker process. Detailed knowledge of ECR or S3 is not pre-requisite to use this notebook. However, you can refer to https://aws.amazon.com/ecr/ and https://aws.amazon.com/s3/ for details of these two services.

**simulation_orchestrator.py**
Manages the end-to-end simulation flow involving data transformation, parameters learning, simulation execution, visualization, etc. Some of its key methods are:

* run: first learns the best parameters for the given entity (state/country) and timelines and then invokes simulation for the projection days
* prep_projection: Finds the best parameter ranges (for time-to-peak and transmission probability) by comparing against other countries and subsequently runs optimization to determine the best parameters
* project: Runs simulation with the parameters found in prep_projection
* simulate: Invokes the simulation module on a small population size and scales the projections with respect to the actual population size of the target state or country

**case_trends_finder.py**
When the COVID-19 cases are spreading in a particular region, it is not possible to estimate the duration and extent of the spread, just by looking at the data of that region alone. A more practical approach is to understand the possibilities from the data of other countries that have exhibited similar trends in the past. This is the purpose of the case_trends_finder module.
Given the latest case-rate growth trend of a region it first identifies the countries that have shown similar trends in the past and eventually plateaued. From these reference countries, it determines the possible time-to-peak, mean/median daily growth rates until the peak, and higher and lower bounds of growth rates. These data points are used by the simulation_orchestrator for parameter optimization and simulation execution. Some of the key methods are:

* run: primary coordinator method that finds the required data points (time-to-peak, growth rates, bounds, etc.) and returns the weighted average using target country's similarity of trend as well as population with the candidate countries
* get_periodic_relative_changes: computes the rate of change (of case count), relative rate of change and the peak/plateau points for countries
* get_top_matches: Finds the best matching countries and match points by comparing relative rate of changes using a sliding window approach

### Core Simulation

**simulation.py**
It is the primary simulation module. It executes the following key tasks to generate the case projections: 

* Initially infect a fraction of the population based on the case rate parameter
* For each day:
    * Update antibody (igg and ign)
    * Update possible infections and symptoms progression of infected populations
    * PCR tests and associated isolation of positive cases
    * Mark population subset as negative following -ve test or prolonged isolation
    * spread the infection considering current infection count, transmission probability and intervention impact
    * Report daily SEIR metrics (Susceptible, Exposed, Infected, Recovered)

**utils.py**
utils module bundles all the simulation related methods that are invoked by simulation.py. Some of the important methods are:

* incidence_rate_curve: It generates the dual-wave Gaussian distributions to control the simulation using the probability of infection spread on a particular day, based on that day's position in the distribution. The mean of the first distribution, which indicates the remaining time to peak of the first wave, was learnt by the simulation_orchestrator and passed as an input parameter.
* population_init_infections: Infects a subset of the population based on the input case rate to initiate the simulation process.
* update_population_infections: Models the daily spread of the infection based on the simulation_curve (probability distribution), transmission probability and interventions. transmission probability parameter is learnt and supplied by the simulation_orchestrator. Intervention scores are optional. Intervention scores as well as intervention's influence level (e.g. 50% or 75%) are also sent by simulation_orchestrator as input parameters.

## Data Sources

This project is a generic framework and we tested it with the data-sources mentioned below. However, this framework can work as-is on any dataset sourced from other agencies, as long as they follow the same schema. Users can choose other data schemas as well if they make necessary changes to the code.

* **Daily, country level intervention data**: Corona Virus Government Response Tracker data curated and published by Blavatnik School of Government, University of Oxford, UK (https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker)
* **Daily confirmed cases for countries**: COVID-19 Data published by Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master)
* **Daily confirmed cases for Indian States**: Data aggregated by covid19india.org from news bulletins and Govt. handles (https://api.covid19india.org/csv/latest/state_wise_daily.csv)
* **Country Populations**: Country population data from the World Bank (https://databank.worldbank.org/reports.aspx?source=2&series=SP.POP.TOTL&country=)

## Configurability

Parameters and constants are configured in **src/config.py**. These can be altered as per requirement.
Configuration parameters represent following 3 groups:

* Simulation parameters: Used in core simulation. Examples: time_to_wait_for_antibody_test, time_for_incubation, time_to_stop_symptoms, time_self_isolation, etc.
* Optimization parameters: Used in determining the optimal parameters for simulation. Examples: includes n_population, fitment_period_max, transmission_prob_range_min, optimization_trials_low, trend_sim_weightage, population_sim_weightage, enable_case_rate_adjustment, etc.
* Infra parameters: Represents links, directories, files, etc. Examples: base_data_dir, country_populations_data, intervention_scores_loc, etc.

Key optimization parameters (from src/config.py) that can be tweaked as per requirements are as follows:

* min_country_conf_case_threshold = 25000 (Consider a country for case rate trend comparison only if its total case count exceeds this threshold)
* match_spans = 7 (Number of time intervals to use while comparing case rate changes between two countries)
* period = 2 (Number of days in each comparison interval of match_spans parameter)
* overall_start_date = '15012020' (Start Date (DDMMYYYY) for considering infection data)
* sim_dist_threshold = 0.2 (Maximum distance (error) threshold for selecting countries with matching case rate trend)
* trend_sim_weightage = 0.25 (Weightage of similarity trend match while selecting reference countries)
* population_sim_weightage = 0.75 (Weightage of population proximity while selecting reference countries)
* min_relevant_countries_count = 5 (Minimum number of reference countries required to enable case rate adjustment based on reference country's trends)
* enable_case_rate_adjustment = False (Enable/Disable reference country based case rate adjustment)
* max_rate_adjustment_factor = 3.0 (Maximum scaling factor of case rate based on trends of reference countries)
* intervention_influence_pctg = 0.75 (Percentage/fraction of influence of the interventions on disease transmission)
* optimize_wave1_weeks=False (whether to optimize the number of weeks to reach the peak of infection Wave-1. If False, the number derived from the reference countries will be used)
* n_population_max = 5000 (Maximum population size for simulation (higher number will increase the duration of simulation cycles))
* n_population = 3000 (Preferred population size for simulation)
* min_initial_infection = 5 (Minimum existing infection count while initiating the simulation (actual infection rate would be scaled up/down to ensure this minimum count))
* fitment_period_max = 21 (Max number of days to fit while optimizing the simulation parameters)
* higher_bound_min = 1.2, lower_bound_min = 0.2 (Default higher and lower bound factors)
* transmission_prob_range_min = 0.00005, transmission_prob_range_max = 0.005 (Min and max range for transmission probability optimization)
* wave1_weeks_default_range_low = 1, wave1_weeks_default_range_high = 5 (Default min and max number of weeks to reach wave-1 peak for optimization)
* optimization_trials_low = 50, optimization_trials_high = 100, optimization_jobs = 8 (Optimization trials (runs) min and max, and job count)

## Running the code

### Environment setup

**Running the Notebooks (interventions_scorer.ipynb and covid19_simulator.ipynb)**
Objective: Development and execution workload on the local machine end-to-end:

* Memory required: 8 GB (minimum), 16 GB (preferred) 
* Install Python 3.6+
* Install the necessary libraries (pip install -r requirements.txt)
* Develop on / run the code baseline using Jupyter Notebook

**Running Notebooks on AWS**
While logged on to your AWS account, click on the link to quick create the AWS CloudFormation Stack for the region you want to run your notebook:
​
<table>
  <tr>
    <th colspan="3">AWS Region</td>
    <th>AWS CloudFormation</td>
  </tr>
  <tr>
    <td>US West</td>
    <td>Oregon</td>
    <td>us-west-2</td>
    <td align="center">
      <a href="https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?templateURL=https://aws-ml-blog.s3.amazonaws.com/artifacts/Covid19-Simulation/covid.yaml&stackName=covid19-simulation-stack">
        <img src="media/launch_button.svg" height="30">
      </a>
    </td>
  </tr>
</table>

**Running SageMaker Notebooks from Local Machine (interventions_scorer_sagemaker.ipynb and covid19_simulator_sagemker.ipynb)**
Objective: Development on Local machine and only execution using AWS SageMaker:
On your local machine:

* Install Docker
* Install (pip) AWS helper python libs: awscli, sagemaker, boto3
* Setup the local machine with AWS access details (via "aws configure" in console)
* Create an S3 bucket for this experiment
* Create an AWS IAM Role with the following policies attached:
    * AmazonEC2ContainerRegistryFullAccess
    * AmazonSageMakerFullAccess
    * AmazonS3FullAccess
* Specify the S3 bucket and IAM role in the SageMaker Notebooks (with _sagemaker suffix)
* Run Docker
* Run your target SageMaker notebooks from Jupyter

***** You can also run the whole project using Jupyter labs in AWS SageMaker directly.**

### Running the code

1. Make necessary configuration alterations in src/config.py (if required) 

2. Calculate Intervention Effectiveness for various countries using interventions_scorer.ipynb:
   2.1. Set LOAD_LATEST_DATA as True to download the latest intervention data from source. Set it as False in case of subsequent runs during the same day.
   2.2. Run the notebook to score and persist the country-level, daily intervention weights (once every day should be good enough). These scores will be used while generating the infection projections.
    
3. Generate infection projections at state / country levels using covid19_simulator.ipynb:
   3.1. Set LOAD_LATEST_DATA as True to download the latest infection data for simulation. Set it as False in case of subsequent runs during the same day.
   3.2. Configure the target states / countries and control params as follows:
```
    # Whether to learn the best params on the latest data or use the parameters learnt last time
    learn_params = True
    # Days to use for simulation parameters optimization
    fitment_days = 14
    # Latest n number of days to leave for testing
    test_days = 5
    # Total projection timeline
    projection_days = 270
    # Parameters related to the target location
    country_code, state, state_population, actual_testing_capacity = 'IND', 'KA', 61095297, 11000
``` 
   3.3. Run the notebook to generate the case projections for the states / countries specified
```     
    run (country_code, state, state_population, actual_testing_capacity, fitment_days, test_days, projection_days, learn_params)
``` 

For SageMaker:
- Use interventions_scorer_sagemaker and covid19_simulator_sagemaker instead of the regular notebooks to run the simulations on AWS SageMaker
- Specify the appropriate S3 bucket and IAM role in the SageMaker notebooks before running them

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

