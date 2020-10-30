# COVID-19 Simulation Toolkit Read-me

## Overview

This software simulates COVID-19 case projections at various regional granularity level. The output is the projection of the total confirmed cases over a specific timeline for a target state or a country, for a given degree of intervention. We have provided a few sample simulations at state and country levels in covid19_simulator.ipynb, covid19_simulator_USA_states.ipynb and covid19_simulator_India_states.ipynb notebooks. 

In terms of the working methodology, this solution first tries to understand the transmission rates and the time to peak of the 1st wave from the daily COVID-19 cases for the target entity (state/country). Next, it selects the best (optimal) parameters using optimization techniques on a stochastic simulation model. The simulation model is subsequently re-invoked with the optimized parameters and intervention scores to generate a case-count projection for a configurable time frame. Refer to the section **Intervention Impact Scoring** for details on how to score the relative effectiveness of different interventions for a country and to the section **Simulation Orchestration** for more information on running the simulations.

The core simulation model is a bottom-up, stochastic approach, in which each individual is simulated separately with variability in the disease progression. Subsequently, individuals are aggregated to generate population level statistics. The variability in the disease model accounts for current uncertainties in the disease progression. This simulation approach incorporates insights from medical journal publications elaborating recurring waves of Influenza pandemics, notably https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0060343. It assumes two waves of infection surges following Gaussian distributions and uses the distributions while generating the infection case-count projections in a probabilistic manner. In absence of any historical data on COVID-19 pandemic, data on Influenza Pandemics has been considered as a similar, relatable baseline in this data-driven approach. 
Additionally, the simulation also incorporates various crucial factors, like transmission probability, testing efficiency, transmission control level, intervention impacts, etc. to come up with more realistic projections.

Though our sample notebooks demonstrate simulations at country and state levels, the approach is generic enough to be applied at other granularity levels, such as, county, district, city, etc.


## License

This project is licensed under the Apache 2.0 License.


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

* run: first learns the best parameters for the given entity (state/country) and timelines and then invokes simulation for the projection timeline
* prep_projection: Finds the best parameter ranges (for transmission control and time-to-peak) via optimization
* project: Runs simulation with the parameters found in prep_projection
* simulate: Invokes the simulation module on a small population size and scales the projections with respect to the actual population size of the target state or country

**case_trends_finder.py**
Our simulator models the disease transmission using underlying gaussian probability distributions. Hence, we need to feed the simulator with the parameters for the distributions, which differ across population groups (countries, states, etc.). We do a sliding window analysis of the smoothened daily confirmed cases data to detect the starting points and the peaks of the 1st and 2nd waves. These information along with statistics on transmission rates are subsequently used to infer the parameters of the distributions.

* get_disease_transmission_stats: primary coordinator method that finds the required data points around the transmission patterns
* get_periodic_relative_changes: computes the rate of change (of case count), relative rate of change and the peak/plateau points for countries


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

* incidence_rate_curve: It generates the dual-wave Gaussian distributions to control the simulation using the probability of infection spread on a particular day, based on that day's position in the distribution. The mean of the first distribution, which indicates the remaining time to peak of the first wave, is extracted/learnt by the simulation_orchestrator and passed as an input parameter.
* population_init_infections: Infects a subset of the population based on the input case rate to initiate the simulation process.
* update_population_infections: Models the daily spread of the infection based on the simulation_curve (probability distribution), transmission control and interventions. transmission control parameter is learnt and supplied by the simulation_orchestrator.



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

Key config parameters (from src/config.py) that can be tweaked as per requirements are as follows:

* gap_weeks_between_disease_waves_default = 25 (default gap between 1st and 2nd disease waves in weeks)
* transmission_prob_default = 0.005 (default transmission probability)
* wave2_peak_factor_default = 3, wave2_spread_factor_default = 1 (Wave-2 distribution parameter scales with respect to wave-1 distribution)
* number_of_people_to_infect_default = 3 (default number of people to be infected by an already infected person)

* min_country_conf_case_threshold = 25000 (Consider a country for case rate trend comparison only if its total case count exceeds this threshold)
* period = 3 (time window (in days) while measuring periodic changes)
* n_population_max = 5000 (Maximum population size for simulation (higher number will increase the duration of simulation cycles))
* n_population = 3000 (Preferred population size for simulation)
* min_initial_infection = 5 (Minimum existing infection count while initiating the simulation (actual infection rate would be scaled up/down to ensure this minimum count))
* transmission_control_range_min = 0.05, transmission_control_range_max = 1.0 (Min and max range for transmission control optimization)
* wave1_weeks_default_range_low = 1, wave1_weeks_default_range_high = 5 (Default min and max number of weeks to reach wave-1 peak for optimization)
* optimization_trials_low = 40, optimization_trials_high = 60 (Optimization/learning trials (runs) min and max)




## Running the code

### **Environment setup**

**Running the Notebooks (interventions_scorer.ipynb and covid19_simulator.ipynb)**
Objective: Development and execution workload on the local machine end-to-end:

* Memory required: 8 GB (minimum), 16+ GB (preferred) 
* Install Python 3.6+
* Install the necessary libraries (pip install -r requirements.txt)
* Develop on / run the code baseline using Jupyter Notebook

**Running SageMaker Notebooks from Local Machine (interventions_scorer_sagemaker.ipynb and covid19_simulator_sagemker.ipynb)**
Objective: Development on Local machine and only execution using Amazon SageMaker:
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

***** You can also run the whole project using Jupyter labs in Amazon SageMaker directly.**

### Code execution

1. Make necessary configuration alterations in src/config.py (if required) 

2. Calculate Intervention Effectiveness for various countries using interventions_scorer.ipynb:
   2.1. Set LOAD_LATEST_DATA as True to download the latest intervention data from source. Set it as False in case of subsequent runs during the same day.
   2.2. Run the notebook to score and persist the country-level, daily intervention weights (once every day should be good enough). These scores will be used while generating the infection projections.
    
3. Generate sample infection projections at country level using covid19_simulator.ipynb (or sample state-level alternatives):
   3.1. Set LOAD_LATEST_DATA as True to download the latest infection data for simulation. Set it as False in case of subsequent runs during the same day.
   3.2. Configure the target states / countries and control params as follows:
```
    # Parameters related to the target location
    country_code, state, state_population, actual_testing_capacity = 'IND', 'KL', 33406061, 6000
    # Future projection timeline (days)
    projection_days = 180
``` 
   3.3. Run the notebook to generate the case projections for the states / countries specified
```     
    run (country_code, state, state_population, actual_testing_capacity, future_projection_days, country_level_projection=False)
``` 

For SageMaker:
- Use interventions_scorer_sagemaker and covid19_simulator_sagemaker instead of the regular notebooks to run the simulations on Amazon SageMaker
- Specify the appropriate S3 bucket and IAM role in the SageMaker notebooks before running them

