import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta
from xgboost import XGBRFRegressor
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


### Configure weights of 3 methods for intervention scoring
### - the 3 weights should sum up to 1.0
### - a method will be ignored if its weight is set to 0.0
intervention_scoring_methods = {'fit_stringency_index':1.0, 
                                'fit_conf_cases':0.0, 
                                'fit_intv_effect':0.0}

#Data source for the whole analysis
data_src = os.path.join(config.base_data_dir, config.oxcgrt_intervention_data_offline)

relevant_columns = ['CountryName', 'CountryCode', 'C1_School closing', 'C2_Workplace closing', 
                    'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 
                    'C6_Stay at home requirements', 'C7_Restrictions on internal movement', 
                    'C8_International travel controls', 'E1_Income support', 'H1_Public information campaigns', 
                    'H2_Testing policy', 'H3_Contact tracing', 'ConfirmedCases', 'ConfirmedDeaths', 
                    'StringencyIndex']

#selected_countries = ['IND', 'USA', 'GBR', 'ITA', 'JPN', 'SGP', 'NLD', 'ISR', 'BEL', 'BRA', 'DEU', 'CUB', 'ESP', 'MEX', 'MYS', 'PHL', 'HUN', 'ZAF']
selected_countries = None

#Select a country only if it has exceeded the conf_cases_threshold
conf_cases_threshold = 10000
#Select records having confirmed cases >= min_case_threshold
min_case_threshold = 0
#window for rollong averages of conf case counts
smoothing_window_len = 3
#number of lags to use for time-series style modeling of conf cases
num_lags = 1
#Skip a few recent dayes data for potential missing values
recent_days_to_skip = 5 
#median incubation period for Covid19
incubation_period = 5
plot_predictions = False


### Fetch and filter cases & intervention data
def get_all_data (data_src):
    data = pd.read_csv(data_src)
    data = data.loc[(data['RegionCode'] == '') | (data['RegionCode'].isnull())]
    
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    data = data.set_index('Date')
    
    data = data[relevant_columns]
    countries = data[['CountryCode', 'CountryName']].drop_duplicates(keep='first')
    country_dict = dict()
    for cc in countries['CountryCode']:
        country_dict[cc] = countries.loc[countries['CountryCode']==cc, 'CountryName'].tolist()[0]

    selected_countries = data.loc[(data['ConfirmedCases'] > conf_cases_threshold),'CountryCode'].unique()
    print ('Countries with more than %d confirmed cases: %d' % (conf_cases_threshold, len(selected_countries)))
    if selected_countries is not None and len(selected_countries)>0:
        data = data.loc[data['CountryCode'].isin(selected_countries)]
    data['ConfirmedCases'] = data['ConfirmedCases'].fillna(method='ffill')
    data = data.loc[data['ConfirmedCases'] >= min_case_threshold]
    
    return data, country_dict

### Filter data for a specific country
def get_country_data (data, country_code, min_threshold=1):
    country_data = data.loc[data['CountryCode'] == country_code]
    country_data = country_data[:-recent_days_to_skip]
    country_data = country_data.loc[country_data['ConfirmedCases'] >= min_threshold]
    country_data = country_data.loc[~country_data.index.duplicated(keep='first')]
    print (f'Data dimension for counry {country_code} is {country_data.shape}')
    return country_data

### Feature engineering
def add_features (country_data, rolling_window=3):
    country_data['Change'] = (country_data['ConfirmedCases'] - country_data['ConfirmedCases'].shift(1)) - 1
    country_data['RateOfChange'] = (country_data['ConfirmedCases'] - country_data['ConfirmedCases'].shift(1)) / (country_data['ConfirmedCases'].shift(1) - country_data['ConfirmedCases'].shift(2))

    country_data['RateOfChange'] = country_data['RateOfChange'].replace([np.inf, -np.inf], np.nan)
    country_data['RateOfChange'] = country_data['RateOfChange'].fillna(0)

    country_data['Change_MA'] = country_data['Change'].rolling(window=rolling_window).mean()
    country_data['RateOfChange_MA'] = country_data['RateOfChange'].rolling(window=rolling_window).mean()
    return country_data

### Get training features and labels
def get_modeling_data (country_data, target_col, nlags=1, fit_conf_cases=False, fit_intv_effect=False):
    country_data = country_data.fillna(method='ffill')
    country_data = country_data.fillna(0) #Fill remaining initial NaN values
    #X = country_data.drop(['CountryName', 'CountryCode', 'ConfirmedCases', 'Change', 'RateOfChange', 'DepVar', 'Change_MA', 'RateOfChange_MA'], axis=1)
    #country_data[target_col] = country_data[target_col].fillna(0.0)
    drop_cols = ['CountryName', 'CountryCode', 'ConfirmedCases', 'ConfirmedDeaths', 'StringencyIndex', 'Change', 'RateOfChange', 'Change_MA', 'RateOfChange_MA']
    X = country_data.drop(drop_cols, axis=1, errors='ignore')
    y = country_data[target_col]
    
    if fit_conf_cases or fit_intv_effect:
        lag_cols = []
        for lag in range(nlags):
            X[target_col + '_' + str(lag+1)] = country_data[target_col].shift(lag)
            X[target_col + '_' + str(lag+1)].fillna(0, inplace=True)
            lag_cols.append (target_col + '_' + str(lag+1))
    
    X1, X2 = None, None
    if fit_intv_effect: 
        X1 = X.drop([col for col in X.columns if col not in lag_cols], axis=1)
        X2 = X.drop([col for col in X.columns if col in lag_cols], axis=1)
            
    return X, X1, X2, y

### Fit the data against an ensemble based regression model and get predictions on the same data
def fit_model (X, y):
    model = XGBRFRegressor(n_estimators=1000, max_depth=7, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    #print (y)
    err_mae = mean_absolute_error(y, y_pred)
    err_rmse = np.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, err_mae, err_rmse

### Measure predicted total cases using model's predictions
def get_predicted_total_cases (country_data, X, y):
    _, y_pred, _, _ = fit_model(X, y)
    total_cases_pred = country_data['ConfirmedCases'].shift() + (y_pred)
    return y_pred, total_cases_pred


def plot01 (country_data, y, y_pred, total_cases_pred):
    fig, ax = plt.subplots(1, 2, figsize=(25, 8))
    ax[0].plot(y.values, label='actual', color='green')
    ax[0].plot(y_pred, label='predicted')
    ax[0].legend()
    ax[0].set_title('Actual Vs Predicted: % Change')

    ax[1].plot(country_data['ConfirmedCases'].values, label='actual', color='green')
    ax[1].plot(total_cases_pred.values, label='predicted')
    ax[1].legend()
    ax[1].set_title('Actual Vs Predicted: Cummulative Cases')
    plt.show();


### Get the mean difference in case count between the following 2 predictions:
###   1. prediction with all features
###   2. mean of the prediction using previous case counts (based on n-lags) and the prediction with interventions
###     having the target intervention (target_intv) turned off
### The resulting mean difference in case count is subsequently interpreted as an indicator of approx impact 
### of the target intervention (target_intv)
def get_case_count_diff (country_data, X1, X2, y, total_cases_pred, target_intv, plot_graphs=False):
    ### Fit & predict with only TimeSeries MA Lag Feature(s)
    y_pred1, total_cases_pred1 = get_predicted_total_cases (country_data, X1, y)

    ### Fit & predict with Interventions (with target_intv set to 0) but without TimeSeries MA Lag Feature(s)'
    y_pred2, total_cases_pred2 = get_predicted_total_cases (country_data, X2, y)

    y_pred3 = pd.DataFrame(zip(y_pred1, y_pred2)).mean(axis=1)
    total_cases_pred3 = country_data['ConfirmedCases'].shift() + (y_pred3).tolist()
#     if plot_graphs:
#         plot01 (y, y_pred3, total_cases_pred3, country_data)
    
    total_cases_diff = None
    country_data['seq'] = [i for i in range(len(country_data))]
    non_zero_indices = country_data.loc[country_data[target_intv] > 0, 'seq'].tolist()
    ### Assuming that the infection will be detectable only after the incubation_period 
    non_zero_indices = [min((v + incubation_period), len(country_data)-1) for v in non_zero_indices]
    country_data.drop(['seq'], axis=1, inplace=True)
    ### Measure the mean difference between the two scenarios
    if len(non_zero_indices) > 0:
        total_cases_diff = np.mean(total_cases_pred3.iloc[non_zero_indices] - total_cases_pred.iloc[non_zero_indices])

    return total_cases_diff

### Measure the approx effectiveness of each intervention for a country
def measure_intervention_effects (country_data, nlags=num_lags, plot_graphs=True):
    target_col = 'Change_MA'
    country_data = add_features (country_data)
    X, X1, X2, y = get_modeling_data (country_data, target_col, nlags=nlags, fit_intv_effect=True)
    # Get prediction with all features (previous case counts and all interventions)
    y_pred, total_cases_pred = get_predicted_total_cases (country_data, X2, y)

    intervention_vs_cases_diff = []
    interventions = X2.columns
    country_data['seq1'] = [i for i in range(len(country_data))]
    for intervention in interventions:
        X2_Temp = X2.copy()
        # Turning an intervention off
        X2_Temp[intervention] = 0
        seq_with_conf_cases = country_data.loc[country_data['ConfirmedCases']>0, 'seq1']
        seq_with_intervention_active = country_data.loc[country_data[intervention]>0, 'seq1']
        delay_in_intervention = 100 * (seq_with_intervention_active.min() - seq_with_conf_cases.min()) / len(seq_with_conf_cases)
        # Get an approx case-count diff with an intervention turned off
        total_cases_diff = get_case_count_diff (country_data, X1, X2_Temp, y, total_cases_pred, intervention, plot_graphs=plot_graphs)
        intervention_vs_cases_diff.append([intervention, total_cases_diff])

    country_data.drop(['seq1'], axis=1, inplace=True)
    country_intv_scores = pd.DataFrame(intervention_vs_cases_diff, columns=['intervention', 'score'])
    min_score, max_score = np.min(country_intv_scores['score']), np.max(country_intv_scores['score'])
    country_intv_scores['score_norm'] = country_intv_scores['score'].apply(lambda x: (x - min_score) / (max_score - min_score))
    #country_intv_scores['score_norm'] = country_intv_scores['score_norm'].max() - country_intv_scores['score_norm']
    return country_intv_scores


### Plot actuals vs predictions
def plot (y, y_pred, country_data):
    fig, ax = plt.subplots(1, 1, figsize=(17, 7))
    ax.plot(y.values, label='actual', color='green')
    ax.plot(y_pred, label='predicted')
    ax.legend()
    ax.set_title('Actual Vs Predicted: % Case Count Change')
    plt.show();
    

### Score interventions (features) using the feature importances measured by the ensemble model used (XGBoost Regressor) 
def score_country_interventions (country_data, nlags=1, fit_conf_cases=False, fit_intv_effect=False, plot_predictions=False):  
    
    if fit_intv_effect:
        return measure_intervention_effects (country_data, nlags=nlags)
    
    target_col = 'Change_MA' if fit_conf_cases else 'StringencyIndex'
    if fit_conf_cases:
        country_data = add_features (country_data)
    X, _, _, y = get_modeling_data (country_data, target_col, nlags=nlags, fit_conf_cases=fit_conf_cases)
    
    model, y_pred, mae, rmse = fit_model (X, y)
    #print (f'MAE: {mae}, RMSE: {rmse} [Predicting Confirmed Case Rate? => {fit_conf_cases}]')
    
    if fit_conf_cases and plot_predictions:
        plot (y, y_pred, country_data)
    
    #plot_importance(model)
    #pyplot.show()
    #feature_importances_dict = model.get_booster().get_score(importance_type='gain')
    feature_importances_dict = model.get_booster().get_fscore()
    feature_importances_list = list()
    for feature in feature_importances_dict:
        if 'Change_MA' in feature:
            continue
        feature_importances_list.append([feature, feature_importances_dict[feature]])

    country_intv_scores = pd.DataFrame(feature_importances_list, columns=['intervention', 'score'])
    min_score, max_score = np.min(country_intv_scores['score']), np.max(country_intv_scores['score'])
    country_intv_scores['score_norm'] = country_intv_scores['score'].apply(lambda x: (x - min_score) / (max_score - min_score))
    #display (country_measures_analysis.head(25))
    
    return country_intv_scores

    
def score_interventions (selected_countries=None): 
    
    assert sum(intervention_scoring_methods.values()) == 1.0, 'Error: Sum of the scoring method weights should be 1.0' 
    
    fit_stringency_index = True if intervention_scoring_methods['fit_stringency_index'] > 0 else False
    fit_conf_cases = True if intervention_scoring_methods['fit_conf_cases'] > 0 else False
    fit_intv_effect = True if intervention_scoring_methods['fit_intv_effect'] > 0 else False
    
    data_all, country_dict = get_all_data (data_src)
    if selected_countries is None or len(selected_countries)==0:
        selected_countries = data_all['CountryCode'].unique()
    
    all_country_intv_scores = pd.DataFrame()
    for country_code in selected_countries:
        print ('* '*10 + f'Scoring interventions for country: {country_dict[country_code]} [{country_code}]')
        country_data = get_country_data (data_all, country_code)
        
        if len(country_data) < 50:
            print ('Not enough data to score interventions . . .\n')
            continue
            
        
        country_scores_merged = None
        if fit_stringency_index:
            country_scores = score_country_interventions (country_data, nlags=num_lags, fit_conf_cases=False, plot_predictions=False)
            country_scores_merged = country_scores
            country_scores_merged.rename(columns={'score_norm': 'score_stringency_idx'}, inplace=True)

        if fit_conf_cases:
            country_scores = score_country_interventions (country_data, nlags=num_lags, fit_conf_cases=True, plot_predictions=False)
            country_scores = country_scores[['intervention', 'score_norm']]
            country_scores.rename(columns={'score_norm': 'score_conf_cases'}, inplace=True)
            country_scores_merged = country_scores if country_scores_merged is None else pd.merge(country_scores_merged, country_scores, on=['intervention'], how='outer')
            
        if fit_intv_effect:
            country_scores = score_country_interventions (country_data, nlags=num_lags, fit_intv_effect=True, plot_predictions=False)
            country_scores = country_scores[['intervention', 'score_norm']]
            country_scores.rename(columns={'score_norm': 'score_intv_effects'}, inplace=True)
            country_scores_merged = country_scores if country_scores_merged is None else pd.merge(country_scores_merged, country_scores, on=['intervention'], how='outer')
        
        if not fit_stringency_index:
            country_scores_merged['score_stringency_idx'] = 'NA' 
        if not fit_conf_cases:
            country_scores_merged['score_conf_cases'] = 'NA'
        if not fit_intv_effect:
            country_scores_merged['score_intv_effects'] = 'NA'
        country_scores_merged['country_code'] = country_code
        country_scores_merged['country_name'] = country_dict[country_code]
        country_scores_merged.fillna(0, inplace=True) 
            
        country_scores_merged['score_final'] = country_scores_merged['score_stringency_idx'] * intervention_scoring_methods['fit_stringency_index'] if fit_stringency_index else 0
        country_scores_merged['score_final'] += country_scores_merged['score_conf_cases'] * intervention_scoring_methods['fit_conf_cases'] if fit_conf_cases else 0
        country_scores_merged['score_final'] += country_scores_merged['score_intv_effects'] * intervention_scoring_methods['fit_intv_effect'] if fit_intv_effect else 0
                
        display_cols = ['country_code', 'country_name', 'intervention', 'score_stringency_idx', 'score_conf_cases', 'score_intv_effects', 'score_final']
        country_scores_merged = country_scores_merged[display_cols].reset_index(drop=True)       
        
        if not config.sagemaker_run:
            display (country_scores_merged[display_cols].sort_values(by='score_final', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Greens'))
        
        all_country_intv_scores = pd.concat([all_country_intv_scores, country_scores_merged], axis=0)

    all_country_intv_scores = all_country_intv_scores.sort_values(
        by=['country_name', 'score_final'], ascending=[True, False]).reset_index(drop=True)
    
    return data_all, selected_countries, all_country_intv_scores
    

def assign_weighted_aggregations (dfx, intervention_scores, target_countries):
    scored_intvs = list(intervention_scores['intervention'].unique())
    for intv in scored_intvs:
        dfx[intv+'_weighted'] = 0
    dfx['aggr_weighted_intv'] = 0
    dfx['aggr_weighted_intv_norm'] = 0
    
    if target_countries is None or len(target_countries)==0:
        target_countries = dfx['CountryCode'].unique()

    for country in target_countries:
        country_intv_weights = intervention_scores.loc[intervention_scores['country_code']==country]
        if len(country_intv_weights) == 0:
            continue
        uniq_vals = []
        for intv in scored_intvs:
            if intv not in country_intv_weights['intervention'].unique():
                continue
            #print (country, intv, len(country_intv_weights.loc[country_intv_weights['intervention'] == intv, 'score_final']))
            intv_weight = country_intv_weights.loc[country_intv_weights['intervention'] == intv, 'score_final'].iloc[0]
            dfx.loc[dfx['CountryCode']==country, intv+'_weighted'] = dfx.loc[dfx['CountryCode']==country, intv] 
            dfx.loc[dfx['CountryCode']==country, intv+'_weighted'] *= intv_weight
            dfx.loc[dfx['CountryCode']==country, intv+'_weighted'].fillna(0, inplace=True)
            
            if dfx.loc[dfx['CountryCode']==country, intv+'_weighted'].isnull().any():
                continue
                
            dfx.loc[dfx['CountryCode']==country, 'aggr_weighted_intv'] += \
                dfx.loc[dfx['CountryCode']==country, intv+'_weighted']

            dfx.loc[dfx['CountryCode']==country, 'aggr_weighted_intv_norm'] = \
                dfx.loc[dfx['CountryCode']==country, 'aggr_weighted_intv']
            if dfx.loc[dfx['CountryCode']==country, 'aggr_weighted_intv'].max() > 0:
                dfx.loc[dfx['CountryCode']==country, 'aggr_weighted_intv_norm'] /= \
                    dfx.loc[dfx['CountryCode']==country, 'aggr_weighted_intv'].max()
    return dfx