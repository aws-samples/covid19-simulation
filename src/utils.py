import numpy as np
import pandas as pd
import urllib.request

from io import StringIO

import config



def toss_coin():
    return np.random.uniform(0, 1, None)


#https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0060343
def incidence_rate_curve(n_days, wave1_weeks, peak_wave1, peak_wave2, mu1, mu2, wave2_spread_factor):
    #sigma = 7 * 4
    sigma = int (7 * max (12, wave1_weeks) / 3)
    
    timeline = max (366, n_days+1)
    prob = np.zeros((timeline, 1))
    for dd in range(len(prob)):
        prob[dd] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (dd - mu1) ** 2 / (2 * sigma ** 2))
    scale = peak_wave1 / max(prob)
    prob1 = scale * prob
    
    #sigma2 = 7 * 2
    sigma2 = sigma * wave2_spread_factor

    prob = np.zeros((timeline, 1))
    for dd in range(len(prob)):
        prob[dd] = 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(- (dd - mu2) ** 2 / (2 * sigma2 ** 2))
    scale = peak_wave2 / max(prob)
    prob2 = scale * prob

    # shift to start in week x
    prob = np.add(prob1, prob2)
    
    prob_cutoff = np.percentile(prob, 50)
    prob[prob == 0] = np.random.uniform(low=0.0, high=prob_cutoff, size=(len(prob[prob == 0]),))

    return [val[0] for val in prob]


# Testing functions
def get_population_symptom_rate(population):
    return np.sum([person.symptoms for person in population]) / len(population)


def send_population_pcr_test(rapid_pcr_tests_available, pcr_tests_sent_per_day, population,
                             current_day):
    # find people who are eligible for a test
    eligible_for_pcr = []
    for person in population:
        eligible_for_pcr.append(-1)
        if config.add_ab:
            if current_day == 0:
                person.antibody_test_administered.append([current_day, person.igg])
            elif len(person.antibody_test_administered) > 0 and person.igg:
                continue

        if person.at_home_isolation:
            continue

        if person.clear_to_work:
            continue

        if is_waiting_pcr_test_result(person):
            continue

        if len(person.pcr_tests_administered) == 0:
            eligible_for_pcr[-1] = 1
        else:
            if len(person.at_home_isolation_administered) > 0 and person.at_home_isolation is None:
                eligible_for_pcr[-1] = 0
            else:
                last_pcr_test_day = person.pcr_tests_administered[-1][0]
                if (current_day - last_pcr_test_day) >= config.time_between_consecutive_pcr_tests:
                    eligible_for_pcr[-1] = 2

    eligible_for_pcr = np.array(eligible_for_pcr)
    for idx in np.where(eligible_for_pcr == 0)[0]:
        rapid_pcr_tests_available, pcr_tests_sent_per_day = \
            send_pcr_test(rapid_pcr_tests_available, pcr_tests_sent_per_day, population[idx], current_day)
    for idx in np.where(eligible_for_pcr == 1)[0]:
        rapid_pcr_tests_available, pcr_tests_sent_per_day = \
            send_pcr_test(rapid_pcr_tests_available, pcr_tests_sent_per_day, population[idx], current_day)
    for idx in np.where(eligible_for_pcr == 2)[0]:
        rapid_pcr_tests_available, pcr_tests_sent_per_day = \
            send_pcr_test(rapid_pcr_tests_available, pcr_tests_sent_per_day, population[idx], current_day)

    return (np.where(eligible_for_pcr == 0)[0].size + np.where(eligible_for_pcr == 1)[0].size +
            np.where(eligible_for_pcr == 2)[0].size), rapid_pcr_tests_available, pcr_tests_sent_per_day
    # print ("DAY: %s, sym:%s, prev sym: %s, asym: %s, cont asym:%s" % (current_day, p1, p2, p3, p4))


def send_pcr_test(rapid_pcr_tests_available, pcr_tests_sent_per_day, person, current_day):
    if rapid_pcr_tests_available <= 0 or is_waiting_pcr_test_result(person):
        return rapid_pcr_tests_available, pcr_tests_sent_per_day

    person.pcr_tests_administered.append([current_day, None])
    # person.timeline.append([current_day, 'pcr'])

    return rapid_pcr_tests_available - 1, pcr_tests_sent_per_day + 1


def update_per_person_result(population, current_day):
    daily_pos = 0
    daily_neg = 0
    pcr_after_isolation = 0
    for person in population:
        if len(person.pcr_tests_administered) > 0:
            # for pcr_test in person.pcr_tests_administered:
            pcr_test = person.pcr_tests_administered[-1]
            # import pdb; pdb.set_trace()

            if pcr_test[1] is None:  # no results yet
                pcr_test_day = pcr_test[0]
                wait_days = current_day - pcr_test_day

                #                 if wait_days >2:
                #                     print ("--- broken ---")
                #                     import pdb; pdb.set_trace()

                if pcr_test[1] is None and wait_days == 0:
                    if person.exposure:
                        person.pcr_tests_administered[-1][1] = True

                        #                         # will never show positive since they were infected prior to Day 0
                        #                         if person.igg and person.infection_contracted[-1] < 0:
                        #                             person.pcr_tests_administered[-1][1] = False
                        #                             daily_neg +=1
                        #                             continue

                        # Override - if antibodies, PCR will be -ve
                        if person.antibody_present or person.igg:
                            person.pcr_tests_administered[-1][1] = False
                            daily_neg += 1
                            pcr_after_isolation += 1
                            continue
                        daily_pos += 1
                    else:
                        person.pcr_tests_administered[-1][1] = False
                        daily_neg += 1

    return daily_pos, daily_neg, pcr_after_isolation


# Pcr batch update
def update_population_batch_pcr_test(population, current_day):
    # update person's test results based on the batch results
    return update_per_person_result(population, current_day)


# Update disease progression

def symptoms_distribution(mu=5, sigma=5):
    mu += np.random.uniform(-1.0, 1.0, 1)[0]  # add half a day randomness
    # sigma += np.random.uniform(-1.0,1.0,1)[0] # add half a day randomness

    prob = np.zeros((60, 1))
    for dd in range(len(prob)):
        prob[dd] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (dd - mu) ** 2 / (2 * sigma ** 2))

    prob[0] = 0
    prob[1] = 0

    for dd in range(18, len(prob)):
        prob[dd] = 0  # prob[18]

    # normalize to sum to 1
    prob = prob / np.sum(prob)

    return prob


def igm_distribution(mu=14, sigma=3):
    mu += np.random.uniform(-1.0, 1.0, None)  # add half a day randomness
    # sigma += np.random.uniform(-1.0,1.0,1)[0] # add half a day randomness

    prob = np.zeros((60, 1))
    if toss_coin() <= 0.99:
        prob[:23] = (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.arange(23) - mu) ** 2 / (2 * sigma ** 2))).reshape(
            23, 1)

        for dd in range(23, len(prob)):
            prob[dd] = 0.0  # prob[20]

        # normalize to sum to 1
        prob = prob / np.sum(prob)

    return prob


def igg_distribution(mu=18, sigma=3):
    # def igg_distribution(mu=24, sigma=7):
    mu += np.random.uniform(-1.0, 1.0, None)  # add half a day randomness
    # sigma += np.random.uniform(-1.0,1.0,1)[0] # add half a day randomness
    ind = int(mu + 3)

    prob = np.zeros((60, 1))
    if toss_coin() <= 0.99:
        prob[:ind] = (
                1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.arange(ind) - mu) ** 2 / (2 * sigma ** 2))).reshape(
            ind, 1)
        prob[ind:] = prob[ind]

        for dd in range(int(mu + 3), len(prob)):
            prob[dd] = prob[int(mu + 3)]

        # normalize to sum to 1
        prob = prob / np.sum(prob)
    return prob


# TEST Availability Schedule
def get_schedule(schedule, day):
    try:
        return schedule[day]
    except KeyError:
        return 0


def get_pcr_schdeule(pcr_schedule, day):
    # assume capacity is added daily.
    # week = day //7
    # return (week+1) * init_test_capacity

    return pcr_schedule[day]


def get_ab_schedule():
    # assume capacity is added daily.
    return 0


class Person:
    def __init__(self, _id):
        self.p_id = _id

        # disease progression variables
        self.carrier = False  # Not contracted
        self.exposure = False  # Contracted but may not be detected yet
        self.infection_contracted = []  # date, injection contracted
        self.symptoms_observed = []  # day when you started showing symptoms
        self.symptoms = False
        self.antibody_present = False  # antibody_present AB Test +ve
        self.symptoms_surpressed_afterwards = False  # symptoms gone after N days
        self.infectious = False
        self.igg = False  # immune...
        self.igm = False

        # observational variables
        self.possible_infection = False  # Questionnaire or Temperature check ...
        self.known_contact_with_covid_case = False

        # risk variables (e.g., social, age, co-morb...)
        self.risk_score = toss_coin()  # 0-1 based on ... use this for PCR testing

        # test variables
        self.pcr_tests_administered = []  # date, result, batch_id
        self.antibody_test_administered = []  # date, result
        self.likelihood_score = 0  # not used now
        self.pcr_test_batch_id = -1  # current batch id. -1 == not in a batch
        self.ab_false_positive = False
        self.pcr_false_negative = False

        # home in isolation or clear work
        # (+ve on AB test and -ve on viral load and 14 day period) - you can't be infected or infect others
        self.clear_to_work = False
        self.at_home_isolation = False  # False = init, True = isolated, None = back from isolation
        self.at_home_isolation_administered = []  # day when the disease is confirmed for self isolation

        # debug variable - person timeline
        self.timeline = []  # 'day,test'; 'day,time'
        self.history = []

        # employment tracker
        self.join_day = -10
        self.exit_day = None

        self.transmission_network = list()

        # igg achieved
        self.igg_achieved = []


def is_waiting_pcr_test_result(person):
    # waiting on any pcr test results
    if len(person.pcr_tests_administered) == 0:
        return None
    if person.pcr_tests_administered[-1][1] is None:
        return True
    return False


def update_population_igg_progression(population, current_day):
    for person in population:
        if person.exposure:
            days_passed = min(59, current_day - person.infection_contracted[-1])
            # igg is cumulative check, once gained, not goes away
            # show_igg_distribution = np.cumsum(igg_distribution())
            # prob = show_igg_distribution[int(days_passed)]
            # if prob > igg_detection_threshold: # some delta error
            if days_passed >= 14:
                person.igg = True
                if len(person.igg_achieved) == 0:
                    person.igg_achieved.append(current_day)
            else:
                person.igg = False
        else:
            person.igg = False


def update_population_igm_progression(population, current_day, igm_detection_threshold=0.001):
    for person in population:
        if person.exposure:
            days_passed = min(59, current_day - person.infection_contracted[-1])
            show_igm_distribution = igm_distribution()
            prob = show_igm_distribution[int(days_passed)]
            if prob > igm_detection_threshold:  # some delta error
                person.igm = True
            else:
                person.igm = False
        else:
            person.igm = False


def infectiousness_distribution(mu=5, sigma=5):
    mu += np.random.uniform(-1.0, 1.0, 1)[0]  # add half a day randomness
    sigma += np.random.uniform(-1.0, 1.0, 1)[0]  # add half a day randomness

    prob = np.zeros((360, 1))
    for dd in range(18):
        prob[dd] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (dd - mu) ** 2 / (2 * sigma ** 2))

    prob[0] = 0
    prob[1] = 0

    for dd in range(18, len(prob)):
        prob[dd] = 0  # prob[18]

    # normalize to sum to 1
    prob = prob / np.sum(prob)

    return prob


def update_population_disease_progression(population, current_day):
    for person in population:
        if person.exposure:
            days_passed = current_day - person.infection_contracted[-1]
            if days_passed <= 14:
                person.infectious = True
            else:
                person.infectious = False


def update_population_symptom_progression(population, current_day, symptoms_threshold=0.08):
    num_symptomatic, num_asymptomatic = 0, 0
    for person in population:
        if person.exposure:
            if person.carrier:
                person.symptoms = False
                num_asymptomatic += 1
            else:
                days_passed = current_day - person.infection_contracted[-1]
                if days_passed <= 14:
                    # already counted when they first get symptoms, do not increment counter
                    prob = person.infectiousness_distribution[int(days_passed)]

                    if prob > symptoms_threshold:  # some delta error
                        # if the symptoms were false before, turned them on now
                        if not person.symptoms:
                            person.symptoms_observed.append(current_day)
                            # turn on the symptoms
                            person.symptoms = True
                            num_symptomatic += 1
                    else:
                        person.symptoms = False
                else:
                    person.symptoms = False
        else:
            person.symptoms = False

    return num_symptomatic, num_asymptomatic


# Update our recommendation, e.g. stay-at-home vs clear-to-work

def update_population_clear_to_work(population):
    num_clear_to_work = 0
    for person in population:
        # PCR positive, 3 week quarantine, PCR -ve again
        if person.at_home_isolation is None and person.pcr_tests_administered[-1][1] is False:
            person.clear_to_work = True
            num_clear_to_work += 1
        # check immunity with Ab and clear
        elif config.add_ab and person.igg:
            person.clear_to_work = True
            num_clear_to_work += 1

    return num_clear_to_work


# isolation if we suspect
def update_population_isolation(population, current_day):
    # isolate any individual we found out has PCR +ve
    isolated = 0
    out_of_quarantine = 0
    in_to_quarantine = 0
    for person in population:

        if person.at_home_isolation is None:  # safe - done with quarantine
            continue

        if len(person.pcr_tests_administered) == 0:
            continue

        if person.at_home_isolation is True:
            isolation_days = person.at_home_isolation_administered[-1][0]
            if (current_day - isolation_days) >= config.time_self_isolation:
                person.at_home_isolation = None
                person.antibody_present = True
                out_of_quarantine += 1
        else:
            # Last PCR test was positive, isolate.
            if len(person.pcr_tests_administered) > 0 and person.pcr_tests_administered[-1][1]:
                if not person.at_home_isolation:
                    person.at_home_isolation_administered.append([current_day, 0])  # (day, PCR_isolation)
                    in_to_quarantine += 1
                person.at_home_isolation = True

        if person.at_home_isolation is True:
            isolated += 1

    # print ("ISOLATION: ", current_day, isolated)
    return out_of_quarantine, isolated, in_to_quarantine


# Update observation state
def update_population_possible_infection(population):
    for person in population:
        if person.symptoms:
            person.possible_infection = True
        else:
            person.possible_infection = False


# Given scores, a map of scores to positive viral load probabilities, a batching mode, and batch sizes
# Simulate the PRC w/ batching protocol
# return the number of tests taken to identify all positives present

# PCR batching logic based on the risk score
# select batching algo from pp
def get_batch_sizes():
    return [1]


# Get the testing rates based on the number of tests available or to be obtained
# Testing rate strategy
def antibody_testing_rate():
    return 1  # antibody_tests_available / tot_population


def pcr_testing_rate():
    return 1  # rapid_pcr_tests_available / tot_population


# Update prevalance rate based on external factors, e.g., social, policy, ...
def update_current_infection_rate():
    # assume infection rate overall remains the same
    return


def update_population_based_on_attrition(population, current_day):
    # add attrition
    number_to_remove = 0
    number_left_with_pcr_pos = 0
    number_left_with_pcr_neg = 0
    number_left_with_no_test = 0
    number_left_exposed_to_virus = 0

    if current_day > 0 and current_day % config.attrition_frequency == 0:
        # permute the population
        temp_randomized_population = np.random.permutation(len(population))

        # consider attrition for people not in quarantine
        randomized_population = list()
        for idx in temp_randomized_population:
            if not population[idx].at_home_isolation:
                randomized_population.append(idx)

        # immunity cutoff
        number_to_remove = int(config.attrition_rate * len(population))

        for ii in range(number_to_remove):
            idx = randomized_population[ii]

            if len(population[idx].pcr_tests_administered) > 0:
                # left with a pos PCR test at some point
                all_neg_test = True
                for tt in population[idx].pcr_tests_administered:
                    if tt[1]:
                        number_left_with_pcr_pos += 1
                        all_neg_test = False
                        break

                if all_neg_test:
                    number_left_with_pcr_neg += 1
            else:
                number_left_with_no_test += 1

            # incidence rate becomes negative due to this
            if population[idx].exposure:
                number_left_exposed_to_virus += 1

            # reset the person, assign a new id
            population[idx].p_id = idx + current_day * 100000  # len(population)

            # disease progression variables
            population[idx].carrier = False  # Not contracted
            population[idx].exposure = False  # Contracted but may not be detected yet
            population[idx].infection_contracted = []  # date, injection contracted
            population[idx].symptoms_observed = []  # day when you started showing symptoms
            population[idx].symptoms = False
            population[idx].antibody_present = False  # antibody_present AB Test +ve
            population[idx].symptoms_surpressed_afterwards = False  # symptoms gone after N days
            population[idx].infectious = False
            population[idx].igg = False  # immune...
            population[idx].igm = False

            # observational variables
            population[idx].possible_infection = False  # Questionnaire or Temperature check ...
            population[idx].known_contact_with_covid_case = False

            # risk variables (e.g., social, age, co-morb...)
            population[idx].risk_score = toss_coin()  # 0-1 based on ... use this for PCR testing

            # test variables
            population[idx].pcr_tests_administered = []  # date, result, batch_id
            population[idx].antibody_test_administered = []  # date, result
            population[idx].likelihood_score = 0  # not used now
            population[idx].pcr_test_batch_id = -1  # current batch id. -1 == not in a batch
            population[idx].ab_false_positive = False
            population[idx].pcr_false_negative = False

            # home in isolation or clear work
            # (+ve on AB test and -ve on viral load and 14 day period) - you can't be infected or infect others
            population[idx].clear_to_work = False
            population[idx].at_home_isolation = False  # False = init, True = isolated, None = back from isolation
            population[idx].at_home_isolation_administered = []  # day when the disease is confirmed for self isolation

            # debug variable - person timeline
            population[idx].timeline = []  # 'day,test'; 'day,time'
            population[idx].history = []

            # employment tracker
            # person start date
            population[idx].exit_day = None
            population[idx].join_day = current_day

    return number_to_remove, number_left_with_pcr_pos, number_left_with_pcr_neg, number_left_with_no_test, \
           number_left_exposed_to_virus


# initialization of population
# NOTE: Assume that we don't know about the asymptomatic people in the warehouse; Need to take incubation period

def population_init_infections(population, incidence_rate_annual, case_population_fraction):
    # permute the population
    randomized_population = np.random.permutation(len(population))

    # immunity cutoff
    number_of_immune = int(config.initial_antibody_immunity_in_population * len(population))

    # give first N percent immunity
    for ii in randomized_population[:number_of_immune]:
        population[ii].exposure = True
        # assign an exposure day greater than immunity build-up start
        infection_day = -np.random.randint(20, 30, 1)[0]
        population[ii].infection_contracted.append(infection_day)
        population[ii].infectiousness_distribution = infectiousness_distribution()

        # assume 2 days after the infection the symptoms may appear
        if toss_coin() < config.infected_and_symptomatic_in_population:
            population[ii].symptoms_observed.append(infection_day + 2)  # happens before day 1

        # give immunity
        population[ii].igg = True
        population[ii].igm = False
        population[ii].infectious = False

    # create people who are currently infected and does not have immunity
    # number_of_new_infections = int(incidence_rate_annual[0] * len(population))
    # Initialize infected population based on current infection rate
    number_of_new_infections = 0
    #if case_population_fraction > 0:
    #    number_of_new_infections = int(case_population_fraction * len(population))
    
    #print (number_of_new_infections)

    # give N percent after immunity
    for ii in randomized_population[number_of_immune:(number_of_immune + number_of_new_infections)]:
        population[ii].exposure = True
        # assign an exposure day within the last 2 weeks
        infection_day = -2  # 2 days ago
        population[ii].infection_contracted.append(infection_day)
        population[ii].infectiousness_distribution = infectiousness_distribution()

        # assume 2 days after the infection the symptoms may appear
        if toss_coin() < config.infected_and_symptomatic_in_population:
            population[ii].symptoms_observed.append(infection_day + 2)  # happens before day 1

        # give immunity
        population[ii].igg = False
        population[ii].igm = False
        population[ii].infectious = True

    # update symptoms
    update_population_symptom_progression(population, 0)

    return number_of_new_infections, number_of_immune


# initialize 20 person to each test for now
def update_rapid_pcr_tests_available(pcr_tests_available):
    return [config.pcr_batch_size for _ in range(pcr_tests_available)], [None for _ in range(pcr_tests_available)]


def get_incidence_rate(df, population):
    df = df.sort_values(by=['date'])
    df_last_week = df.iloc[-8:]
    print(df_last_week)
    rate = int(np.mean(np.diff(df_last_week.cases))) / population / float(config.infected_and_symptomatic_in_population)
    print('Avg new cases = %d, rate = %.4f\n' % (np.mean(np.diff(df_last_week.cases)), rate))

    return rate


def get_population(filename, fc_code):
    fc_roster_df = pd.read_csv(filename, header=1)
    fc_roster_df = fc_roster_df.fillna(method='ffill')
    fc_roster_df.head()

    n_population = 0
    for val, pop in zip(fc_roster_df['Site ID'], fc_roster_df['Sum of Active Roster']):
        if fc_code in val:
            print(val, pop)
            n_population = int(pop.replace(',', ''))
            break

    return n_population


def get_pcr_schedule(n_days):
    timeline = max (365, n_days)
    return [420] * timeline


def get_model_outcome_metrics(population, day):
    susceptible, infectious_at_work, recovered = 0, 0, 0
    daily_infections_at_work, days_spent_infected_at_work = 0, 0
    num_total_exposed = 0
    num_total_quarantine_administered = 0

    for person in population:

        if not person.exposure:
            susceptible += 1

        if person.exposure and not person.at_home_isolation and not person.igg:
            infectious_at_work += 1
            infected_days = day - person.infection_contracted[-1]
            if infected_days <= 1:
                daily_infections_at_work += 1

        if person.igg:
            recovered += 1

            # number of days between infection and quarantine
            if len(person.igg_achieved) > 0:
                days_infected = person.igg_achieved[0] - person.infection_contracted[-1]
                days_spent_infected_at_work += days_infected

        if person.exposure:
            num_total_exposed += 1

        if len(person.at_home_isolation_administered) > 0:
            num_total_quarantine_administered += 1

    return susceptible, infectious_at_work, recovered, daily_infections_at_work, days_spent_infected_at_work, \
           num_total_exposed, num_total_quarantine_administered


def update_population_infections(learning_phase, total_active, incidence_rate_annual, population, current_day, transmission_prob,
                                 transmission_control, intervention_influence_pctg, intervention_score=None):
    # ASSUMPTION: Newly joining are not infected or have immunity, we ignore infected asymptomatics joining in
    bin_based_transmission_prob = transmission_prob  # 0.005
    
    intervention_score = intervention_score if intervention_score >= 0.1 else 0.1

    overall_transmission_possibility = 1
    
    # Model intervention influence on transmission. transmission_control represents the transmission manageability via
    # interventions.
    if intervention_score is not None and intervention_score > 0:
        overall_transmission_possibility = (1 - transmission_control) + (transmission_control * (1 - intervention_score))
        
    # basic number
    number_of_people_to_infect = config.number_of_people_to_infect_default

    incidence_rate_today = incidence_rate_annual[current_day]

    num_new_infections = 0
    num_new_transmission_infections = 0
    
    # transmission over the weekdays only
    for person in population:
        if current_day % 7 < 5:
            transmission_probability = overall_transmission_possibility * bin_based_transmission_prob
            # Is exposed person still in the FC?
            if person.exposure and not person.at_home_isolation:
                infected_days = current_day - person.infection_contracted[-1]
                if 2 < infected_days <= 10:
                    transmission_probability = transmission_probability * infected_days

                    # only within transmission probability
                    if toss_coin() < transmission_probability:
                        # N different persons instead of the same person
                        subpopulation_infected = np.random.choice(population, number_of_people_to_infect,
                                                                  replace=False)
                        for person_infected in subpopulation_infected:
                            if not person_infected.exposure and not person_infected.igg and \
                                    (person_infected.join_day < current_day):
                                person_infected.exposure = True
                                person_infected.infection_contracted.append(current_day)
                                person_infected.infectiousness_distribution = infectiousness_distribution()
                                num_new_transmission_infections += 1
                                person.transmission_network.append([person_infected.p_id, current_day])
                                continue

        if not person.exposure and not person.igg and (person.join_day < current_day):
            toss_coin_prob = toss_coin()
            if total_active >= 2:
                active_case_fraction_log = (np.log(total_active) / np.log(len(population)))
            else:
                active_case_fraction_log = (np.log(2) / np.log(len(population)))
                
            if toss_coin_prob <= incidence_rate_today * overall_transmission_possibility * active_case_fraction_log:
                person.exposure = True
                person.infection_contracted.append(current_day)
                person.infectiousness_distribution = infectiousness_distribution()
                num_new_infections += 1

    return num_new_infections, num_new_transmission_infections


def get_county_info(fips: str) -> (str, str, int):
    df = pd.read_csv(config.county_population_filename)

    county_col = int(fips[2:])
    state_col = int(fips[0:2])

    df_sel = df.loc[(df['COUNTY'] == county_col) & (df['STATE'] == state_col)]

    return df_sel['STNAME'].values[0], df_sel['CTYNAME'].values[0].replace('County', '').strip(), \
           df_sel['POPESTIMATE2019'].values[0]


def download_latest_data(url, file_path=None):
    with urllib.request.urlopen(url) as response:
        data = response.read()  # a `bytes` object

        if file_path is not None:
            with open(file_path, 'wb') as f:
                f.write(data)
        else:
            return pd.read_csv(StringIO(data.decode("utf-8")))
