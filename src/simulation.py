import numpy as np
import config


from utils import Person, incidence_rate_curve, get_pcr_schedule, population_init_infections, \
    update_population_igg_progression, update_population_igm_progression, update_population_disease_progression, \
    update_population_symptom_progression, update_population_possible_infection, pcr_testing_rate, \
    update_rapid_pcr_tests_available, send_population_pcr_test, update_population_batch_pcr_test, \
    update_population_isolation, update_population_clear_to_work, update_population_based_on_attrition, \
    get_model_outcome_metrics, get_pcr_schdeule, update_population_infections


class Simulation:
    def __init__(self, state_population, n_days, wave1_weeks, weeks_between_waves, case_rate, active_case_population_fraction,
                 daily_case_population_fraction, testing_capacity, transmission_strength, 
                 transmission_prob=config.transmission_prob_default, intervention_influence_pctg=config.intervention_influence_pctg_default, 
                 wave2_peak_factor=config.wave2_peak_factor_default, wave2_spread_factor=config.wave2_spread_factor_default, 
                 log_results=True):
        # state, county, county_population = get_county_info(fips)
        self.active_case_population_fraction = active_case_population_fraction
        self.daily_case_population_fraction = daily_case_population_fraction
        self.transmission_strength = transmission_strength
        self.transmission_prob = transmission_prob
        self.testing_capacity = testing_capacity
        self.log_results = log_results
        self.wave1_weeks = wave1_weeks
        self.intervention_influence_pctg = intervention_influence_pctg
        
        peak_wave1 = case_rate  # get_incidence_rate(df_state, state_population)
        peak_wave2 = wave2_peak_factor * peak_wave1
        mu1 = wave1_weeks * 7
        mu2 = mu1 + weeks_between_waves * 7

        self.incidence_rate_annual = incidence_rate_curve(n_days, wave1_weeks, peak_wave1, peak_wave2, mu1, mu2, wave2_spread_factor)
        self.prevalence_rate_annual = np.cumsum(self.incidence_rate_annual)

        self.pcr_schedule_ = get_pcr_schedule(n_days)
        self.pcr_schedule = [int(1 + state_population / 14) for _ in self.pcr_schedule_]
        

    @staticmethod
    def set_config(
                   add_ab=False,
                   batching=True,
                   ab_cost=28,
                   ab_cost_cheap=28,
                   pcr_cost=77,
                   attrition_rate=0.05,
                   attrition_frequency=14,
                   time_to_wait_for_antibody_test=0,
                   time_to_wait_for_pcr_test_results=2,
                   time_for_incubation=2,
                   time_to_stop_symptoms=8,
                   time_self_isolation=14,
                   ab_false_positive_rate=1.0,
                   pcr_false_negative_rate=1.0,
                   pcr_batch_size=1,
                   init_test_capacity=1000,
                   init_ab_test_capacity=100,
                   ab_tests_sent_per_day=0,
                   time_between_consecutive_pcr_tests=14,
                   pcr_test_result_positive_probability=0.01,
                   initial_antibody_immunity_in_population=0,
                   infected_and_symptomatic_in_population=0.14,
                   symptom_covid_vs_flu=0,
                   awaiting_on_pcr_test_result=0):
        config.batching = batching
        config.ab_cost = ab_cost
        config.ab_cost_cheap = ab_cost_cheap
        config.pcr_cost = pcr_cost
        config.attrition_frequency = attrition_frequency
        config.attrition_rate = attrition_rate
        config.time_between_consecutive_pcr_tests = time_between_consecutive_pcr_tests
        config.time_for_incubation = time_for_incubation
        config.time_to_stop_symptoms = time_to_stop_symptoms
        config.time_to_wait_for_antibody_test = time_to_wait_for_antibody_test
        config.time_to_wait_for_pcr_test_results = time_to_wait_for_pcr_test_results
        config.time_self_isolation = time_self_isolation
        config.ab_false_positive_rate = ab_false_positive_rate
        config.pcr_false_negative_rate = pcr_false_negative_rate
        config.pcr_batch_size = pcr_batch_size
        config.init_test_capacity = init_test_capacity
        config.init_ab_test_capacity = init_ab_test_capacity
        config.ab_tests_sent_per_day = ab_tests_sent_per_day
        config.pcr_test_result_positive_probability = pcr_test_result_positive_probability
        config.initial_antibody_immunity_in_population = initial_antibody_immunity_in_population
        config.infected_and_symptomatic_in_population = infected_and_symptomatic_in_population
        config.symptom_covid_vs_flu = symptom_covid_vs_flu
        config.awaiting_on_pcr_test_result = awaiting_on_pcr_test_result
        config.add_ab = add_ab

    def run(self, learning_phase, n_days, n_population, intervention_scores=None):
        total_infections = 0
        total_recovered = 0
        total_active = 0
        new_infections = list()

        pcr_test_tracker = []  # results every day

        # INITIALIZE - Day -1
        population = [Person(i) for i in range(n_population)]

        # initialize exposure
        num_new_infections, num_immune_before_day_0 = population_init_infections(population,
                                                                                 self.incidence_rate_annual,
                                                                                 self.daily_case_population_fraction)

        # no transmissions initially
        num_new_transmission_infections = 0
        
        # current tests available in the system - add according to schedule
        # rapid_pcr_tests_available = 0
        # antibody_tests_available = 0
        
        if self.log_results:
            print('~' * 10 + 'SIMULATION CONFIGURATION' + '~' * 10)
            print(f"Population = {n_population}, "
                  f"PCR Cap Per Day: {config.init_test_capacity}, "
                  f"AB Cap Per Day: {config.init_ab_test_capacity}, "
                  f"AB FPR: {1 - config.ab_false_positive_rate}, "
                  f"PCR FPN: {1 - config.pcr_false_negative_rate}")
            print('-' * 20)
            print(' ')

            print('Initial immunity =', config.initial_antibody_immunity_in_population)
            print('Attrition rate =', config.attrition_rate)

        report_out_list = list()

        for day in range(n_days):
            # need to add here
            num_total_new_cases = sum([num_new_infections, num_new_transmission_infections])

            pcr_tests_sent_per_day = 0

            # update antibody
            update_population_igg_progression(population, day)
            update_population_igm_progression(population, day)

            # update disease progression
            # num_new_infections = update_population_disease_progression(population, day)
            update_population_disease_progression(population, day)

            # update disease progression
            # update_population_infectiousness_progression(population, day)

            # update symptoms
            update_population_symptom_progression(population, day)

            # update possible infection based on symptoms observed
            update_population_possible_infection(population)

            # ADD schedule for tests
            # rapid_pcr_tests_available = get_pcr_schdeule(self.pcr_schedule, day) #assume daily influx of tests
            #             if day <=60:
            #                 rapid_pcr_tests_available = int(0.2*n_population/14) + 1
            #             else:
            #                 rapid_pcr_tests_available = int(n_population/14) + 1
            rapid_pcr_tests_available = self.testing_capacity
            if self.log_results:
                print("-->", rapid_pcr_tests_available)
            # rapid_pcr_tests_available = int(1 + n_population/14.0)
            pcr_testing_rate()

            # allocate batch testing and update the pcr batch allocations
            current_pcr_allocation, current_pcr_results = update_rapid_pcr_tests_available(rapid_pcr_tests_available)
            pcr_test_tracker.append([current_pcr_allocation, current_pcr_results])

            # Send PCR tests
            eligible_for_pcr_count, rapid_pcr_tests_available, pcr_tests_sent_per_day = \
                send_population_pcr_test(rapid_pcr_tests_available, pcr_tests_sent_per_day,
                                         population, day)

            # Run PCR simulation () - collect the PCR Test Results
            # n_pos_pcr_results = simulate_pcr_test(population, day)

            # update PCR test results for everyone
            daily_pos, daily_neg, pcr_after_isolation = update_population_batch_pcr_test(population, day)

            # update isolation at population level to account for isolating the batch
            out_of_quarantine, _, sent_to_quarantine = update_population_isolation(population, day)

            # update clear to work
            clear_to_work = update_population_clear_to_work(population)

            # add attrition rate
            num_left_at_end_of_the_day, number_left_with_pcr_pos, number_left_with_pcr_neg, \
            number_left_with_no_test, number_left_exposed_to_virus = update_population_based_on_attrition(population,
                                                                                                          day)

            # get the metrics at the end of the day before infecting others
            susceptible, infectious_at_work, recovered, daily_infections_at_work, days_spent_infected_at_work, \
            num_total_exposed, num_total_quarantine_administered = get_model_outcome_metrics(
                population, day)

            if config.add_ab:
                daily_ab_tests = 0
                for person in population:
                    if len(person.antibody_test_administered) > 0 and person.antibody_test_administered[-1][0] == day:
                        daily_ab_tests += 1

            else:
                daily_ab_tests = 0
            
            if self.log_results:
                print(" ")
                print("Day: %d, Week: %d, Incidence rate: %.4f"
                      % (int(day), day / 7, self.incidence_rate_annual[day]))
                print("Susceptible (at work): %d, Infectious (at work): %d, Clear (at work): %d" % (
                    susceptible, infectious_at_work, clear_to_work))
                print("Quarantine done: %d, Sent to isolation (at home) = %d" % (out_of_quarantine, sent_to_quarantine))
                print("PCR (daily): %d, PCR Pos(daily): %d, PCR Neg(daily): %d"
                      % (pcr_tests_sent_per_day, daily_pos, daily_neg))
                print("Number of assoc left: %d, PCR pos: %d, PCR neg: %d, No PCR: %d"
                      % (num_left_at_end_of_the_day, number_left_with_pcr_pos, number_left_with_pcr_neg,
                         number_left_with_no_test))
                print(f"Total Infections so far: {total_infections}")
                print('-' * 20)

            metrics_dict = {'Day': day,
                            'Population': n_population,
                            'Max Daily Capacity on PCR Tests': config.init_test_capacity,
                            'Clear to work': clear_to_work,
                            'Infected working in FC and not in quarantine': infectious_at_work,
                            'Daily PCR Tests': pcr_tests_sent_per_day,
                            'Daily Ab Tests': daily_ab_tests,
                            'Susceptible': susceptible,
                            'Daily PCR Positive': daily_pos,
                            'Daily PCR Negative': daily_neg,
                            'Daily PCR Capacity': get_pcr_schdeule(self.pcr_schedule, day),
                            'Daily New Infection': num_total_new_cases,
                            # 'Daily Total Exposed':num_total_exposed,
                            'Daily New Infection County': num_new_infections,
                            'Daily New Infection Transmission': num_new_transmission_infections,
                            'Attrition': num_left_at_end_of_the_day,
                            'Attrition PCR Pos': number_left_with_pcr_pos,
                            'Attrition PCR Neg': number_left_with_pcr_neg,
                            'Attrition No Test': number_left_with_no_test,
                            'Attrition Exposed': number_left_exposed_to_virus,
                            'Back From Quarantine': out_of_quarantine,
                            'Sent To Quarantine': sent_to_quarantine,
                            'Total Quarantine Administered': num_total_quarantine_administered,
                            'Initial immune': num_immune_before_day_0}
            new_infections.append(num_total_new_cases)
            total_infections += num_total_new_cases
            total_recovered += recovered
            init_active = int(self.active_case_population_fraction * len(population))
            
            wave1_spread_factor = self.wave1_weeks / 4
            active_population_contrib_window = max (5, int(np.round(5 * wave1_spread_factor)))
            total_active = max(0, int(np.round(init_active * (active_population_contrib_window - day) / active_population_contrib_window))) + np.sum(new_infections[-min(7,day):-min(2,day)])
            
            report_out_list.append(metrics_dict)
            
            intervention_score = 0
            if intervention_scores is not None:
                end_day = max(0, day - 1)
                start_day = max(0, end_day - 2)
                intervention_score = np.mean(intervention_scores[start_day:day]) if start_day != end_day else intervention_scores[start_day]
                #intervention_score = np.mean(intervention_scores)
            # NEEDS to be here because will impact the current day if above...
            # [!!IMP!!] add new infections, for the next day

            num_new_infections, num_new_transmission_infections = update_population_infections(learning_phase, 
                total_active, self.incidence_rate_annual, population, day, self.transmission_prob, self.transmission_strength, self.intervention_influence_pctg, intervention_score=intervention_score)


        return [], report_out_list, population
