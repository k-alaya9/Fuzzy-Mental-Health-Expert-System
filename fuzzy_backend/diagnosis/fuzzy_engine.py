import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# inputs
academic_workload = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'academic_workload')
social_relationships = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'social_relationships')
average_sleep = ctrl.Antecedent(np.arange(0, 13, 1), 'average_sleep')
financial_concerns = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'financial_concerns')
academic_pressure = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'academic_pressure')
age = ctrl.Antecedent(np.arange(17, 31, 1), 'age')

academic_workload['light'] = fuzz.trimf(academic_workload.universe, [0, 0, 2.5])
academic_workload['moderate'] = fuzz.trimf(academic_workload.universe, [1.5, 2.5, 4])
academic_workload['heavy'] = fuzz.trimf(academic_workload.universe, [3, 5, 5])

social_relationships['poor'] = fuzz.gaussmf(social_relationships.universe, 1, 0.8)
social_relationships['average'] = fuzz.gaussmf(social_relationships.universe, 2.5, 0.8)
social_relationships['strong'] = fuzz.gaussmf(social_relationships.universe, 4, 0.8)


average_sleep['low'] = fuzz.trapmf(average_sleep.universe, [0, 0, 3, 6])
average_sleep['adequate'] = fuzz.trapmf(average_sleep.universe, [5, 6, 8, 10])
average_sleep['oversleep'] = fuzz.trapmf(average_sleep.universe, [9, 10, 12, 12])


financial_concerns['low'] = fuzz.trimf(financial_concerns.universe, [0, 0, 2])
financial_concerns['medium'] = fuzz.trimf(financial_concerns.universe, [1.5, 2.5, 4])
financial_concerns['high'] = fuzz.trimf(financial_concerns.universe, [3, 5, 5])


academic_pressure['low'] = fuzz.trapmf(academic_pressure.universe, [0, 0, 1.5, 2])
academic_pressure['medium'] = fuzz.trapmf(academic_pressure.universe, [1.5, 2.5, 3.5, 4])
academic_pressure['high'] = fuzz.trapmf(academic_pressure.universe, [3.5, 4.2, 5, 5])


age['young'] = fuzz.gaussmf(age.universe, 18, 1.5)
age['typical'] = fuzz.gaussmf(age.universe, 22, 2)
age['older'] = fuzz.gaussmf(age.universe, 27, 2)

# outputs

depression = ctrl.Consequent(np.arange(0, 5.1, 0.1), 'depression')
anxiety = ctrl.Consequent(np.arange(0, 5.1, 0.1), 'anxiety')
isolation = ctrl.Consequent(np.arange(0, 5.1, 0.1), 'isolation')
future_insecurity = ctrl.Consequent(np.arange(0, 5.1, 0.1), 'future_insecurity')

# --- Depression ---
depression['none'] = fuzz.trapmf(depression.universe, [0, 0, 0.5, 1.2])
depression['mild'] = fuzz.trimf(depression.universe, [0.8, 1.8, 2.5])
depression['moderate'] = fuzz.trimf(depression.universe, [2.2, 3.0, 3.8])
depression['severe'] = fuzz.trapmf(depression.universe, [3.5, 4.2, 5, 5])

# --- Anxiety ---
anxiety['low'] = fuzz.gaussmf(anxiety.universe, 1, 0.7)
anxiety['medium'] = fuzz.gaussmf(anxiety.universe, 2.5, 0.7)
anxiety['high'] = fuzz.gaussmf(anxiety.universe, 4, 0.7)

# --- Isolation ---
isolation['connected'] = fuzz.trapmf(isolation.universe, [0, 0, 0.8, 1.5])
isolation['neutral'] = fuzz.trimf(isolation.universe, [1.2, 2.5, 3.8])
isolation['isolated'] = fuzz.trapmf(isolation.universe, [3.5, 4.2, 5, 5])

# --- Future Insecurity ---
future_insecurity['secure'] = fuzz.trapmf(future_insecurity.universe, [0, 0, 1, 2])
future_insecurity['unsure'] = fuzz.trimf(future_insecurity.universe, [1.5, 2.8, 4])
future_insecurity['very_insecure'] = fuzz.trapmf(future_insecurity.universe, [3.5, 4.2, 5, 5])
# Rules

rule1 = ctrl.Rule(academic_workload['heavy'] & social_relationships['poor'] & average_sleep['low'],
                  consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

rule2 = ctrl.Rule(academic_workload['moderate'] & social_relationships['average'] & average_sleep['adequate'],
                  consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

rule3 = ctrl.Rule(academic_workload['light'] & social_relationships['strong'] & average_sleep['adequate'],
                  consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

rule4 = ctrl.Rule(financial_concerns['high'] & academic_pressure['high'],
                  consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

rule5 = ctrl.Rule(age['older'] & social_relationships['poor'],
                  consequent=[depression['moderate'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

rule6 = ctrl.Rule(academic_workload['heavy'] & financial_concerns['medium'] & average_sleep['low'],
                  consequent=[depression['moderate'], anxiety['high'], isolation['isolated'], future_insecurity['unsure']])

rule7 = ctrl.Rule(social_relationships['strong'] & average_sleep['oversleep'],
                  consequent=[depression['mild'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

rule8 = ctrl.Rule(academic_pressure['medium'] & financial_concerns['low'] & age['young'],
                  consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['secure']])

rule9 = ctrl.Rule(academic_workload['moderate'] & social_relationships['poor'] & average_sleep['adequate'],
                  consequent=[depression['moderate'], anxiety['high'], isolation['neutral'], future_insecurity['unsure']])

rule10 = ctrl.Rule(financial_concerns['medium'] & academic_pressure['medium'] & age['typical'],
                   consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

rule11 = ctrl.Rule(academic_workload['light'] & social_relationships['average'] & average_sleep['low'],
                   consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

rule12 = ctrl.Rule(financial_concerns['high'] & average_sleep['low'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

rule13 = ctrl.Rule(social_relationships['poor'] & average_sleep['low'] & academic_pressure['high'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

rule14 = ctrl.Rule(academic_workload['moderate'] & social_relationships['strong'] & average_sleep['adequate'],
                   consequent=[depression['mild'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

rule15 = ctrl.Rule(age['young'] & financial_concerns['low'] & social_relationships['strong'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])
rule16 = ctrl.Rule(academic_pressure['low'] & financial_concerns['low'] & social_relationships['strong'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

rule17 = ctrl.Rule(academic_workload['heavy'] & social_relationships['average'] & average_sleep['low'],
                   consequent=[depression['moderate'], anxiety['high'], isolation['neutral'], future_insecurity['unsure']])

rule18 = ctrl.Rule(age['typical'] & financial_concerns['medium'] & academic_pressure['high'],
                   consequent=[depression['moderate'], anxiety['high'], isolation['neutral'], future_insecurity['unsure']])

rule19 = ctrl.Rule(social_relationships['poor'] & average_sleep['adequate'] & academic_pressure['medium'],
                   consequent=[depression['moderate'], anxiety['medium'], isolation['isolated'], future_insecurity['unsure']])

rule20 = ctrl.Rule(academic_workload['light'] & financial_concerns['low'] & average_sleep['oversleep'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])
# قاعدة 21
rule21 = ctrl.Rule(academic_workload['heavy'] & financial_concerns['high'] & average_sleep['low'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

# قاعدة 22
rule22 = ctrl.Rule(social_relationships['poor'] & academic_pressure['high'] & age['older'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

# قاعدة 23
rule23 = ctrl.Rule(average_sleep['adequate'] & social_relationships['average'] & academic_workload['moderate'],
                   consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 24
rule24 = ctrl.Rule(financial_concerns['medium'] & academic_pressure['medium'] & age['typical'],
                   consequent=[depression['moderate'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 25
rule25 = ctrl.Rule(age['young'] & social_relationships['strong'] & average_sleep['oversleep'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

# قاعدة 26
rule26 = ctrl.Rule(academic_workload['light'] & financial_concerns['low'] & academic_pressure['low'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

# قاعدة 27
rule27 = ctrl.Rule(social_relationships['poor'] & average_sleep['low'] & academic_workload['moderate'],
                   consequent=[depression['moderate'], anxiety['high'], isolation['isolated'], future_insecurity['unsure']])

# قاعدة 28
rule28 = ctrl.Rule(financial_concerns['high'] & academic_pressure['medium'] & average_sleep['low'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

# قاعدة 29
rule29 = ctrl.Rule(academic_pressure['low'] & social_relationships['strong'] & average_sleep['adequate'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

# قاعدة 30
rule30 = ctrl.Rule(age['older'] & financial_concerns['medium'] & social_relationships['average'],
                   consequent=[depression['moderate'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 31
rule31 = ctrl.Rule(academic_workload['heavy'] & social_relationships['poor'] & academic_pressure['high'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

# قاعدة 32
rule32 = ctrl.Rule(financial_concerns['low'] & average_sleep['oversleep'] & age['young'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

# قاعدة 33
rule33 = ctrl.Rule(academic_pressure['medium'] & social_relationships['average'] & average_sleep['adequate'],
                   consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 34
rule34 = ctrl.Rule(social_relationships['strong'] & financial_concerns['low'] & average_sleep['adequate'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

# قاعدة 35
rule35 = ctrl.Rule(age['typical'] & academic_workload['moderate'] & academic_pressure['medium'],
                   consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 36
rule36 = ctrl.Rule(average_sleep['low'] & social_relationships['poor'] & financial_concerns['high'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

# قاعدة 37
rule37 = ctrl.Rule(academic_workload['light'] & social_relationships['strong'] & financial_concerns['low'],
                   consequent=[depression['none'], anxiety['low'], isolation['connected'], future_insecurity['secure']])

# قاعدة 38
rule38 = ctrl.Rule(academic_pressure['high'] & age['older'] & financial_concerns['medium'],
                   consequent=[depression['moderate'], anxiety['high'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 39
rule39 = ctrl.Rule(social_relationships['average'] & academic_workload['moderate'] & average_sleep['adequate'],
                   consequent=[depression['mild'], anxiety['medium'], isolation['neutral'], future_insecurity['unsure']])

# قاعدة 40
rule40 = ctrl.Rule(academic_workload['heavy'] & average_sleep['low'] & financial_concerns['high'],
                   consequent=[depression['severe'], anxiety['high'], isolation['isolated'], future_insecurity['very_insecure']])

rules_list = []
for i in range(1, 41):
    rules_list.append(eval(f"rule{i}"))
mental_health_ctrl = ctrl.ControlSystem(rules_list)
mental_health_sim = ctrl.ControlSystemSimulation(mental_health_ctrl)
def run_fuzzy_diagnosis(data):
    mental_health_sim.input['academic_workload'] = data['academic_workload']
    mental_health_sim.input['social_relationships'] = data['social_relationships']
    mental_health_sim.input['average_sleep'] = data['average_sleep']
    mental_health_sim.input['financial_concerns'] = data['financial_concerns']
    mental_health_sim.input['academic_pressure'] = data['academic_pressure']
    mental_health_sim.input['age'] = data['age']

    mental_health_sim.compute()

    return {
        'depression': (mental_health_sim.output['depression']*100)/5,
        'anxiety': (mental_health_sim.output['anxiety']*100)/5,
        'isolation':( mental_health_sim.output['isolation']*100)/5,
        'future_insecurity': (mental_health_sim.output['future_insecurity']*100)/5,
    }