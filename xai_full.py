# Author: Andria Nicolaou (nicolaou.andria@ucy.ac.cy)

from datetime import datetime
from rule_extraction import *
from rule_selection import *
from errors_dilemmas import *
from priorities import *
from priorities_evaluation import *

def run_xai():
    # Save the starting time of the script
    now = datetime.now()
    start = now.strftime("Start time: %Y-%m-%d %H:%M:%S %Z%z")
    
    # Create root folder
    root_path = './xai-output'
    # Check whether root directory already exists
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    
    with open('xai-output/execution-time.txt', 'w') as f:
      f.write(start)
    
    # Path of selected data
    selected_path = './data_processed/'
    
    # Path of predictions
    prediction_path = 'Please check the priorities_evaluation.py line 145'
    
    # Define the target name
    target_name = 'label'
    
    # Define the target groups
    group1 = 'Sepsis'
    group2 = 'NoSepsis'
    
    # Define the column names
    column_labels = ['replete_fiber_10' , 'WBC_3_diff', 'arterial_blood_pressure_mean_8_diff', 'glucerna15_0', 'furosemide_lasix_0',
    'tandem_heart_flow_8', 'fibersourceHN_0', 'arterial_blood_pressure_mean_10_diff', 'RRApacheIIValue_5', 'sodium_wholeblood_19',
    'arterial_blood_pressure_mean_18', 'pinsp_hamilton_range', 'PH_arterial_10', 'potassium_whole_blood_max', 'nepro_0', 'SV_arterial_13',
    'osmolite15_0', 'calcium_non_ionized_range', 'temperature_celsius_range', 'arterial_blood_pressure_systolic_11_diff', 'O2_flow_23',
    'shock_index_18_diff', 'WBC_1_diff', 'arterial_blood_pressure_systolic_22_diff', 'promote_0', 'urinary_tract_infection_not_specified',
    'arterial_blood_pressure_systolic_14_diff', 'SV_arterial_min', 'dextrose5_0', 'heart_rate_14_diff', 'shock_index_9_diff', 'shock_index_14_diff',
    'spontRR_range', 'arterial_blood_pressure_systolic_23', 'milrinone_0', 'temperature_celsius_0', 'jevity15_0', 'promoteFiber_0', 'heparinSodium_0',
    'furosemide_Lasix_250_50_0', 'negativeInspForce_mean', 'heartRate_16_diff', 'repleteFiber_0', 'PH_arterial_22', 'GCS_eyeOpening1_diff',
    'arterial_blood_pressure_systolic_19_diff', 'arterial_blood_pressure_mean_11', 'propofol_0', 'arterial_blood_pressure_systolic_16', 'gcs_sum_23',
    'O2_flow_mean', 'dexmedetomidine_precedex_0', 'TFCd_NICOM_max', 'jevity12_0', 'pf_ratio_a_range', 'arterial_blood_pressure_systolic_min',
    'tandem_heart_flow_10', 'nacl09_1', 'calcium_gluconateCRRT_0', 'PH_arterial_0', 'fspn_high_2', 'BUN_3_diff', 'WBC_2_diff', 'BUN_4_diff',
    'glucerna12_0', 'hosp_to_icu', 'phosphorous_range', 'BUN_2_diff', 'nacl09_0', 'BUN_1_diff', 'label', 'Index']
    
    
    # --------- Rule extraction ---------
    print('Rule extraction..')
    
    # Extract rules from the first target group
    auc1, fidelity1, rules1 = rule_extraction(group1, target_name, column_labels, selected_path, replace=False)
    
    # Extract rules from the second target group
    auc2, fidelity2, rules2 = rule_extraction(group2, target_name, column_labels, selected_path, replace=True)
    
    
    # --------- Rule selection ---------
    print('Rule selection..')
    
    # Select the rule list of the model with high fidelity and high auc
    selected_loop = rule_selection(auc1, fidelity1, auc2, fidelity2)
    
    # Apply the selected rules regarding the first target group on both training and evaluation sets
    apply_rules(group1, target_name, column_labels, selected_path, selected_loop, rules1, replace=False)
    
    # Apply the selected rules regarding the second target group on both training and evaluation sets
    apply_rules(group2, target_name, column_labels, selected_path, selected_loop, rules2, replace=True)
    
    
    #--------- Calculate errors and dilemmas ---------
    print('Calculate errors and dilemmas..')
    
    errors(group1, group2, target_name)
    dilemmas(group1, group2)
    
    # Get the unique cases of errors and dilemmas on both training and evaluation sets
    errors_dilemmas(selected_loop)
    
    
    #--------- Find priorities ---------
    print('Find priorities..')
    
    # Find the priority rules based on dilemma cases from training set
    priorities(group2, 'training')
    
    # Find the priority rules based on dilemma cases from evaluation set
    priorities(group2, 'evaluation')
    
    # Get the intersection of priority rules from both training and evaluation sets
    # Aim: Return a dataframe including the priority rules that will be used in evaluation
    intersection_priorities()
    
    
    #--------- Evaluate priorities ---------
    print('Evaluate priorities..')
    
    # Get the selected priority rules based on the selected loop and return the resolved indices
    evaluation(group1, group2, selected_loop, selected_path)
    
    # Evaluate the selected priority rules on predictions
    prediction(column_labels, selected_path)
    
    
    # Save the ending time of the script
    now = datetime.now()
    end = now.strftime("\nEnd time: %Y-%m-%d %H:%M:%S")
    
    with open('xai-output/execution-time.txt', 'a') as f:
      f.write(end)
