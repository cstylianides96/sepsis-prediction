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
    prediction_path = 'Please check the priorities_evaluation.py line 147'
    
    # Define the target name
    target_name = 'label'
    
    # Define the target groups
    group1 = 'Sepsis'
    group2 = 'NoSepsis'
    
    # Define the column names
    column_labels = ['Arterial Blood Pressure diastolic_diff_0', 'hosp_to_icu', 'Glucose (serum)_range', 
                     'GCS - Verbal Response_max', 'Arterial Blood Pressure systolic_range', 'Arterial Blood Pressure mean_diff_0',
                     'WBC_range', 'Arterial Blood Pressure systolic_diff_0', 'BUN_diff_0', 'BUN_diff_1',
                     'Arterial Blood Pressure systolic_21', 'Glucose (serum)_diff_1', 'GCS - Verbal Response_23',
                     'Hematocrit (serum)_diff_2', 'Arterial Blood Pressure mean_diff_19', 'BUN_diff_2', 'GCS - Motor Response_0',
                     'Arterial Blood Pressure diastolic_diff_2', 'LDH_diff_0,Sodium (serum)_mean', 'Arterial Blood Pressure mean_14',
                     'Glucose (serum)_diff_0', 'Fibrinogen_14', 'Arterial Blood Pressure mean_diff_10', 'LDH_diff_5', 
                     'Magnesium_3', 'Differential-Lymphs_diff_1', 'Sodium (serum)_diff_1', 'Glucose (serum)_diff_2', 
                     'Creatinine (serum)_mean', 'Alkaline Phosphate_diff_18', 'Differential-Lymphs_diff_0', 
                     'Respiratory Rate_range', 'LDH_min', 'Differential-Lymphs_diff_22', 'Arterial Blood Pressure systolic_diff_20',
                     'Heart Rate_diff_0', 'Hematocrit (serum)_4', 'C Reactive Protein (CRP)_diff_12', 'Phosphorous_range',
                     'label', 'index']
    
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
