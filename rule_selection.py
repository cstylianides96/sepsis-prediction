# Author: Andria Nicolaou (nicolaou.andria@ucy.ac.cy)

from rule_metrics import *

def rule_selection(auc1, fidelity1, auc2, fidelity2):

    # Calculate the average fidelity and average training AUC
    auc_list = []
    fidelity_list = []

    for i in range(len(auc1)):
      avg_auc = (auc1[i] + auc2[i])/2
      avg_fidelity = (fidelity1[i] + fidelity2[i])/2

      auc_list.append(avg_auc)
      fidelity_list.append(avg_fidelity)

    # Find the loop with the highest avg. fidelity and avg. AUC
    best_loop = 0
    best_auc = auc_list[0]
    best_fidelity = fidelity_list[0]

    for i in range(len(auc_list)):
      if auc_list[i] > best_auc and fidelity_list[i] > best_fidelity:
          best_auc = auc_list[i]
          best_fidelity = fidelity_list[i]
          best_loop = i
      elif auc_list[i] >= best_auc and fidelity_list[i] > best_fidelity:
          best_auc = auc_list[i]
          best_fidelity = fidelity_list[i]
          best_loop = i
      elif auc_list[i] > best_auc and fidelity_list[i] >= best_fidelity:
          best_auc = auc_list[i]
          best_fidelity = fidelity_list[i]
          best_loop = i

    return best_loop



def apply_rules(group, target_name, column_labels, selected_path, selected_loop, rules, replace):

    # Create empty Excel files
    with pd.ExcelWriter('xai-output/rule-extraction/rules-training' + group + '.xlsx', mode='w') as writer1:
        pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')
    with pd.ExcelWriter('xai-output/rule-extraction/rules-evaluation' + group + '.xlsx', mode='w') as writer2:
        pd.DataFrame().to_excel(writer2, sheet_name='Loop 1')

    for i in range(1, 41):

      # Read training data
      train = pd.read_csv(selected_path + 'train_' + str(i) + '_encoded.csv')

      # Rename column names of training data
      train.columns = column_labels

      # Read evaluation data
      test = pd.read_csv(selected_path + 'test_' + str(i) + '_encoded.csv')

      # Rename column names of evaluation data
      test.columns = column_labels

      # Change the positive class
      if replace == True:
          train[target_name] = train[target_name].replace({1: 0, 0: 1})
          test[target_name] = test[target_name].replace({1: 0, 0: 1})

      # Create dataframes including the data applied by the selected rules
      df1 = rule_dataframe(group, train, rules[selected_loop])
      df2 = rule_dataframe(group, test, rules[selected_loop])


      # Save the dataframes to Excel file
      with pd.ExcelWriter('xai-output/rule-extraction/rules-training' + group + '.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
            df1.to_excel(writer1, sheet_name='Loop ' + str(i))
      with pd.ExcelWriter('xai-output/rule-extraction/rules-evaluation' + group + '.xlsx', mode='a', if_sheet_exists='overlay') as writer2:
            df2.to_excel(writer2, sheet_name='Loop ' + str(i))