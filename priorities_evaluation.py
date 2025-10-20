# Author: Andria Nicolaou (nicolaou.andria@ucy.ac.cy)

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import ast


def evaluation(group1, group2, selected_loop, selected_path):

  with pd.ExcelWriter('xai-output/priorities/selected-priority-rules.xlsx', mode='w') as writer2:
    pd.DataFrame().to_excel(writer2, sheet_name='Loop 1')

  with pd.ExcelWriter('xai-output/priorities/resolved-indices-evaluation.xlsx', mode='w',) as writer3:
    pd.DataFrame().to_excel(writer3, sheet_name='Loop 1')

  with pd.ExcelWriter('xai-output/priorities/possible-new-dilemmas.xlsx', mode='w') as writer4:
    pd.DataFrame().to_excel(writer4, sheet_name='Loop 1')

  with open('xai-output/priorities/priorities-info.txt', 'a') as f:
    f.write("\n\n\n--------------------------------------")
    f.write(f'\nSummary of the selected priority rules\n')
    f.write("--------------------------------------")


  # Read the selected priority rules (plus 1 as indexing starts from 0)
  priority_rules_selected = pd.read_excel('xai-output/priorities/priority-rules.xlsx',sheet_name='Loop ' + str(selected_loop + 1))

  for i in range(1, 41):

    # Step 1: Find the selected priority rules in each loop

    # Read priority rules
    priority_rules = pd.read_excel('xai-output/priorities/priority-rules.xlsx', sheet_name='Loop ' + str(i))

    # Get the intersection of selected priority rules
    merged_df = pd.merge(priority_rules_selected, priority_rules, on=['Priority Rules', 'Other Rules'],how='inner')

    # Select the column indices of the priority rules (not the selected) to note the resolved dilemma cases
    selected_df = merged_df[['Priority Rules', 'Other Rules', 'Indices Training_y', 'Indices Evaluation_y']]

    # Rename the columns to remove the '_y' suffix
    selected_df = selected_df.rename(columns={'Indices Training_y': 'Indices Training',
                                        'Indices Evaluation_y': 'Indices Evaluation'})

    # Create a new column to the dataframe saving the rule prediction
    def predict_rule(priority_rule):
      if isinstance(priority_rule, str):  # Ensure the input is a string
        if priority_rule.startswith('R'):
          return group1
        elif priority_rule.startswith('N'):
          return group2
      return ""  # Return empty string for non-matching or non-string values

    # Apply the previous function that creates the column 'Rule Prediction'
    selected_df['Rule Prediction'] = selected_df['Priority Rules'].apply(predict_rule)

    # Convert string representation of 'Indices Evaluation' lists to actual lists
    selected_df['Indices Evaluation'] = selected_df['Indices Evaluation'].apply(ast.literal_eval)

    # Save the DataFrame to Excel file
    with pd.ExcelWriter('xai-output/priorities/selected-priority-rules.xlsx', mode='a', if_sheet_exists='overlay') as writer2:
      selected_df.to_excel(writer2, sheet_name='Loop ' + str(i), index=False)


    # Step 2: Get the resolved evaluation indices
    all_indices = []
    for indices in selected_df['Indices Evaluation']:
      all_indices.extend(indices)  # Extend the list with all the indices

    resolved_indices = list(set(all_indices))  # Create a set that contains the unique indices
    resolved_indices.sort()

    # Create a Dataframe including the resolved evaluation indices
    resolved_indices_df = pd.DataFrame(resolved_indices, columns=['Index'])

    # Save the DataFrame to Excel file
    with pd.ExcelWriter('xai-output/priorities/resolved-indices-evaluation.xlsx', mode='a', if_sheet_exists='overlay') as writer3:
      resolved_indices_df.to_excel(writer3, sheet_name='Loop ' + str(i), index=False)

    with open('xai-output/priorities/priorities-info.txt', 'a') as f:
      f.write(f"\n\nLoop: {str(i)}")
      f.write(f"\n{len(resolved_indices_df)} records of the evaluation set can be resolved by applying {len(selected_df)} selected priority rules of the training set")


    # Step 3: Find the possible new dilemmas
    conflicting_rules_list = []
    for index, row in selected_df.iterrows():
      priority_rule = row['Priority Rules']
      other_rule = row['Other Rules']

      if(((selected_df['Priority Rules'] == other_rule) & (selected_df['Other Rules'] == priority_rule)).any()):
        conflicting_rules_list.append(row)

    # Create a Dataframe including the conflicting rules
    conflicting_rules_df = pd.DataFrame(conflicting_rules_list)

    if conflicting_rules_df.empty:
      possible_dilemmas_df = pd.DataFrame()

    else:
      possible_dilemmas = []
      for indices in conflicting_rules_df['Indices Evaluation']:
        possible_dilemmas.extend(indices)  # Extend the list with all the indices

      possible_dilemmas = list(set(possible_dilemmas))  # Create a set that contains the unique dilemma indices
      possible_dilemmas.sort()

      # Create a Dataframe including the indices of possible dilemmas
      possible_dilemmas_df = pd.DataFrame(possible_dilemmas, columns=['Index'])

    # Save the DataFrame to Excel file
    with pd.ExcelWriter('xai-output/priorities/possible-new-dilemmas.xlsx', mode='a', if_sheet_exists='overlay') as writer4:
      possible_dilemmas_df.to_excel(writer4, sheet_name='Loop ' + str(i), index=False)

    with open('xai-output/priorities/priorities-info.txt', 'a') as f:
      f.write(f"\nPossible New Dilemmas: {len(possible_dilemmas_df)}")



def prediction(column_labels, selected_path):

  accuracy = []
  sensitivity = []
  specificity = []
  precision = []
  negative_predictive_value = []

  # Create empty Excel files
  with pd.ExcelWriter('xai-output/priorities/resolved-indices-prediction.xlsx', mode='w',) as writer:
    pd.DataFrame().to_excel(writer, sheet_name='Loop 1')

  with open('xai-output/priorities/prediction-info.txt', 'w') as f:
    f.write("Evaluate priorities using predictions\n")
    f.write("-------------------------------------")

  for i in range(1, 41):

    # Read testing data to get the true labels
    test = pd.read_csv(selected_path + 'test_' + str(i) + '_encoded.csv')
    # Rename column names of evaluation data
    test.columns = column_labels
    # Get the true labels
    labels = test['label']

    # Read prediction data to get the predicted labels
    df = pd.read_csv(selected_path + 'GBM_obs24_pred12_feat70_balanced_pred' + str(i) + '_3_encoded.csv')

    # Get the predictions
    predictions = df['pred']

    # Add the actual label column to the prediction data
    df['label'] = test['label']

    # Get the dilemmas of evaluation set
    test_dilemma = pd.read_excel('xai-output/rule-extraction/dilemmas-evaluation.xlsx', sheet_name='Loop ' + str(i))
    dilemmas_evaluation = test_dilemma['Index']

    # Find dilemmas in the wrong predictions
    wrong_predictions = df.loc[df['pred'] != df['label'], 'index']
    dilemmas = pd.DataFrame(pd.merge(wrong_predictions, dilemmas_evaluation, left_on='index', right_on='Index')['Index']).drop_duplicates(keep='first',ignore_index=True)

    with open('xai-output/priorities/prediction-info.txt', 'a') as f:
      f.write(f'\n\nLoop {str(i)}:')
      f.write(f'\n{len(dilemmas)} records presented dilemmas on the classification predictions:\n')

    # Check for similar indices between prediction dilemmas and resolved indices
    possible_resolved_indices = pd.read_excel('xai-output/priorities/resolved-indices-evaluation.xlsx', sheet_name='Loop ' + str(i))
    similar_indices = dilemmas[dilemmas['Index'].isin(possible_resolved_indices['Index'])]

    # Check for new dilemmas between prediction dilemmas and possible new dilemmas
    possible_new_dilemmas = pd.read_excel('xai-output/priorities/possible-new-dilemmas.xlsx', sheet_name='Loop ' + str(i))
    if possible_new_dilemmas.empty:
      new_dilemmas = pd.DataFrame()
      resolved_indices = similar_indices

    else:
      new_dilemmas = dilemmas[dilemmas['Index'].isin(possible_new_dilemmas['Index'])]

      # Drop the new dilemmas indices
      resolved_indices = similar_indices[~similar_indices['Index'].isin(new_dilemmas['Index'])]

    false_positive = 0
    false_negative = 0

    with (open('xai-output/priorities/prediction-info.txt', 'a') as f):
      if len(similar_indices) > 0:
          f.write(f"{len(similar_indices)} of them have the possibility to be resolved,")
          f.write(f" but {len(new_dilemmas)} of them are new dilemmas")
          f.write(f"\nThus {len(resolved_indices)} can be resolved")

          # Add the prediction column to the testing data
          test['prediction'] = df['pred']

          # Get the rows that correspond to the similar indices
          resolved_indices_df = test[test['Index'].isin(resolved_indices['Index'])]

          # Reset the index
          resolved_indices_df = resolved_indices_df.reset_index(drop=True)

          # Check each row if the case is false positive or false negative
          for index, row in resolved_indices_df.iterrows():
            if row['label'] == 0 and row['prediction'] == 1:
              false_positive += 1
            elif row['label'] == 1 and row['prediction'] == 0:
              false_negative += 1
          f.write(f'; {false_positive} false positives and {false_negative} false negatives')

          # Save the DataFrame to an Excel file
          with pd.ExcelWriter('xai-output/priorities/resolved-indices-prediction.xlsx', mode='a', if_sheet_exists='overlay') as writer:
            resolved_indices_df.to_excel(writer, sheet_name='Loop ' + str(i))

      else:
        f.write("none of them can be resolved")

    # Calculate confusion metrics
    acc, sen, spe, prec, npv = calculate_confusion_matrix(labels, predictions, false_positive, false_negative)

    # Calculate average confusion metrics
    accuracy.append(acc)
    sensitivity.append(sen)
    specificity.append(spe)
    precision.append(prec)
    negative_predictive_value.append(npv)

  with open('xai-output/priorities/prediction-info.txt', 'a') as f:
    f.write('\n\n------------Average------------')
    f.write('\nAccuracy= {:.5f}\nSensitivity= {:.5f}\nSpecificity= {:.5f}\nPrecision= {:.5f}\nNegative Predictive Value= {:.5f}\n\n'
    .format(np.nanmean(accuracy), np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(precision), np.nanmean(negative_predictive_value)))
    f.write('------Standard deviation-------')
    f.write('\nAccuracy= {:.5f}\nSensitivity= {:.5f}\nSpecificity= {:.5f}\nPrecision= {:.5f}\nNegative Predictive Value= {:.5f}\n\n'
    .format(np.nanstd(accuracy), np.nanstd(sensitivity), np.nanstd(specificity), np.nanstd(precision), np.nanstd(negative_predictive_value)))



def calculate_confusion_matrix(labels, predictions, false_positive, false_negative):

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0

    with open('xai-output/priorities/prediction-info.txt', 'a') as f:
      f.write(f'\n\nConfusion matrix (before applying priorities): ' +
                '\nTrue Positives= ' + str(tp) +
                '\nFalse Positives= ' + str(fp) +
                '\nTrue Negatives= ' + str(tn) +
                '\nFalse Negatives= ' + str(fn) +
                '\n\nAccuracy= ' + str(round(accuracy, 5)) +
                '\nSensitivity= ' + str(round(sensitivity, 5)) +
                '\nSpecificity= ' + str(round(specificity, 5)) +
                '\nPrecision= ' + str(round(precision, 5)) +
                '\nNegative Predictive Value= ' + str(round(npv, 5)))

      # Calculate metrics after applying priorities
      tp2 = tp + false_negative
      fp2 = fp - false_positive
      tn2 = tn + false_positive
      fn2 = fn - false_negative
      acc2 = (tp2 + tn2) / (tp2 + tn2 + fp2 + fn2)
      sen2 = tp2 / (tp2 + fn2) if (tp2 + fn2) != 0 else 0
      spe2 = tn2 / (tn2 + fp2) if (tn2 + fp2) != 0 else 0
      prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) != 0 else 0
      npv2 = tn2 / (tn2 + fn2) if (tn2 + fn2) != 0 else 0
      f.write(f'\n\nConfusion matrix (after applying priorities): ' +
                '\nTrue Positives= ' + str(tp2) +
                '\nFalse Positives= ' + str(fp2) +
                '\nTrue Negatives= ' + str(tn2) +
                '\nFalse Negatives= ' + str(fn2) +
                '\n\nAccuracy= ' + str(round(acc2, 5)) +
                '\nSensitivity= ' + str(round(sen2, 5)) +
                '\nSpecificity= ' + str(round(spe2, 5)) +
                '\nPrecision= ' + str(round(prec2, 5)) +
                '\nNegative Predictive Value= ' + str(round(npv2, 5)))

      return acc2, sen2, spe2, prec2, npv2