<<<<<<< HEAD
# Author: Andria Nicolaou (nicolaou.andria@ucy.ac.cy)

import pandas as pd
import os

def errors(group1, group2, target_name):

  # Create empty Excel files
  with pd.ExcelWriter('xai-output/rule-extraction/errors-training.xlsx', mode='w') as writer1:
      pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')
  with pd.ExcelWriter('xai-output/rule-extraction/errors-evaluation.xlsx', mode='w') as writer2:
      pd.DataFrame().to_excel(writer2, sheet_name='Loop 1')

  for i in range(1, 41):

    # Read rule dataframes of both target groups extracted during training
    df1 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df2 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Get the errors on training
    train_error = df.loc[df[target_name] == 0].drop(columns=[df.columns[0]]).reset_index(drop=True)

    # Read rule dataframes of both target groups extracted during evaluation
    df3 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df4 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df3, df4], ignore_index=True)
    # Get the errors on evaluation
    test_error = df.loc[df[target_name] == 0].drop(columns=[df.columns[0]]).reset_index(drop=True)

    # Save the dataframes to Excel file
    with pd.ExcelWriter('xai-output/rule-extraction/errors-training.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
      train_error.to_excel(writer1, sheet_name='Loop ' + str(i))
    with pd.ExcelWriter('xai-output/rule-extraction/errors-evaluation.xlsx', mode='a', if_sheet_exists='overlay') as writer2:
      test_error.to_excel(writer2, sheet_name='Loop ' + str(i))


def dilemmas(group1, group2):

  # Create empty Excel files
  with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-training.xlsx', mode='w') as writer1:
      pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')
  with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-evaluation.xlsx', mode='w') as writer2:
      pd.DataFrame().to_excel(writer2, sheet_name='Loop 1')

  for i in range(1, 41):

    # Read rule dataframes of both target groups extracted during training
    df1 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df2 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop(columns=[df.columns[0]])

    # Group by the index
    groups = df.groupby(df['Index'])
    # Filter groups to find those where the 'Prediction' column has more than one unique value
    train_dilemma = groups.filter(lambda x: len(x['Prediction'].unique()) > 1).reset_index(drop=True)

    # Read rule dataframes of both target groups extracted during evaluation
    df3 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df4 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df3, df4], ignore_index=True)
    df = df.drop(columns=[df.columns[0]])

    # Group by the index
    groups = df.groupby(df['Index'])
    # Filter groups to find those where the 'Prediction' column has more than one unique value
    test_dilemma = groups.filter(lambda x: len(x['Prediction'].unique()) > 1).reset_index(drop=True)

    # Save the dataframes to Excel file
    with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-training.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
      train_dilemma.to_excel(writer1, sheet_name='Loop ' + str(i))
    with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-evaluation.xlsx', mode='a', if_sheet_exists='overlay') as writer2:
      test_dilemma.to_excel(writer2, sheet_name='Loop ' + str(i))


def errors_dilemmas(selected_loop):

  # Note the selected loop (plus 1 as indexing starts from 0) in the text file
  with open('xai-output/rule-extraction/dilemmas-info.txt', 'w') as f:
    f.write("\n\nSummary of errors and dilemmas (Selected loop: " + str(selected_loop) + ")\n")
    f.write("------------------------------")

  for i in range(1, 41):

    # Read dataframes including errors and dilemmas extracted during training and testing
    train_error = pd.read_excel('xai-output/rule-extraction/errors-training.xlsx', sheet_name='Loop ' + str(i))
    test_error = pd.read_excel('xai-output/rule-extraction/errors-evaluation.xlsx', sheet_name='Loop ' + str(i))
    train_dilemma = pd.read_excel('xai-output/rule-extraction/dilemmas-training.xlsx', sheet_name='Loop ' + str(i))
    test_dilemma = pd.read_excel('xai-output/rule-extraction/dilemmas-evaluation.xlsx', sheet_name='Loop ' + str(i))

    # Get the unique cases of errors on training
    number_train_errors = train_error['Index'].nunique()
    unique_train_errors = train_error['Index'].unique()

    # Get the unique cases of errors on testing
    number_test_errors = test_error['Index'].nunique()
    unique_test_errors = test_error['Index'].unique()

    # Get the unique cases of dilemmas on training
    number_train_dilemmas = train_dilemma['Index'].nunique()
    unique_train_dilemmas = train_dilemma['Index'].unique()

    # Get the unique cases of dilemmas on testing
    number_test_dilemmas = test_dilemma['Index'].nunique()
    unique_test_dilemmas = test_dilemma['Index'].unique()

    # Find the common cases between errors and dilemmas
    train_intersection = len(set(unique_train_errors).intersection(set(unique_train_dilemmas)))
    test_intersection = len(set(unique_test_errors).intersection(set(unique_test_dilemmas)))

    # Get the unique cases of errors on both training and testing
    errors_training = number_train_errors - train_intersection
    errors_evaluation = number_test_errors - test_intersection

    with open('xai-output/rule-extraction/dilemmas-info.txt', 'a') as f:
      f.write("\n\nLoop: " + str(i) +
            "\nErrors (training): " + str(errors_training) +
            "\nDilemmas (training): " + str(number_train_dilemmas) +
            "\nErrors (evaluation): " + str(errors_evaluation) +
            "\nDilemmas (evaluation): " + str(number_test_dilemmas))

  # Create folder
  path = os.path.join('./xai-output', 'priorities')
  # Check whether directory already exists
  if not os.path.exists(path):
    os.mkdir(path)

  with open('xai-output/priorities/priorities-info.txt', 'w') as f:
    f.write("Summary of priority rules\n")
=======
# Author: Andria Nicolaou (nicolaou.andria@ucy.ac.cy)

import pandas as pd
import os

def errors(group1, group2, target_name):

  # Create empty Excel files
  with pd.ExcelWriter('xai-output/rule-extraction/errors-training.xlsx', mode='w') as writer1:
      pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')
  with pd.ExcelWriter('xai-output/rule-extraction/errors-evaluation.xlsx', mode='w') as writer2:
      pd.DataFrame().to_excel(writer2, sheet_name='Loop 1')

  for i in range(1, 41):

    # Read rule dataframes of both target groups extracted during training
    df1 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df2 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Get the errors on training
    train_error = df.loc[df[target_name] == 0].drop(columns=[df.columns[0]]).reset_index(drop=True)

    # Read rule dataframes of both target groups extracted during evaluation
    df3 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df4 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df3, df4], ignore_index=True)
    # Get the errors on evaluation
    test_error = df.loc[df[target_name] == 0].drop(columns=[df.columns[0]]).reset_index(drop=True)

    # Save the dataframes to Excel file
    with pd.ExcelWriter('xai-output/rule-extraction/errors-training.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
      train_error.to_excel(writer1, sheet_name='Loop ' + str(i))
    with pd.ExcelWriter('xai-output/rule-extraction/errors-evaluation.xlsx', mode='a', if_sheet_exists='overlay') as writer2:
      test_error.to_excel(writer2, sheet_name='Loop ' + str(i))


def dilemmas(group1, group2):

  # Create empty Excel files
  with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-training.xlsx', mode='w') as writer1:
      pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')
  with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-evaluation.xlsx', mode='w') as writer2:
      pd.DataFrame().to_excel(writer2, sheet_name='Loop 1')

  for i in range(1, 41):

    # Read rule dataframes of both target groups extracted during training
    df1 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df2 = pd.read_excel('xai-output/rule-extraction/rules-training' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop(columns=[df.columns[0]])

    # Group by the index
    groups = df.groupby(df['Index'])
    # Filter groups to find those where the 'Prediction' column has more than one unique value
    train_dilemma = groups.filter(lambda x: len(x['Prediction'].unique()) > 1).reset_index(drop=True)

    # Read rule dataframes of both target groups extracted during evaluation
    df3 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group1) + '.xlsx', sheet_name='Loop ' + str(i))
    df4 = pd.read_excel('xai-output/rule-extraction/rules-evaluation' + str(group2) + '.xlsx', sheet_name='Loop ' + str(i))

    # Concat the rule dataframes
    df = pd.concat([df3, df4], ignore_index=True)
    df = df.drop(columns=[df.columns[0]])

    # Group by the index
    groups = df.groupby(df['Index'])
    # Filter groups to find those where the 'Prediction' column has more than one unique value
    test_dilemma = groups.filter(lambda x: len(x['Prediction'].unique()) > 1).reset_index(drop=True)

    # Save the dataframes to Excel file
    with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-training.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
      train_dilemma.to_excel(writer1, sheet_name='Loop ' + str(i))
    with pd.ExcelWriter('xai-output/rule-extraction/dilemmas-evaluation.xlsx', mode='a', if_sheet_exists='overlay') as writer2:
      test_dilemma.to_excel(writer2, sheet_name='Loop ' + str(i))


def errors_dilemmas(selected_loop):

  # Note the selected loop (plus 1 as indexing starts from 0) in the text file
  with open('xai-output/rule-extraction/dilemmas-info.txt', 'w') as f:
    f.write("\n\nSummary of errors and dilemmas (Selected loop: " + str(selected_loop) + ")\n")
    f.write("------------------------------")

  for i in range(1, 41):

    # Read dataframes including errors and dilemmas extracted during training and testing
    train_error = pd.read_excel('xai-output/rule-extraction/errors-training.xlsx', sheet_name='Loop ' + str(i))
    test_error = pd.read_excel('xai-output/rule-extraction/errors-evaluation.xlsx', sheet_name='Loop ' + str(i))
    train_dilemma = pd.read_excel('xai-output/rule-extraction/dilemmas-training.xlsx', sheet_name='Loop ' + str(i))
    test_dilemma = pd.read_excel('xai-output/rule-extraction/dilemmas-evaluation.xlsx', sheet_name='Loop ' + str(i))

    # Get the unique cases of errors on training
    number_train_errors = train_error['Index'].nunique()
    unique_train_errors = train_error['Index'].unique()

    # Get the unique cases of errors on testing
    number_test_errors = test_error['Index'].nunique()
    unique_test_errors = test_error['Index'].unique()

    # Get the unique cases of dilemmas on training
    number_train_dilemmas = train_dilemma['Index'].nunique()
    unique_train_dilemmas = train_dilemma['Index'].unique()

    # Get the unique cases of dilemmas on testing
    number_test_dilemmas = test_dilemma['Index'].nunique()
    unique_test_dilemmas = test_dilemma['Index'].unique()

    # Find the common cases between errors and dilemmas
    train_intersection = len(set(unique_train_errors).intersection(set(unique_train_dilemmas)))
    test_intersection = len(set(unique_test_errors).intersection(set(unique_test_dilemmas)))

    # Get the unique cases of errors on both training and testing
    errors_training = number_train_errors - train_intersection
    errors_evaluation = number_test_errors - test_intersection

    with open('xai-output/rule-extraction/dilemmas-info.txt', 'a') as f:
      f.write("\n\nLoop: " + str(i) +
            "\nErrors (training): " + str(errors_training) +
            "\nDilemmas (training): " + str(number_train_dilemmas) +
            "\nErrors (evaluation): " + str(errors_evaluation) +
            "\nDilemmas (evaluation): " + str(number_test_dilemmas))

  # Create folder
  path = os.path.join('./xai-output', 'priorities')
  # Check whether directory already exists
  if not os.path.exists(path):
    os.mkdir(path)

  with open('xai-output/priorities/priorities-info.txt', 'w') as f:
    f.write("Summary of priority rules\n")
>>>>>>> 18936624597853e0f0418198783eb81f0c56b6ac
    f.write("-------------------------")