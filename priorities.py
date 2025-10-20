# Author: Andria Nicolaou (nicolaou.andria@ucy.ac.cy)

import pandas as pd


def priorities(group2, data):

    # Create empty Excel file
    with pd.ExcelWriter('xai-output/priorities/priority-rules-' + str(data) + '.xlsx', mode='w') as writer1:
        pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')

    with open('xai-output/priorities/priorities-info.txt', 'a') as f:
      f.write(f'\n\nSet: {str(data)}')

    for i in range(1, 41):

        # Read rule dataframes of both target groups extracted during training
        df = pd.read_excel('xai-output/rule-extraction/dilemmas-' + str(data) + '.xlsx', sheet_name='Loop ' + str(i))

        # Rename the rules of second group
        def modify_rule(row):
            if row['Prediction'] == group2:
                return 'N' + row['Rule']
            else:
                return row['Rule']

        # Modify the 'Rule' column by applying the previous function
        df['Rule'] = df.apply(modify_rule, axis=1)

        # Drop the first column
        df = df.drop(columns=[df.columns[0]])

        # Sort by 'Index' column
        df_sorted = df.sort_values('Index')

        # Group by 'Index' to get the number of dilemma records
        grouped_df = df_sorted.groupby('Index').apply(lambda x: x.to_dict('Records')).reset_index(name='Records')

        with open('xai-output/priorities/priorities-info.txt', 'a') as f:
          f.write(f'\n\nLoop {str(i)}:')
          f.write(f"\n{len(grouped_df)} records presented dilemmas on the {data} set:")

        # Step 1: Analyze each record to identify its priority rules
        index_list = []
        priority_rules_list = []
        other_rules_list = []
        for index, row in grouped_df.iterrows():
            records = row['Records']
            priority_rules = []
            other_rules = []
            # If the record has label = 1 then its rule is a priority rule
            for record in records:
                if record['label'] == 1:
                    priority_rules.append(record['Rule'])
                else:
                    other_rules.append(record['Rule'])

            # Save the index, priority rules and other rules in the lists
            for priority_rule in priority_rules:
                for other_rule in other_rules:
                    index_list.append(row['Index'])
                    priority_rules_list.append(priority_rule)
                    other_rules_list.append(other_rule)

        # Create a new DataFrame including indices, priority rules and other rules
        priority_rules_df = pd.DataFrame({
            'Index': index_list,
            'Priority Rules': priority_rules_list,
            'Other Rules': other_rules_list
        })

        # Group by 'Priority Rules' and 'Other Rules' and aggregate
        priority_rules_df = priority_rules_df.groupby(['Priority Rules', 'Other Rules'])['Index'].apply(list).reset_index()

        # Rename the 'Index' column to 'Indices' for clarity
        priority_rules_df = priority_rules_df.rename(columns={'Index': 'Indices'})

        # Save the DataFrame to Excel file
        with pd.ExcelWriter('xai-output/priorities/priority-rules-' + str(data) + '.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
          priority_rules_df.to_excel(writer1, sheet_name='Loop ' + str(i), index=False)


        # Step 2: Get the resolved indices after applying the priority rules
        all_indices = []
        for indices in priority_rules_df['Indices']:
            all_indices.extend(indices)  # Extend the list with all the unique indices

        resolved_indices = list(set(all_indices)) # Create a set that contains the indices
        resolved_indices.sort()


        with open('xai-output/priorities/priorities-info.txt', 'a') as f:
          f.write(f"\n{len(resolved_indices)} of them can be resolved by generating {len(priority_rules_df)} priority rules")



def intersection_priorities():

  # Create empty Excel file
  with pd.ExcelWriter('xai-output/priorities/priority-rules.xlsx', mode='w') as writer1:
    pd.DataFrame().to_excel(writer1, sheet_name='Loop 1')

  for i in range(1, 41):

    # Read priority rule dataframes of both training and evaluation
    priority_rules_training = pd.read_excel('xai-output/priorities/priority-rules-training.xlsx', sheet_name='Loop ' + str(i))
    priority_rules_evaluation = pd.read_excel('xai-output/priorities/priority-rules-evaluation.xlsx', sheet_name='Loop ' + str(i))

    # Get the intersection of dataframes
    merged_df = pd.merge(priority_rules_training, priority_rules_evaluation, on=['Priority Rules', 'Other Rules'], how='inner')

    # Rename the columns
    merged_df = merged_df.rename(columns={'Indices_x': 'Indices Training', 'Indices_y': 'Indices Evaluation'})

    # Save the DataFrame to Excel file
    with pd.ExcelWriter('xai-output/priorities/priority-rules.xlsx', mode='a', if_sheet_exists='overlay') as writer1:
      merged_df.to_excel(writer1, sheet_name='Loop ' + str(i), index=False)
