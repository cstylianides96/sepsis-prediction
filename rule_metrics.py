import pandas as pd

def metrics(target_name, data, rules):

    coverage = []
    confidence = []

    for rule in rules:

        cov = 0
        conf = 0

        # Check the rule if it has more than one condition
        if str(rule).find('&') == -1:

            # Split the parts of the rule
            feature, operator, value = rule.split()

            # Calculate coverage and confidence of the rule by grouping the training data in which the rule is applied
            if operator == '<=':
                cov = len(data.loc[data[feature] <= float(value)])
                conf = len(data.loc[(data[target_name] == 1) & (data[feature] <= float(value))])
            elif operator == '>':
                cov = len(data.loc[data[feature] > float(value)])
                conf = len(data.loc[(data[target_name] == 1) & (data[feature] > float(value))])
            elif operator == '>=':
                cov = len(data.loc[data[feature] > float(value)])
                conf = len(data.loc[(data[target_name] == 1) & (data[feature] > float(value))])
            elif operator == '<':
                cov = len(data.loc[data[feature] > float(value)])
                conf = len(data.loc[(data[target_name] == 1) & (data[feature] > float(value))])

        # Cases where the rule has more than one condition
        else:

            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            df_cov = pd.DataFrame()
            df_conf = pd.DataFrame()

            # Split the rule to its conditions
            conditions = rule.split('&')

            # Calculate coverage and confidence of the rule by applying it to the training data
            for condition in conditions:
                # Split the parts of the rule
                feature, operator, value = condition.split()

                if operator == '<=':
                    df1 = data.loc[data[feature] <= float(value)]
                    df2 = data.loc[(data[target_name] == 1) & (data[feature] <= float(value))]
                elif operator == '>':
                    df1 = data.loc[data[feature] > float(value)]
                    df2 = data.loc[(data[target_name] == 1) & (data[feature] > float(value))]
                elif operator == '>=':
                    df1 = data.loc[data[feature] > float(value)]
                    df2 = data.loc[(data[target_name] == 1) & (data[feature] > float(value))]
                elif operator == '<':
                    df1 = data.loc[data[feature] > float(value)]
                    df2 = data.loc[(data[target_name] == 1) & (data[feature] > float(value))]


                # Concatenate the right cases
                df_cov = pd.concat([df_cov, df1])
                df_conf = pd.concat([df_conf, df2])

            # Check the cases that are applied by all the conditions of the rule
            for i_cov in df_cov.index.value_counts():
                if i_cov == len(conditions):
                    cov = cov + 1

            for i_conf in df_conf.index.value_counts():
                if i_conf == len(conditions):
                    conf = conf + 1

        coverage.append(cov)
        confidence.append(conf)

    return coverage, confidence



def rule_dataframe(group, data, rules):

    df = pd.DataFrame()
    df_final = pd.DataFrame()

    for rule in rules:

        # Check the rule if it has more than one condition
        if str(rule).find('&') == -1:

            # Split the parts of the rule
            feature, operator, value = rule.split()

            if operator == '<=':
                df = data.loc[data[feature] <= float(value)]
            elif operator == '>':
                df = data.loc[data[feature] > float(value)]

        # Cases where the rule has more than one condition
        else:

            # Split the rule to its conditions
            conditions = rule.split('&')

            # Initialize the dataframe with the data
            df = data

            for condition in conditions:

                # Split the parts of the rule
                feature, operator, value = condition.split()

                if operator == '<=':
                    df = df.loc[data[feature] <= float(value)]
                elif operator == '>':
                    df = df.loc[data[feature] > float(value)]


        # Create a column defining the rule
        df = df.copy()
        df['Rule']= 'R' + str(rules.index(rule))
        # Concatenate the cases
        df_final = pd.concat([df_final, df])

    # Create a column defining the prediction
    df_final['Prediction'] = group

    # Create a column defining the index
    #df_final['Index'] = df_final.index

    return df_final
