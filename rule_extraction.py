from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score,  precision_score, confusion_matrix
from te2rules.explainer import ModelExplainer
import numpy as np
from metrics import *
import os

def rule_extraction(group, target_name, column_labels, selected_path, replace):

    # Create folder
    path = os.path.join('./xai-output', 'rule-extraction')
    # Check whether directory already exists
    if not os.path.exists(path):
        os.mkdir(path)

    with open('xai-output/rule-extraction/rules-info' + group + '.txt', 'w') as f:
        f.write(target_name + '\n')

    auc_list = []
    fidelity_list = []
    rule_list = []

    for i in range(1, 41):

      # Read training data
      train = pd.read_csv(selected_path + 'train_' + str(i) + '_encoded.csv')

      # Rename column names of training data
      train.columns = column_labels

      # Read evaluation data
      test = pd.read_csv(selected_path + 'test_' + str(i) + '_encoded.csv')

      # Rename column names of evaluation data
      test.columns = column_labels

      # Select the appropriate model
      model = GradientBoostingClassifier(n_estimators=80, random_state=123)

      # Change the positive class
      if replace == True:
          train[target_name] = train[target_name].replace({1: 0, 0: 1})
          test[target_name] = test[target_name].replace({1: 0, 0: 1})

      # Separate the data from the target
      train_target = train[target_name]
      train_data = train.drop(columns=[target_name, 'Index'], axis=1)
      test_target = test[target_name]
      test_data = test.drop(columns=[target_name, 'Index'], axis=1)

      # Print the final number of data on each set
      with open('xai-output/rule-extraction/rules-info' + group + '.txt', 'a') as f:
          f.write('\nLoop: {}\n'.format(i))
          f.write('\nNumber of training data in positive class {} and negative class {}'
                  .format((train_target == 1).sum(), (train_target == 0).sum()))
          f.write('\nNumber of testing data in positive class {} and negative class {}\n\n'
                  .format((test_target == 1).sum(), (test_target == 0).sum()))

          f.write(str(model))

          # Model training
          model.fit(train_data, train_target)

          # Get training probabilities
          probs = model.predict_proba(train_data)[:,1]

          # Find accuracy and AUC on training and testing
          scores1 = model.score(train_data, train_target)
          scores2 = model.score(test_data, test_target)
          auc1 = roc_auc_score(train_target, probs)
          auc2 = roc_auc_score(test_target, model.predict_proba(test_data)[:, 1])
          f.write('\nTraining Accuracy= ' + str(round(scores1, 2)) +
                  '\nEvaluation Accuracy= ' + str(round(scores2, 2)) +
                  '\nTraining AUC= ' + str(round(auc1, 5)) +
                  '\nEvaluation AUC= ' + str(round(auc2, 5)))


          # Adjust the appropriate threshold of classification predictions
          def adjust_threshold(target, probabilities):

            # Get fpr, tpr, thresholds
            fpr, tpr, thresholds = roc_curve(target, probabilities)
            roc_df = pd.DataFrame(zip(thresholds, tpr, fpr), columns=['thresholds', 'sen', 'fpr']).sort_values(by='sen',ascending=False).reset_index(drop=True)

            # Specify threshold such that sensitivity >= 0.9
            roc_df_90 = roc_df.loc[roc_df['sen'] >= 0.9]

            # Get the minimum threshold (threshold of last row and first column)
            threshold = roc_df_90.iloc[-1, 0]
            #print('Threshold= ' + str(threshold)) # 0.013716252891950019

            return np.where(probs >= threshold, 1, 0)

          train_predict = adjust_threshold(train_target, probs)

          # Calculate sensitivity (recall, true positive rate), specificity (true negative rate), precision and negative predictive value
          cm = confusion_matrix(train_target, train_predict)
          TN, FP, FN, TP = cm.ravel()
          sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
          specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
          precision = precision_score(train_target, train_predict)
          npv = TN / (TN + FN) if (TN + FN) != 0 else 0
          f.write('\n\nTraining metrics:' +
                  '\nSensitivity= ' + str(round(sensitivity, 5)) +
                  '\nSpecificity= ' + str(round(specificity, 5)) +
                  '\nPrecision= ' + str(round(precision, 5)) +
                  '\nNegative Predictive Value= ' + str(round(npv, 5)) +
                  '\n\nTrue Positives= ' + str(TP) +
                  '\nFalse Positives= ' + str(FP) +
                  '\nTrue Negatives= ' + str(TN) +
                  '\nFalse Negatives= ' + str(FN))


          # Rule extraction using training data
          model_explainer = ModelExplainer(model=model, feature_names=list(train_data.columns))
          rules = model_explainer.explain(X=list(train_data.values), y=train_predict.tolist(), num_stages = 5)

          print('\nCalculate coverage and rule accuracy..\n')
          # Calculate coverage and confidence of each rule on training data
          coverage, confidence = metrics(target_name, train, rules)
          # Calculate coverage and confidence of each rule on evaluation data
          coverage2, confidence2 = metrics(target_name, test, rules)

          # Summarize the findings
          f.write('\n\n' + str(len(rules)) + ' rules found\n')
          for j in range(len(rules)):
              f.write('\nRule ' + str(j) + ': ' + str(rules[j]) +
                      '\n\n\tTraining: Covered ' + str(coverage[j]) + ' from ' + str(len(train[target_name])) + ' subjects. Covered confidently ' + str(confidence[j]) +
                      '\n\tCoverage= ' + str(round(coverage[j]/len(train[target_name]), 2)) +
                      '\n\tAccuracy= ' + str(round(confidence[j]/coverage[j], 2) if coverage[j] != 0 else 'n/a') +
                      '\n\n\tEvaluation: Covered ' + str(coverage2[j]) + ' from ' + str(len(test[target_name])) + ' subjects. Covered confidently ' + str(confidence2[j]) +
                      '\n\tCoverage= ' + str(round(coverage2[j]/len(test[target_name]), 2)) +
                      '\n\tAccuracy= ' + str(round(confidence2[j]/coverage2[j], 2) if coverage2[j] != 0 else 'n/a') +'\n')

          # Get fidelity of the extracted rules
          fidelity, positive_fidelity, negative_fidelity = model_explainer.get_fidelity()

          f.write('\nThe rules explain ' + str(round(fidelity * 100, 2)) + '% of the overall predictions of the model')
          f.write('\nThe rules explain ' + str(round(positive_fidelity * 100, 2)) + '% of the overall positive predictions of the model')
          f.write('\nThe rules explain ' + str(round(negative_fidelity * 100, 2)) + '% of the overall negative predictions of the model\n\n')

          # Save the auc and fidelity metrics to lists
          auc_list.append(auc1)
          fidelity_list.append(fidelity)

          # Save rules to lists
          rule_list.append(rules)

    return auc_list, fidelity_list, rule_list