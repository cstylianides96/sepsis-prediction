import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def plot_all_sepsis():
    cohort = pd.read_csv('data_processed/sepsis3_processed.csv')
    # print(cohort)
    # print(cohort['label'].value_counts())  # 0:58558, 1:14623
    # print(len(cohort[cohort['label']==1]['subject_id'].unique()))  # 13260 unique cases

    sepsis3 = cohort[cohort['label']==1]
    sepsis3 = sepsis3.sort_values(by='hours_after_adm').reset_index(drop=True)
    sepsis3['cumsum'] = 1
    sepsis3['cumsum'] = sepsis3['cumsum'].cumsum()
    # print(sepsis3.head())

    # PLot Cumulative Freq vs Hours after ICU admission
    plt.plot(sepsis3['hours_after_adm'], sepsis3['cumsum'])
    plt.xlabel('Hours After ICU Admission', fontsize=16)
    plt.ylabel('Cumulative Frequency', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(left=0, right=15)
    plt.title('Total Patients Experiencing Sepsis After ICU Admission', fontsize=18)
    plt.tight_layout()
    #plt.savefig('plots/patients_hours_after_adm_cumsum')
    plt.show()

    # Plot boxplot
    plt.boxplot(np.asarray(sepsis3['hours_after_adm']))
    plt.ylabel('Hours After ICU Admission', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 10)
    plt.tight_layout()
    #plt.savefig('plots/patients_hours_after_adm_boxplot')
    plt.show()

    cases_per_hour = sepsis3.groupby(['hours_after_adm']).size().reset_index()
    # print(cases_per_hour)
    #plt.bar(cases_per_hour.iloc[:8, 0]-0.2, cases_per_hour.iloc[:8, 1], width=0.4, label='Cases')
    plt.bar(cases_per_hour.iloc[:, 0], cases_per_hour.iloc[:, 1])
    plt.xlabel('Hours After ICU Admission', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.xlim(left=0, right=9)
    plt.title('Patients Experiencing Sepsis After ICU Admission', fontsize=16)
    plt.legend(prop={'size': 14})
    for val_x, val_y in zip(cases_per_hour.iloc[:, 0], cases_per_hour.iloc[:, 1]):
        plt.annotate(str(val_y), (val_x, val_y), fontsize=10)
    plt.tight_layout()
    #plt.savefig('plots/patients_hours_after_adm_bar')
    plt.show()

    # Sepsis-Stats
    mean_hours = sepsis3['hours_after_adm'].mean()
    median_hours = sepsis3['hours_after_adm'].median()
    # print(sepsis3['hours_after_adm'].quantile([0.25, 0.5, 0.75]))
    # print(sepsis3['hours_after_adm'].mode())

    grouped=cohort.groupby(['hours_after_adm']).size().reset_index()
    grouped.columns = ['hours_after_adm', 'patients']
    grouped['cumsum'] = grouped['patients'].cumsum()
    # print(grouped.iloc[:7, :])


def plot_feat_importances(model_name, obs_win, pred_win, n_feat, model):
    model_feats = model.feature_names_in_.tolist()
    X_cols = model_feats

    itemids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'label']]
    icd10_codes = pd.read_csv('data_raw/icd10cm_codes_2024.csv')

    importances = model.feature_importances_
    indices = np.argsort(importances)
    items = [X_cols[i] for i in indices]
    print(items)
    labels = []
    for item in items:
        if ('_' in item) and (item[-1].isdigit()):  # ending in window number
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit():  # itemid (chartevent/inputevent/outputevent/procedureevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else: #ratios
                labels.append(item)
        elif ('_' in item) and (item.rsplit('_', 1)[1] in ['mean', 'median', 'min', 'max', 'sd', 'range', 'sum']): #stats for temporal features
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit(): #itemid (chartevent/inputevent/outputevent/procedureevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else: #ratios
                labels.append(item)
        elif item in icd10_codes['icd10_code'].tolist():  # diagnosis
            label = icd10_codes.loc[icd10_codes['icd10_code'] == item]['label'].values[0]
            labels.append(label)
        elif 'diff' in item:
            itemid = item.rsplit('_', 2)[0]
            if itemid.isdigit():
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 2)[-2] + '_' + item.rsplit('_', 2)[-1]
                labels.append(label)
            else: #gcs_sum
                labels.append(item)
        else:  # gender/ethnicity/age/adm_to_pred/hosp_to_icu
            labels.append(item)
    #print(labels)
    #print(importances[indices])
    plt.figure()
    plt.barh( labels, importances[indices], color='b', align='center')
    #plt.yticks(range(len(indices)), labels, fontsize=6)
    plt.xticks(fontsize=10)
    plt.xlabel('Relative Importance', fontsize=10)
    plt.title('Feature Importances', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_imp_3')
    plt.show()



def histogram_plots():
    df = pd.read_csv('DFtrain_GBM_obs24_pred12_feat70_clean_3.csv').iloc[:, :-1]
    for feats in range(0, 70, 20):
        df_sub = df.iloc[:, feats:feats+20]
        n_features = 20

        n_cols = 5  # Number of columns for subplots
        n_rows = math.ceil(n_features/n_cols)  # Calculate number of rows needed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        axes = axes.flatten()  # Flatten to 1D for easier indexing

        for i, feature in enumerate(df_sub.columns):
            if i < len(axes):  # Check that i is within bounds
                axes[i].hist(df_sub[feature], bins=20, edgecolor='black')
                axes[i].set_title(feature)
            else:
                print(f"Not enough axes for feature {feature}, skipping.")

        plt.tight_layout()
        plt.show()
