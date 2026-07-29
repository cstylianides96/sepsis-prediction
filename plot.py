# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

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


def plot_feat_importances(model_name, n_feat, model):
    model_feats = model.feature_names_in_.tolist()
    X_cols = model_feats

    itemids = pd.read_csv('/data_raw/d_items.csv')[['itemid', 'label']]
    icd10_codes = pd.read_csv('/data_raw/icd10cm_codes_2024.csv')

    importances = model.feature_importances_
    indices = np.argsort(importances)
    items = [X_cols[i] for i in indices]
    print(items)
    labels = []
    for item in items:
        if ('_' in item) and (item[-1].isdigit() and ('diff' not in item)):  # ending in window number
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit():  # itemid (chartevent)
                label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
                label = label + '_' + item.rsplit('_', 1)[1]
                labels.append(label)
            else: #ratios
                labels.append(item)
        elif ('_' in item) and (item.rsplit('_', 1)[1] in ['mean', 'min', 'max','range']): #stats for temporal features
            itemid = item.rsplit('_', 1)[0]
            if itemid.isdigit(): #itemid (chartevent)
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
                label = label + '_' + item.rsplit('_', 2)[1] + '_' + item.rsplit('_', 2)[2]
                labels.append(label)
            else: #gcs_sum
                labels.append(item)
        else:  # gender/age/hosp_to_icu
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
    plt.savefig('/plots/' + model_name + '_feat' + str(n_feat) + '_importances.png')
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


def box_plots():
    ensemble_metrics = pd.read_csv('/results/ENSEMBLE_results_balanced.csv')
    ensemble_metrics = ensemble_metrics[['test_auc','test_sen_90','test_spec_90','test_precision_90','test_npv_90','test_sen_yuden','test_spec_yuden','test_precision_yuden','test_npv_yuden']]
    metrics = ensemble_metrics.columns.tolist()
    n = len(metrics)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    # boxplot style props
    boxprops = dict(facecolor='#a6cee3', color='black')
    medianprops = dict(color='red')
    whiskerprops = dict(color='black')
    capprops = dict(color='black')
    flierprops = dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.6)

    for i, col in enumerate(metrics):
        data = ensemble_metrics[col].dropna()
        if i >= len(axes):
            break
        if data.empty:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes, fontsize=8)
            axes[i].axis('off')
            continue

        # compute stats
        q1 = data.quantile(0.25)
        median = data.median()
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        mean = data.mean()
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        std = data.std()

        # draw boxplot with styling
        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops,
                        flierprops=flierprops)

        # add stats textbox in axes coordinates
        stats_text = (
            f"median: {median:.2f}\n"
            f"IQR: {iqr:.2f}\n"
            f"mean: {mean:.2f}\n"
            f"range: {data_range:.2f}\n"
            f"std: {std:.2f}"
        )
        axes[i].text(0.03, 0.97, stats_text, transform=axes[i].transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        axes[i].set_title(col, fontsize=12)
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    # Turn off any unused subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('/plots/ensemble_metrics_boxplots.png', dpi=200)
    plt.show()
