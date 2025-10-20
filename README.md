# Early Sepsis Prediction Using Interpretable Models

### Paper
[Citation]

### Aim
ML pipeline for sepsis diagnosis 12 hours in advance, by using 24 hours of clinical data (MIMIC-IV) and applying ML, DL, and Ensemble models. The pipeline is supported by a rule-based explainability method and argumentation-based reasoning.

### System Specifications
Parallel data preprocessing was performed on an HPC cluster running Rocky Linux 8.5, featuring multiple compute nodes
with AMD EPYC 7313 CPUs, up to 512 GB RAM and managed via SLURM. Experiments were conducted on a local workstation
running Ubuntu 22.04.5 LTS with Linux kernel 6.8.0. The system was equipped with an Intel Core i9-12900K CPU (16 cores,
24 threads, up to 5.2 GHz) and 62 GB of RAM. Analyses were run on Python >= 3.10. Deep learning models were implemented
using keras-core with the TensorFlow backend and executed on the CPU.

------------------------------------------------------------------------------------------------------------------------
### Steps to use this repository
1. Install all required packages from the **requirements.txt** file.
2. Create the following directories: 'data_raw', 'data_per_patient', 'data_processed', 'models', 'plots', 'results', 
'predictions', 'thresholds', 'xai-output'.
3. Download raw [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/) data and create the ['sepsis3'](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/sepsis)
table. Save them in the 'data_raw' directory. 
4. Run **pipeline_main.py** for the full pipeline.

The final GBM and LSTM models discussed in the paper are provided in the 'models' directory. Data in the form they were
inputted in the models are also provided in the 'data_processed' directory.

------------------------------------------------------------------------------------------------------------------------

### Functions used in pipeline_main.py
**create_cohort()**: Creates cases and controls cohort using MIMIC-IV v2.2 and the 'sepsis3' table where cases are
identified according to the Sepsis-3 definition. Generates *sepsis3_processed.csv*.

**preproc_raw_feat()**: Converts ICD-9 codes to ICD-10 codes, removes outliers lower than the 2nd percentile and above 
the 98th percentile and imputs with the value at the 2nd and 98th percentile, respectively.
Generates diagnoses, input, output, procedure and charted data, and their summaries.

**create_data_per_patient()**: Timestep checks, resampling, imputation. Generates *labels_sepsis.csv* and 3 csvs for each
patient: *demo.csv* (demographic), *dynamic.csv* (temporal), *static.csv* (diagnoses).

**create_datasets_imputed()**: Combines patient csvs into single dataset, selects cases and controls, performs
one-hot-encoding, KNN imputation for missing charted data, feature engineering, train-test split, feature selection.
Generates *obs24_pred12_processed.csv*, *obs24_pred12_imputed.csv*, *obs24_pred12_stayids.csv*,
*obs24_pred12_ratios.csv*, *obs24_pred12_gcs.csv*, *obs24_pred12_stats.csv*, *obs24_pred12_imputed_train.csv*,
*obs24_pred12_imputed_test.csv*,  *obs24_pred12_imputed_train_fs2.csv*, *obs24_pred12_imputed_train_fs2_1000.csv*.

**create_datasets_scorsys()**: Further feature engineering on obs24_pred12_imputed.csv features according to scoring systems
and clinicians' suggestions, encoding of specific features, train-test split, feature selection on the new training set. 
Generates *os24_pred12.csv*, *obs24_pred12_pfratio.csv*, *obs24_pred12_shockindex.csv*, *obs24_pred12_temp.csv*, *obs24_pred12_map.csv*, 
*obs24_pred12_fspn.csv*, *obs24_pred12_qsofa.csv*, *obs24_pred12_dop.csv*, *obs24_pred12_dopepinephnor_a.csv*, 
*obs24_pred12_dopepinephnor_b.csv*, *obs24_pred12_stats.csv*, *obs24_pred12_dig.csv*, *obs24_pred12_all.csv*, *obs24_pred12_all_changes*.

**create_datasets_clean()**: Concatenates features generated in the previous 2 functions and removes duplicates, encodes as
binary features coming from device measurements*, encodes as binary input events, train-test split.
Generates *obs24_pred12.csv*, *obs24_pred12_3.csv*, *obs24_pred12_train_3.csv*, *obs24_pred12_test_3.csv*.

*We assume that any chartevent variable belonging in the 'Impella', 'IABP', 'PiCCO', 'NICOM', 'ECMO', 'Centrimag', 
'Cardiovascular (Pacer Data)', 'Dialysis', 'Access Lines - Invasive', 'Hemodynamics', 'Skin - Impairment', 'Treatments', 
'Pain / Sedation', or 'Alarms' category with more than 60% missingness across patients came from specialized devices 
used by the minority of ICU patients. Therefore, these variables were Missing Not At Random (MNAR) and were not suitable 
for imputation. We encoded them as binary depending on whether a value exists/the device was used for a patient. 
Any remaining missing chartevent data were imputed using the K-Nearest Neighbour imputation method.

**data_subjects()**: Records subjects per class and data split. Generates *data_subjects.csv*.

**run_ml()**: Runs traditional ML models. RF, GBM, XGB, LGBM, AdaBoost, MLP supported. Model name, observation window, 
prediction window, number of cross validation splits and number of most important features for refitting model are required.
Generates models, prediction probabilities, performance results, importance plots, ROC and Precision-Recall curves.

**itemid_to_name_dataset()**: Converts all itemid names of variables to their labels, Model name, observation window,
prediction window, number of features used and model path name are required. Generates
*DFtrain_GBM_obs24_pred12_feat70_clean_3.csv*, *DFtest_GBM_obs24_pred12_feat70_clean_3.csv*,
*BEST_set5_GBM_obs24_pred12_feat70_clean_3.csv* with all feature names in order of importance and
*BEST_set5_GBM_obs24_pred12_feat70_unique_clean_3.csv* with unique feature names and their mimic-iv category in order of
importance.

**dataset_stats()**: Prints descriptive stats on the entire dataset (train and test sets).

**categorize()**: Encodes specific features according to [Med Calc](https://www.mdcalc.com/) and associate sites. 
Preliminary step for XAI. Generates *DFtrain_GBM_obs24_pred12_feat70_clean_3_encoded.csv* and 
*DFtest_GBM_obs24_pred12_feat70_clean_3_encoded.csv*.

**histogram_plots()**: Creates histogram plots for training set.

**df_train_sets_balanced()**: Generates 41 balanced training sets.
**df_test_sets_balanced()**: Generates 41 balanced test sets.

**run_ml_balanced()**: Runs GBM over 41 balanced training sets. Model name and number of cross validation splits are required.
Generates 41 *_balanced_probX_3.csv* files in the 'predictions' directory, *_results_balanced_3.csv* in
the 'results' directory, and a thresholds file in the 'thresholds' directory.

**run_ml_balanced_encoded()**: Same functionality as run_ml_balanced() based on the encoded datasets.

**run_ml_average()**: Prints average metrics across the 41 balanced datasets.

**probs_to_pred_thres_90()**: Generates predicted labels out of probability labels for XAI priorities evaluation in the 41
balanced datasets in the 'predictions' directory. Model name is required.

**run_dl()**: Runs LSTM, CNN, CNN-LSTM and TCN models over the 41 balanced training sets. Model name, learning rate, 
epoch number, batch size and model try number are required. 

**overall_results_DL()**: Extracts AUC and loss values of the 41st train and val sets (end of training) and the average  
AUC and loss values across the 41 test sets. Generates *'results_DL.csv'*.

**overall_results_DL_updated()**: Generates more performance metrics of selected models. A model names list and their
corresponding selected versions list are required. Generates *results_DL_new.csv*.

**run_ensemble()**: Averages predictions of GBM and H-LSTM to provide performance metrics for the 41 test sets and print
the mean performance across them.

**plot_all_metrics_ensemble()**: Plots ROC, Precision-Recall, Calibration and Net Benefit Curves for the 41 datasets in 
4 subplots. Generates *GBM-LSTM_obs24_pred12_ALL_PLOTS.png*.

**run_xai()**: Performs Explainable AI: rule extraction and selection. Argumentation-based reasoning is implemented in 
Prolog and executed using Gorgias Cloud. To run the argumentation algorithm, you'll need to construct the argumentation 
theory using the findings obtained from rule extraction and selection. More information 
on how to use Gorgias Cloud can be found [here](http://gorgiasb.tuc.gr/GorgiasCloud.html).
