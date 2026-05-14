# Early Sepsis Prediction Using Interpretable Models

### Paper
[Citation]

### Aim
ML pipeline for sepsis diagnosis 12 hours in advance, by using 24 hours of clinical data (MIMIC-IV) and applying ML, DL, and Ensemble models. The pipeline is supported by a rule-based explainability method and argumentation-based reasoning. The pipeline is externally validated on the eICU dataset.

### System Specifications
Parallel data preprocessing was performed on an HPC cluster running Rocky Linux 8.5, featuring multiple compute nodes
with AMD EPYC 7313 CPUs, up to 512 GB RAM and managed via SLURM. Experiments were conducted on a local workstation
running Ubuntu 22.04.5 LTS with Linux kernel 6.8.0. The system was equipped with an Intel Core i9-12900K CPU (16 cores,
24 threads, up to 5.2 GHz) and 62 GB of RAM. Analyses were run on Python >= 3.10. Deep learning models were implemented
using keras-core with the TensorFlow backend and executed on the CPU.

------------------------------------------------------------------------------------------------------------------------
### Steps to use this repository
1. Install all required packages from the **requirements.txt** file.
2. Create the following directories: 'data_raw', 'data_raw_eicu_v2.0', 'data_processed', 'data_processed_eicu', 'models', 'plots', 'results', 
'predictions', 'xai-output'.
3. Download raw [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/) data and create the ['sepsis3'](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/sepsis)
table. Save them in the 'data_raw' directory. 
4. Download raw [eICU v2.0](https://physionet.org/content/eicu-crd/2.0/) data. Save them in the 'data_raw_eicu_v2.0' directory.
5. Run **pipeline_main.py** for the full pipeline.

MIMIC-IV data as they were used in models after processing are provided in the 'data_processed' directory.

------------------------------------------------------------------------------------------------------------------------

### Functions used in pipeline_main.py

**create_cohort()**: Creates cases and controls cohort using MIMIC-IV v2.2 and the 'sepsis3' table where cases are
identified according to the Sepsis-3 definition. Generates *sepsis3_processed.csv*.

**extract_data()**: Extracts charted data, diagnoses and demographics. Generates *sepsis_chartevents.csv*, *sepsis_diagnoses.csv*, *sepsis_demographics.csv*.


**preprocess()**: Cleaning, temporal data handling, feature engineering, data splits and feature selection.

**run_gbm(n_feat_imp=40)**: Feature selection using a GBM model. FInal iteration where the 40 most important features were selected from the last 50 and were chosen for final model based on AUC change on validation dataset. Model, results, predictions, importance plot are saved.
    
**gbm_feat_selection()**: Selects final features from datasets. Generates *train_selected_feat40.csv*, *val_selected_feat40.csv*, *test_selected_feat40.csv*.
    
**data_subjects()**: Records subjects per class and data split. Generates *data_subjects.csv* and *data_subjects_percentages.csv*.

**dataset_stats()**: Prints descriptive stats on the entire dataset (train and test sets).

**demo_stats()**: Prints descriptive stats of demographic variables on the entire dataset (train and test sets).

**create_balanced_datasets(encoded=False)**: Splits train, validation and test sets into balanced sets (40 equal sized balanced datasets for each set).

**run_ml_balanced(encoded=False)**: Runs a GBM model with 5-fold CV on each balanced dataset. Saves results and predictions for each.

**run_ml_average(encoded=False)**: Prints average of results across the 40 datasets.

**probs_to_pred()**: Generates predicted labels out of probability labels for XAI priorities evaluation in the 40 datasets. Model name (GBM, LSTM, GBM-LSTM) is required.
 
**run_dl()**: Runs DL models. Requires  model name (LSTM, 1DCNN, TCN, 1DCNN-LSTM), observation window (24), prediction window (12), learning rate, number of epochs, batch size, and number of model try.

**overall_results_DL()**: Extracts results at the end of each model training (at 40th training and validation sets) and average test set results (across the 40 datasets). Generates *DL_results_balanced.csv*.

**overall_results_DL_updated()**: Updates results of selected models with more metrics. Requires a list of the selected model names and a list of their corresponding model tries. Generates *DL_results_balanced_updated.csv*.

**run_ensemble()**: Saves average predictions of ensemble model and its results.

**plot_all_metrics_ensemble()**: Plots ROC, Precision-Recall, Calibration and Net Benefit Curves for the 40 datasets in 4 subplots. Generates *GBM-LSTM_balanced_allplots.png*
    
**itemid_to_name_dataset()**: Converts all itemids of the selected variables to their labels as a preprocessing step for XAI. Saves the new dataset with the updated column names. Saves feature names from most to least important and the unique features with the category they belong to.

**categorize()**: Encodes specific features according to [Med Calc](https://www.mdcalc.com/) and associate sites as a preprocessing step for XAI. Saves encoded datasets.

**run_xai()**: Performs Explainable AI: rule extraction and selection. Argumentation-based reasoning is implemented in 
Prolog and executed using Gorgias Cloud. To run the argumentation algorithm, you'll need to construct the argumentation 
theory using the findings obtained from rule extraction and selection. More information 
on how to use Gorgias Cloud can be found [here](http://gorgiasb.tuc.gr/GorgiasCloud.html).

**create_cohort_eICU()**: Creates cohort from the eICU dataset identidying cases and controls in the same way as for MIMIC-IV.

**extract_data_eICU()**: Extracts corresponding eICU variables.

**preprocess_eICU()**: Preprocess eICU data in the same way MIMIC-IV data were preprocessed.

**ensemble_eICU()**: Runs the GBM and LSTM models developped on MIMIC-IV, averages predictions and computes metrics for the ensemble. Saves results.
