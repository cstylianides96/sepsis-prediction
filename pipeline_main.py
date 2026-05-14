# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

from cohort_creation import create_cohort
from cohort_creation_eICU import create_cohort_eICU
from data_extraction import extract_data
from data_extraction_eICU import extract_data_eICU
from data_processing2 import preprocess
from data_processing_eICU import preprocess_eICU
from GBM_feat_selection import run_gbm, gbm_feat_selection
from subjects import data_subjects
from descriptive_stats import dataset_stats, demo_stats
from xai_preprocess import itemid_to_name_dataset, categorize
from balanced_datasets import create_balanced_datasets
from probs_to_pred import probs_to_pred
from ML_balanced import run_ml_balanced, run_ml_average
from DL_balanced import run_dl
from results_DL import overall_results_DL, overall_results_DL_updated
from ensemble import run_ensemble
from model_evaluation import plot_all_metrics_ensemble
from xai_full import run_xai
from ML_eICU import ensemble_eICU


def run_pipeline():
    create_cohort()
    extract_data()
    preprocess()
    run_gbm(n_feat_imp=40)
    gbm_feat_selection()
    
    data_subjects()
    dataset_stats()
    demo_stats()

    create_balanced_datasets(encoded=False) 
    run_ml_balanced(encoded=False) 
    run_ml_average(encoded=False)
    probs_to_pred('GBM') 

    run_dl(model_name='IDCNN', obs_win=24, pred_win=12, lr=0.001, epochs=60, batch_size=32, model_try='10') #params for final model in code
    overall_results_DL()
    overall_results_DL_updated(['1DCNN', '1DCNN-LSTM', 'LSTM', 'TCN'], [15, 8, 15, 6])
    probs_to_pred('LSTM') 

    run_ensemble()
    probs_to_pred('ENSEMBLE') 
    plot_all_metrics_ensemble()
    
    itemid_to_name_dataset()
    categorize()
    create_balanced_datasets(encoded=True)
    run_xai()

    create_cohort_eICU()
    extract_data_eICU()
    preprocess_eICU()
    ensemble_eICU()
run_pipeline()
# paths in files, upload to github
