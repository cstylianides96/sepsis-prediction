from cohort import create_cohort
from raw_feat_preproc import preproc_raw_feat
from data_per_patient import create_data_per_patient
from dataset_part1 import create_datasets_imputed
from dataset_part2 import create_datasets_scorsys
from dataset_part3 import create_datasets_clean, categorize
from subjects import data_subjects
from descriptive_stats import dataset_stats
from ML import run_ml
from itemid_to_name import itemid_to_name_dataset
from plot import histogram_plots
from balanced_datasets import df_train_sets_balanced, df_test_sets_balanced, probs_to_pred_thres_90
from ML_balanced import run_ml_balanced, run_ml_balanced_encoded, run_ml_average
from DL_model_balanced import run_dl
from results_DL import overall_results_DL, overall_results_DL_updated
from ensemble import run_ensemble
from model_evaluation import plot_all_metrics_ensemble
from xai_full import run_xai


def run_pipeline():
    create_cohort()
    preproc_raw_feat()
    create_data_per_patient()
    create_datasets_imputed()
    create_datasets_scorsys()
    create_datasets_clean()
    data_subjects()
    run_ml(model_name='GBM', obs_win = 24, pred_win = 12, n_splits = 5, n_feat_imp = 70)
    itemid_to_name_dataset('GBM', 24, 12, '70', 'models/GBM_obs24_pred12_feat80(70)_3.pkl') #
    dataset_stats()
    categorize()
    histogram_plots()
    df_train_sets_balanced()
    df_test_sets_balanced()
    run_ml_balanced()
    run_ml_balanced_encoded('GBM', obs_win = 24, pred_win = 12, n_splits = 5)#
    run_ml_average()
    probs_to_pred_thres_90('GBM')
    run_dl(model_name='TCN', obs_win=24, pred_win=12, lr=0.001, epochs=60, batch_size=32, model_try='5')
    overall_results_DL()
    overall_results_DL_updated(['1DCNN', '1DCNN-LSTM', 'LSTM', 'TCN'], [1, 3, 1, 3])
    probs_to_pred_thres_90('LSTM')
    run_ensemble()
    probs_to_pred_thres_90('ENSEMBLE')
    plot_all_metrics_ensemble()
    run_xai()


#models, data, tables

