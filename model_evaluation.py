# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import pandas as pd
import numpy as np
from sklearn.metrics import (precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score, accuracy_score,
                             PrecisionRecallDisplay, RocCurveDisplay, precision_score, recall_score,
                             classification_report, ConfusionMatrixDisplay, auc, average_precision_score, brier_score_loss)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.calibration import calibration_curve


def youden_index(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold) # .astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1


def evaluate(prob, actual, model_name, obs_win, pred_win, n_feat, acc):

    auc = np.round(roc_auc_score(actual, prob), 5)

    # ROC
    fpr, tpr, thresholds = roc_curve(actual, prob)
    # save thresholds (SEN, FPR)
    sen_fpr_thres = pd.DataFrame(zip(thresholds, tpr, fpr),
                                 columns=['thresholds', 'sen', 'fpr']).sort_values(by='sen',
                                                                                   ascending=False).reset_index(drop=True)
    if acc==False: # imbalanced dataset
        display_roc = RocCurveDisplay.from_predictions(
            actual, prob, name=model_name, plot_chance_level=True)
        display_roc.ax_.set_title(
            "ROC: " + model_name + ', obs ' + str(obs_win) + ', pred ' + str(pred_win) + ', feat ' + str(n_feat),
            fontsize=16)
        plt.savefig(
            'plots/' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_ROC_3')


    # PRC
    precision, recall, thresholds = precision_recall_curve(actual, prob)
    # save thresholds (PRE, SEN)
    pre_rec_thres = pd.DataFrame(zip(thresholds, recall, precision),
                                 columns=['thresholds', 'recall', 'precision']).sort_values(by='recall',
                                                                        ascending=False).reset_index(drop=True)
    if acc==False: # imbalanced dataset
        display_prc = PrecisionRecallDisplay.from_predictions(actual, prob, name=model_name, plot_chance_level=True)
        display_prc.ax_.set_title(
            'PRC: ' + model_name + ', obs ' + str(obs_win) + ', pred ' + str(pred_win) + ', feat ' + str(n_feat),
            fontsize=16)  # AP is Weighted Average Precision of thresholds
        plt.savefig(
            'plots/' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_PRC_3')


    # threshold such that recall>=0.9 (SEN, FPR)
    sen_fpr_thres_90 = sen_fpr_thres.loc[sen_fpr_thres['sen'] >= 0.9]
    threshold_90 = sen_fpr_thres_90.iloc[-1, 0]
    pred_opt = (prob >= threshold_90)
    cm = confusion_matrix(actual, pred_opt)
    tn, fp, fn, tp = cm.ravel()
    #print(tn, fp, fn, tp)
    sen_90 = np.round(recall_score(actual, pred_opt), 5)  # tp / (tp + fn)  # Recall/TPR/Sensitivity(TP/(TP+FN))
    spec_90 = np.round((tn / (tn + fp)), 5)  # Specificity / TNR(TN / (TN + FP))
    precision_90 = np.round(precision_score(actual, pred_opt), 5)
    npv_90 = np.round((tn / (tn + fn)), 5)  # NPV  (TN/(TN+FN))
    if acc:
        acc_90 = np.round(accuracy_score(actual, pred_opt), decimals=5)

    # threshold that maximizes Youden Index
    youden_indices = [youden_index(actual, prob, threshold) for threshold in sen_fpr_thres['thresholds']]
    optimal_idx = np.argmax(youden_indices)
    threshold_yuden = sen_fpr_thres.iloc[optimal_idx, 0]
    pred_opt = (prob >= threshold_yuden)
    cm = confusion_matrix(actual, pred_opt)
    tn, fp, fn, tp = cm.ravel()
    sen_yuden = np.round(recall_score(actual, pred_opt), 5)  # tp / (tp + fn)  # Recall/TPR/Sensitivity(TP/(TP+FN))
    spec_yuden = np.round((tn / (tn + fp)), 5)  # Specificity / TNR(TN / (TN + FP))
    precision_yuden = np.round(precision_score(actual, pred_opt), 5)
    npv_yuden = np.round((tn / (tn + fn)), 5)  # NPV  (TN/(TN+FN))
    if acc:
        acc_yuden = np.round(accuracy_score(actual, pred_opt), decimals=5)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.savefig('plots/' + model_name + '_obs' + str(obs_win) + '_pred' + str(pred_win) + '_feat' + str(n_feat) + '_CM')
    # plt.show()
    # print(classification_report(actual, pred))
    if acc:
        return auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, threshold_90, threshold_yuden, acc_90, acc_yuden
    else:
        return auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, threshold_90, threshold_yuden


def plot_all_metrics_ensemble():
    # Load Data
    y_test_df = pd.DataFrame()
    for idx in range(0, 40):
        df_test = pd.read_csv(f'data_processed/test_{idx + 1}.csv').iloc[:, :-1]
        y_test = df_test.iloc[:, -1]
        y_test_df = pd.concat([y_test_df, y_test], axis=1)
    y_test_df.columns = list(range(1, 41))

    prob_avg_df = pd.read_csv('predictions/GBM-LSTM_obs24_pred12_prob_balanced_3.csv')

    # Set up Subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    ax_roc = axs[0, 0]
    ax_prc = axs[0, 1]
    ax_cal = axs[1, 0]
    ax_dca = axs[1, 1]

    # ROC Curve
    roc_scores = []
    for idx in range(0, 40):
        prob = prob_avg_df.iloc[:, idx].dropna()
        y = y_test_df.iloc[:, idx].dropna()
        fpr, tpr, _ = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        roc_scores.append(roc_auc)
        ax_roc.plot(fpr, tpr, alpha=0.7)

    mean_auc = np.mean(roc_scores)
    auc_proxy = mlines.Line2D([], [], color='none', label=f'Mean AUC: {mean_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier')
    ax_roc.legend(handles=[ax_roc.lines[-1], auc_proxy], loc='lower right', fontsize=14)
    ax_roc.set_title('ROC Curve', fontsize=16)
    ax_roc.set_xlabel('False Positive Rate', fontsize=14)
    ax_roc.set_ylabel('True Positive Rate', fontsize=14)
    ax_roc.grid(True)
    ax_roc.tick_params(axis='both', labelsize=14)

    # Precision-Recall Curve
    avg_prec_scores = []
    for idx in range(0, 40):
        prob = prob_avg_df.iloc[:, idx].dropna()
        y = y_test_df.iloc[:, idx].dropna()
        precision, recall, _ = precision_recall_curve(y, prob)
        avg_prec = average_precision_score(y, prob)
        avg_prec_scores.append(avg_prec)
        ax_prc.plot(recall, precision, alpha=0.7)

    mean_avg_prec = np.mean(avg_prec_scores)
    avg_prec_proxy = mlines.Line2D([], [], color='none', label=f'Mean Avg Precision: {mean_avg_prec:.2f}')
    ax_prc.legend(handles=[avg_prec_proxy], loc='lower left', fontsize=14)
    ax_prc.set_title('Precision-Recall Curve', fontsize=16)
    ax_prc.set_xlabel('Recall', fontsize=14)
    ax_prc.set_ylabel('Precision', fontsize=14)
    ax_prc.grid(True)
    ax_prc.tick_params(axis='both', labelsize=14)

    # Calibration Plot
    brier_scores = []
    for idx in range(0, 40):
        prob = prob_avg_df.iloc[:, idx].dropna()
        y = y_test_df.iloc[:, idx].dropna()
        prob_true, prob_pred = calibration_curve(y, prob, n_bins=10, strategy='uniform')
        brier = brier_score_loss(y, prob)
        brier_scores.append(brier)
        ax_cal.plot(prob_pred, prob_true, marker='o', alpha=0.7)

    mean_brier = np.mean(brier_scores)
    brier_proxy = mlines.Line2D([], [], color='none', label=f'Mean Brier Score: {mean_brier:.2f}')
    ax_cal.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    ax_cal.legend(handles=[ax_cal.lines[-1], brier_proxy], loc='lower right', fontsize=14)
    ax_cal.set_title('Calibration Plot', fontsize=16)
    ax_cal.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax_cal.set_ylabel('Fraction of Positives', fontsize=14)
    ax_cal.grid(True)
    ax_cal.tick_params(axis='both', labelsize=14)

    # Decision Curve Analysis
    thresholds = np.linspace(0.01, 0.99, 100)
    all_net_benefits = []

    for idx in range(0, 40):
        prob = prob_avg_df.iloc[:, idx].dropna()
        y = y_test_df.iloc[:, idx].dropna()
        N = len(y)
        net_benefits = []

        for pt in thresholds:
            y_pred = (prob >= pt).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            net_benefit = (tp / N) - (fp / N) * (pt / (1 - pt))
            net_benefits.append(net_benefit)

        ax_dca.plot(thresholds, net_benefits, alpha=0.4)
        all_net_benefits.append(net_benefits)

    net_benefit_matrix = np.array(all_net_benefits)
    avg_net_benefit_across_sets = np.mean(net_benefit_matrix, axis=0)
    event_rate = np.mean(np.concatenate([y_test_df.iloc[:, idx].dropna().values for idx in range(41)]))

    # Treat all and none
    treat_all = event_rate - (1 - event_rate) * (thresholds / (1 - thresholds))
    ax_dca.plot(thresholds, treat_all, linestyle='--', color='green', label='Treat All')
    ax_dca.plot(thresholds, np.zeros_like(thresholds), linestyle='--', color='red', label='Treat None')

    # Add Net Benefit labels
    avg_net_benefit_01 = avg_net_benefit_across_sets[9]
    avg_net_benefit_02 = avg_net_benefit_across_sets[19]
    avg_net_benefit_03 = avg_net_benefit_across_sets[29]

    proxy1 = mlines.Line2D([], [], color='none', label=f'NB at 0.1: {avg_net_benefit_01:.2f}')
    proxy2 = mlines.Line2D([], [], color='none', label=f'NB at 0.2: {avg_net_benefit_02:.2f}')
    proxy3 = mlines.Line2D([], [], color='none', label=f'NB at 0.3: {avg_net_benefit_03:.2f}')

    ax_dca.legend(handles=[ax_dca.lines[-2], ax_dca.lines[-1], proxy1, proxy2, proxy3], loc='lower right', fontsize=14)
    ax_dca.set_title('Decision Curve Analysis', fontsize=16)
    ax_dca.set_xlabel('Threshold Probability', fontsize=14)
    ax_dca.set_ylabel('Net Benefit', fontsize=14)
    ax_dca.grid(True)
    ax_dca.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.savefig('plots/GBM-LSTM_obs24_pred12_ALL_PLOTS.png', dpi=300)
    plt.show()
