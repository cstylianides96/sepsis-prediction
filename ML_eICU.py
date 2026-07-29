# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

from typing import Concatenate
import pandas as pd
from prompt_toolkit.input import Input
from sklearn.ensemble import GradientBoostingClassifier
from DL_balanced2 import preprocess_temporal
from model_evaluation import evaluate
from keras_core.metrics import AUC
import os
from keras_core.callbacks import EarlyStopping
from keras_core.callbacks import ModelCheckpoint
import pandas as pd
from keras_core import Model
from keras_core import Input
from keras_core.layers import LSTM, BatchNormalization, Dropout, Dense, Concatenate, MultiHeadAttention, GlobalAveragePooling1D
from keras_core.callbacks import EarlyStopping
from keras_core.callbacks import ModelCheckpoint
from keras_core.metrics import AUC
from keras_core.optimizers import Adam
from sklearn.model_selection import train_test_split
import joblib
import random
import numpy as np
import tensorflow as tf


# Avoid hard crashes from missing CUDA/cuDNN runtime libraries.
# Opt-in GPU usage only when explicitly requested:
#   SEPSIS_USE_GPU=1 /path/to/python /DL_balanced.py
if os.environ.get('SEPSIS_USE_GPU', '0') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Set random seed for reproducibility
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def ml_eICU(): 

    prob_ml_list = []
    auc_list = []
    sen_90_list = []
    spec_90_list = []
    precision_90_list = []
    npv_90_list = []
    sen_yuden_list = []
    spec_yuden_list = []
    precision_yuden_list = []
    npv_yuden_list = []
    thres_90_list = []
    thres_yuden_list = []
    for idx in range(0, 338): 
        print(idx+1, '/', 338)
        df_train_ml = pd.read_csv('/data_processed_eicu/train_' + str(idx+1) + '.csv')
        feats_ml = joblib.load('/models/GBM_feat_selection_feat50(40).pkl').feature_names_in_.tolist() + ['label']
        df_train_ml = df_train_ml[feats_ml]
        param_grid = {'learning_rate': 0.1, # median
                'n_estimators': 250, # mode
                'subsample': 0.9, # median
                'max_depth': 6, # mode
                'max_features': 1} # same for all
        model_ml = GradientBoostingClassifier(**param_grid, random_state=123)
        model_ml.fit(df_train_ml.iloc[:, :-1], df_train_ml.iloc[:, -1])
        df_test_ml = pd.read_csv('/data_processed_eicu/test_' + str(idx+1) + '.csv')
        df_test_ml = df_test_ml[feats_ml]

        prob_ml = model_ml.predict_proba(df_test_ml.iloc[:, :-1])[:, 1]
        auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, \
        thres_90, thres_yuden = evaluate(prob_ml, df_test_ml.iloc[:, -1], acc=False)
        # print(prob_ml)
        print(auc)
        prob_ml_list.append(prob_ml)
        auc_list.append(auc)
        sen_90_list.append(sen_90)
        spec_90_list.append(spec_90)
        precision_90_list.append(precision_90)
        npv_90_list.append(npv_90)
        sen_yuden_list.append(sen_yuden)
        spec_yuden_list.append(spec_yuden)
        precision_yuden_list.append(precision_yuden)
        npv_yuden_list.append(npv_yuden)
        thres_90_list.append(thres_90)
        thres_yuden_list.append(thres_yuden)
    print('mean auc:', sum(auc_list)/len(auc_list))  # mean auc: 0.8898847633136088
    prob_ml_df = pd.DataFrame(prob_ml_list).transpose()
    print(prob_ml_df)

    return prob_ml_df, auc_list, sen_90_list, spec_90_list, precision_90_list, npv_90_list, sen_yuden_list, spec_yuden_list, precision_yuden_list, npv_yuden_list, \
    thres_90_list, thres_yuden_list


def dl_eICU():

    prob_dl_list = []
    auc_list = []
    sen_90_list = []
    spec_90_list = []
    precision_90_list = []
    npv_90_list = []
    sen_yuden_list = []
    spec_yuden_list = []
    precision_yuden_list = []
    npv_yuden_list = []
    thres_90_list = []
    thres_yuden_list = []

    print(1, '/', 338)
    df_train_dl = pd.read_csv('/data_processed_eicu/train_' + str(1) + '_norm.csv') 
    feats_df = pd.read_csv('/data_processed/train_X_1_norm.csv').iloc[:, :-1].columns.tolist()
    df_train_dl = df_train_dl[feats_df]
    X_df_train_dl = df_train_dl.iloc[:, :-1]
    y = df_train_dl.iloc[:, -1]
    X_t = X_df_train_dl.iloc[:, :-10]
    print(X_t.columns)
    X_s = X_df_train_dl.iloc[:, -10:]
    print(X_s.columns)
    y_np = y.to_numpy()

    df_val_dl = pd.read_csv('/data_processed_eicu/val_' + str(1) + '_norm.csv')
    df_val_dl = df_val_dl[feats_df]
    X_df_val_dl = df_val_dl.iloc[:, :-1]
    y = df_val_dl.iloc[:, -1]
    X_t_val = X_df_val_dl.iloc[:, :-10]
    print(X_t_val.columns)  
    X_s_val = X_df_val_dl.iloc[:, -10:]
    print(X_s_val.columns)
    y_np_val = y.to_numpy()

    X_t_np = preprocess_temporal(X_t, 24)
    X_s_np = X_s.to_numpy()
    X_t_val_np = preprocess_temporal(X_t_val, 24)
    X_s_val_np = X_s_val.to_numpy()
    
    temporal_input = Input(shape=(X_t_np.shape[1], X_t_np.shape[2]), name='input_temporal')
    static_input = Input(shape=(X_s.shape[1],), name='input_static')

    x_static = Dense(32, activation='relu')(static_input)
    x_static = BatchNormalization()(x_static)
    x_static = Dropout(0.2)(x_static)

    x_temporal = temporal_input
    x_temporal = LSTM(32, return_sequences=True)(x_temporal)
    x_temporal = BatchNormalization()(x_temporal)
    x_temporal = LSTM(32, return_sequences=True)(x_temporal)
    x_temporal = BatchNormalization()(x_temporal)
    x_temporal = MultiHeadAttention(num_heads=3, key_dim=4)(x_temporal, x_temporal)
    residual = Dense(32)(temporal_input)
    x_temporal = x_temporal + residual
    x_temporal = GlobalAveragePooling1D()(x_temporal)

    x = Concatenate()([x_temporal, x_static])
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[temporal_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[AUC()])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('/models/LSTM_eICU_' + str(2) + '.keras', monitor='val_loss', save_best_only=True, mode='min') 
    model.fit([X_t_np, X_s_np],
            y_np,
            validation_data=([X_t_val_np, X_s_val_np], y_np_val),
            epochs=60,
            batch_size=32,
            callbacks=[checkpoint, early_stop],verbose=1)

    for idx in range(1, 338):
        print(idx+1, '/', 338)

        df_train_dl = pd.read_csv('/data_processed_eicu/train_' + str(idx+1) + '_norm.csv') 
        feats_df = pd.read_csv('/data_processed/train_X_1_norm.csv').iloc[:, :-1].columns.tolist()
        df_train_dl = df_train_dl[feats_df]
        X_df_train_dl = df_train_dl.iloc[:, :-1]
        y = df_train_dl.iloc[:, -1]
        X_t = X_df_train_dl.iloc[:, :-10]
        print(X_t.columns)
        X_s = X_df_train_dl.iloc[:, -10:]
        print(X_s.columns)
        y_np = y.to_numpy()
        
        df_val_dl = pd.read_csv('/data_processed_eicu/val_' + str(idx+1) + '_norm.csv') 
        df_val_dl = df_val_dl[feats_df]
        X_df_val_dl = df_val_dl.iloc[:, :-1]
        y = df_val_dl.iloc[:, -1]
        X_t_val = X_df_val_dl.iloc[:, :-10]
        print(X_t_val.columns)  
        X_s_val = X_df_val_dl.iloc[:, -10:]
        print(X_s_val.columns)
        y_np_val = y.to_numpy()
    
        X_t_np = preprocess_temporal(X_t, 24)
        X_s_np = X_s.to_numpy()
        X_t_val_np = preprocess_temporal(X_t_val, 24)
        X_s_val_np = X_s_val.to_numpy()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('/models/LSTM_eICU_' + str(2) + '.keras', monitor='val_loss', save_best_only=True, mode='min') 
        model.fit([X_t_np, X_s_np],
                y_np,
                validation_data=([X_t_val_np, X_s_val_np], y_np_val),
                epochs=60,
                batch_size=32,
                callbacks=[checkpoint, early_stop],verbose=1)

    for idx in range(338): 
        print(f'\n>>> Dataset chunk {idx + 1} / {338}')
        df_test_dl = pd.read_csv('/data_processed_eicu/test_' +str(idx+1)+ '_norm.csv')
        df_test_dl = df_test_dl[feats_df]
        X_df_test_dl = df_test_dl.iloc[:, :-1]
        y = df_test_dl.iloc[:, -1]
        X_t = X_df_test_dl.iloc[:, :-10]
        print(X_t.columns)
        X_s = X_df_test_dl.iloc[:, -10:]
        print(X_s.columns)
        y_np = y.to_numpy()

        X_t_np = preprocess_temporal(X_t, 24)
        X_s_np = X_s.to_numpy()

        prob_dl = model.predict([X_t_np, X_s_np])[:, 0]
        auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, \
        thres_90, thres_yuden = evaluate(prob_dl, y, acc=False)
        print(auc)

        prob_dl_list.append(prob_dl)
        auc_list.append(auc)
        sen_90_list.append(sen_90)
        spec_90_list.append(spec_90)
        precision_90_list.append(precision_90)
        npv_90_list.append(npv_90)
        sen_yuden_list.append(sen_yuden)
        spec_yuden_list.append(spec_yuden)
        precision_yuden_list.append(precision_yuden)
        npv_yuden_list.append(npv_yuden)
        thres_90_list.append(thres_90)
        thres_yuden_list.append(thres_yuden)

    print('mean auc:', sum(auc_list)/len(auc_list)) #mean auc: 0.7365152662721891
    prob_dl_df = pd.DataFrame(prob_dl_list).transpose()
    print(prob_dl_df)

    return prob_dl_df, auc_list, sen_90_list, spec_90_list, precision_90_list, npv_90_list, sen_yuden_list, spec_yuden_list, precision_yuden_list, npv_yuden_list, \
    thres_90_list, thres_yuden_list


def ensemble_eICU():

    results = pd.DataFrame(columns=['model', 'test_auc', 
                                'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                                'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden', 'test_npv_yuden'])
    prob_ml_df, auc_ml_list, sen_90_ml_list, spec_90_ml_list, precision_90_ml_list, npv_90_ml_list, sen_yuden_ml_list, spec_yuden_ml_list, precision_yuden_ml_list, npv_yuden_ml_list, \
    thres_90_ml_list, thres_yuden_ml_list = ml_eICU()
    prob_dl_df, auc_dl_list, sen_90_dl_list, spec_90_dl_list, precision_90_dl_list, npv_90_dl_list, sen_yuden_dl_list, spec_yuden_dl_list, precision_yuden_dl_list, npv_yuden_dl_list, \
    thres_90_dl_list, thres_yuden_dl_list = dl_eICU()

    results = pd.concat([
        pd.DataFrame({'model': 'ML_eICU_balanced_test', 
                      'test_auc': sum(auc_ml_list)/len(auc_ml_list), 'test_sen_90': sum(sen_90_ml_list)/len(sen_90_ml_list), 'test_spec_90': sum(spec_90_ml_list)/len(spec_90_ml_list), 'test_precision_90': sum(precision_90_ml_list)/len(precision_90_ml_list), 'test_npv_90': sum(npv_90_ml_list)/len(npv_90_ml_list),
                      'test_sen_yuden': sum(sen_yuden_ml_list)/len(sen_yuden_ml_list), 'test_spec_yuden': sum(spec_yuden_ml_list)/len(spec_yuden_ml_list), 'test_precision_yuden': sum(precision_yuden_ml_list)/len(precision_yuden_ml_list), 'test_npv_yuden': sum(npv_yuden_ml_list)/len(npv_yuden_ml_list)}, index=[0]),
        pd.DataFrame({'model': 'DL_eICU_balanced_test', 
                      'test_auc': sum(auc_dl_list)/len(auc_dl_list), 'test_sen_90': sum(sen_90_dl_list)/len(sen_90_dl_list), 'test_spec_90': sum(spec_90_dl_list)/len(spec_90_dl_list), 'test_precision_90': sum(precision_90_dl_list)/len(precision_90_dl_list), 'test_npv_90': sum(npv_90_dl_list)/len(npv_90_dl_list),
                      'test_sen_yuden': sum(sen_yuden_dl_list)/len(sen_yuden_dl_list), 'test_spec_yuden': sum(spec_yuden_dl_list)/len(spec_yuden_dl_list), 'test_precision_yuden': sum(precision_yuden_dl_list)/len(precision_yuden_dl_list), 'test_npv_yuden': sum(npv_yuden_dl_list)/len(npv_yuden_dl_list)}, index=[0])
    ], ignore_index=True)

    # Ensemble - soft voting - average of predictions
    prob_avg = (prob_ml_df + prob_dl_df) / 2
    print(prob_avg)

    auc_list = []
    sen_90_list = []
    spec_90_list = []
    precision_90_list = []
    npv_90_list = []
    sen_yuden_list = []
    spec_yuden_list = []
    precision_yuden_list = []
    npv_yuden_list = []
    thres_90_list = []
    thres_yuden_list = []

    for idx in range(0, 338): 
        print(idx+1, '/', 338)
        y = pd.read_csv('/data_processed_eicu/test_' +str(idx+1)+'.csv')['label']
        prob = prob_avg.iloc[:, idx]

        auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90, thres_yuden = evaluate(prob, y, acc=False)
        print(auc)
        auc_list.append(auc)
        sen_90_list.append(sen_90)
        spec_90_list.append(spec_90)
        precision_90_list.append(precision_90)
        npv_90_list.append(npv_90)
        sen_yuden_list.append(sen_yuden)
        spec_yuden_list.append(spec_yuden)
        precision_yuden_list.append(precision_yuden)
        npv_yuden_list.append(npv_yuden)
        thres_90_list.append(thres_90)
        thres_yuden_list.append(thres_yuden)

    results = pd.concat([results, pd.DataFrame({'model': 'ENSEMBLE_eICU_balanced_test', 
                            'test_auc': sum(auc_list)/len(auc_list), 'test_sen_90': sum(sen_90_list)/len(sen_90_list), 'test_spec_90': sum(spec_90_list)/len(spec_90_list), 'test_precision_90': sum(precision_90_list)/len(precision_90_list), 'test_npv_90': sum(npv_90_list)/len(npv_90_list),
                            'test_sen_yuden': sum(sen_yuden_list)/len(sen_yuden_list), 'test_spec_yuden': sum(spec_yuden_list)/len(spec_yuden_list), 'test_precision_yuden': sum(precision_yuden_list)/len(precision_yuden_list), 'test_npv_yuden': sum(npv_yuden_list)/len(npv_yuden_list)}, index=[0])], ignore_index=True)
    results.to_csv('/results/ENSEMBLE_results_eICU_balanced_test.csv', index=False)
