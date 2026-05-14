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


# Avoid hard crashes from missing CUDA/cuDNN runtime libraries.
# Opt-in GPU usage only when explicitly requested:
#   SEPSIS_USE_GPU=1 /path/to/python /DL_balanced.py
if os.environ.get('SEPSIS_USE_GPU', '0') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def ml_eICU(): 

    df_train_ml = pd.read_csv('/data_processed_eicu/train_df_balanced.csv')
    feats_ml = joblib.load('/models/GBM_feat_selection_feat50(40).pkl').feature_names_in_.tolist() + ['label']
    df_train_ml = df_train_ml[feats_ml]
    param_grid = {'learning_rate': 0.1, # median
            'n_estimators': 250, # mode
            'subsample': 0.9, # median
            'max_depth': 6, # mode
            'max_features': 1} # same for all
    model_ml = GradientBoostingClassifier(**param_grid, random_state=123)
    model_ml.fit(df_train_ml.iloc[:, :-1], df_train_ml.iloc[:, -1])
    df_test_ml = pd.read_csv('/data_processed_eicu/test_df_balanced.csv') 
    df_test_ml = df_test_ml[feats_ml]

    prob_ml = model_ml.predict_proba(df_test_ml.iloc[:, :-1])[:, 1]
    auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, \
    thres_90, thres_yuden = evaluate(prob_ml, df_test_ml.iloc[:, -1], acc=False)
    # print(prob_ml)
    print(auc) #0.82 , 0.82 balanced test

    return prob_ml, auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, \
    thres_90, thres_yuden


def dl_eICU():

    df_train_dl = pd.read_csv('/data_processed_eicu/train_df_norm_balanced.csv')
    feats_df = pd.read_csv('/data_processed/train_X_1_norm.csv').iloc[:, :-1].columns.tolist()
    df_train_dl = df_train_dl[feats_df]
    X_df_train_dl = df_train_dl.iloc[:, :-1]
    y = df_train_dl.iloc[:, -1]
    X_t = X_df_train_dl.iloc[:, :-10]
    print(X_t.columns)
    X_s = X_df_train_dl.iloc[:, -10:]
    print(X_s.columns)
    y_np = y.to_numpy()

    X_t_np = preprocess_temporal(X_t, 24)
    X_s_np = X_s.to_numpy()
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

    X_t_train, X_t_val, X_s_train, X_s_val, y_train, y_val = train_test_split(
        X_t_np,
        X_s_np,
        y_np,
        test_size=0.3,
        random_state=123,
        stratify=y_np,
        shuffle=True,
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('/models/LSTM_eICU_' + str(1) + '.keras', monitor='val_loss', save_best_only=True, mode='min') 
    model.fit([X_t_train, X_s_train],
            y_train,
            validation_data=([X_t_val, X_s_val], y_val),
            epochs=60,
            batch_size=32,
            callbacks=[checkpoint, early_stop],verbose=1)

    
    df_test_dl = pd.read_csv('/data_processed_eicu/test_df_norm_balanced.csv') 
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
    print(auc) #0.76657 #0.76682 balanced test

    return prob_dl, auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, \
    thres_90, thres_yuden
# train auc: 0.9836 - train loss: 0.1513 - val_auc: 0.9324 - val_loss: 0.4510
# 8 epochs




def ensemble_eICU():


    results = pd.DataFrame(columns=['model', 'test_auc', 
                                'test_sen_90', 'test_spec_90', 'test_precision_90','test_npv_90', 
                                'test_sen_yuden', 'test_spec_yuden', 'test_precision_yuden', 'test_npv_yuden', 
                                'thres_90', 'thres_yuden'])
    prob_ml, auc_ml, sen_90_ml, spec_90_ml, precision_90_ml, npv_90_ml, sen_yuden_ml, spec_yuden_ml, precision_yuden_ml, npv_yuden_ml, \
    thres_90_ml, thres_yuden_ml = ml_eICU()
    prob_dl, auc_dl, sen_90_dl, spec_90_dl, precision_90_dl, npv_90_dl, sen_yuden_dl, spec_yuden_dl, precision_yuden_dl, npv_yuden_dl, \
    thres_90_dl, thres_yuden_dl = dl_eICU()

    results = pd.concat([
        pd.DataFrame({'model': 'ML_eICU_balanced_test', 
                      'test_auc': auc_ml, 'sen_90': sen_90_ml, 'spec_90': spec_90_ml, 'precision_90': precision_90_ml, 'npv_90': npv_90_ml,
                      'sen_yuden': sen_yuden_ml, 'spec_yuden': spec_yuden_ml, 'precision_yuden': precision_yuden_ml, 'npv_yuden': npv_yuden_ml,
                      'thres_90': thres_90_ml, 'thres_yuden': thres_yuden_ml}, index=[0]),
        pd.DataFrame({'model': 'DL_eICU_balanced_test', 
                      'test_auc': auc_dl, 'sen_90': sen_90_dl, 'spec_90': spec_90_dl, 'precision_90': precision_90_dl, 'npv_90': npv_90_dl,
                      'sen_yuden': sen_yuden_dl, 'spec_yuden': spec_yuden_dl, 'precision_yuden': precision_yuden_dl, 'npv_yuden': npv_yuden_dl,
                      'thres_90': thres_90_dl, 'thres_yuden': thres_yuden_dl}, index=[0])
    ], ignore_index=True)

    y = pd.read_csv('/data_processed_eicu/test_df_norm_balanced.csv')['label']

    # Ensemble - soft voting - average of predictions
    prob = (prob_ml + prob_dl) / 2
    # print(prob)

    auc, sen_90, spec_90, precision_90, npv_90, sen_yuden, spec_yuden, precision_yuden, npv_yuden, thres_90, thres_yuden = evaluate(prob, y, acc=False)
    print(auc) #0.81891, 0.81827 balanced test

    results = pd.concat([results, pd.DataFrame({'model': 'ENSEMBLE_eICU_balanced_test', 
                            'test_auc': auc, 'sen_90': sen_90, 'spec_90': spec_90, 'precision_90': precision_90, 'npv_90': npv_90,
                            'sen_yuden': sen_yuden, 'spec_yuden': spec_yuden, 'precision_yuden': precision_yuden, 'npv_yuden': npv_yuden,
                            'thres_90': thres_90, 'thres_yuden': thres_yuden}, index=[0])], ignore_index=True)
    results.to_csv('/results/ENSEMBLE_results_eICU_balanced_test.csv', index=False)

