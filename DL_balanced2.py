# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import os
import random
import numpy as np

# Avoid hard crashes from missing CUDA/cuDNN runtime libraries.
# Opt-in GPU usage only when explicitly requested:
#   SEPSIS_USE_GPU=1 /path/to/python PIPELINE2/DL_balanced.py
if os.environ.get('SEPSIS_USE_GPU', '0') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import pandas as pd
from keras_core import Model
from keras_core import Input
from keras_core.layers import LSTM, Conv1D, Flatten, BatchNormalization, Dropout, Dense, MaxPool1D, Concatenate, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras_core.callbacks import EarlyStopping
from keras_core.callbacks import ModelCheckpoint
from keras_core.models import load_model
from keras_core.metrics import AUC
from keras_core.optimizers import Adam
import joblib


# Set random seed for reproducibility
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def split_temporal_static(obs_win):

    model_feats = joblib.load('PIPELINE2/models/GBM_feat_selection_feat50(40).pkl').feature_names_in_
    print(model_feats)
    temporal_selected = []
    static = []
    wins = list(range(obs_win))

    for col in model_feats:
        if ('_'in col) and col[-1].isdigit(): # windows, diff, ratio
            col = col.rsplit('_', 2)[0]
            temporal_selected.append(col)
        else:
            static.append(col)
    temporal_selected = list(set(temporal_selected)) #unique temporal
    print(temporal_selected)
    print(len(temporal_selected)) #16
    print(static)
    print(len(static)) #10

    temporal = [] #all timesteps of temporal
    for temp in temporal_selected:
        for w in wins:
            temporal.append(temp+'_'+str(w))
    print(temporal) 

    # all features
    temporal_static = temporal + static + ['label']
    df_train = pd.read_csv('PIPELINE2/data_processed/train_merged_imputed_flattened_aggregated_binary.csv')
    label = df_train.pop('label')
    df_train['label'] = label
    y_df_train = df_train.iloc[:, -1]

    df_val = pd.read_csv('PIPELINE2/data_processed/val_merged_imputed_flattened_aggregated_binary.csv')
    label = df_val.pop('label')
    df_val['label'] = label
    y_df_val = df_val.iloc[:, -1]

    df_test = pd.read_csv('PIPELINE2/data_processed/test_merged_imputed_flattened_aggregated_binary.csv')
    label = df_test.pop('label')
    df_test['label'] = label
    y_df_test = df_test.iloc[:, -1]

    X_df_train_temporal = df_train[temporal]
    X_df_train_static = df_train[static]
    X_df_val_temporal = df_val[temporal]
    X_df_val_static = df_val[static]
    X_df_test_temporal = df_test[temporal]
    X_df_test_static = df_test[static]

    # Reorder columns
    ordered_cols = sorted(X_df_train_temporal.columns, key=lambda x: (int(x.split('_')[-1]), x.split('_')[0]))
    X_df_train_temporal = X_df_train_temporal[ordered_cols]
    X_df_val_temporal = X_df_val_temporal[ordered_cols]
    X_df_test_temporal = X_df_test_temporal[ordered_cols]
    print(len(ordered_cols)) #16*24=384
    print(static) #10
    #
    X_df_train_temporal.to_csv('PIPELINE2/data_processed/train_X_temporal.csv', index=False)
    X_df_val_temporal.to_csv('PIPELINE2/data_processed/val_X_temporal.csv', index=False)
    X_df_test_temporal.to_csv('PIPELINE2/data_processed/test_X_temporal.csv', index=False)
    X_df_train_static.to_csv('PIPELINE2/data_processed/train_X_static.csv', index=False)
    X_df_val_static.to_csv('PIPELINE2/data_processed/val_X_static.csv', index=False)
    X_df_test_static.to_csv('PIPELINE2/data_processed/test_X_static.csv', index=False)
    y_df_train.to_csv('PIPELINE2/data_processed/train_y.csv', index=False)
    y_df_val.to_csv('PIPELINE2/data_processed/val_y.csv', index=False)
    y_df_test.to_csv('PIPELINE2/data_processed/test_y.csv', index=False)


def normalize():
    X_df_train_temporal = pd.read_csv('PIPELINE2/data_processed/train_X_temporal.csv')
    X_df_val_temporal = pd.read_csv('PIPELINE2/data_processed/val_X_temporal.csv')
    X_df_test_temporal = pd.read_csv('PIPELINE2/data_processed/test_X_temporal.csv')
    X_df_train_static = pd.read_csv('PIPELINE2/data_processed/train_X_static.csv')
    X_df_val_static = pd.read_csv('PIPELINE2/data_processed/val_X_static.csv')
    X_df_test_static = pd.read_csv('PIPELINE2/data_processed/test_X_static.csv')

    scaler_temporal = MinMaxScaler()
    scaler_static = MinMaxScaler()
    scaler_temporal.fit(X_df_train_temporal)
    scaler_static.fit(X_df_train_static)
    for df, name, scaler in [(X_df_train_temporal, 'train_X_temporal', scaler_temporal), (X_df_val_temporal, 'val_X_temporal', scaler_temporal), (X_df_test_temporal, 'test_X_temporal', scaler_temporal), 
                             (X_df_train_static, 'train_X_static', scaler_static), (X_df_val_static, 'val_X_static', scaler_static), (X_df_test_static, 'test_X_static', scaler_static)]:
        print(df.describe())
        df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)
        print(df_norm.describe())
        df_norm.to_csv('PIPELINE2/data_processed/' + name + '_norm.csv', index=False)


def df_sets_balanced():

    X_df_train_temporal = pd.read_csv('PIPELINE2/data_processed/train_X_temporal_norm.csv')
    X_df_val_temporal = pd.read_csv('PIPELINE2/data_processed/val_X_temporal_norm.csv')
    X_df_test_temporal = pd.read_csv('PIPELINE2/data_processed/test_X_temporal_norm.csv')
    X_df_train_static = pd.read_csv('PIPELINE2/data_processed/train_X_static_norm.csv')
    X_df_val_static = pd.read_csv('PIPELINE2/data_processed/val_X_static_norm.csv')
    X_df_test_static = pd.read_csv('PIPELINE2/data_processed/test_X_static_norm.csv')
    y_df_train = pd.read_csv('PIPELINE2/data_processed/train_y.csv')
    y_df_val = pd.read_csv('PIPELINE2/data_processed/val_y.csv')
    y_df_test = pd.read_csv('PIPELINE2/data_processed/test_y.csv')

    df_train = pd.concat([X_df_train_temporal, X_df_train_static, y_df_train], axis=1)
    df_train['index'] = range(0, len(df_train))
    df_val = pd.concat([X_df_val_temporal, X_df_val_static, y_df_val], axis=1)
    df_val['index'] = range(0, len(df_val))
    df_test = pd.concat([X_df_test_temporal, X_df_test_static, y_df_test], axis=1)
    df_test['index'] = range(0, len(df_test))

    cases_train = df_train.loc[df_train['label'] == 1]
    controls_train = df_train.loc[df_train['label'] == 0]
    cases_val = df_val.loc[df_val['label'] == 1]
    controls_val = df_val.loc[df_val['label'] == 0]
    cases_test = df_test.loc[df_test['label']==1]
    controls_test = df_test.loc[df_test['label']==0]

    splits_size_train, splits_size_val, splits_size_test =  len(cases_train), len(cases_val), len(cases_test)
    splits_train = [controls_train[i:i + splits_size_train] for i in range(0, len(controls_train), splits_size_train)]
    splits_val = [controls_val[i:i + splits_size_val] for i in range(0, len(controls_val), splits_size_val)]
    splits_test = [controls_test[i:i + splits_size_test] for i in range(0, len(controls_test), splits_size_test)]

    # temporal, static, y, index
    for idx, split in enumerate(splits_train):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_train = pd.concat([split, cases_train], axis=0).reset_index(drop=True)
        print(df_train)
        df_train.to_csv('PIPELINE2/data_processed/train_X_' + str(idx+1) + '_norm.csv', index=False) # delete files without norm (previous preprocessing)

    for idx, split in enumerate(splits_val):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_val = pd.concat([split, cases_val], axis=0).reset_index(drop=True)
        print(df_val)
        df_val.to_csv('PIPELINE2/data_processed/val_X_' + str(idx+1) + '_norm.csv', index=False)

    # temporal, static, y, index
    for idx, split in enumerate(splits_test):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_test = pd.concat([split, cases_test], axis=0).reset_index(drop=True)
        print(df_test)
        df_test.to_csv('PIPELINE2/data_processed/test_X_' + str(idx+1) + '_norm.csv', index=False)


def load_data(idx):

    DATA_DIR = 'PIPELINE2/data_processed'

    def read(file):
        return pd.read_csv(os.path.join(DATA_DIR, file))

    X_df_train = read(f'train_X_{idx + 1}_norm.csv').iloc[:, :-2] # temporal, static
    X_df_val = read(f'val_X_{idx + 1}_norm.csv').iloc[:, :-2] # temporal, static
    X_df_test = read(f'test_X_{idx + 1}_norm.csv').iloc[:, :-2] # temporal, static
    y_df_train = read(f'train_X_{idx + 1}.csv')['label']
    y_df_val = read(f'val_X_{idx + 1}.csv')['label']
    y_df_test = read(f'test_X_{idx + 1}.csv')['label']

    return X_df_train, X_df_val, X_df_test, y_df_train, y_df_val, y_df_test


def preprocess_temporal(X, obs_win):
    n_features = int(X.shape[1] / obs_win)
    print(n_features)
    return X.to_numpy().reshape(len(X), obs_win, n_features)

def build_model(model_name, X_train_t, X_train_s, obs_win, lr):
    X_t_np = preprocess_temporal(X_train_t, obs_win)
    temporal_input = Input(shape=(X_t_np.shape[1], X_t_np.shape[2]), name='input_temporal')
    static_input = Input(shape=(X_train_s.shape[1],), name='input_static')

    def apply_common(x):
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x

    x_static = apply_common(static_input)
    x_temporal = temporal_input

    if model_name == 'MLP':
        x_temporal = Flatten()(temporal_input)
        x_temporal = apply_common(x_temporal)

    elif model_name == 'LSTM':
        x_temporal = LSTM(32, return_sequences=True)(x_temporal)
        x_temporal = BatchNormalization()(x_temporal)
        x_temporal = LSTM(32, return_sequences=True)(x_temporal)
        x_temporal = BatchNormalization()(x_temporal)
        x_temporal = MultiHeadAttention(num_heads=3, key_dim=4)(x_temporal, x_temporal)
        residual = Dense(32)(temporal_input)
        x_temporal = x_temporal+residual
        # x_temporal = Flatten()(x_temporal)
        x_temporal = GlobalAveragePooling1D()(x_temporal)
      

    elif model_name == '1DCNN':
        x_temporal = Conv1D(32, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = Conv1D(32, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = Conv1D(32, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = MultiHeadAttention(num_heads=4, key_dim=4)(x_temporal, x_temporal)
        residual = Conv1D(32, kernel_size=1, padding='same')(temporal_input)
        x_temporal = x_temporal + residual
        x_temporal = Flatten()(x_temporal)

    elif model_name == 'TCN':
        for _ in range(3):
            x_temporal = Conv1D(16, 3, activation='relu', dilation_rate=2, padding='causal')(x_temporal)
            x_temporal = BatchNormalization()(x_temporal)
        # residual = Conv1D(16, kernel_size=1, padding='same')(temporal_input)
        # x_temporal = x_temporal + residual
        x_temporal = MultiHeadAttention(num_heads=3, key_dim=4)(x_temporal, x_temporal)
        x_temporal = Flatten()(x_temporal)

    elif model_name == '1DCNN-LSTM':
        x_temporal = Conv1D(32, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = Conv1D(8, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = Conv1D(8, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = LSTM(32, return_sequences=True)(x_temporal)
        x_temporal = BatchNormalization()(x_temporal)
        # x_temporal = LSTM(32, return_sequences=False)(x_temporal)
        # x_temporal = BatchNormalization()(x_temporal)
        x_temporal = MultiHeadAttention(num_heads=3, key_dim=4)(x_temporal, x_temporal)
        residual = Conv1D(32, kernel_size=1, padding='same')(temporal_input)
        x_temporal = x_temporal + residual
        # x_temporal = Flatten()(x_temporal)
        x_temporal = GlobalAveragePooling1D()(x_temporal)



    x = Concatenate()([x_temporal, x_static])
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[temporal_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[AUC()])
    return model


def fit(model_name, obs_win, lr, epochs, batch_size, model_try):

    results = pd.DataFrame(columns=['train_loss', 'train_auc', 'val_loss', 'val_auc']) #, 'test_loss', 'test_auc']) 

    print(f'\n>>> Dataset chunk {0 + 1} / {40}')
    X_df_train, X_df_val, X_df_test, y_df_train, y_df_val, y_df_test = load_data(0)
    X_train_t = X_df_train.iloc[:, :-10]
    print('temporal', X_train_t)
    X_train_s = X_df_train.iloc[:, -10:]
    X_t_np = preprocess_temporal(X_train_t, obs_win)
    X_s_np = X_train_s.to_numpy()
    y_np = y_df_train.to_numpy().ravel()

    X_val_t = X_df_val.iloc[:, :-10]
    X_val_s = X_df_val.iloc[:, -10:]
    X_t_val = preprocess_temporal(X_val_t, obs_win)
    X_s_val = X_val_s.to_numpy()
    y_val = y_df_val.to_numpy().ravel()

    model = build_model(model_name, X_train_t, X_train_s, obs_win, lr)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        [X_t_np, X_s_np],
        y_np,
        validation_data=([X_t_val, X_s_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],  # checkpoint],
        # class_weight=class_weights,
        verbose=1)

    train_loss = history.history['loss'][-1]
    train_auc = history.history['auc'][-1]
    val_loss = history.history['val_loss'][-1]
    val_auc = history.history['val_auc'][-1]

    results.loc[len(results)] = [train_loss, train_auc, val_loss, val_auc]

    for idx in range(1, 40):
        print(f'\n>>> Dataset chunk {idx + 1} / {40}')
        X_df_train, X_df_val, X_df_test, y_df_train, y_df_val, y_df_test = load_data(idx)
        X_train_t = X_df_train.iloc[:, :-10]
        X_train_s = X_df_train.iloc[:, -10:]
        X_t_np = preprocess_temporal(X_train_t, obs_win)
        X_s_np = X_train_s.to_numpy()
        y_np = y_df_train.to_numpy().ravel()

        X_val_t = X_df_val.iloc[:, :-10]
        X_val_s = X_df_val.iloc[:, -10:]
        X_t_val = preprocess_temporal(X_val_t, obs_win)
        X_s_val = X_val_s.to_numpy()
        y_val = y_df_val.to_numpy().ravel()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('PIPELINE2/models/' + str(model_name) + '_' + str(model_try) + '.keras', monitor='val_loss', save_best_only=True, mode='min') 
        #saves model as trained on last dataset with weights where val loss is minimum, every training continues from this, the final model is used for every test set predictions

        history = model.fit(
            [X_t_np, X_s_np],
            y_np,
            validation_data=([X_t_val, X_s_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            #class_weight=class_weights,
            verbose=1)

        train_loss = history.history['loss'][-1] #uses final epoch weights
        train_auc = history.history['auc'][-1]
        val_loss = history.history['val_loss'][-1]
        val_auc = history.history['val_auc'][-1]

        # best_epoch = np.argmin(history.history['val_loss'])
        # train_loss = history.history['loss'][best_epoch]
        # train_auc = history.history['AUC'][best_epoch]
        # val_loss = history.history['val_loss'][best_epoch]
        # val_auc = history.history['val_AUC'][best_epoch]

        results.loc[len(results)] = [train_loss, train_auc, val_loss, val_auc]

        if idx==39: # plot for last dataset, end of training/learning
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.legend()
            plt.title('Training vs Validation Loss')
            plt.savefig('PIPELINE2/plots/DL_results_balanced_' + model_name + '_' + str(model_try) + '.png')
            plt.show()

        # last epoch weights are used for each dataset evaluation (could report best weights)
        results.to_csv('PIPELINE2/results/DL_results_balanced_' + model_name + '_' + str(model_try) + '.csv', index=False)


def predict(model_name, obs_win, model_try):

    results = pd.read_csv('PIPELINE2/results/DL_results_balanced_' + model_name + '_' + str(model_try) + '.csv')
    test_loss_list = []
    test_auc_list = []
    probs = pd.DataFrame()
    for idx in range(40):
        print(f'\n>>> Dataset chunk {idx + 1} / {40}')
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(idx)
        X_test_t = X_test.iloc[:, :-10]
        X_test_s = X_test.iloc[:, -10:]
        X_t_np = preprocess_temporal(X_test_t, obs_win)
        X_s_np = X_test_s.to_numpy()
        y_np = y_test.to_numpy().ravel()

        # model with weights from lower val loss on final dataset is used for predictions on test set of all datasets (checkpoint callback)
        model_path = 'PIPELINE2/models/' + str(model_name) + '_' + str(model_try) + '.keras'
        print(f"Loading model from: {model_path}")
        model = load_model(
            model_path,
            compile=True,
            custom_objects={'AUC': AUC()})  # uses weights from lower loss on final val data

        test_loss, test_auc = model.evaluate(  # uses model with weights from lower loss on final val data
            [X_t_np, X_s_np], y_np, verbose=0)
        print("Returned metrics:", test_loss, test_auc) #averaged across batches (verbose returns metrics of last batch)
        # print("Metric names:", model.metrics_names)
        test_loss_list.append(test_loss)
        test_auc_list.append(test_auc)

        prob = pd.DataFrame(model.predict([X_t_np, X_s_np]))
        probs = pd.concat([probs, prob], axis=1)
    probs.columns = list(range(1, 41))
    probs.to_csv(
        'PIPELINE2/predictions/' + str(model_name) + '_' + str(model_try) + '_balanced_prob' + '.csv', index=False)

    results['test_loss'] = test_loss_list
    results['test_auc'] = test_auc_list
    print('Average Test Loss: ', results['test_loss'].mean().round(2))
    print('Average Test AUC: ', results['test_auc'].mean().round(2))

    results.to_csv('PIPELINE2/results/DL_results_balanced_' + model_name + '_' + str(model_try) + '.csv', index=False)


def run_dl(model_name, obs_win, lr, epochs, batch_size, model_try):
    split_temporal_static(obs_win)
    normalize()
    df_sets_balanced()
 
    fit(model_name=model_name, obs_win=obs_win, lr=lr, epochs=epochs, batch_size=batch_size, model_try=model_try)
    predict(model_name, obs_win, model_try)


# run_dl(model_name='1DCNN-LSTM', obs_win=24, lr=0.001, epochs=60, batch_size=32, model_try='9')
#  1dcnn10 auc 0.80, loss 0.56

# lstm 7 onwards new preprocessing
# 1dcnn 12 onwards new preprocessing
# tcn 6 onwards new preprocessing
# 1dcnn-lstm 6 onwards new preprocessing
