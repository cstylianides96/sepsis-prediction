# Author: Charithea Stylianides (c.stylianides@cyens.org.cy)

import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from keras_core import Model
from keras_core import Input
from keras_core.layers import LSTM, Conv1D, Flatten, BatchNormalization, Dropout, Dense, MaxPool1D, Concatenate, MultiHeadAttention
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras_core.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
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


def split_temporal_static(obs_win, pred_win):

    model_feats = joblib.load('models_set5_clean/GBM_obs24_pred12_feat80(70)_3.pkl').feature_names_in_
    print(model_feats)
    temporal_selected = []
    static = []
    wins = list(range(obs_win))

    for col in model_feats:
        if ('_'in col) and col[-1].isdigit():
            col = col.rsplit('_', 1)[0]
            temporal_selected.append(col)
        else:
            static.append(col)
    temporal_selected = list(set(temporal_selected)) #unique temporal
    print(temporal_selected)
    print(len(temporal_selected))
    print(static)
    print(len(static))

    temporal = [] #all timesteps of temporal
    for temp in temporal_selected:
        for w in wins:
            temporal.append(temp+'_'+str(w))
    #print(temporal, static)

    # all features
    temporal_static = temporal + static + ['label']
    cleandir = pd.read_csv('data_processed/obs24_pred12_train_3.csv').columns
    cols_not_in_cleandir = list(set(temporal_static)-set(cleandir))
    print(cols_not_in_cleandir)
    df_train_missdir = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_train.csv', usecols = cols_not_in_cleandir)
    cols_in_cleandir = list(set(temporal_static)-set(cols_not_in_cleandir))
    print(cols_in_cleandir)
    df_train_cleandir = pd.read_csv('data_processed/obs24_pred12_train_3.csv', usecols=cols_in_cleandir)

    df_train_all = pd.concat([df_train_missdir, df_train_cleandir], axis=1)
    label = df_train_all.pop('label')
    df_train_all['label'] = label
    y_df_train = df_train_all.iloc[:, -1]

    df_test_missdir = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_imputed_test.csv',
        usecols=cols_not_in_cleandir)
    df_test_cleandir = pd.read_csv('data_processed/obs24_pred12_test_3.csv', usecols=cols_in_cleandir)

    df_test_all = pd.concat([df_test_missdir, df_test_cleandir], axis=1)
    label = df_test_all.pop('label')
    df_test_all['label'] = label
    y_df_test = df_test_all.iloc[:, -1]

    X_df_train_temporal = df_train_all[temporal]
    X_df_train_static = df_train_all[static]
    X_df_test_temporal = df_test_all[temporal]
    X_df_test_static = df_test_all[static]

    encoded_mnar_charted = ['229627', '224145', '228006', '224322', '220073', '224310', '223771', '220058', '227627', '228381', '227716', '227989', '228184', '224654', '226499', '228866', '220060', '227537', '229677', '228162', '224318', '229626', '220765', '224921', '224153', '229247', '227628', '223767', '228369', '224314', '224951', '228004', '227990', '220074', '224752', '228185', '227717', '223768', '227543', '224144', '229365', '229236', '228872', '224954', '224311', '228178', '220292', '226457', '225980', '228620', '224918', '220293', '224846', '224562', '223772', '224845', '224315', '224152', '225183', '229364', '228151', '228368', '224317', '224309', '229363', '228723', '227538', '220072', '220063', '224952', '228182', '224920', '228179', '224191', '229262', '228377', '224751', '220059', '224953', '228005', '229669', '228180', '228382', '228621', '224842', '224154', '220061', '224919', '228374', '228159', '224917', '229235', '224652', '228370', '227546', '228724', '228392', '229280', '224150', '228158', '223773', '220066', '228375', '227775', '229668', '220056', '229277', '228152', '229248', '229263', '227066', '224149', '224916']
    itemids = pd.read_csv('data_raw/d_items.csv')[['itemid', 'linksto']]
    inputids = [str(item) for item in itemids.loc[itemids['linksto']=='inputevents']['itemid'].tolist()]
    print(inputids)
    print('encoded and input events:', encoded_mnar_charted+inputids)
    for col in X_df_train_temporal.columns:
        for event in encoded_mnar_charted+inputids:
            itemid = col.rsplit('_', 1)[0]
            if itemid==event:
                X_df_train_temporal[col]= X_df_train_temporal[col].apply(lambda x: 0 if x == 0 else 1)
                print(f"Converted to binary: {col}")

    for col in X_df_test_temporal.columns:
        for event in encoded_mnar_charted + inputids:
            itemid = col.rsplit('_', 1)[0]
            if itemid == event:
                X_df_test_temporal[col] = X_df_test_temporal[col].apply(lambda x: 0 if x == 0 else 1)
                print(f"Converted to binary: {col}")

    # Reorder columns
    ordered_cols = sorted(X_df_train_temporal.columns, key=lambda x: (int(x.split('_')[-1]), x.split('_')[0]))
    X_df_train_temporal = X_df_train_temporal[ordered_cols]
    X_df_test_temporal = X_df_test_temporal[ordered_cols]
    print(len(ordered_cols))
    print(static)
    #
    X_df_train_temporal.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_X_temporal.csv', index=False)
    X_df_test_temporal.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_X_temporal.csv', index=False)
    X_df_train_static.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_X_static.csv', index=False)
    X_df_test_static.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_X_static.csv', index=False),
    y_df_train.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_y.csv', index=False)
    y_df_test.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_y.csv', index=False)


def df_sets_balanced(obs_win, pred_win):

    X_df_train_temporal = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_X_temporal.csv')
    print(X_df_train_temporal.shape)
    X_df_test_temporal = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_X_temporal.csv')
    print(X_df_test_temporal.shape)
    X_df_train_static = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_X_static.csv')
    print(X_df_train_static.shape)
    X_df_test_static = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_X_static.csv')
    print(X_df_test_static.shape)
    y_df_train = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_y.csv')
    print(y_df_train.shape)
    y_df_test = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_y.csv')
    print(y_df_test.shape)

    df_train = pd.concat([X_df_train_temporal, X_df_train_static, y_df_train], axis=1)
    df_train['index'] = range(0, len(df_train))
    df_test = pd.concat([X_df_test_temporal, X_df_test_static, y_df_test], axis=1)
    df_test['index'] = range(0, len(df_test))

    cases_train = df_train.loc[df_train['label'] == 1]
    controls_train = df_train.loc[df_train['label'] == 0]
    cases_test = df_test.loc[df_test['label']==1]
    controls_test = df_test.loc[df_test['label']==0]

    splits_size_train, splits_size_test =  len(cases_train), len(cases_test)
    splits_train = [controls_train[i:i + splits_size_train] for i in range(0, len(controls_train), splits_size_train)]
    splits_test = [controls_test[i:i + splits_size_test] for i in range(0, len(controls_test), splits_size_test)]

    # temporal, static, y, index
    for idx, split in enumerate(splits_train):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_train = pd.concat([split, cases_train], axis=0).reset_index(drop=True)
        print(df_train)
        df_train.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_train_X_' + str(idx+1) + '.csv', index=False)

    # temporal, static, y, index
    for idx, split in enumerate(splits_test):
        print(f"Part {idx + 1}: {split}")
        split = split.reset_index(drop=True)
        df_test = pd.concat([split, cases_test], axis=0).reset_index(drop=True)
        print(df_test)
        df_test.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_test_X_' + str(idx+1) + '.csv', index=False)


def normalize(obs_win, pred_win):

    # temporal, static
    for idx in range(0, 41):
        print(idx + 1, '/', 41)

        X_df_train = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(
            pred_win) + '_train_X_' + str(idx + 1) + '.csv').iloc[:, :-2]  # remove index,y

        X_df_test = pd.read_csv('data_processed/obs' + str(obs_win) + '_pred' + str(
            pred_win) + '_test_X_' + str(idx + 1) + '.csv').iloc[:, :-2]  # remove index,y

        for df, name in [(X_df_train, 'train_X'), (X_df_test, 'test_X')]:
            print(df.describe())
            scaler = MinMaxScaler()
            scaler.fit(df)
            df=pd.DataFrame(scaler.transform(df), columns=df.columns)
            print(df.describe())
            df.to_csv('data_processed/obs' + str(obs_win) + '_pred' + str(pred_win) + '_' + name + '_' + str(idx + 1) + '_norm.csv', index=False)


def load_data(obs_win, pred_win, idx):

    DATA_DIR = 'data_processed'

    def read(file):
        return pd.read_csv(os.path.join(DATA_DIR, file))

    X_df_train = read(f'obs{obs_win}_pred{pred_win}_train_X_{idx + 1}_norm.csv') # temporal, static
    X_df_test = read(f'obs{obs_win}_pred{pred_win}_test_X_{idx + 1}_norm.csv') # temporal, static
    y_df_train = read(f'obs{obs_win}_pred{pred_win}_train_X_{idx + 1}.csv')['label']
    y_df_test = read(f'obs{obs_win}_pred{pred_win}_test_X_{idx + 1}.csv')['label']

    return X_df_train, X_df_test, y_df_train, y_df_test


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
        x_temporal = MultiHeadAttention(num_heads=2, key_dim=16)(x_temporal, x_temporal)
        residual = Dense(32)(temporal_input)
        x_temporal = x_temporal+residual
        x_temporal = Flatten()(x_temporal)

    elif model_name == '1DCNN':
        x_temporal = Conv1D(32, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = Conv1D(8, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = Conv1D(8, 3, activation='relu')(x_temporal)
        x_temporal = MaxPool1D(2)(x_temporal)
        x_temporal = MultiHeadAttention(num_heads=2, key_dim=4)(x_temporal, x_temporal)
        residual = Conv1D(8, kernel_size=1, padding='same')(temporal_input)
        x_temporal = x_temporal + residual
        x_temporal = Flatten()(x_temporal)

    elif model_name == 'TCN':
        for _ in range(2):
            x_temporal = Conv1D(32, 3, activation='relu', dilation_rate=2, padding='causal')(x_temporal)
            x_temporal = BatchNormalization()(x_temporal)
        residual = Conv1D(32, kernel_size=1, padding='same')(temporal_input)
        x_temporal = x_temporal + residual
        x_temporal = MultiHeadAttention(num_heads=2, key_dim=16)(x_temporal, x_temporal)
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
        x_temporal = LSTM(32, return_sequences=True)(x_temporal)
        x_temporal = BatchNormalization()(x_temporal)
        x_temporal = MultiHeadAttention(num_heads=2, key_dim=16)(x_temporal, x_temporal)
        residual = Conv1D(32, kernel_size=1, padding='same')(temporal_input)
        x_temporal = x_temporal + residual
        x_temporal = Flatten()(x_temporal)


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


def fit(model_name, obs_win, pred_win, lr, epochs, batch_size, model_try):

    results = pd.DataFrame(columns=['train_loss', 'train_auc', 'val_loss', 'val_auc']) #, 'test_loss', 'test_auc'])

    print(f'\n>>> Dataset chunk {0 + 1} / {40}')
    X_train, X_test, y_train, y_test = load_data(obs_win, pred_win, 0)
    X_train_t = X_train.iloc[:, :-33]
    X_train_s = X_train.iloc[:, -33:]
    X_t_np = preprocess_temporal(X_train_t, obs_win)
    X_s_np = X_train_s.to_numpy()
    y_np = y_train.to_numpy().ravel()

    model = build_model(model_name, X_train_t, X_train_s, obs_win, lr)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    X_t_train, X_t_val, X_s_train, X_s_val, y_train, y_val = train_test_split(
        X_t_np, X_s_np, y_np,
        test_size=0.3,
        stratify=y_np,
        random_state=42)

    history = model.fit(
        [X_t_train, X_s_train],
        y_train,
        validation_data=([X_t_val, X_s_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],  # checkpoint],
        # class_weight=class_weights,
        verbose=1)

    # if restore_best_weights=False
    train_loss = history.history['loss'][-1]
    train_auc = history.history['auc'][-1]
    val_loss = history.history['val_loss'][-1]
    val_auc = history.history['val_auc'][-1]

    results.loc[len(results)] = [train_loss, train_auc, val_loss, val_auc]

    for idx in range(1, 40):
        print(f'\n>>> Dataset chunk {idx + 1} / {40}')
        X_train, X_test, y_train, y_test = load_data(obs_win, pred_win, idx)

        X_train_t = X_train.iloc[:, :-33]
        X_train_s = X_train.iloc[:, -33:]
        X_t_np = preprocess_temporal(X_train_t, obs_win)
        X_s_np = X_train_s.to_numpy()
        y_np = y_train.to_numpy().ravel()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('models/' + str(model_name) + '_' + str(model_try) + '.keras', monitor='val_loss', save_best_only=True, mode='min') #final-run model used for test set performance
        #saves model as trained on last dataset with weights where val loss is minimum, this is the model used for every dataset predictions

        X_t_train, X_t_val, X_s_train, X_s_val, y_train, y_val = train_test_split(
            X_t_np, X_s_np, y_np,
            test_size=0.3,
            stratify=y_np,
            random_state=42)

        history = model.fit(
            [X_t_train, X_s_train],
            y_train,
            validation_data=([X_t_val, X_s_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            #class_weight=class_weights,
            verbose=1)

        train_loss = history.history['loss'][-1] #uses final weights
        train_auc = history.history['auc'][-1]
        val_loss = history.history['val_loss'][-1]
        val_auc = history.history['val_auc'][-1]

        # best_epoch = np.argmin(history.history['val_loss'])
        # train_loss = history.history['loss'][best_epoch]
        # train_auc = history.history['AUC'][best_epoch]
        # val_loss = history.history['val_loss'][best_epoch]
        # val_auc = history.history['val_AUC'][best_epoch]

        results.loc[len(results)] = [train_loss, train_auc, val_loss, val_auc]

        if idx==39:
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.legend()
            plt.title('Training vs Validation Loss')
            plt.savefig('plots/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_balanced_'
                        + model_name + '_' + str(model_try) + '.png')
            plt.show()

        results.to_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_balanced_'
                       + model_name + '_' + str(model_try) + '.csv', index=False)


def predict(model_name, obs_win, pred_win, model_try):

    results = pd.read_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_balanced_'
                   + model_name + '_' + str(model_try) + '.csv')
    test_loss_list = []
    test_auc_list = []
    probs = pd.DataFrame()
    for idx in range(40):
        print(f'\n>>> Dataset chunk {idx + 1} / {40}')
        X_train, X_test, y_train, y_test = load_data(obs_win, pred_win, idx)
        X_test_t = X_test.iloc[:, :-33]
        X_test_s = X_test.iloc[:, -33:]
        X_t_np = preprocess_temporal(X_test_t, obs_win)
        X_s_np = X_test_s.to_numpy()
        y_np = y_test.to_numpy().ravel()

        # model on final train-val dataset
        model_path = 'models/' + str(model_name) + '_' + str(model_try) + '.keras'
        print(f"Loading model from: {model_path}")
        model = load_model(
            model_path,
            compile=True,
            custom_objects={'AUC': AUC()})  # uses weights from lower loss on final val data

        test_loss, test_auc = model.evaluate(  # uses model with weights from lower loss on final val data
            [X_t_np, X_s_np], y_np, verbose=0)
        print("Returned metrics:", test_loss, test_auc) #averaged across batches (verbose return metrics of last batch)
        # print("Metric names:", model.metrics_names)
        test_loss_list.append(test_loss)
        test_auc_list.append(test_auc)

        prob = pd.DataFrame(model.predict([X_t_np, X_s_np]))
        probs = pd.concat([probs, prob], axis=1)
    probs.columns = list(range(1, 41))
    probs.to_csv(
        'predictions/' + str(model_name) + '_' + str(model_try) + '_obs' + str(obs_win) + '_pred' + str(pred_win) +
        '_balanced_prob' + '_3.csv', index=False)

    results['test_loss'] = test_loss_list
    results['test_auc'] = test_auc_list
    print('Average Test Loss: ', results['test_loss'].mean().round(2))
    print('Average Test AUC: ', results['test_auc'].mean().round(2))

    results.to_csv('results/obs' + str(obs_win) + '_pred' + str(pred_win) + '_results_balanced_'
                   + model_name + '_' + str(model_try) + '.csv', index=False)


def run_dl(model_name, obs_win, pred_win, lr, epochs, batch_size, model_try):
    split_temporal_static(obs_win, pred_win)
    df_sets_balanced(obs_win, pred_win)
    normalize(obs_win, pred_win)

    fit(model_name=model_name, obs_win=obs_win, pred_win=pred_win, lr=lr, epochs=epochs, batch_size=batch_size, model_try=model_try)
    predict(model_name, obs_win, pred_win, model_try)


#run_dl(model_name='1DCNN', obs_win=24, pred_win=12, lr=0.001, epochs=60, batch_size=32, model_try='2')
