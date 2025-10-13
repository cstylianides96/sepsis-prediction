import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def create_data_per_patient():

    data = pd.read_csv("data_processed/sepsis3_processed.csv.gz", compression='gzip', header=0, index_col=None)

    cond, cond_per_adm = generate_cond(data)
    print('cond created')
    proc = generate_proc(data)
    print('proc created')
    out = generate_out(data)
    print('out created')
    chart = generate_chart(data)
    print('chart created')
    meds = generate_meds(data)
    print('meds created')

    los, data, hids, cond, meds, proc, out, chart = sepsis_length(data, cond, meds, proc, out, chart, include_time=4, predW=3)

    final_meds, final_proc, final_out, final_chart, los = make_equal_intervals(los, meds, proc, out, chart, bucket=1)
    create_csvs(hids, data, cond, final_meds, final_proc, final_out, final_chart, los, include_time=4, predW=3, impute='Median')
    print('DATA PER PATIENT AND LABELS CREATED')


def generate_cond(data):
    cond = pd.read_csv("data_processed/preproc_diag_icu_sepsis.csv.gz", compression='gzip', header=0, index_col=None)
    cond = cond[cond['stay_id'].isin(data['stay_id'])]
    cond_per_adm = cond.groupby('stay_id').size().max()

    return cond, cond_per_adm


def generate_proc(data):
    proc = pd.read_csv("data_processed/preproc_proc_icu_sepsis.csv.gz", compression='gzip', header=0, index_col=None)
    # Choose proc of stayids in sepsis3_processed.csv
    proc = proc[proc['stay_id'].isin(data['stay_id'])]
    # Remove where event time is before icu admission
    proc[['start_days', 'dummy', 'start_hours']] = proc['event_time_from_admit'].str.split(pat=' ', n=-1, expand=True)
    proc[['start_hours', 'min', 'sec']] = proc['start_hours'].str.split(pat=':', n=-1, expand=True)
    proc['start_time'] = pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
    proc = proc.drop(columns=['start_days', 'dummy', 'start_hours', 'min', 'sec'])
    proc = proc[proc['start_time'] >= 0]

    # Remove where event time is after discharge time
    proc = pd.merge(proc, data[['stay_id', 'los']], on='stay_id', how='left')
    proc['sanity'] = proc['los']-proc['start_time']
    proc = proc[proc['sanity'] > 0]
    del proc['sanity']

    return proc


def generate_out(data):
    out = pd.read_csv("data_processed/preproc_out_icu_sepsis.csv.gz", compression='gzip', header=0, index_col=None)
    # Choose out of stayids in sepsis3_processed.csv
    out = out[out['stay_id'].isin(data['stay_id'])]
    # Remove where event time is before icu admission
    out[['start_days', 'dummy', 'start_hours']] = out['event_time_from_admit'].str.split(pat=' ', n=-1, expand=True)
    out[['start_hours', 'min', 'sec']] = out['start_hours'].str.split(pat=':', n=-1, expand=True)
    out['start_time'] = pd.to_numeric(out['start_days'])*24+pd.to_numeric(out['start_hours'])
    out = out.drop(columns=['start_days', 'dummy', 'start_hours', 'min', 'sec'])
    out = out[out['start_time'] >= 0]

    # Remove where event time is after discharge time
    out = pd.merge(out,data[['stay_id', 'los']], on='stay_id', how='left')
    out['sanity'] = out['los']-out['start_time']
    out = out[out['sanity'] > 0]
    del out['sanity']

    return out


def generate_chart(data):
    chunksize = 5000000
    final = pd.DataFrame()
    for chart in tqdm(pd.read_csv("data_processed/preproc_chart_icu_sepsis.csv.gz", compression='gzip',
                                  header=0, index_col=None, chunksize=chunksize)):
        # Choose chart of stayids in sepsis3_processed.csv
        chart = chart[chart['stay_id'].isin(data['stay_id'])]
        # Remove where event time is before icu admission
        chart[['start_days', 'dummy', 'start_hours']] = chart['event_time_from_admit'].str.split(pat=' ', n=-1, expand=True)
        chart[['start_hours', 'min', 'sec']] = chart['start_hours'].str.split(pat=':', n=-1, expand=True)
        chart['start_time'] = pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
        chart = chart.drop(columns=['start_days', 'dummy', 'start_hours', 'min', 'sec', 'event_time_from_admit'])
        chart = chart[chart['start_time'] >= 0]

        # Remove where event time is after discharge time
        chart = pd.merge(chart, data[['stay_id', 'los']], on='stay_id', how='left')
        chart['sanity'] = chart['los']-chart['start_time']
        chart = chart[chart['sanity'] > 0]
        del chart['sanity']
        del chart['los']

        if final.empty:
            final = chart
        else:
            final = pd.concat([final, chart], axis=0, ignore_index=True)  # final.append(chart, ignore_index=True)

    # final.to_csv("data_processed/preproc_chart_icu_sepsis.csv.gz")
    # final = pd.read_csv("data_processed/preproc_chart_icu_sepsis.csv.gz", compression='gzip',
    #             header=0, index_col=None)
    # chart = final
    return final

def generate_meds(data):
    meds = pd.read_csv("data_processed/preproc_med_icu_sepsis.csv.gz", compression='gzip', header=0, index_col=None)
    # Remove where event start time is after event stop time
    meds[['start_days', 'dummy', 'start_hours']] = meds['start_hours_from_admit'].str.split(pat=' ', n=-1, expand=True)
    meds[['start_hours', 'min', 'sec']] = meds['start_hours'].str.split(pat=':', n=-1, expand=True)
    meds['start_time'] = pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
    meds[['start_days', 'dummy', 'start_hours']] = meds['stop_hours_from_admit'].str.split(pat=' ', n=-1, expand=True)
    meds[['start_hours', 'min', 'sec']] = meds['start_hours'].str.split(pat=':', n=-1, expand=True)
    meds['stop_time'] = pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
    meds = meds.drop(columns=['start_days', 'dummy', 'start_hours', 'min', 'sec'])
    meds['sanity'] = meds['stop_time']-meds['start_time']
    meds = meds[meds['sanity'] > 0]
    del meds['sanity']

    # Choose meds of stayids in sepsis3_processed.csv
    meds = meds[meds['stay_id'].isin(data['stay_id'])]
    meds = pd.merge(meds, data[['stay_id', 'los']], on='stay_id', how='left')

    # Remove where start time is after end of visit
    meds['sanity'] = meds['los']-meds['start_time']
    meds = meds[meds['sanity'] > 0]
    del meds['sanity']

    # Any stop_time after end of visit is set at end of visit
    meds.loc[meds['stop_time'] > meds['los'], 'stop_time'] = meds.loc[meds['stop_time'] > meds['los'], 'los']
    del meds['los']

    meds['rate'] = meds['rate'].apply(pd.to_numeric, errors='coerce')
    meds['amount'] = meds['amount'].apply(pd.to_numeric, errors='coerce')

    return meds


def sepsis_length(data, cond, meds, proc, out, chart, include_time, predW):
    los = 500 #max case label at 492 hours after adm

    # random controls chosen at the size of cohortcases*(totalcontrols/totalcases)
    # ratio_cases_contr = len(data.loc[data['label']==0])/len(data.loc[data['label']==1])
    # contr_index_list = data.loc[data['label']==0].index.values.tolist()
    # n_cases = len(data[data['hours_after_adm'] == (include_time+predW)])
    # rand_contr_ind = random.sample(population=contr_index_list, k=int(n_cases*ratio_cases_contr))
    # contr = data.iloc[rand_contr_ind, :].reset_index(drop=True)

    # All controls with LOS equal or more than obs win
    contr = data.loc[data['label'] == 0]
    contr = contr.loc[contr['los'] >= 24] #max obs win
    # cases: patients experiencing sepsis (include_time + predW)(>=7)hours after admission (minimum combination)
    cases = data[(data['hours_after_adm'] == (include_time+predW)) | (data['hours_after_adm'] > (include_time+predW))]
    data = pd.concat([contr, cases], axis=0)
    data = data.reset_index(drop=True)

    hids = data['stay_id'].unique()
    print(include_time, predW, len(hids))
    cond = cond[cond['stay_id'].isin(data['stay_id'])]
    #data['los'] = include_time

    # Choose events of stayids in cohort &
    # Remove any patient whose event started after end of obs win (after include_time)
    # MEDS
    meds = meds[meds['stay_id'].isin(data['stay_id'])]
    #meds = meds[meds['start_time'] <= include_time]
    #meds.loc[meds.stop_time > include_time, 'stop_time'] = include_time

    # PROCS
    proc = proc[proc['stay_id'].isin(data['stay_id'])]
    #proc = proc[proc['start_time'] <= include_time]

    # OUT
    out = out[out['stay_id'].isin(data['stay_id'])]
    #out = out[out['start_time'] <= include_time]

    # CHART
    chart = chart[chart['stay_id'].isin(data['stay_id'])]
    #chart = chart[chart['start_time'] <= include_time]

    return los, data, hids, cond, meds, proc, out, chart #new cases


def make_equal_intervals(los, meds, proc, out, chart, bucket):
    final_meds = pd.DataFrame()
    final_proc = pd.DataFrame()
    final_out = pd.DataFrame()
    final_chart = pd.DataFrame()

    meds = meds.sort_values(by=['start_time'])
    proc = proc.sort_values(by=['start_time'])
    out = out.sort_values(by=['start_time'])
    chart = chart.sort_values(by=['start_time'])

    t = 0
    # Resampling in bins of size=bucket
    for i in tqdm(range(1, los+1, bucket)):
        # MEDS
        # within bucket (interval, bin): mean of rate and amount, ignoring missing values
        sub_meds = (meds[(meds['start_time'] >= i) & (meds['start_time'] < i+bucket)]
                    .groupby(['stay_id', 'itemid', 'orderid']).agg({'stop_time': 'max', 'subject_id': 'max',
                                                                    'rate': np.nanmean, 'amount': np.nanmean}))
        sub_meds = sub_meds.reset_index()
        sub_meds['start_time'] = t
        sub_meds['stop_time'] = sub_meds['stop_time']/bucket
        if final_meds.empty:
            final_meds = sub_meds
        else:
            final_meds = pd.concat([final_meds, sub_meds], axis=0)  # final_meds.append(sub_meds)

        # PROC
        # within bucket (interval, bin): creates itemids per stay_id
        sub_proc = (proc[(proc['start_time'] >= i) & (proc['start_time'] < i+bucket)]
                    .groupby(['stay_id', 'itemid']).agg({'subject_id': 'max'}))
        sub_proc = sub_proc.reset_index()
        sub_proc['start_time'] = t
        if final_proc.empty:
            final_proc = sub_proc
        else:
            final_proc = pd.concat([final_proc, sub_proc], axis=0)  # final_proc.append(sub_proc)

        # OUT
        # within bucket (interval, bin): creates itemids per stay_id
        sub_out = (out[(out['start_time'] >= i) & (out['start_time'] < i+bucket)]
                   .groupby(['stay_id', 'itemid']).agg({'subject_id': 'max'}))
        sub_out = sub_out.reset_index()
        sub_out['start_time'] = t
        if final_out.empty:
            final_out = sub_out
        else:
            final_out = pd.concat([final_out, sub_out], axis=0)  # final_out.append(sub_out)

        # CHART
        # within bucket (interval, bin): mean of values, ignoring missing values
        sub_chart = (chart[(chart['start_time'] >= i) & (chart['start_time'] < i+bucket)]
                     .groupby(['stay_id', 'itemid']).agg({'valuenum': np.nanmean}))
        sub_chart = sub_chart.reset_index()
        sub_chart['start_time'] = t
        if final_chart.empty:
            final_chart = sub_chart
        else:
            final_chart = pd.concat([final_chart, sub_chart], axis=0)  # final_chart.append(sub_chart)

        t = t+1
    print("bucket", bucket)
    los = int(los/bucket)

    return final_meds, final_proc, final_out, final_chart, los


def create_csvs(hids, data, cond, meds, proc, out, chart, los, include_time, predW, impute):  # across time intervals

    hids = list(set(hids))
    print(los)
    labels_csv = pd.DataFrame(columns=['stay_id', 'label'])
    labels_csv['stay_id'] = pd.Series(hids)
    labels_csv['label'] = 0

    for hid in hids:
        grp = data[data['stay_id'] == hid]

        labels_csv.loc[labels_csv['stay_id'] == hid, 'label'] = int(grp['label'])
    labels_csv.to_csv('labels_sepsis.csv', index=False)

    for start in range(0, len(hids), 1000):
        end = start + 1000
        hids_chunk = hids[start:end]

        for hid in tqdm(hids_chunk):
            grp = data[data['stay_id'] == hid]
            demo_csv = grp[['age', 'gender', 'race', 'insurance']]
            if not os.path.exists("data_per_patient/"+str(hid)):
                os.makedirs("data_per_patient/"+str(hid))
            demo_csv.to_csv('data_per_patient/'+str(hid) + '/demo.csv', index=False)

            dyn_csv = pd.DataFrame()

            # MEDS (amount)
            feat = meds['itemid'].unique()
            df2 = meds[meds['stay_id'] == hid]
            if df2.shape[0] == 0:
                # if no stayid from sepsis3_processed.csv takes any medication, amount=0 (med itemids x time intervals)
                amount = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                amount = amount.fillna(0)
                amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])
            else:  # if stayids take medication, amount not 0
                # fill rate (med itemids x time intervals)
                rate = df2.pivot_table(index='start_time', columns='itemid', values='rate')
                # fill amount (meds itemids x time intervals)
                amount = df2.pivot_table(index='start_time', columns='itemid', values='amount')
                df2 = df2.pivot_table(index='start_time', columns='itemid', values='stop_time')
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.ffill()
                df2 = df2.fillna(0)  # ffill & fill with 0 for stop time

                rate = pd.concat([rate, add_df])
                rate = rate.sort_index()
                rate = rate.ffill()
                rate = rate.fillna(-1)  # ffill & fill with -1 for rate

                amount = pd.concat([amount, add_df])
                amount = amount.sort_index()
                amount = amount.ffill()
                amount = amount.fillna(-1)  # ffill & fill with -1 for  amount
                # print(df2.head())
                df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                df2[df2 > 0] = 1
                df2[df2 < 0] = 0
                rate.iloc[:, 0:] = df2.iloc[:, 0:]*rate.iloc[:, 0:]
                amount.iloc[:, 0:] = df2.iloc[:, 0:]*amount.iloc[:, 0:]
                # print(df2.head())

                feat_df = pd.DataFrame(columns=list(set(feat)-set(amount.columns)))
                amount = pd.concat([amount, feat_df], axis=1)

                amount = amount[feat]
                amount = amount.fillna(0)
                amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])

            if dyn_csv.empty:
                dyn_csv = amount
            else:
                dyn_csv = pd.concat([dyn_csv, amount], axis=1)

                print(hid, amount)

            # PROCS (binary)
            feat = proc['itemid'].unique()
            df2 = proc[proc['stay_id'] == hid]
            if df2.shape[0] == 0:
                # if no stayid from sepsis3_processed.csv has procedure, fill with 0 (proc itemids x time intervals)
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
            else:
                df2.loc[:, 'val'] = 1 #binary (no amount)
                df2 = df2.pivot_table(index='start_time', columns='itemid', values='val')
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)  # fill with 0 for proc
                df2[df2 > 0] = 1

                feat_df = pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])

            if dyn_csv.empty:
                dyn_csv = df2
            else:
                dyn_csv = pd.concat([dyn_csv, df2], axis=1)

            # OUT (binary)
            feat = out['itemid'].unique()
            df2 = out[out['stay_id'] == hid]
            if df2.shape[0] == 0:
                # if no stayid from sepsis3_processed.csv has output, fill with 0 (out itemids x time intervals)
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
            else:
                df2.loc[:, 'val']=1 #binary output values (no amount)
                df2 = df2.pivot_table(index='start_time', columns='itemid', values='val')
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)  # fill with 0 for output
                df2[df2 > 0] = 1

                feat_df = pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])

            if dyn_csv.empty:
                dyn_csv = df2
            else:
                dyn_csv = pd.concat([dyn_csv, df2], axis=1)

            # CHART (valuenum)
            feat = chart['itemid'].unique()
            df2 = chart[chart['stay_id'] == hid]
            if df2.shape[0] == 0:
                # if no stayid from sepsis3_processed.csv has chart, fill with 0 (chart itemids x time intervals)
                # val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat) -------------
                # val = val.fillna(0)-------------
                val = pd.DataFrame(np.nan, index=los, columns=feat) #-----------------
                val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])
            else:
                val = df2.pivot_table(index='start_time', columns='itemid', values='valuenum') #valuenum: numeric value (use this)

                df2.loc[:, 'val'] = 1 #binary for 'value'
                df2 = df2.pivot_table(index='start_time', columns='itemid', values='val') #value: text+numeric
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)
                df2[df2 > 0] = 1
                df2[df2 < 0] = 0

                val = pd.concat([val, add_df])
                val = val.sort_index()
                if impute == 'Mean':
                    val = val.ffill()
                    val = val.bfill()
                    val = val.fillna(val.mean())  # ffill, bfill, fill with mean for missing chart values
                elif impute == 'Median':
                    val = val.ffill()
                    val = val.bfill()
                    val = val.fillna(val.median())  # ffill, bfill, fill with median for missing chart values

                feat_df = pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val = pd.concat([val, feat_df], axis=1)

                val = val[feat]
                val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])

            if dyn_csv.empty:
                dyn_csv = val
            else:
                dyn_csv = pd.concat([dyn_csv, val], axis=1)

            # Save temporal data_processed to csv
            dyn_csv.to_csv('data_per_patient/'+str(hid) +'/dynamic.csv', index=False)

            # COND (binary)
            feat = cond['new_icd_code'].unique()
            grp = cond[cond['stay_id'] == hid]
            if grp.shape[0] == 0:
                # if stayid from sepsis3_processed.csv has no cond, fill with 0 (cond itemids x 1row)
                feat_df = pd.DataFrame(np.zeros([1, len(feat)]), columns=feat)
                grp = feat_df.fillna(0)
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
            else:
                grp.loc[:, 'val'] = 1 #binary
                grp = grp.drop_duplicates()
                grp = grp.pivot(index='stay_id', columns='new_icd_code', values='val').reset_index(drop=True)
                feat_df = pd.DataFrame(columns=list(set(feat)-set(grp.columns)))
                grp = pd.concat([grp, feat_df],  axis=1)
                grp = grp.fillna(0)  # fill with 0 for missing diagnoses
                grp = grp[feat]
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])

            grp.to_csv('data_per_patient/'+str(hid) +'/static.csv', index=False)
