import pandas as pd
import numpy as np
import joblib
import pickle
import time
from timeit import default_timer as timer
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import warnings
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from arch.bootstrap import MCS



def calculate_metrics(df):
    actual = df['real'].values
    model_columns = df.columns[1:]  # 'Actual'을 제외한 모델 컬럼

    performance_dict = {'RMSE': [], 'MAE': [], 'MAPE': []}

    for model in model_columns:
        preds = df[model].values
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        mape = mean_absolute_percentage_error(actual, preds)

        performance_dict['RMSE'].append(rmse)
        performance_dict['MAE'].append(mae)
        performance_dict['MAPE'].append(mape)

    performance_df = pd.DataFrame(performance_dict, index=model_columns)
    return performance_df.T



def strategic_sampling(df_options: pd.DataFrame, n_samples) -> pd.DataFrame:


    df_options = df_options.copy()

    df_options['money_bin'] = pd.qcut(
        df_options['moneyness'], q=5, labels=False, duplicates='drop'
    )
    df_options['T_bin'] = pd.qcut(
        df_options['steps'], q=5, labels=False, duplicates='drop'
    )

    valid_bins = df_options['money_bin'].nunique() * df_options['T_bin'].nunique()
    if valid_bins == 0:
        return pd.DataFrame()

    samples_per_cell = int(n_samples / valid_bins)

    df_options['2d_bin'] = df_options.apply(lambda row: (row['money_bin'], row['T_bin']), axis=1)

    def sample_from_2d_bin(group):
        if len(group) < samples_per_cell:
            return group
        return group.sample(samples_per_cell)

    sampled_df = df_options.groupby('2d_bin', group_keys=False).apply(sample_from_2d_bin)

    return sampled_df.drop(columns=['money_bin', 'T_bin', '2d_bin']).index


def test_indexes(option_dat, res_df, flag, n_samples):

    fdat = option_dat[option_dat['cp_flag'] == flag]

    res_list = []
    for sub_ in range(2014, 2024):
        temp = fdat[pd.to_datetime(fdat['date']).apply(lambda x:x.year) >= sub_]
        temp = temp[pd.to_datetime(temp['date']).apply(lambda x:x.year) < sub_ + 1]
        check = res_df[res_df.index.isin(temp.index)]
        check = temp.loc[check.index]
        np.random.seed(42)

        sample_indices = [list(strategic_sampling(check, n_samples)) for _ in range(20)]

        res_list.append(list(sample_indices))

    return res_list


def yearly_res(test_idx, res_df):


    res_list = []
    for i in range(len(test_idx)):
        temp_res = []
        for j in range(len(test_idx[0])):
            temp_idx = test_idx[i][j]
            # check = res_df.loc[temp_idx]
            check = res_df.loc[res_df.index.isin(temp_idx)]
            temp_per = calculate_metrics(check)
            temp_res.append(temp_per)
        res_list.append(temp_res)

    # all_idx = set([item for sublist in test_idx for item in sublist])
    # check = res_df.loc[all_idx]
    # temp_per = calculate_metrics(check)
    # res_list.append(temp_per)

    return res_list

def yearly_mcs(test_idx, res_df):

    res_list = []
    for i in range(len(test_idx)):
        temp_res = []
        for j in range(len(test_idx[0])):
            temp_idx = test_idx[i][j]
            check = res_df.loc[temp_idx]
            actual = check['real'].values

            model_columns = check.columns[1:]
            performance_dict = {'MSE': []}
            for model in model_columns:
                preds = check[model].values
                mse = mean_squared_error(actual, preds)
                performance_dict['MSE'].append(mse)
            temp_per = pd.DataFrame(performance_dict, index=model_columns).T
            # temp_per = calculate_metrics(check)
            temp_res.append(temp_per)
        temp_res = pd.concat(temp_res)

        temp_res = MCS(temp_res, size = 0.1, reps = 2000, block_size = None, seed = 42)

        temp_res.compute()


        temp_res = pd.DataFrame(temp_res.pvalues).T

        #temp_res = temp_res[['MC','TGAN','QGAN','SGAN','MC_N','TGAN_T','QGAN_T','SGAN_T']]
        res_list.append(temp_res)

    return res_list



opdat = joblib.load('data/processed/opdat.pkl')
spc = joblib.load('results/all_results/PRICE/spc.pkl').dropna()
spp = joblib.load('results/all_results/PRICE/spp.pkl').dropna()

call_index = test_indexes(opdat, spc, 'C', 25)
put_index = test_indexes(opdat, spp,'P', 25)


spc_y = yearly_res(call_index, spc.astype(float))
spc_y_m = [sum(ymean)/len(ymean) for ymean in spc_y]
spp_y = yearly_res(put_index, spp.astype(float))
spp_y_m = [sum(ymean)/len(ymean) for ymean in spp_y]

spc_y_m = pd.concat(spc_y_m)
spp_y_m = pd.concat(spp_y_m)

mcs_c = yearly_mcs(call_index, spc.dropna().astype(float))
mcs_p = yearly_mcs(put_index, spp.dropna().astype(float))



whole_call_s = [[samples for year in spc_y for samples in year]]
whole_put_s = [[samples for year in spp_y for samples in year]]
whole_samples = [whole_call_s[0] + whole_put_s[0]]

whole_mean_c = [sum(ymean)/len(ymean) for ymean in whole_call_s]
whole_mean_p = [sum(ymean)/len(ymean) for ymean in whole_put_s]
whole_mean = [sum(ymean)/len(ymean) for ymean in whole_samples]
whole_res = pd.concat([whole_mean_c[0], whole_mean_p[0], whole_mean[0]])

mcs_whole = pd.concat(yearly_mcs([[item for sublist in call_index for item in sublist]], spc.dropna().astype(float))+
                       yearly_mcs([[item for sublist in put_index for item in sublist]], spp.dropna().astype(float))+
                       yearly_mcs([[item for sublist in call_index for item in sublist] + [item for sublist in put_index for item in sublist]],
                                  pd.concat([spc.dropna().astype(float), spp.dropna().astype(float)])))