import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import copy
import os
import random
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

from mpl_toolkits.mplot3d import Axes3D
from numba import njit

warnings.filterwarnings(action='ignore')


def calculate_metrics(df):
    actual = df['real'].values
    model_columns = df.columns[1:]  # 'Actual'을 제외한 모델 컬럼

    performance_dict = {'RMSE': [], 'MAE': [], 'MAPE': []}

    for model in model_columns:
        preds = df[model].values
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        mape = mean_absolute_percentage_error(actual, preds)  # MAPE는 퍼센트로 계산

        performance_dict['RMSE'].append(rmse)
        performance_dict['MAE'].append(mae)
        performance_dict['MAPE'].append(mape)

    # 성능 데이터프레임 생성
    performance_df = pd.DataFrame(performance_dict, index=model_columns)
    return performance_df.T


def delta_results(opdat, flag):

    if flag =='C':
        flag_dir = 'CALL/'
    else:
        flag_dir = 'PUT/'

    u_dir = 'results/pricing_results/Delta/'


    sblist = opdat[opdat['cp_flag'] == flag]['symbol'].unique()

    sb_data = opdat[opdat['symbol'].isin(sblist[0:])].groupby('symbol')
    p_label = []
    for sb, temp in sb_data:
        p_label.append(temp['delta'])

    p_label = pd.concat(p_label)

    m_dir = ['TimeGAN/', 'QuantGAN/', 'SigCWGAN/', 'MC/']

    GAN_all = []
    for m_ in m_dir:
        file_dir = u_dir + m_ + flag_dir
        flist = os.listdir(file_dir)
        flist = sorted(flist, key=lambda x: int(x[:-4].split('s')[1]))
        GANres = []
        for fdir in flist:
            tres = joblib.load((file_dir + fdir))
            GANres = GANres + tres
        GANres = [item for sublist in GANres for item in sublist]
        GAN_all.append(GANres)
    GAN_all = pd.DataFrame(list(zip(*GAN_all)))


    resdf = pd.DataFrame(columns=['real','TGAN', 'QGAN', 'SGAN','MC'], index = p_label.index)

    resdf['real'] = p_label.values
    resdf.iloc[:len(GAN_all), 1:] = GAN_all.values

    return resdf


@njit
def find_valid_and_last(date_ord, product_id):
    n = len(date_ord)
    valid = np.zeros(n, dtype=np.bool_)


    for i in range(n - 1):
        if product_id[i] == product_id[i + 1]:
            delta = date_ord[i + 1] - date_ord[i]
            if delta in [1,2,3]:
                valid[i] = True

    for i in range(1, n):
        if product_id[i] != product_id[i - 1]:
            valid[i - 1] = True
    valid[n - 1] = True

    return valid


def get_valid_idx_numba_fast(df):

    date_ord = df["date"].values.astype("datetime64[D]").astype(np.int64)
    product_id = df["symbol"].astype("category").cat.codes.values.astype(np.int64)


    order = np.lexsort((date_ord, product_id))
    date_ord_sorted = date_ord[order]
    product_id_sorted = product_id[order]

    mask_sorted = find_valid_and_last(date_ord_sorted, product_id_sorted)

    mask = np.zeros(len(df), dtype=np.bool_)
    mask[order] = mask_sorted
    return df[mask]


def test_indexes(option_dat, res_df, flag):

    fdat = option_dat[option_dat['cp_flag'] == flag]


    res_list = []
    for sub_ in range(2014, 2024):
        temp = fdat[pd.to_datetime(fdat['date']).apply(lambda x:x.year) >= sub_]
        temp = temp[pd.to_datetime(temp['date']).apply(lambda x:x.year) < sub_ + 1]
        check = res_df[res_df.index.isin(temp.index)]
        check = temp.loc[check.index]
        np.random.seed(42)

        sample_indices = [list(strategic_sampling(check)) for _ in range(20)]

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


def strategic_sampling(df_options: pd.DataFrame) -> pd.DataFrame:


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
    samples_per_cell = int(2000 / valid_bins)


    df_options['2d_bin'] = df_options.apply(lambda row: (row['money_bin'], row['T_bin']), axis=1)

    def sample_from_2d_bin(group):
        if len(group) < samples_per_cell:
            return group
        return group.sample(samples_per_cell)

    sampled_df = df_options.groupby('2d_bin', group_keys=False).apply(sample_from_2d_bin)

    return sampled_df.drop(columns=['money_bin', 'T_bin', '2d_bin']).index



opdat = joblib.load('data/processed/opdat.pkl')

sdc = delta_results(opdat, 'C')
sdp = delta_results(opdat, 'P')

bsm_c = joblib.load('results/pricing_results/Delta/BSM/CALL/res.pkl')
sdc = pd.concat([sdc, bsm_c], axis=1).dropna()

bsm_p = joblib.load('results/pricing_results/Delta/BSM/PUT/res.pkl')
sdp = pd.concat([sdp, bsm_p], axis=1).dropna()



opdat = opdat[opdat['symbol'].map(opdat['symbol'].value_counts()) >= 2]
opdat = opdat.sort_values(by=['symbol','date'])
opdat2 = get_valid_idx_numba_fast(opdat)
opdat2 = opdat2[opdat2['symbol'].map(opdat2['symbol'].value_counts()) >= 2]
opdat2 = opdat2.sort_values(by=['symbol','date'])



sb_data = opdat2[opdat2['cp_flag']=='C'].groupby('symbol')
sample_index_c = []
sample_price_c = []
for sb, temp in tqdm(sb_data):
    sample_ = temp[['index_price']]
    sample_ = 1-(sample_.shift(-1)/sample_)
    sample_ = sample_.dropna().index
    temp2 = temp[['index_price', 'price']]
    temp2 = temp2.shift(-1) - temp2
    temp2 = temp2['price']/temp2['index_price']
    temp2 = temp2.loc[sample_]
    sample_index_c.extend(temp2.index)
    sample_price_c.extend(temp2.values)

sb_data = opdat2[opdat2['cp_flag']=='P'].groupby('symbol')
sample_index_p = []
sample_price_p = []
for sb, temp in tqdm(sb_data):
    sample_ = temp[['index_price']]
    sample_ = 1-(sample_.shift(-1)/sample_)
    sample_ = sample_.dropna().index
    temp2 = temp[['index_price', 'price']]
    temp2 = temp2.shift(-1) - temp2
    temp2 = temp2['price']/temp2['index_price']
    temp2 = temp2.loc[sample_]
    sample_index_p.extend(temp2.index)
    sample_price_p.extend(temp2.values)


call_sample = pd.DataFrame(index = sample_index_c, data = sample_price_c)
call_sample[['MC_N', 'TGAN_IV', 'QGAN_IV',
       'SGAN_IV', 'HESTON', 'BSM']] = sdc.loc[call_sample.index,[ 'MC_N', 'TGAN_IV', 'QGAN_IV',
       'SGAN_IV', 'HESTON', 'BSM']]
call_sample.columns = ['real','MC_N', 'TGAN_IV', 'QGAN_IV',
       'SGAN_IV', 'HESTON', 'BSM']

put_sample = pd.DataFrame(index = sample_index_p, data = sample_price_p)
put_sample[['MC_N', 'TGAN_IV', 'QGAN_IV',
       'SGAN_IV', 'HESTON', 'BSM']] = sdp.loc[put_sample.index, [ 'MC_N', 'TGAN_IV', 'QGAN_IV',
       'SGAN_IV', 'HESTON', 'BSM']]
put_sample.columns = ['real','MC_N', 'TGAN_IV', 'QGAN_IV',
       'SGAN_IV', 'HESTON', 'BSM']


call_sample = call_sample[call_sample['real'].between(-1,1, inclusive=True)]
put_sample = put_sample[put_sample['real'].between(-1,1, inclusive=True)]

call_res = calculate_metrics(call_sample)
put_res = calculate_metrics(put_sample)


cidx = test_indexes(opdat2, call_sample, 'C')
pidx = test_indexes(opdat2, put_sample, 'P')


sdc_y = yearly_res(cidx, call_sample.astype(float))
sdc_y_m = [sum(ymean)/len(ymean) for ymean in sdc_y]
sdp_y = yearly_res(pidx, put_sample.astype(float))
sdp_y_m = [sum(ymean)/len(ymean) for ymean in sdp_y]

sdc_y_m = pd.concat(sdc_y_m)
sdp_y_m = pd.concat(sdp_y_m)


mcs_c = yearly_mcs(cidx, call_sample.dropna().astype(float))
mcs_p = yearly_mcs(pidx, put_sample.dropna().astype(float))


a = latex_transform_paper2(sdc_y_m, pd.concat(mcs_c))
b = latex_transform_paper2(sdp_y_m, pd.concat(mcs_p))

# a.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/delta_res_call.xlsx')
# b.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/delta_res_put.xlsx')

whole_call_s = [[samples for year in sdc_y for samples in year]]
whole_put_s = [[samples for year in sdp_y for samples in year]]
whole_samples = [whole_call_s[0] + whole_put_s[0]]

whole_mean_c = [sum(ymean)/len(ymean) for ymean in whole_call_s]
whole_mean_p = [sum(ymean)/len(ymean) for ymean in whole_put_s]
whole_mean = [sum(ymean)/len(ymean) for ymean in whole_samples]
whole_res = pd.concat([whole_mean_c[0], whole_mean_p[0], whole_mean[0]])
mcs_whole = pd.concat(yearly_mcs([[item for sublist in cidx for item in sublist]], call_sample.dropna().astype(float))+
                       yearly_mcs([[item for sublist in pidx for item in sublist]], put_sample.dropna().astype(float))+
                       yearly_mcs([[item for sublist in cidx for item in sublist] + [item for sublist in pidx for item in sublist]],
                                  pd.concat([call_sample.dropna().astype(float), put_sample.dropna().astype(float)])))

wres = latex_transform_paper_w2(whole_res, ['Call', 'Put','ALL'], mcs_whole)


# wres.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/whole_delta.xlsx')
wres.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/whole_delta05.xlsx')

all_call = [item for year in cidx for sublist in year for item in sublist]

all_put = [item for year in pidx for sublist in year for item in sublist]


# mlist = [0.85 ,0.90, 0.95, 0.99, 1.01 ,1.05, 1.10 ,1.15]
# mlist = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
mlist = [0.85, 0.94, 0.98, 1.02 ,1.06 ,1.15]
spc_money = moneyness_performance(opdat, all_call, call_sample, mlist)
spc_money = pd.concat(spc_money)

spc_money_mcs = pd.concat(moneyness_mcs(opdat, all_call, call_sample, mlist))
spc_money_mcs = spc_money_mcs[['MC_N', 'TGAN_IV', 'QGAN_IV', 'SGAN_IV',
       'HESTON', 'BSM']]


spp_money = moneyness_performance(opdat, all_put, put_sample, mlist)
spp_money = pd.concat(spp_money)

spp_money_mcs = pd.concat(moneyness_mcs(opdat, all_put, put_sample, mlist))
spp_money_mcs = spp_money_mcs[['MC_N', 'TGAN_IV', 'QGAN_IV', 'SGAN_IV',
       'HESTON', 'BSM']]


cmres_m = latex_transform_moneyness_p(spc_money, spc_money_mcs, mlist)
pmres_m = latex_transform_moneyness_p(spp_money, spp_money_mcs, mlist)

# cmres_m.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/call_delta_money.xlsx')
# pmres_m.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/put_delta_money.xlsx')
cmres_m.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/call_delta_money05.xlsx')
pmres_m.to_excel('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_SUM/put_delta_money05.xlsx')



C:/Users/user/Downloads/HEDGEPAPER/REVISION/DELTA_RES/call_real_delta.pkl