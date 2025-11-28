import pandas as pd
import numpy as np
import os
import joblib
import time
from timeit import default_timer as timer
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings(action='ignore')



# modelnm = 'TimeGAN/'
# modelnm = 'QuantGAN/'
modelnm = 'SigCWGAN/'

dir_ = 'noise_path/' + modelnm
pathlist = os.listdir(dir_)
pathlist = [x for x in pathlist if '.pkl' in x]
pathlist = sorted(pathlist, key = lambda x: int(x.split('_')[-1].split('.')[0][5:]))
pathlist = [joblib.load((dir_ + x)).T for x in pathlist]




def GAN_simulation(rf_, div_, ivs_,  num_steps, m_path):

    mean = rf_
    cur_div = div_
    std_dev = ivs_

    simulations = m_path[:num_steps]
    simulations = ((mean - cur_div) - ((std_dev ** 2) / 2)) + simulations * std_dev
    simulations = simulations.cumsum(axis=0)

    return simulations


def price_cal(sprice, rf_, strike, cp_flag, num_steps, simulated_data):


    total_price = np.exp(simulated_data[-1]) * sprice

    up_sum = total_price * 1.01
    mid_sum = total_price * 1.0
    low_sum = total_price * 0.99


    path_length = num_steps

    discount = rf_
    if cp_flag =='C':
        up_sum = np.maximum(up_sum - strike, 0) * np.exp(- ( discount * path_length) )

        mid_sum = np.maximum(mid_sum - strike, 0) * np.exp(- (discount * path_length))

        low_sum = np.maximum(low_sum - strike, 0) * np.exp(- (discount * path_length))

        delta = min(1, (up_sum.mean() - low_sum.mean()) / (sprice * 0.02))

    else:
        up_sum = np.maximum(strike - up_sum, 0) * np.exp(- (discount * path_length))

        mid_sum = np.maximum(strike - mid_sum, 0) * np.exp(- (discount * path_length))

        low_sum = np.maximum(strike - low_sum, 0) * np.exp(- (discount * path_length))

        delta = max(-1, (up_sum.mean() - low_sum.mean()) / (sprice * 0.02))

    price = mid_sum.mean()

    return round(price,6), round(delta,6)


opdat = joblib.load('data/processed/opdat.pkl')

# call_list = opdat[opdat['cp_flag']=='C']['symbol'].unique()
put_list = opdat[opdat['cp_flag']=='P']['symbol'].unique()
num_simulations = 10000

# sb_data = opdat[opdat['symbol'].isin(call_list)].groupby('symbol')
sb_data = opdat[opdat['symbol'].isin(put_list)].groupby('symbol')
del opdat

all_price = []
all_delta = []
i=0
for sb, temp in tqdm(sb_data):

    check_date = pd.to_datetime(temp['exdate'].iloc[0])
    strike_ = temp['strike_price'].iloc[0]/1000
    cp_ = temp['cp_flag'].iloc[0]
    indexes_ = temp['index_price']
    dates_ = temp['date']
    rfs = temp['df']
    divs = temp['div']
    ivs = temp['iv_mean']
    Edates_ = temp['steps']
    date_model = zip(rfs, divs, ivs, Edates_, [pathlist[x] for x in temp['model_idx']])

    simulated_results = [

        GAN_simulation(rf_, div_, iv_, ns_, t_path)
        for rf_, div_, iv_, ns_, t_path in date_model
    ]
    date_model2 = zip(rfs, Edates_, indexes_, simulated_results)

    price_results = [
        price_cal(sprice_, rf_, strike_, cp_, ns_, sim_data)

        if len(sim_data) > 0 else np.nan

        for rf_, ns_, sprice_, sim_data in date_model2
    ]
    prc_res_ = [x[0] for x in price_results]
    delta_res_ = [x[1] for x in price_results]
    i+=1
    all_price.append(prc_res_)
    all_delta.append(delta_res_)
    temp.drop(temp.index, inplace=True)
    del sb, temp, check_date, strike_, cp_, dates_, price_results,\
        simulated_results, date_model, date_model2, prc_res_, delta_res_,\
        indexes_, Edates_

    if (i+1) % 100 == 0:
        # joblib.dump(all_price, ('results/pricing_results/Price/' +modelnm+ 'CALL/res' + str(i//100) + '.pkl') )
        # joblib.dump(all_delta, (
        #             'results/pricing_results/Delta/'+modelnm+ 'CALL/res' + str(i // 100) + '.pkl'))

        joblib.dump(all_price, ('results/pricing_results/Price/' + modelnm + 'PUT/res' + str(i // 100) + '.pkl'))
        joblib.dump(all_delta, (
                'results/pricing_results/Delta/' + modelnm + 'PUT/res' + str(i // 100) + '.pkl'))

        all_price.clear()
        all_delta.clear()
        gc.collect()

if len(all_price)>0:
    # joblib.dump(all_price, ('results/pricing_results/Price/' + modelnm + 'CALL/res' + str(1 + i // 100) + '.pkl'))
    # joblib.dump(all_delta, (
    #         'results/pricing_results/Delta/' + modelnm + 'CALL/res' + str(i // 100) + '.pkl'))

    joblib.dump(all_price, ('results/pricing_results/Price/' + modelnm + 'PUT/res' + str(1 + i // 100) + '.pkl'))
    joblib.dump(all_delta, (
            'results/pricing_results/Delta/' + modelnm + 'PUT/res' + str(i // 100) + '.pkl'))
