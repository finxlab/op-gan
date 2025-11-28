import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib
import warnings
import os
from arch.bootstrap import MCS

warnings.filterwarnings(action='ignore')


def price_results(opdat, flag):

    if flag =='C':
        flag_dir = 'CALL/'
    else:
        flag_dir = 'PUT/'

    u_dir = 'results/pricing_results/Price/'


    sblist = opdat[opdat['cp_flag'] == flag]['symbol'].unique()

    sb_data = opdat[opdat['symbol'].isin(sblist[0:])].groupby('symbol')
    p_label = []
    for sb, temp in sb_data:
        p_label.append(temp['price'])

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



opdat = joblib.load('data/processed/opdat.pkl')
spc = price_results(opdat, 'C')
spp = price_results(opdat, 'P')
bsm_c = joblib.load('results/pricing_results/Price/BSM/CALL/res.pkl')
spc = pd.concat([spc, bsm_c], axis=1).dropna()

bsm_p = joblib.load('results/pricing_results/Price/BSM/PUT/res.pkl')
spp = pd.concat([spp, bsm_p], axis=1).dropna()


joblib.dump(spc, 'results/all_results/PRICE/spc.pkl')
joblib.dump(spp, 'results/all_results/PRICE/spp.pkl')