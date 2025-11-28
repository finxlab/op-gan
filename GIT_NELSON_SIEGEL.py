import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(action='ignore')

def nelson_siegel(params, T):

    beta0, beta1, beta2, lam = params

    T_adjusted = np.where(T == 0, 1e-6, T)

    term1 = beta0
    term2 = beta1 * (1 - np.exp(-T_adjusted / lam)) / (T_adjusted / lam)
    term3 = beta2 * ((1 - np.exp(-T_adjusted / lam)) / (T_adjusted / lam) - np.exp(-T_adjusted / lam))

    return term1 + term2 + term3


def sse_function(params, maturities, yields):

    model_yields = nelson_siegel(params, maturities)
    sse = np.sum((yields - model_yields) ** 2)
    return sse

riskfree = pd.read_csv('data/raw/risk_free.csv', index_col = 0)
riskfree.index = pd.DatetimeIndex(riskfree.index)
riskfree = riskfree/100


maturities = np.array([21/252, 42/252, 63/365, 84/252, 126/252, 1, 2, 3, 5, 7, 10, 20, 30])

matdict = dict(zip(riskfree.columns, maturities))

daily_rates = []
for day_ in tqdm(range(riskfree.shape[0])):
    yields = riskfree.iloc[day_].dropna()

    # 추정할 파라미터들의 초기값 설정
    # 초기값에 따라 최적화 결과가 달라질 수 있으므로 적절한 값을 주는 것이 중요합니다.
    # 일반적으로 [장기, 단기, 중기, 수렴 속도]에 대한 추정치를 넣습니다.

    beta0_initial = yields[-1] # 가장 긴 만기(30년)의 수익률
    beta1_initial = yields['1 Yr'] - yields[-1] # 1년 만기 수익률 - 30년 만기 수익률
    beta2_initial = 0.0 # 일반적으로 0으로 시작
    lambda_initial = 1.0 # 1.0 ~ 3.0 사이의 값

    initial_params = np.array([beta0_initial, beta1_initial, beta2_initial, lambda_initial])

    temp_mat = [matdict[x] for x in yields.index]

    result = minimize(sse_function, initial_params, args=(temp_mat, yields), method='L-BFGS-B')

    if result.success:
        estimated_betas = result.x[:3]  # 베타 0, 1, 2 추출
        estimated_lambda = result.x[3]


        estimated_yields = nelson_siegel(result.x, np.array([m_/252 for m_ in range(2, 255)]))

        daily_rates.append(estimated_yields)
    else:
        print("최적화에 실패했습니다.")
        print(result.message)


yield_curve = pd.DataFrame(daily_rates)
yield_curve.columns = range(2,255)
yield_curve.index = riskfree.index

yield_curve.to_csv('data/processed/rfcurve.csv')
