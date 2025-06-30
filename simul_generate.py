import sys
import collections
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
import datetime
import gc
from functools import reduce
from scipy import linalg
from pathlib import Path
# SIMUL_GEN_PATH = Path(__file__).resolve()
# PROJ_ROOT_PATH = SIMUL_GEN_PATH.parent.parent.parent
# sys.path.insert(0, os.fspath(PROJ_ROOT_PATH))

class CFN(object):

    SCN = u'Сценарий'
    DAT = u'Отчетная дата'
    ZCY = u'Кривая бескупонной доходности (zcyc, годовое начисление)'
    SPT = u'Сценарная короткая ставка (shortrate, годовое начисление)'
    SPT_exp = u'Сценарная короткая ставка до периода принятия решения, далее ожидание (shortrate_exp, годовое начисление)'

    HPR = u'Сценарный рост цен на недвижимость'
    HPI = u'Сценарный индекс цен на недвижимость'
    SNG = u'Сезонный фактор'
    DFS = u'Сценарное дисконтирование'
    DFM = u'Рыночное дисконтирование'

        
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    for i in reversed(range(1, n)):
        ret = np.append(ret, a[-i:].mean())

    return ret
def round_df_dict(df, n=9):
    def get_round(val, s):
        if isinstance(val, (int, float)):
            return round(val, s)
        else:
            return val

    if isinstance(df, pd.DataFrame):
        for key in df.columns:
            df[key] = df[key].apply(get_round, args=(n,))

    elif isinstance(df, dict):
        def dict_round(cc, s):
            for k, v in cc.items():
                if isinstance(v, collections.abc.Mapping):
                    cc[k] = dict_round(cc.get(k, {}), n)
                else:
                    cc[k] = get_round(v, s)
            return cc

        df = dict_round(df, n)
    return df

def month_diff(dt1, dt2):
    return (dt1.year - dt2.year) * 12 + dt1.month - dt2.month

# y2000 = np.datetime64('2000-01-01', 'M')
# month = np.timedelta64(1, 'M')

def G_t(t, **kwargs):
    return G_t_fixed(t, kwargs['TAU'], kwargs['B0'], kwargs['B1'], kwargs['B2'], kwargs['G1'], kwargs['G2'],
                     kwargs['G3'], kwargs['G4'], kwargs['G5'], kwargs['G6'], kwargs['G7'], kwargs['G8'], kwargs['G9'])


def G_t_fixed(t, tau, b0, b1, b2, g1, g2, g3, g4, g5, g6, g7, g8, g9):
    '''
    k = 1.6
    a = np.zeros(9)
    b = np.zeros(9)

    a[0] = 0.0
    a[1] = 0.6
    for i in range(1, 8):
        a[i + 1] = a[i] + a[1] * k**i

    b[0] = a[1]
    for i in range(0, 8):
        b[i + 1] = b[i] * k
    '''

    a = np.array([0., 0.6, 1.56, 3.096, 5.5536, 9.48576, 15.777216, 25.8435456, 41.94967296])
    b = np.array([0.6, 0.96, 1.536, 2.4576, 3.93216, 6.291456, 10.0663296, 16.10612736, 25.769803776]) ** 2

    exp_t_tau = np.exp(-t / tau)
    return (
            b0
            + (b1 + b2) * (tau / t) * (1.0 - exp_t_tau)
            - b2 * exp_t_tau
            + g1 * np.exp(-(t - a[0]) ** 2 / b[0])
            + g2 * np.exp(-(t - a[1]) ** 2 / b[1])
            + g3 * np.exp(-(t - a[2]) ** 2 / b[2])
            + g4 * np.exp(-(t - a[3]) ** 2 / b[3])
            + g5 * np.exp(-(t - a[4]) ** 2 / b[4])
            + g6 * np.exp(-(t - a[5]) ** 2 / b[5])
            + g7 * np.exp(-(t - a[6]) ** 2 / b[6])
            + g8 * np.exp(-(t - a[7]) ** 2 / b[7])
            + g9 * np.exp(-(t - a[8]) ** 2 / b[8])
    )

def ZCYC(t, **kwargs):
    return G_t(t, **kwargs) * 0.0001

class CCIR(object):

    def __init__(self, start_date, n_year=30, n_sim=1000, delta=0, r_rate=ZCYC, forward_shift_months=0, cir_version='cir++', **kbd_params):
        # forward_shift months forward evaluation
        self.forward_shift = forward_shift_months

        # Шаг модели, 1 месяц
        self.dt = 1.0 / 12.0
        self.period_num = int(1.0 / self.dt)

        # Мера моделирования, Q (spot), T (forward)
        self.t_measure = float(False)

        self.cir_version = cir_version
        # Параметры сетки
        # Шаг модели
        self.dt = 1.0 / 12.0

        self.r_rate = r_rate
        self.r_args = kbd_params
        self.r_delta = float(delta) * 0.0001

        self.r = lambda t: self.r_rate(t, **self.r_args) + self.r_delta

        # --- Количество периодов всего (T), долгосрочной ставки (LT), количество симуляций (S) ---

        self.LT = 5 * self.period_num
        self.T = int(n_year * self.period_num) + self.LT + self.forward_shift

        self.T1 = self.T + 1
        self.S = n_sim
        self.P = 2

        self.eps = 1e-6
        # self.t = np.linspace(0.0, self.T * self.dt, self.T1)
        self.t = np.zeros(int(self.T1))
        for i in range(self.T):
            self.t[i + 1] = self.t[i] + self.dt

        # --- Множитель единиц измерения величин вне модели (в базе данных, up/downstream моделях)
        # --- относительно используемых в модели абсолютных величин, т.е. при U = 100 1% в базе записывается как 1.0
        self.U = 100.0

        # --- Результаты моделирования (X), вектор независимых переменных VAR (Y) ---
        self.X = np.zeros((self.T1, self.P + 1, self.S))
        self.X_exp = np.zeros((self.T1, self.P + 1, self.S))
        self.Y = np.zeros((5, self.S))
        self.xs = np.zeros((self.T1, self.S))
        self.xs_exp = np.zeros((self.T1, self.S))

        # --- Параметры VAR ---
        self.var_i = None
        self.var_a = None
        self.var_s = None
        self.Corr = None
        self.Chol = None

        # self.Z = np.zeros((self.T1, self.S))

        # --- Случайные шоки ---
        self.dz = np.zeros((self.T, self.P, self.S))

        # --- Параметры модели CIR ---
        self.cir_a = None
        self.cir_s = None
        self.cir_h = None
        self.a = None
        self.s = None
        self.h = None
        self.theta0 = None

        self.theta = np.zeros(self.T)
        self.dB = np.zeros((self.T1, self.T1))

        self.A = np.zeros(self.T1)
        self.B = np.zeros(self.T1)
        self.phi = np.zeros(self.T1)

        # Генерация сетки модели
        # self.t = np.linspace(0.0, self.T * self.dt, self.T1)
        self.t = np.zeros(self.T1)
        for i in range(self.T):
            self.t[i + 1] = self.t[i] + self.dt
        self.eps = 1e-6

        # Рыночные ставки
        self.rt = np.zeros(self.T1)
        self.f_mkt = np.zeros(self.T1)

        self.rt[0] = self.r(self.eps)
        self.rt[1:] = self.r(self.t[1:])

        self.f_mkt[0] = self.rt[0]
        self.f_mkt[1:] = self.f_market(self.t[1:])

        self.Start_Date = np.datetime64(start_date, 'M')
        self.M1_Date = np.timedelta64(1, 'M')
        self.Period_Date = np.arange(self.Start_Date, self.Start_Date + (self.T1 - self.forward_shift) * self.M1_Date,
                                     self.M1_Date)
        self.Period_Date_Max = self.Start_Date + (self.T1 - self.forward_shift - self.LT) * self.M1_Date - 1


    def set_cir(self, cir_a=None, cir_s=None, cir_ax=None, cir_sx=None, cir_tx=None):

        # дрифт
        if cir_a is not None:
            self.cir_a = cir_a
            self.a = cir_ax if cir_ax is not None else cir_a / self.dt

        # волатильность
        if cir_s is not None:
            self.cir_s = cir_s
            self.s = cir_sx if cir_sx is not None else cir_s * np.sqrt(self.period_num)

        self.theta0 = cir_tx

        # "cir_a"   : 0.068269,
        # "cir_s"   : 0.028034,

        # вспомогательные переменные
        
        self.ss = self.s * self.s
        self.cir_h = (self.cir_a * self.cir_a + 2.0 * self.cir_s * self.cir_s) ** 0.5
        self.h = (self.a * self.a + 2.0 * self.ss) ** 0.5

        # theta0 > sigma^2 / 2a для обеспечения положительности симуляционных ставок
        if self.theta0 is None:
            self.theta0 = np.max([self.ss / (2 * self.a), self.rt[0], (self.a + self.h) * self.rt[0] / (2 * self.a)])
        self.atheta = self.a * self.theta0

        # начальное значение для моделируемого ряда x(t), где r(t) = x(t) + phi(t) согласовано с рыночной кривой
        self.x0 = self.f_mkt[0]
        # self.theta0 * self.h / self.a / 2.0
        # a% от исходного значения играет роль общего роста внутренней кривой x(t) на всем горизонте


    def f_market(self, t):
        return self.r(t) + 0.5 * t * (self.r(t + self.eps) - self.r(t - self.eps)) / self.eps

    # CIR++

    def calc_phi_AB(self):

        # общие переменные
        h2 = self.h * 2.0
        a_h = self.a + self.h
        exp_ht = np.exp(np.dot(self.h, self.t))  # T+1
        exp_ht_1 = exp_ht - 1
        denominator = h2 + a_h * exp_ht_1

        # Калибровка на рыночную форвардную кривую
        # phi(t) = f_market(0, t) - f_cir(0, t),
        # f_mkt, f_cir - мгновенные t форварда в момент времени 0

        const_part = (2.0 * self.atheta * exp_ht_1) / denominator
        x_part = (h2 * h2 * exp_ht) / (denominator ** 2)
        f_cir = const_part + self.x0 * x_part

        if self.cir_version == 'cir++':
            self.phi = self.f_mkt - f_cir
        elif self.cir_version == 'cir':
            self.phi = np.zeros(self.T1)

        # Аффинная временная структура
        # P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

        # self.A(T-t) = A(t, T)
        self.A = (h2 * np.exp(np.dot(0.5 * a_h, self.t)) / denominator) ** (2 * self.atheta / self.ss)

        # self.B(T-t) = B(t, T)
        self.B = 2 * exp_ht_1 / denominator

        return 1

    # MC симуляция для модели CIR++
    def run_cirpp_mc(self, ds_period=None):
        self.calc_phi_AB()

        self.xs[0][:] = self.X[0][0, :] - self.phi[0]  # = self.x0
        self.xs_exp[0][:] = self.xs[0][:]

        for i in range(1, self.T1):
            # Схема Эйлера, симуляция в Q (current account) или T (zero coupon bond) мере

            dz_per_dt = self.dz[i - 1][0, :] / np.sqrt(self.period_num)
            dz_part = self.s * self.xs[i - 1][:] ** 0.5 * dz_per_dt
            # dz2_part  = 0.25 * self.ss * (dz_per_dt * dz_per_dt - self.dt) # Milstein = Euler + dz2_part, to test vs Euler
            self.xs[i][:] = self.xs[i - 1] + \
                            (self.atheta - (self.a + self.B[self.T - (i - 1)] * self.ss * self.t_measure) * self.xs[
                                i - 1]) * self.dt + \
                            dz_part  # + dz2_part
            if ds_period is None:
                pass
            else:
                if i < ds_period:
                    self.xs_exp[i][:] = self.xs[i][:]
                else:
                    self.xs_exp[i][:] = self.xs_exp[i - 1] + \
                                    (self.atheta - (self.a + self.B[self.T - (i - 1)] * self.ss * self.t_measure) * self.xs_exp[
                                        i - 1]) * self.dt

            # Ограничиваем модельные безрисковые ставки снизу eps
            self.X[i][0, :] = np.maximum(self.xs[i] + self.phi[i], self.eps)
            self.X_exp[i][0, :] = np.maximum(self.xs_exp[i] + self.phi[i], self.eps)

        return 1


    def Run(self, mir_k, new_seed=None, scr_seeds=None, method='cir++', ds_period=None):

        # Генерация шоков
        if scr_seeds is None:
            np.random.seed(new_seed)
            self.dz[:] = np.random.normal(size=(self.T, self.P, self.S))
        else:
            for i in range(self.S):
                np.random.seed(scr_seeds[i])
                self.dz[:, :, i] = np.random.normal(size=(self.T, self.P))

        # Начальные значения
        self.X[0][0, :] = self.rt[0]  # short rate
        
        self.X_exp[0][0, :] = self.rt[0]  # short rate

        # Симуляции Монте-Карло
        self.run_cirpp_mc(ds_period)

        # Переход к форвардной оценке: забываем симуляции до форвадной даты, укорачиваем время
        if self.forward_shift > 0:
            self.T -= self.forward_shift
            self.T1 -= self.forward_shift
            self.t = self.t[:-self.forward_shift]
            self.rt = self.rt[self.forward_shift:]
            self.A = self.A[self.forward_shift:]
            self.B = self.B[self.forward_shift:]
            self.phi = self.phi[self.forward_shift:]
            self.X = self.X[self.forward_shift:, :, :]
            self.X_exp = self.X_exp[self.forward_shift:, :, :]

        # --- Подготовка результатов для модели досрочных погашений ---
        # Ряды для каждого из MC сценариев c шагом self.dt по времени
        size_S_T1 = (self.S, self.T1)

        # S(t) - сценарная реализованная короткая ставка (годовое начисление)
        St = np.zeros(size_S_T1)
        St[:] = np.exp(self.X[:, 0, :].T) - 1.0

        # S_exp(t) - сценарная реализованная короткая ставка до ds_period, далее ожидаемая ставка (годовое начисление)
        St_exp = np.zeros(size_S_T1)
        St_exp[:] = np.exp(self.X_exp[:, 0, :].T) - 1.0


        # G(t) - сценарная реализованная доходность (непрерывное начисление)
        Gt = np.zeros(size_S_T1)
        Gt[:, 0] = self.X[0][0, :]
        for i in range(1, self.T1):
            Gt[:, i] = (self.t[i - 1] * Gt[:, i - 1] + self.dt * self.X[i][0, :]) / self.t[i]

        # Y(t) - сценарная реализованная доходность (годовое начисление)
        Yt = np.zeros(size_S_T1)
        Yt[:] = np.exp(Gt[:, :]) - 1.0

        # D(t) - сценарное реализованное дисконтирование
        Dt = np.zeros(size_S_T1)
        Dt[:] = (1.0 + Yt[:, :]) ** -self.t

        # L(t) - сценарная реализованная доходность за LT / self.period_num лет вперед (непрерывное начисление)
        Lt = np.zeros(size_S_T1)
        for j in range(0, self.T - self.LT + 1):
            Lt[:, j] = np.sum(self.X[j + 1: j + self.LT + 1, 0, :], axis=0) / self.LT

        # R(t) - спот ставка в момент времени t_i = i * self.dt на LT / self.period_num лет (непрерывное начисление)
        if method == 'cir++':
            Rt = np.zeros(size_S_T1)
            t_range = range(self.T - self.LT + 1)
            ln_up = Dt[:, t_range] * self.A[self.LT:] * np.exp(-self.B[self.LT:] * self.x0)
            ln_down = Dt[:, self.LT:] * self.A[self.LT] * self.A[t_range] * np.exp(-self.B[t_range] * self.x0)
            Rt[:, t_range] = (np.log(ln_up / ln_down) - self.B[self.LT] * (self.phi[t_range] - self.rt[t_range])) / \
                             self.t[self.LT]

        # Z(t) - рыночная безрисковая доходность (годовое начисление)
        Zt = np.zeros(size_S_T1)
        Zt[:] = np.exp(self.rt.reshape((1, -1))) - 1.0

        # DZt - рыночное дисконтирование
        DZt = np.zeros(size_S_T1)
        DZt[:] = (1.0 + Zt[:, :]) ** -self.t

        # Структура результатов модели процентных ставок, вход для модели досрочных погашений
        df = pd.DataFrame({
            CFN.SCN: np.repeat(np.arange(self.S), self.T1),  # np.array([range(self.S)] * self.T1).T.ravel(),
            CFN.DAT: np.tile(self.Period_Date, self.S),  # self.Period_Date.tolist() * self.S,
            CFN.ZCY: Zt.ravel() * self.U,
            CFN.SPT: St.ravel() * self.U, 
            CFN.SPT_exp: St_exp.ravel() * self.U,

            CFN.SNG: np.concatenate([(np.mod((self.Period_Date - np.datetime64('2000-01-01', 'M')).astype(int),
                                             12) + 1).astype(float)] * self.S),
            CFN.DFS: Dt.ravel(),
            CFN.DFM: DZt.ravel()
        })
        df.set_index([CFN.SCN, CFN.DAT], inplace=True)
        df.sort_index(inplace=True)
        df = df.loc[pd.IndexSlice[:, :self.Period_Date_Max], :].copy(deep=True)
        return df, self.dz


def f_rates(t, r):
    rates = np.zeros(len(t))
    rates[0] = r[0]
    for i in range(1, len(r)):
        rates[i] = (r[i] * t[i] - r[i - 1] * t[i - 1]) / (t[i] - t[i - 1])
    return rates


def run(tt, shock=0, seed=None, cir_version='cir++', ds_period=None,
        number_of_monte_carlo_scenarios=None,
        history=None,
        mortgage_rates_models=None,
        zcyc_values=None, 
        use_standart_zcyc=None,
        evaluation_date=None,
        cir_sx=None,
        cir_a=None,
        cir_s=None,
        cir_ax=None,
        cir_tx=None,
        coefficients=None,
        bet_models=1.0,):    
    """
    Функция расчета CIR модели с явной передачей параметров
    
    Parameters:
    -----------
    tt : int
        Горизонт расчета
    shock : float
        Шок ставки в базисных пунктах
    seed : int
        Seed для генератора случайных чисел
    cir_version : str
        Версия CIR модели ('cir++' или 'cir') 
    ds_period : int
        Период принятия решения
    number_of_monte_carlo_scenarios : int
        Количество Монте-Карло сценариев
    history : pd.DataFrame
        Исторические данные
    mortgage_rates_models : float
        Параметр модели ставок по ипотеке
    zcyc_values : dict
        Значения кривой доходности
    use_standart_zcyc : bool
        Использовать стандартную кривую
    evaluation_date : datetime
        Дата оценки
    cir_sx, cir_a, cir_s, cir_ax, cir_tx : float
        Параметры CIR модели
    coefficients : dict
        Коэффициенты модели
    bet_models : float
        Параметр бета модели

    """
    
    # Обработка шока ставки
    if shock != 0 and shock is not None:
        print(f'Shocking CIR by {shock} b.p.')
        shock = shock / 10000
        
        # Создаем CIR без шока для получения базовых ставок
        cir = CCIR(
            start_date=evaluation_date,
            n_year=25,
            n_sim=number_of_monte_carlo_scenarios,
            delta=0,
            r_rate=ZCYC if use_standart_zcyc else np.interp,
            cir_version=cir_version,
            **coefficients #._asdict()
        )
        
        fwrd_rates = f_rates(cir.t, cir.rt)
        fwrd_rates += shock

        ZCYCValues = []
        for val in zip(cir.t, zcyc_rates(cir.t, fwrd_rates)):
            ZCYCValues.append({"X": val[0], "Y": val[1] * 100})
        # cir.free()
        
        zcyc_values = ZCYCValues
        use_standart_zcyc = False

    # Создаем основной CIR объект
    cir = CCIR(
        start_date=evaluation_date,
        n_year=tt,
        n_sim=number_of_monte_carlo_scenarios,
        delta=0,
        r_rate=ZCYC if use_standart_zcyc else np.interp,
        cir_version=cir_version,
        **coefficients #._asdict()
    )

    # Устанавливаем параметры CIR
    cir.set_cir(
        cir_a=cir_a,
        cir_s=cir_s * bet_models,
        cir_ax=cir_ax,
        cir_sx=cir_sx,
        cir_tx=cir_tx
    )

    # Запускаем расчет
    arr, dz = cir.Run(mortgage_rates_models, new_seed=seed, method='cir++', ds_period=ds_period)

    # Очистка памяти
    # cir.free()

    # Подготовка результатов
    arr['period'] = arr.index.codes[1]
    arr = arr[arr['period'] < tt * 12]
    arr['Сценарная короткая ставка (shortrate, годовое начисление)'] /= 100
    arr['Сценарная короткая ставка до периода принятия решения, далее ожидание (shortrate_exp, годовое начисление)'] /= 100

    if ds_period is None:
        arr = np.array(arr.reset_index().sort_values(by=['period', 'Сценарий'])[
            ['Сценарная короткая ставка (shortrate, годовое начисление)', 'period', 'Сценарий']
        ].set_index(['Сценарий', 'period']).unstack())
        return arr, None, dz
    else:
        arr1 = np.array(arr.reset_index().sort_values(by=['period', 'Сценарий'])[
            ['Сценарная короткая ставка (shortrate, годовое начисление)', 'period', 'Сценарий']
        ].set_index(['Сценарий', 'period']).unstack())
        
        arr2 = np.array(arr.reset_index().sort_values(by=['period', 'Сценарий'])[
            ['period', 'Сценарий', 'Сценарная короткая ставка до периода принятия решения, далее ожидание (shortrate_exp, годовое начисление)']
        ].set_index(['Сценарий', 'period']).unstack())

        return arr1, arr2, dz


def zcyc_rates(t, r):
    rates = np.zeros(len(t))
    rates[0] = r[0]
    for i in range(1, len(r)):
        rates[i] = (rates[i - 1] * t[i - 1] + r[i] * (t[i] - t[i - 1])) / (t[i])
    return rates


class Simulator:
    def __init__(self, sim_params, free=False):
        self.sim_params = sim_params
        # self.free = free
        seed = 10
        self.rnd = np.random.RandomState(seed)
        # self.rnd.set_state(sim_params['rnd_state'])


    def convert_macro_results(self, res_vec, hist_data, cir_delay, macro_month_delay, T1, T2, delay_m, planned=''):

        ########
        par = res_vec['inflation'].copy()
        res_vec['inflation'] = np.power(1 + res_vec['inflation'], 1 / 4) - 1
        nsims = len(res_vec['inflation'])
        IR_new = np.zeros((nsims, len(res_vec['inflation'][0]) // 3))
        IR = np.zeros((nsims, len(par[0]) // 3))
        for j in range(nsims):
            if delay_m < 0:
                infl_m = np.concatenate([hist_data['cpi_yoy'][-12 + delay_m:], par[j][:]])
                infl_q = np.concatenate([(np.power(1 + hist_data['cpi_yoy'][-12 + delay_m:], 1 / 4) - 1), res_vec['inflation'][j][:]])
            else:
                infl_m = np.concatenate([hist_data['cpi_yoy'][-12 - macro_month_delay:], par[j][:]])
                infl_q = np.concatenate([(np.power(1 + hist_data['cpi_yoy'][-12 - macro_month_delay:], 1 / 4) - 1), res_vec['inflation'][j][:]])

            IR_new[j] = infl_q[::3][:len(IR_new[j])]
            IR[j] = infl_m[::3][:len(IR[j])]

        inflation_matrix = 1 + IR_new.copy()
        inflation_matrix = np.hstack([np.full((inflation_matrix.shape[0], 1), 1 + np.mean(hist_data['cpi_yoy'][-12:])), inflation_matrix])
        CPI_new = np.cumprod(inflation_matrix, axis=1)

        #########
        # CR_ind
        par = np.power(1 + res_vec['cost'], 1 / 4) - 1
        hist_cost_qoq = np.array(hist_data['cost_qoq'])
        CR = np.array(
            [np.concatenate([hist_cost_qoq[delay_m:], par[j]])[:T2] if delay_m < 0 else np.concatenate([hist_cost_qoq[-macro_month_delay:], par[j]])[:T1] for j in range(nsims)])

        CR = CR[:, ::3]

        common_len = min(CR.shape[1], IR_new.shape[1])
        CR = CR[:, :common_len]
        IR_new = IR_new[:, :common_len]
        CR_w_infl = np.log((CR + 1) / (IR_new + 1))
        CR_Idx = np.ones_like(CR_w_infl)
        for i in range(1, CR_w_infl.shape[1]):
            CR_Idx[:, i] = np.exp(CR_w_infl[:, i]) * CR_Idx[:, i - 1]

        CR_Idx = (pd.DataFrame(CR_Idx)).unstack().reset_index().rename({0: 'CostIdx_RF' + planned}, axis=1)
        CR = CR.astype(np.float64)
        IR_new = IR_new.astype(np.float64)

        #########
        # CIR

        if cir_delay > 0:
            start_cir = max(0, -macro_month_delay)
            end_cir = T2 + start_cir
        else:
            start_cir = 0
            end_cir = T2
        cir = res_vec['cir'][:, start_cir:end_cir]
        CIR_ZR = (np.exp(cir[:, ::3]) * np.exp(cir[:, 1::3]) * np.exp(cir[:, 2::3])) ** (1.0 / 12.0) - 1
        CIR_ZR = pd.DataFrame(CIR_ZR).unstack().reset_index().rename({0: 'CIR_ZR' + planned}, axis=1)
        #########
        # KEY
        KEY = res_vec['key_rate']
        if cir_delay > 0:
            KEY = KEY[:, cir_delay:T2 + cir_delay]
        KEY = KEY[:, ::3]
        KEY = pd.DataFrame(KEY).unstack().reset_index().rename({0: 'Key_rate' + planned}, axis=1)
        ######

        # HPI_ind квартальные данные годового роста цен
        par = np.power(1 + res_vec['hpi'], 1 / 4) - 1

        hist_hpi_pr_qoq = np.array(hist_data['hpi_pr_qoq'])

        HPI = np.array([np.concatenate([hist_hpi_pr_qoq[delay_m:], par[j]])[:T2] if delay_m < 0 else np.concatenate([hist_hpi_pr_qoq[-macro_month_delay:], par[j]])[:T1] for j in
                        range(nsims)])
        
        HPI = HPI[:, ::3]
        common_len = min(HPI.shape[1], IR_new.shape[1])
        HPI = HPI[:, :common_len]
        IR_new = IR_new[:, :common_len]
        HPI_w_infl = np.log((HPI + 1) / (IR_new + 1))

        HPI_Idx = np.ones_like(HPI_w_infl)
        for i in range(1, HPI_w_infl.shape[1]):
            HPI_Idx[:, i] = np.exp(HPI_w_infl[:, i]) * HPI_Idx[:, i - 1]
        HPI_Idx = (pd.DataFrame(HPI_Idx)).unstack().reset_index().rename({0: 'PrimeIdx_RF' + planned}, axis=1)

        # MR
        par = res_vec['mortgage_rate']
        hist_MR = np.array(hist_data['MR'])
        MR = np.array([moving_average(np.concatenate([hist_MR[(-3 + delay_m):], par[j][-delay_m:]]))[3:T1 + 3] if delay_m < 0 else moving_average(
            np.concatenate([hist_MR[-macro_month_delay:], par[j]]))[3:T1 + 3] for j in range(nsims)])
        MR = MR[:, ::3]
        MR = pd.DataFrame(MR).unstack().reset_index().rename({0: 'Mortgage_rate' + planned}, axis=1)
        ########

        CPI_new = pd.DataFrame(CPI_new).unstack().reset_index().rename({0: 'CPI' + planned}, axis=1)
        IR = pd.DataFrame(IR).unstack().reset_index().rename({0: 'Inflation_rate' + planned}, axis=1)
        #########
        RUO_1m = res_vec['RUO_1m']
        RUO_1m = pd.DataFrame(RUO_1m).unstack().reset_index().rename({0: 'RUO_1m' + planned}, axis=1)
        RUO_3m = res_vec['RUO_3m']
        RUO_3m = pd.DataFrame(RUO_3m).unstack().reset_index().rename({0: 'RUO_3m' + planned}, axis=1)

        RUO_6m = res_vec['RUO_6m']
        RUO_6m = pd.DataFrame(RUO_6m).unstack().reset_index().rename({0: 'RUO_6m' + planned}, axis=1)
        
        # Добавляем обработку новых переменных
        NPL_FIN = res_vec['NPL_FIN']
        NPL_FIN = pd.DataFrame(NPL_FIN).unstack().reset_index().rename({0: 'NPL_FIN' + planned}, axis=1)

        NPL_ret = res_vec['NPL_ret']
        NPL_ret = pd.DataFrame(NPL_ret).unstack().reset_index().rename({0: 'NPL_ret' + planned}, axis=1)

        NPL_corp = res_vec['NPL_corp']
        NPL_corp = pd.DataFrame(NPL_corp).unstack().reset_index().rename({0: 'NPL_corp' + planned}, axis=1)

        
        return IR, CPI_new, MR, HPI_Idx, CR_Idx, CIR_ZR, KEY, RUO_1m, RUO_3m, RUO_6m, NPL_FIN, NPL_ret, NPL_corp

    def generate_macro3(self, irshock=5000, free=False):
        print("plan_max ",self.sim_params['plan_max'] )
        plan_max = int(self.sim_params['plan_max'])
        quartal_delay = self.sim_params['quartal_delay']
        month_delay = self.sim_params['month_delay']
        macro_data = self.sim_params['macro_data']
        model_specs = self.sim_params['cc']
        rnd = self.rnd
        print("rnd:",rnd) 
        Corr = self.sim_params['Corr']
        port_batch = self.sim_params['port_batch']
        N_scr = self.sim_params['N_scr']
        display("N_scr", N_scr)
        e_start = self.sim_params['e_start']
        e_end = self.sim_params['e_end']
        macroseedlist = None
        # macroseedlist = [1]
        seed = 10
        macro_date = self.sim_params['macro_date']
        volatility = self.sim_params['volatility']
        cir_version = self.sim_params.get('cir_version', None)
        financing_decision_periods = self.sim_params.get('financing_decision_periods', None)        
        portfolio_params = self.sim_params['portfolio_params']
        eval_start_str = self.sim_params['evaluation_start']
        hist_data = self.sim_params['hist_data']
        # macro = self.sim_params['macro']
        macro = self.sim_params['History_old']
        History = self.sim_params['History']
        UseStandartZCYC = self.sim_params['UseStandartZCYC']
        MortgageRatesModels = self.sim_params['MortgageRatesModels']
        NumberOfMonteCarloScenarios = self.sim_params['NumberOfMonteCarloScenarios']
        display("NumberOfMonteCarloScenarios",NumberOfMonteCarloScenarios)
        ZCYCValues = self.sim_params['ZCYCValues']
        Coefficients = self.sim_params['Coefficients']
        print("Coefficients:",Coefficients)
        EvaluationDate = self.sim_params['EvaluationDate']
        cir_sx = self.sim_params['cir_sx']
        cir_a = self.sim_params['cir_a']
        cir_tx = self.sim_params['cir_tx']
        cir_s = 0.032
        cir_ax = None
        cir_sx = None
        # cir_a 0.064
        # cir_tx 0.08
        
        [round_df_dict(df) for df in [macro_data, model_specs, Corr]]
        print("## Generate Macro ##")
        print('seed: ' + str(seed))

        if macroseedlist is None:
            mchunklen = divmod(N_scr, port_batch)[0] + math.ceil(divmod(N_scr, port_batch)[1] / port_batch)
            macroseedlist = []
            macroseedlist = [r for r in (rnd.randint(1, mchunklen * 10) for _ in range(mchunklen)) if r not in macroseedlist]
        print('macroseedlist: ' + str(macroseedlist))

        start = next((i for i in range(len(macroseedlist)) if i * port_batch <= e_start < (i + 1) * port_batch), 0)
        end = next((i for i in range(len(macroseedlist)) if i * port_batch < e_end <= (i + 1) * port_batch), len(macroseedlist) - 1)

        macro_sims = list(range(start, end + 1))
        tt = int(math.ceil(plan_max / 4) + 2)
        macrofull = port_batch >= N_scr
        export_list = []
        sim_id_start = 0  # Инициализация переменной sim_id_start

        for i in range(len(macro_sims)):
            m = macro_sims[i]
            # seed = macroseedlist[m]
            nsims = ((m != len(macroseedlist) - 1) * port_batch + (m == len(macroseedlist) - 1) * (N_scr - m * port_batch)) * (not macrofull) + macrofull * N_scr
            # Месяц оценки проекта
            eval_start = datetime.date(*map(int, pd.to_datetime(eval_start_str).strftime('%Y.%m.%d').split(".")))

            # Последний месяц макры
            macro_eval_start = datetime.date.fromtimestamp(int(pd.to_datetime(macro_data['Period_Date'].max()).value / 1000000000))
            # Последний исторический месяц ставки

            try:
                cir_hist_start = datetime.date.fromtimestamp(int(History.index[-1].value / 1000000000)) #оригинал
            except:
                History = History.set_index('Date')
                cir_hist_start = datetime.date.fromtimestamp(int(History.index[-1].value / 1000000000))
            
            # cir_hist_start = datetime.date.fromtimestamp(int(History['Date'].iloc[-1].value / 1000000000)) #для экселя
            
            # Разница между макро и ставкой
            cir_macro_delay = month_diff(macro_eval_start, cir_hist_start)

            # Разница между входной макро и оценкой проекта
            macro_month_delay = month_diff(macro_eval_start, eval_start)

            # Разница между ставкой и оценкой проекта
            cir_month_delay = month_diff(cir_hist_start, eval_start)

            # Разница между входной макро и исторической макрой
            macro_hist_delay = month_diff(macro_eval_start, macro_date)
            # irshock = 500
            print("## CCIR and montecarlo ##")

            cir_data, cir_planned, r_shocks = run(
                tt=tt + max(cir_macro_delay, -cir_month_delay) / 12,
                shock=irshock, 
                cir_version=cir_version,
                seed=(seed),
                ds_period=financing_decision_periods,
                number_of_monte_carlo_scenarios=NumberOfMonteCarloScenarios,
                history=History,
                mortgage_rates_models=MortgageRatesModels,
                zcyc_values=ZCYCValues,
                use_standart_zcyc=UseStandartZCYC,
                evaluation_date = EvaluationDate,
                cir_sx=cir_sx,
                cir_a=cir_a,
                cir_s=cir_s,
                cir_ax=cir_ax,
                cir_tx=cir_tx,
                coefficients=Coefficients)
            
            r_shocks = np.array([[r_shocks[j][0][i] for j in range(len(cir_data[0]))] for i in range(len(cir_data))])
            T1 = int((tt - macro_month_delay / 12) * 12)

            no_shock_period = -min(cir_macro_delay, 0) + (macro_month_delay if macro_month_delay > 0 else 0)
            shock_start = macro_month_delay if macro_month_delay > 0 else 0
            shock_end = T1 + shock_start

            macro_r_shocks = np.array([np.concatenate([np.zeros(no_shock_period), r_shocks[j]])[shock_start:shock_end] for j in range(len(r_shocks))])

            delay = max(-cir_macro_delay, cir_month_delay)
            if delay > 0:
                r_fact = cir_data[:, :delay]
                cir_data = np.array([np.concatenate([r_fact[j], cir_data[j]]) for j in range(len(cir_data))])

        start_cir = macro_month_delay if macro_month_delay > 0 else 0
        end_cir = T1 + start_cir

        macro_cir = np.array([cir_data[j][start_cir:end_cir] for j in range(len(cir_data))])

        T2 = T1 + macro_month_delay
        delay_m = min(0, macro_hist_delay, macro_month_delay)
        Seed =seed
        print("Seed:",Seed) 
        np.random.seed(Seed)
        shocks = {
            'CPI': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'HPI': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'CR': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'GDP': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'MR': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'HPI_PR': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'CostRate': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'KEY': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            
            'NPL_FIN': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'NPL_ret': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'NPL_corp': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'RUO_3m': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'RUO_1m': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12]),
            'RUO_6m': rnd.normal(0, 1, [N_scr, portfolio_params['Macro_horizon'] * 12])
        }

        shock_input = [shocks[key][:, :T1] for key in ['CPI', 'HPI', 'CR', 'GDP', 'MR', 'HPI_PR', 'CostRate', 'KEY','RUO_3m', 'RUO_1m', 'RUO_6m', 'NPL_FIN', 'NPL_ret','NPL_corp']]
        shocks_CPI, shocks_HPI, shocks_CR, shocks_GDP, shocks_MR, shocks_HPI_PR, shocks_CostRate, shocks_KEY,  shocks_RUO_3m, shocks_RUO_1m, shocks_RUO_6m, shocks_NPL_FIN, shocks_NPL_ret, shocks_NPL_corp = shock_input
        
        try:
            Corr =Corr.set_index('Unnamed: 0')
            del Corr.index.name
        except Exception:
            pass
            
        Cholesky = linalg.cholesky(Corr, check_finite=True, lower=True)
      
        shocks = {}

        shock_names = ["macro_r", "Corp_Rate", "MR", "cost_m_yoy", "cpi_yoy", "gdp_yoy", "hpi_pr_yoy", "hpi_yoy", "key_rate",'RUO_3m', 'RUO_1m', 'RUO_6m', 'NPL_FIN', 'NPL_ret','NPL_corp']

        shocks_in = [macro_r_shocks, shocks_CR, shocks_MR, shocks_CostRate, shocks_CPI, shocks_GDP, shocks_HPI_PR, shocks_HPI, shocks_KEY, shocks_RUO_3m, shocks_RUO_1m, shocks_RUO_6m,  shocks_NPL_FIN, shocks_NPL_ret, shocks_NPL_corp]

        shocks_array = np.array(shocks_in)
        shocks_result = np.einsum('ij,jkl->ikl', Cholesky, shocks_array)

        shocks = {name: shocks_result[i] for i, name in enumerate(shock_names)}

        T = tt - macro_month_delay / 12 + 15 / 12

        model_specs = self.sim_params['cc']
        

        variables = {
            'cpi_yoy': np.zeros((N_scr, int(12 * T))),
            'rate': np.zeros((N_scr, int(12 * T))),
            'gdp_yoy': np.zeros((N_scr, int(12 * T))),
            'key_rate': np.zeros((N_scr, int(12 * T))),
            'MR': np.zeros((N_scr, int(12 * T))),
            'Corp_Rate': np.zeros((N_scr, int(12 * T))),
            'hpi_yoy': np.zeros((N_scr, int(12 * T))),
            'hpi_pr_yoy': np.zeros((N_scr, int(12 * T))),
            'cost_m_yoy': np.zeros((N_scr, int(12 * T))),
            'infl_me': np.zeros((N_scr, int(12 * T))),
            'i1': np.zeros((N_scr, int(12 * T))),
            'i2': np.zeros((N_scr, int(12 * T))),
            'infl_i1': np.zeros((N_scr, int(12 * T))),
            'infl_i2': np.zeros((N_scr, int(12 * T))),
            
            'RUO_1m': np.zeros((N_scr, int(12 * T))),
            'RUO_3m': np.zeros((N_scr, int(12 * T))),
            'RUO_6m': np.zeros((N_scr, int(12 * T))),
            'NPL_FIN': np.zeros((N_scr, int(12 * T))),
            'NPL_ret': np.zeros((N_scr, int(12 * T))),
            'NPL_corp': np.zeros((N_scr, int(12 * T))),
        }
        for var in ['cpi_yoy', 'rate', 'gdp_yoy', 'key_rate', 'MR', 'Corp_Rate', 'hpi_yoy', 'hpi_pr_yoy', 'cost_m_yoy', 'RUO_3m' ,'RUO_6m','NPL_ret','NPL_FIN','NPL_corp']:   
            variables[var][:, -15:] = macro_data[var][-15:]
        # индекс гос поддержки ипотеки ( объем поддерки (руб) / стоимсость 1 кв.м (руб)) - расчет описан в specs_debug.ipynd
        # ind3MR = [0.005, 0.005, 0.004, 0.004, 0.004, 0.004] + [0.002] * (int(12 * T) - 15 - 12) + [0.003, 0.001, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.004, 0.005, 0.004, 0.004, 0.005]
        # ind3MR = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002] + [0.002] * (int(12 * T) - 15 - 12) + [0.005, 0.002, 0.002, 0.003, 0.003, 0.003, 0.006, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001 ]
        ind3MR = [0.003, 0.003, 0.003, 0.003, 0.003, 0.003] + [0.003] * (int(12 * T) - 15 - 12) + [0.005, 0.002, 0.002, 0.003, 0.003, 0.003, 0.006, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001 ]
        
        # [0.003, 0.001, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.004, 0.005, 0.004, 0.004, 0.005] данные с 2022-12-31 по 2023-12-31
        # [0.005, 0.002, 0.002, 0.003, 0.003, 0.003, 0.006, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001 ] анные с 2023-12-31 по 2023-12-31
        
        print("len(macro_data['cpi_yoy'][-15:]):", len(macro_data['cpi_yoy']))
        print("len(ind3MR) :", len(ind3MR))
        print("(int(12 * T) - 15)/12 =", (12 * T - 15)/12)

        
        
        import ast
        for key in model_specs.keys():
            if type(model_specs[key]) == str:
                model_specs[key] = ast.literal_eval(model_specs[key])
   
        
        if type(volatility) == dict:
            volatility = pd.DataFrame.from_dict(volatility, orient='index').T
        
        for t in range(0, int(12 * T) - 15): 
            variables['rate'][:, t] = macro_cir[:, t]
            variables['cpi_yoy'][:, t] = model_specs['cpi_yoy']['const'] + volatility['cpi_yoy'].item() * shocks['cpi_yoy'][:, t] + \
                                         model_specs['cpi_yoy']['cpi_yoy_L1'] * variables['cpi_yoy'][:, t - 1] + \
                                         model_specs['cpi_yoy']['rate_D3'] * (variables['rate'][:, t] - variables['rate'][:, t - 3])

            variables['infl_me'][:, t] = (np.mean(variables['cpi_yoy'][:, t - 11:], axis=1))

            variables['i1'][:, t] = (variables['infl_me'][:, t] > 0.05) * 1
            variables['i2'][:, t] = (variables['infl_me'][:, t] <= 0.05) * 1
            variables['infl_i1'][:, t] = variables['cpi_yoy'][:, t] * variables['i1'][:, t]
            variables['infl_i2'][:, t] = variables['cpi_yoy'][:, t] * variables['i2'][:, t]

            variables['key_rate'][:, t] = (
                    variables['rate'][:, t] +
                    model_specs['key_rate']['const'] +
                    model_specs['key_rate']['cpi_yoy_L2'] * variables['cpi_yoy'][:, t - 2] +
                    model_specs['key_rate']['cpi_yoy_D1'] * (variables['cpi_yoy'][:, t] - variables['cpi_yoy'][:, t - 1]) +
                    model_specs['key_rate']['rate_Ma3'] * np.mean(
                variables['rate'][:, t - 3:] if t >= 6 else
                np.concatenate([
                    variables['rate'][:, max(0, t - 3):],
                    np.zeros((variables['rate'].shape[0], max(0, 3 - t)))
                ], axis=1),
                axis=1
            ) +
                    volatility['key_rate'].item() * shocks['key_rate'][:, t])

            variables['gdp_yoy'][:, t] = (
                    model_specs['gdp_yoy']['const'] +
                    model_specs['gdp_yoy']['gdp_yoy_L12'] * variables['gdp_yoy'][:, t - 12] +
                    model_specs['gdp_yoy']['cpi_yoy_Ma6'] * np.mean(
                variables['cpi_yoy'][:, t - 6:] if t >= 12 else
                np.concatenate([
                    variables['cpi_yoy'][:, t - 6:], variables['cpi_yoy'][:, :t]], axis=1), axis=1) + \
                    model_specs['gdp_yoy']['key_rate_spread_rate_Ma3'] * np.mean(
                variables['key_rate'][:, t - 3:] - variables['rate'][:, t - 3:] if t >= 6 else
                np.concatenate([
                    variables['key_rate'][:, t - 3:] - variables['rate'][:, t - 3:], variables['key_rate'][:, : t] - variables['rate'][:, : t],
                ], axis=1),
                axis=1
            ) +
                    volatility['gdp_yoy'].item() * shocks['gdp_yoy'][:, t])

            variables['MR'][:, t] = variables['key_rate'][:, t] + \
                                    model_specs['MR']['const'] + model_specs['MR']['gdp_yoy_L3'] * variables['gdp_yoy'][:, t - 3] + \
                                    model_specs['MR']['key_rate_spread_rate'] * (variables['key_rate'][:, t] - variables['rate'][:, t]) + \
                                    (model_specs['MR']['ind3MR_Ma12'] * (
                                        np.mean(ind3MR[max(0, t - 12 + 1):t + 1]) if t >= 12 - 1 else np.mean(ind3MR[int(max((t - 12), (-12 / 2))):] + ind3MR[:t + 1]))) + \
                                    volatility['MR'].item() * shocks['MR'][:, t]

            variables['Corp_Rate'][:, t] = variables['key_rate'][:, t] + np.maximum( \
                model_specs['Corp_Rate']['const'] + \
                model_specs['Corp_Rate']['MR_spread_key_rate'] * (variables['MR'][:, t] - variables['key_rate'][:, t]) + \
                model_specs['Corp_Rate']['cpi_yoy_L1'] * variables['cpi_yoy'][:, t - 1] + \
                volatility['Corp_Rate'].item() * shocks['Corp_Rate'][:, t], 0)

            variables['hpi_pr_yoy'][:, t] = (
                    model_specs['hpi_pr_yoy']['const'] +
                    model_specs['hpi_pr_yoy']['hpi_pr_yoy_L2'] * variables['hpi_pr_yoy'][:, t - 2] +
                    model_specs['hpi_pr_yoy']['cpi_yoy_D3'] * (variables['cpi_yoy'][:, t] - variables['cpi_yoy'][:, t - 3]) +
                    model_specs['hpi_pr_yoy']['gdp_yoy_L3'] * variables['gdp_yoy'][:, t - 3] +
                    volatility['hpi_pr_yoy'].item() * shocks['hpi_pr_yoy'][:, t])

            variables['cost_m_yoy'][:, t] = (
                    model_specs['cost_m_yoy']['const'] +
                    model_specs['cost_m_yoy']['cost_m_yoy_L3'] * variables['cost_m_yoy'][:, t - 3] +
                    model_specs['cost_m_yoy']['hpi_pr_yoy_D2'] * (variables['hpi_pr_yoy'][:, t] - variables['hpi_pr_yoy'][:, t - 2]) +
                    model_specs['cost_m_yoy']['hpi_pr_yoy_L2'] * variables['hpi_pr_yoy'][:, t - 2] + \
                    volatility['cost_m_yoy'].item() * shocks['cost_m_yoy'][:, t]
            )

            variables['hpi_yoy'][:, t] = ( \
                        model_specs['hpi_yoy']['const'] + \
                        model_specs['hpi_yoy']['cost_m_yoy_L1'] * variables['cost_m_yoy'][:, t - 1] + \
                        model_specs['hpi_yoy']['hpi_pr_yoy_L1'] * variables['hpi_pr_yoy'][:, t - 1] + \
                        volatility['hpi_yoy'].item() * shocks['hpi_yoy'][:, t])
            
            variables['RUO_3m'][:, t] = variables['key_rate'][:, t] +  np.maximum(0, \
                            model_specs['RUO_3m_spread_key_rate']['const']  + \
                            model_specs['RUO_3m_spread_key_rate']['Corp_Rate_spread_key_rate_L3'] *  (variables['Corp_Rate'][:, t -3] - variables['key_rate'][:, t-3]) +\
                            model_specs['RUO_3m_spread_key_rate']['MR_spread_key_rate_D1'] * ((variables['MR'][:, t] - variables['key_rate'][:, t]) - (variables['MR'][:, t -1] - variables['key_rate'][:, t -1])) + \
                            model_specs['RUO_3m_spread_key_rate']['RUO_3m_spread_key_rate_L1']* (variables['RUO_3m'][:, t -1] - variables['key_rate'][:, t-1]) + \
                            model_specs['RUO_3m_spread_key_rate']['key_rate_spread_rate_L3'] *  (-variables['rate'][:, t-3] + variables['key_rate'][:, t-3] ) + \
                            np.maximum(0,volatility['RUO_3m_spread_key_rate'].item()* shocks['RUO_3m'][:, t]))

            variables['RUO_1m'][:, t] = model_specs['RUO_1m']['RUO_3m_L3']*variables['RUO_3m'][:, t-3] + model_specs['RUO_1m']['const'] +\
                            model_specs['RUO_1m']['cpi_yoy_D3'] * (variables['cpi_yoy'][:, t] - variables['cpi_yoy'][:, t -3] ) + \
                            model_specs['RUO_1m']['RUO_3m_D1']* (variables['RUO_3m'][:, t] - variables['RUO_3m'][:, t-1]) +\
                            np.maximum(0, volatility['RUO_1m'].item() * shocks['RUO_1m'][:, t])
                                        
            variables['RUO_6m'][:, t] = model_specs['RUO_6m']['RUO_3m_D1'] *(variables['RUO_3m'][:, t] -variables['RUO_3m'][:, t-1]) + \
                model_specs['RUO_6m']['const'] + model_specs['RUO_6m']['RUO_3m_L3'] * variables['RUO_3m'][:, t -3] +\
                np.maximum(0, volatility['RUO_6m'].item() * shocks['RUO_6m'][:, t])
            
            
           
            variables['NPL_corp'][:, t] = np.maximum(0,(
                model_specs['NPL_corp']['const'] +
                model_specs['NPL_corp']['NPL_corp_L1'] * variables['NPL_corp'][:, t - 1] +
                model_specs['NPL_corp']['Corp_Rate_L3'] * (variables['key_rate'][:, t - 3]) +\
                volatility['NPL_corp'].item() * shocks['NPL_corp'][:, t]
            ))

            variables['NPL_FIN'][:, t] = np.maximum(0,(
                model_specs['NPL_FIN']['const'] +
                model_specs['NPL_FIN']['NPL_corp_D6'] * (variables['NPL_corp'][:, t] - variables['NPL_corp'][:, t - 6]) +
                model_specs['NPL_FIN']['NPL_FIN_L1'] * variables['NPL_FIN'][:, t - 1] +
                model_specs['NPL_FIN']['NPL_corp_L3'] * variables['NPL_corp'][:, t - 3] +
                model_specs['NPL_FIN']['gdp_yoy_L3'] * variables['gdp_yoy'][:, t - 3] +
                model_specs['NPL_FIN']['Corp_Rate_L1'] * variables['Corp_Rate'][:, t - 1] +
                volatility['NPL_FIN'].item() * shocks['NPL_FIN'][:, t]
            ))

            
            variables['NPL_ret'][:, t] = np.maximum(0,(
                model_specs['NPL_ret']['const'] + model_specs['NPL_ret']['NPL_ret_L3']*variables['NPL_ret'][:, t -3] +\
                model_specs['NPL_ret']['key_rate_spread_rate_L3'] * (variables['key_rate'][:, t - 3] - variables['rate'][:, t - 3]) +\
                model_specs['NPL_ret']['NPL_corp_spread_NPL_ret_L1'] * (variables['NPL_corp'][:, t - 1] - variables['NPL_ret'][:, t - 1]) +\
                volatility['NPL_ret'].item() * shocks['NPL_ret'][:, t]
            )) 
        
        sim_list = [variables[var] for var in ['cpi_yoy', 'gdp_yoy', 'key_rate', 'MR', 'Corp_Rate', 'hpi_yoy', 'hpi_pr_yoy', 'cost_m_yoy', 'RUO_1m', 'RUO_3m', 'RUO_6m', 'NPL_FIN', 'NPL_ret', 'NPL_corp']]
        sim_list_names = ['cpi_yoy', 'gdp_yoy', 'key_rate', 'MR', 'Corp_Rate', 'hpi_yoy', 'hpi_pr_yoy', 'cost_m_yoy', 'RUO_1m', 'RUO_3m', 'RUO_6m', 'NPL_FIN', 'NPL_ret', 'NPL_corp']
        

        
        
        
        res_vec = dict(zip(sim_list_names, sim_list))
    
        display("macro_cir", macro_cir)
        res_vec = {
            'cpi_yoy': res_vec['cpi_yoy'][:, :-15],
            'gdp_yoy': res_vec['gdp_yoy'][:, :-15],
            'inflation': res_vec['cpi_yoy'][:, :-15],
            'cost': res_vec['cost_m_yoy'][:, :-15],
            'cir': macro_cir,
            'hpi': res_vec['hpi_pr_yoy'][:, :-15],
            'mortgage_rate': res_vec['MR'][:, :-15],
            'key_rate': res_vec['key_rate'][:, :-15],
            'hpi_second': res_vec['hpi_yoy'][:, :-15],
            'MR': res_vec['MR'][:, :-15],
            
            'RUO_1m': res_vec['RUO_1m'][:, :-15],
            'RUO_3m': res_vec['RUO_3m'][:, :-15],
            'RUO_6m': res_vec['RUO_6m'][:, :-15],
            'NPL_FIN': res_vec['NPL_FIN'][:, :-15],
            'NPL_ret': res_vec['NPL_ret'][:, :-15],
            'NPL_corp': res_vec['NPL_corp'][:, :-15],

        }
        cir_delay = max(-cir_macro_delay, cir_month_delay)
        IR, CPI, MR, HPI_Idx, CR_Idx, CIR_ZR, KEY, RUO_1m, RUO_3m, RUO_6m, NPL_FIN, NPL_ret, NPL_corp = self.convert_macro_results(res_vec, macro_data, cir_delay, macro_month_delay, T1, T2, delay_m)

        export = reduce(lambda left, right: pd.merge(left, right, on=['level_0', 'level_1'], how='outer'),
                            [IR, CPI, MR, HPI_Idx, CR_Idx, CIR_ZR, KEY, RUO_1m, RUO_3m, RUO_6m, NPL_FIN, NPL_ret, NPL_corp])

        export = export[export.level_0 < plan_max]

        if not macrofull:

            if m == start:
                export = export[export['level_1'] >= e_start]
            if m == end:
                export = export[export['level_1'] < e_end + (e_end == 0) * N_scr]
            export.level_1 = np.tile(
                np.arange(m * port_batch + e_start, min((m + 1) * port_batch + e_start, N_scr)),
                plan_max)
            elen = len(export[export.level_0 == 0])
            export['sim_id'] = np.tile(np.arange(sim_id_start, sim_id_start + elen), plan_max)
            sim_id_start += elen

        export = export.rename({'level_0': 'period', 'level_1': 'macro_sim_id'}, axis=1)
        export['period'] += 1

        return export, res_vec
