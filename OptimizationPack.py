import numpy as np
import pandas as pd
import random
import math
from copy import deepcopy
import datetime
import time
import sys

# ----------------------------------------------FILES WITH DATA---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

atm_data = pd.read_csv('./data/lecture4/atm_data.csv', index_col=0, parse_dates=True)  # это файлик smsb
costs = pd.read_csv('./data/lecture4/costs.csv', index_col=0, encoding='cp1251')
atm_dm_availability = pd.read_csv('./data/lecture4/atm_dm_availability.csv', index_col=0)
atm_dm_availability = atm_dm_availability[atm_dm_availability.availability == 'Encash']

# atm intraday
atm_intraday = pd.read_csv('./data/lecture4/atm_intraday.csv', index_col=0, parse_dates=True)
atm_intraday = atm_intraday.rename(columns={'new_value': 'sum'})
atm_intraday['date_id'] = pd.to_datetime(atm_intraday['date_id'])
atm_intraday['dow'] = atm_intraday['date_id'].dt.weekday
atm_intraday = atm_intraday[atm_intraday.date_id != '2018-01-13']
atm_intraday.index = range(atm_intraday.shape[0])
atm_intraday['new_sum'] = np.where(atm_intraday['sum'] < 0, 0, atm_intraday['sum'])

# -----------------------------------------------GLOBAL VARIABLES-------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

average_denomination = 2200  # dm; for ipt 1700
machine_epsilon = 10

# -------------------------------------------------CASH CENTER----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class CashCenter(object):

    """
        Класс моделирующий состояние КИЦа.

    Поля:
        atm_ids: список ID диспенсеров
        dm_ids: список ID депозитных модулей
        ipt_ids: список ID ИПТ
        times: начало и конец рассматриваемоего периода, список из двух дат (дата представляется строкой, например
            '2018-02-20')
        power: значения мощности КИЦ для каждого дня из рассматриваемого периода, numpy.array
        priority: приоритеты выделения мощности для каждого типа устройств, словарь.
         ключи: 'atm', 'dm', 'ipt' значения: доли от единицы. При этом для 'dm' приоритет указывается свободно от 'atm',
         так как предусмотрена функция совместной инкассации

    Скрытые поля:
        atm_capacity: максимальная вместимость для каждого диспенсера, numpy.array
        dm_capacity: максимальная вместимость для каждого депозитного модуля, numpy.array
        ipt_capacity: максимальная вместимость для каждого ИПТ, numpy.array
        dm_cassete: количество кассет для каждого депозитного модуля, numpy.array
        ipt_cassete: количество кассет для каждого ИПТ, numpy.array
        f_rate: ставка фондирования

        n_atms: количество диспенсеров
        n_dms: количество депозитных модулей
        n_ipts: количество ИПТ
        n_days: количество дней в рассматриваемом периоде

        atm_predictions: таблица из предсказаний клиентских снятий за рассматриваемый период
            для каждого диспенсера, numpy.array
        atm_facts: таблица из фактических значений клиентских снятий за рассматриваемый период
            для каждого диспенсера, numpy.array
        atm_initial_balances: входящий остаток для каждого диспенсера, numpy.array
        atm_cost_encash: стоимость инкассации для каждого диспенсера, numpy.array
        atm_cost_kassa: стоимость работы кассы для каждого диспенсера, numpy.array
        atm_intraday_distribution: распределение клиентского спроса для каждого диспенсера,
            для каждого дня недели, numpy.array
        atm_first_encash_mask: маска первых инкассаций диспенсеров, numpy.array
        atm_availability_mask: маска допуступности деспенсеров для инкассации, numpy.array
        atm_full_mask: синхронизированная (итоговая) маска диспенсеров, numpy.array

        dm_predictions_banknotes: таблица из предсказаний клиентских поступлений в количестве банкнот
            за рассматриваемый период для каждого депозитного модуля, numpy.array
        dm_predictions_money: таблица из предсказаний клиентских поступлений за рассматриваемый период
            для каждого депозитного модуля, numpy.array
        dm_facts_banknotes: таблица из фактических значений клиентских поступлений в количестве банкнот
            за рассматриваемый период для каждого депозитного модуля, numpy.array
        dm_facts_money: таблица из фактических клиентских поступлений за рассматриваемый период
            для каждого депозитного модуля, numpy.array
        dm_initial_balances_banknotes:  входящий остаток в количестве банкнот
            для каждого депозитного модуля, numpy.array
        dm_initial_balances_money: входящий остаток для каждого депозитного модуля, numpy.array
        dm_cost_encash: стоимость инкассации для каждого депозитного модуля, numpy.array
        dm_cost_kassa: стоимость работы кассы для каждого депозитного модуля, numpy.array
        dm_first_encash_mask: маска первых инкассаций депозитных модулей, numpy.array
        dm_availability_mask: маска допуступности депозитных модулей для инкассации, numpy.array
        dm_atm_state_mask: маска совместной (с диспенсером) инкассации депозитных модулей, numpy.array
        dm_full_mask: синхронизированная (итоговая) маска диспенсеров, numpy.array

        ipt_predictions_banknotes: таблица из предсказаний клиентских поступлений в количестве банкнот
            за рассматриваемый период для каждого ИПТ, numpy.array
        ipt_predictions_money: таблица из предсказаний клиентских поступлений за рассматриваемый период
            для ИПТ, numpy.array
        ipt_facts_banknotes: таблица из фактических значений клиентских поступлений в количестве банкнот
            за рассматриваемый период для каждого ИПТ, numpy.array
        ipt_facts_money: таблица из фактических клиентских поступлений за рассматриваемый период
            для каждого ИПТ, numpy.array
        ipt_initial_balances_banknotes: входящий остаток в количестве банкнот
            для каждого ИПТ, numpy.array
        ipt_initial_balances_money:  входящий остаток для каждого ИПТ, numpy.array
        ipt_cost_encash: стоимость инкассации для каждого депозитного модуля, numpy.array
        ipt_cost_kassa: стоимость работы кассы для каждого депозитного модуля, numpy.array
        ipt_first_encash_mask: маска первых инкассаций ИПТ, numpy.array
        ipt_availability_mask: маска допуступности ИПТ для инкассации, numpy.array
        ipt_full_mask: синхронизированная (итоговая) маска ИПТ, numpy.array


        atm_table: 2х-мерная таблица (состояние atm), каждая строка которой, реализует вариант инкассирования диспенсера
        (1 - грузим, 0 - нет)
        dm_table = 2х-мерная таблица (состояние dm), каждая строка которой, реализует вариант инкассирования депозитного
            модуля (1 - выгружаем, 0 - нет)
        ipt_table: 2х-мерная таблица (состояние ipt), каждая строка которой, реализует вариант инкассирования ИПТ
        (1 - выгружаем, 0 - нет)
    """

    def __init__(self,
                 atm_ids=None,
                 dm_ids=None,
                 ipt_ids=None,
                 times=None,
                 power=None,
                 priority=None,
                 atm_capacity=None,
                 dm_capacity=None,
                 ipt_capacity=None,
                 dm_cassette=None,
                 ipt_cassette=None,
                 f_rate=0.1):

        if times is not None:
            self.times = times
            self._n_days = (pd.to_datetime(self.times[1]) - pd.to_datetime(self.times[0])).days + 1
        else:
            sys.exit('YOU MUST CHOOSE SOME TIME PERIOD!')

        self._devices_order = []

        self.atm_ids = atm_ids
        self.atm_capacity = atm_capacity

        if atm_ids is not None and atm_capacity is not None:
            self._n_atms = len(atm_ids)
            self._devices_order.append('atm')
            if len(atm_capacity) != self._n_atms:
                sys.exit('len(atm_capacity) != self._n_atms')

            self._atm_predictions = None
            self._atm_facts = None
            self._atm_initial_balances = None
            self._atm_cost_encash = None
            self._atm_cost_kassa = None
            self._atm_first_encash_mask = np.zeros(shape=(self._n_atms, self._n_days))
            self._atm_availability_mask = np.zeros(shape=(self._n_atms, self._n_days))
            self._atm_full_mask = np.zeros(shape=(self._n_atms, self._n_days))
            self._atm_masks_order = None
            self._atm_mode_order = None

            self.atm_table = None

            self._atm_intraday_distribution = None
            self._atm_intraday_facts = None
            self._atm_intraday_predictions = None
            print('ATM BLOCK: +')
        else:
            self.atm_ids = None
            self.atm_capacity = None
            print('ATM BLOCK: -!')

        self.dm_ids = dm_ids
        self.dm_capacity = dm_capacity
        self.dm_cassette = dm_cassette

        if dm_ids is not None and dm_capacity is not None and dm_cassette is not None:
            self._n_dms = len(dm_ids)
            self._devices_order.append('dm')
            if len(dm_capacity) != self._n_dms or len(dm_cassette) != self._n_dms:
                sys.exit('len(dm_capacity) != self._n_dms or len(dm_cassette) != self._n_dms')

            self._dm_predictions_banknotes = None
            self._dm_predictions_money = None
            self._dm_facts_banknotes = None
            self._dm_facts_money = None
            self._dm_initial_balances_banknotes = None
            self._dm_initial_balances_money = None
            self._dm_cost_encash = None
            self._dm_cost_kassa = None
            self._dm_first_encash_mask = None
            self._dm_availability_mask = None
            self._dm_atm_state_mask = None
            self._dm_full_mask = None

            self.dm_table = None

            self._dm_intraday_distribution = None
            self._dm_intraday_facts_money = None
            self._dm_intraday_facts_banknotes = None
            self._dm_intraday_predictions_money = None
            self._dm_intraday_predictions_banknotes = None
            print('DM BLOCK: +')
        else:
            self.dm_ids = None
            self.dm_capacity = None
            self.dm_cassette = None
            print('DM BLOCK: -')

        self.ipt_ids = ipt_ids
        self.ipt_capacity = ipt_capacity
        self.ipt_cassette = ipt_cassette

        if ipt_ids is not None and ipt_capacity is not None and ipt_cassette is not None:
            self._n_ipts = len(ipt_ids)
            self._devices_order.append('ipt')
            if len(ipt_capacity) != self._n_ipts or len(ipt_cassette) != self._n_ipts:
                sys.exit('len(ipt_capacity) != self._n_ipts or len(ipt_cassette) != self._n_ipts')

            self._ipt_predictions_banknotes = None
            self._ipt_predictions_money = None
            self._ipt_facts_banknotes = None
            self._ipt_facts_money = None
            self._ipt_initial_balances_banknotes = None
            self._ipt_initial_balances_money = None
            self._ipt_cost_encash = None
            self._ipt_cost_kassa = None
            self._ipt_first_encash_mask = None
            self._ipt_availability_mask = None
            self._ipt_full_mask = None

            self.ipt_table = None

            self._ipt_intraday_distribution = None
            self._ipt_intraday_facts_money = None
            self._ipt_intraday_facts_banknotes = None
            self._ipt_intraday_predictions_money = None
            self._ipt_intraday_predictions_banknotes = None
            print('IPT BLOCK: +')
        else:
            self.ipt_ids = None
            self.ipt_capacity = None
            self.ipt_cassette = None
            print('IPT BLOCK: -')

        if not set(self._devices_order):
            sys.exit('YOUR CASH CENTER HAVE TO CONSIST MORE OR EQUAL TO ONE BLOCK!')

        if power is not None and len(power) == self._n_days:
            self.power = np.array(power)
        else:
            sys.exit('self.power MUST HAVE SIZE = self._n_days')

        if priority is not None and \
                        set(self._devices_order) == set(priority.keys()) and np.sum(list(priority.values())) == 1:
            self.priority = priority
        else:
            sys.exit('set(self._devices_order) != set(priority.keys()) or np.sum(list(priority.values())) != 1')

        if 0.0 <= f_rate <= 1.0:
            self.f_rate = f_rate
        else:
            sys.exit('f_rate MUST BE IN [0,1]')

    @staticmethod
    def transform_state(state):

        """
            Вспомогательная функция, которая из стандартной таблицы инкассаций получает таблицу, элементы которой -
            числа, на сколько дней вперед мы инкассируем исходя из состояния state

            Принимает: таблицу state из 0 и 1

            Возвращает: таблицу со значениями
                0 (если на этом месте стоял 0 в исходной таблице)
                k (если на этом месте стояла 1 в исходной таблице и после нее (k-1) нолик), k >= 1
        """

        state_new = deepcopy(state)
        n_atms, n_days = state_new.shape
        for i in range(n_atms):
            for j in range(n_days):
                if state_new[i, j] == 1 and j != n_days - 1:
                    m = 1
                    k = j + 1
                    while state_new[i, k] == 0:
                        m += 1
                        k += 1
                        if k == n_days:
                            break
                    state_new[i, j] = m
        return state_new

    def _set_cash_flow(self, kind='atm'):

        """
            Устанавливает параметры _{kind}_predictions,  _{kind}_facts ,  _{kind}_initial_balances значениями из
            подгруженных таблиц (по умолчанию).

            Принимает:
                kind: 'atm', 'dm' , 'ipt'
         """

        if kind == 'atm':
            tmp = np.zeros(shape=(self._n_atms, self._n_days))
            for index, ID in enumerate(self.atm_ids):
                tmp[index, :] = atm_data[atm_data['ATM_ID'] == ID][self.times[0]:self.times[1]]['CLIENT_OUT']
            self._atm_facts = (np.ceil(tmp / 10) * 10).astype('int64')

            # tmp = np.zeros(shape=(self._n_atms, self._n_days))
            # for index, ID in enumerate(self.atm_ids):
            #     tmp[index, :] = atm_predictions[atm_predictions['ATM_ID'] == ID][self.times[0]:self.times[1]][
            #         'prediction']
            # self._atm_predictions = (np.ceil(tmp / 10) * 10).astype('int64')
            #self._atm_predictions = self._atm_facts

            # tmp = np.zeros(self._n_atms)
            # for index, ID in enumerate(self.atm_ids):
            #     tmp[index] = atm_data[atm_data['ATM_ID'] == ID][self.times[0]:self.times[1]]['INITIAL_BALANCE'].iloc[0]
            # self._atm_initial_balances = (np.floor(tmp / 10) * 10).astype('int64')


            # вот тут по-хорошему должно быть считывание внутридневных данных для ATM,
            # а только потом подсчет внутридневного распределения!
            # но так как нормальных данных до сих пор нет, то сначала на том, что есть,
            # считаем распределение, а только потом экстраполируем это на реальные данные <-- короче,
            # сейчас все наоборот

        if kind == 'dm':
            tmp = np.zeros(shape=(self._n_dms, self._n_days))
            for index, ID in enumerate(self.dm_ids):
                tmp[index, :] = dm_data[dm_data['ATM_ID'] == ID][self.times[0]:self.times[1]]['DEPOSITS']
            self._dm_facts_money = (np.ceil(tmp / 10) * 10).astype('int64')

            tmp = np.zeros(shape=(self._n_dms, self._n_days))
            for index, ID in enumerate(self.dm_ids):
                tmp[index, :] = dm_data[dm_data['ATM_ID'] == ID][self.times[0]:self.times[1]]['NDEPOSITS']
            self._dm_facts_banknotes = np.ceil(tmp).astype('int64')

            tmp = np.zeros(shape=(self._n_dms, self._n_days))
            for index, ID in enumerate(self.dm_ids):
                tmp[index, :] = dm_predictions[dm_predictions['ATM_ID'] == ID][self.times[0]:self.times[1]][
                    'prediction_deposits']
            self._dm_predictions_money = (np.ceil(tmp / 10) * 10).astype('int64')

            tmp = np.zeros(shape=(self._n_dms, self._n_days))
            for index, ID in enumerate(self.dm_ids):
                tmp[index, :] = dm_predictions[dm_predictions['ATM_ID'] == ID][self.times[0]:self.times[1]][
                    'prediction']
            self._dm_predictions_banknotes = np.ceil(tmp).astype('int64')

            tmp = np.zeros(self._n_dms)
            for index, ID in enumerate(self.dm_ids):
                tmp[index] = dm_data[dm_data['ATM_ID'] == ID][self.times[0]:self.times[1]]['OPEN_BAL'][0]
            self._dm_initial_balances_money = (np.floor(tmp / 10) * 10).astype('int64')

            tmp = np.zeros(self._n_dms)
            for index, ID in enumerate(self.dm_ids):
                tmp[index] = dm_data[dm_data['ATM_ID'] == ID][self.times[0]:self.times[1]]['NOPEN_BAL'][0]
            self._dm_initial_balances_banknotes = np.ceil(tmp).astype('int64')

        if kind == 'ipt':
            tmp = np.zeros(shape=(self._n_ipts, self._n_days))
            for index, ID in enumerate(self.ipt_ids):
                tmp[index, :] = ipt_data[ipt_data['id'] == ID][self.times[0]:self.times[1]]['DEPOSITS']
            self._ipt_facts_money = (np.ceil(tmp / 10) * 10).astype('int64')

            tmp = np.zeros(shape=(self._n_ipts, self._n_days))
            for index, ID in enumerate(self.ipt_ids):
                tmp[index, :] = ipt_data[ipt_data['id'] == ID][self.times[0]:self.times[1]]['NDEPOSITS']
                self._ipt_facts_banknotes = np.ceil(tmp).astype('int64')

            tmp = np.zeros(shape=(self._n_ipts, self._n_days))
            for index, ID in enumerate(self.ipt_ids):
                tmp[index, :] = ipt_predictions[ipt_predictions['ATM_ID'] == ID][self.times[0]:self.times[1]][
                    'prediction_deposits']
            self._ipt_predictions_money = (np.ceil(tmp / 10) * 10).astype('int64')

            tmp = np.zeros(shape=(self._n_ipts, self._n_days))
            for index, ID in enumerate(self.ipt_ids):
                tmp[index, :] = ipt_predictions[ipt_predictions['ATM_ID'] == ID][self.times[0]:self.times[1]][
                    'prediction']
            self._ipt_predictions_banknotes = np.ceil(tmp).astype('int64')

            tmp = np.zeros(self._n_ipts)
            for index, ID in enumerate(self.ipt_ids):
                tmp[index] = ipt_data[ipt_data['id'] == ID][self.times[0]:self.times[1]]['OPEN_BAL'][0]
            self._ipt_initial_balances_money = (np.floor(tmp / 10) * 10).astype('int64')

            tmp = np.zeros(self._n_ipts)
            for index, ID in enumerate(self.ipt_ids):
                tmp[index] = ipt_data[ipt_data['id'] == ID][self.times[0]:self.times[1]]['NOPEN_BAL'][0]
            self._ipt_initial_balances_banknotes = np.ceil(tmp).astype('int64')

    def _set_costs(self, kind='atm'):

        """
            Устанавливает параметры _{kind}_cost_encash,  _{kind}_cost_kassa значениями из
            подгруженных таблиц (по умолчанию).

            Принимает:
                kind: 'atm', 'dm' , 'ipt'
         """

        if kind == 'atm':
            tmp = np.zeros(self._n_atms)
            for index, ID in enumerate(self.atm_ids):
                tmp[index] = costs[costs['ATM_ID'] == ID]['NEW_EXPENSE']
            self._atm_cost_encash = np.ceil(tmp).astype('int64')

            tmp = np.zeros(self._n_atms)
            for index, ID in enumerate(self.atm_ids):
                tmp[index] = costs[costs['ATM_ID'] == ID]['NEW_KASSA']
            self._atm_cost_kassa = np.ceil(tmp).astype('int64')

        if kind == 'dm':
            tmp = np.zeros(self._n_dms)
            for index, ID in enumerate(self.dm_ids):
                tmp[index] = costs[costs['ATM_ID'] == ID]['NEW_EXPENSE']
            self._dm_cost_encash = np.ceil(tmp).astype('int64')

            tmp = np.zeros(self._n_dms)
            for index, ID in enumerate(self.dm_ids):
                tmp[index] = costs[costs['ATM_ID'] == ID]['NEW_KASSA']
            self._dm_cost_kassa = np.ceil(tmp).astype('int64')

        if kind == 'ipt':
            tmp = np.zeros(self._n_ipts)
            for index, ID in enumerate(self.ipt_ids):
                tmp[index] = costs[costs['ATM_ID'] == ID]['NEW_EXPENSE']
            self._ipt_cost_encash = np.ceil(tmp).astype('int64')

            tmp = np.zeros(self._n_ipts)
            for index, ID in enumerate(self.ipt_ids):
                tmp[index] = costs[costs['ATM_ID'] == ID]['NEW_KASSA']
            self._ipt_cost_kassa = np.ceil(tmp).astype('int64')

    def _set_intraday_distribution(self, kind='atm'):

        """
            Устанавливает {kind}_intraday_distribution значениями из подгруженных таблиц (по умолчанию).

            Принимает:
                kind: 'atm', 'dm', 'ipt'
        """

        if kind == 'atm':
            list_ids = self.atm_ids
            n_ids = self._n_atms
            dia = atm_intraday[atm_intraday.ATM_ID.isin(list_ids)]

        elif kind == 'dm':
            list_ids = self.dm_ids
            n_ids = self._n_dms
            dia = dm_intraday[dm_intraday.ATM_ID.isin(list_ids)]

        elif kind == 'ipt':
            list_ids = self.ipt_ids
            n_ids = self._n_ipts
            dia = ipt_intraday[ipt_intraday.ATM_ID.isin(list_ids)]

        else:
            sys.exit('Kind have to take only one of three available values: atm, dm or ipt, but you put: ' + str(kind))

        dia = dia.sort_values(['ATM_ID', 'dow', 'hour'])

        dia_each = dia.groupby(by=['ATM_ID', 'dow', 'hour'])['new_sum'].mean().reset_index()
        dia_each.index = range(dia_each.shape[0])

        for _id in list_ids:
            if _id not in dia_each.ATM_ID.unique():
                print('There is not intraday data for ID = ' + str(_id) + ', type = ' + kind + '\n' +
                      'The intraday distribution was set as mean for all ' + kind + 's')

        A1 = []
        for i_id in range(n_ids):
            A2 = []
            for i_dow in range(7):
                d_app = np.array(dia_each[(dia_each.ATM_ID == list_ids[i_id]) & (dia_each.dow == i_dow)]['new_sum'])
                if len(d_app) == 24 and np.all(d_app != np.inf):
                    A2.append(list(np.ceil(d_app)))
                else:
                    A2.append([0] * 24)
            A1.append(A2)
        A1 = np.array(A1)

        for i_id in range(n_ids):
            for i_dow in range(7):
                sum_day = np.sum(A1[i_id, i_dow])
                if sum_day != 0:
                    A1[i_id, i_dow] = A1[i_id, i_dow] / sum_day
                else:
                    A1[i_id, i_dow] = np.array([0] * 24)

        dia_all = dia.groupby(by=['dow', 'hour'])['new_sum'].mean().reset_index()
        dia_all.index = range(dia_all.shape[0])
        A2 = []
        for i_dow in range(7):
            d_app = np.array(dia_all[dia_all.dow == i_dow]['new_sum'])
            if len(d_app) == 24 and np.all(d_app != np.inf):
                A2.append(list(np.ceil(d_app)))
            else:
                A2.append([0] * 9 + [1] * 12 + [0] * 3)
        A2 = np.array(A2)

        for i_dow in range(7):
            sum_day = np.sum(A2[i_dow])
            if sum_day != 0:
                A2[i_dow] = A2[i_dow] / sum_day
            else:
                A2[i_dow] = np.array([0.0] * 9 + [1/12] * 12 + [0.0] * 3)

        for i in range(n_ids):
            for j in range(7):
                if np.sum(A1[i, j]) < 0.1:
                    A1[i, j] = A2[j].copy()
                m1, arg_m1 = np.max(A1[i, j]), np.argmax(A1[i, j])
                A1[i, j, arg_m1] = 1000
                for k in range(24):
                    if k != arg_m1:
                        A1[i, j, k] = np.ceil(A1[i, j, k] * 1000)
                        A1[i, j, arg_m1] -= A1[i, j, k]

        if kind == 'atm':
            self._atm_intraday_distribution = A1 / 1000

        elif kind == 'dm':
            self._dm_intraday_distribution = A1 / 1000

        elif kind == 'ipt':
            self._ipt_intraday_distribution = A1 / 1000


    def _get_hour(self, index, kind='atm', facts=False):  # для facts = True пока не очень корректно

        """
            Ищет час, когда закончатся средства в банкомате в первый день (для atm)
            или когда переполнятся dm и ipt (исходя из внутредневного распределения спроса и начального баланса на
            00:00? - время требует уточнения)

            Принимает:
                index: номер строки, соответствующей диспенсеру, в таблицах
                    _atm_intraday_distribution, _atm_predictions, _atm_facts,
                    _atm_initial_balances; int
                kind: 'atm', 'dm', 'ipt'; str
                facts: True/False; boolean; если facts = False, тогда это лишь прогноз (сделанный в 00:00) часа
                следующего дня, когда закончатся деньги, если facts = True, тогда это реальный час, когда закончатся
                деньги (или не закончатся)

            Возвращает:
                i_hour: час, в которой в первый день закончится self._atm_initial_balance или переполятся
                dm и ipt, int в диапазоне [0, 23]. Деньги, соответственно, заканчиваются между [i_hour, i_hour + 1)
                (или переполянется dm и ipt);
                если деньги не закончатся в первый день - тогда возвращаем i_hour = 24
        """

        if kind == 'atm':

            if facts:
                tmp = self._atm_facts[index, :]
            else:
                tmp = self._atm_predictions[index, :]

            # день недели, который соответствует первой дате
            # рассматриваемого промежутка
            i_dow = pd.to_datetime(self.times[0]).weekday()
            # распределение в этот день
            day_dist = self._atm_intraday_distribution[index, i_dow]
            cons = 0
            flag = False
            for i_hour, hd in enumerate(day_dist):
                cons += hd * tmp[0]
                if cons > self._atm_initial_balances[index]:
                    flag = True
                    break
            if flag:
                return i_hour
            else:
                return 24

        if kind == 'dm':
            return 24

        if kind == 'ipt':
            return 24

    def _row_upload(self, index, row, timestamp='day'):

        """
            Формирует строку инкассаций диспенсера

            Принимает:
                index: номер строки, соответствующей диспенсеру, в таблицах
                    _atm_intraday_distribution, _atm_predictions, _atm_facts,
                    _atm_initial_balances; int
                row: строка из нулей и единиц (1 - инкассируем, 0 - нет), numpy.array
                facts: если False загрузка формируется по предсказаниям, если True по фактическим
                    значениям (необходимо для backtest'ов / валидации), boolean

            Возвращает:
                upload: строка соответствующая row, где вместо единиц записаны суммы загрузок,
                    numpy.array
        """

        state_atm_tr = self.transform_state(row.reshape(1, -1))[0]
        upload = np.zeros(self._n_days)

        if timestamp == 'day':
            i_hour_pred = 0
            for i_day in range(self._n_days):
                upload[i_day] = np.sum(self._atm_predictions[index, i_day:int(state_atm_tr[i_day]) + i_day])

        elif timestamp == 'intraday':
            i_dow = pd.to_datetime(self.times[0]).weekday()
            day_dist = self._atm_intraday_distribution[index, i_dow]
            i_hour_pred = self._get_hour(index, kind='atm', facts=False)

            if row[0] == 1 and i_hour_pred != 24:
                upload[0] = np.sum(self._atm_predictions[index, :int(state_atm_tr[0])]) - \
                            np.sum(day_dist[:i_hour_pred]) * self._atm_predictions[index, 0]  # загружаем в i_hour
            else:
                upload[0] = np.sum(self._atm_predictions[index, :int(state_atm_tr[0])]) #  загружаем или нет в 00:00

            for i_day in range(1, self._n_days):
                upload[i_day] = np.sum(self._atm_predictions[index, i_day:int(state_atm_tr[i_day]) + i_day])

        return upload, i_hour_pred

    def _row_lost(self, index, row, kind='atm', facts=False, timestamp='day'):

        """
            Формирует строку исходящих остатков для объекта НДО (АТМ, ДМ, ИПТ)

            Принимает:
                index: номер строки, соответствующий объекту НДО, в таблицах
                    _{kind}_intraday_distribution, _{kind}_predictions_{money/banknotes},
                    _{kind}_facts_{money/banknotes}, _{kind}_initial_balances_{money/banknotes}; int
                row: строка из нулей и единиц (1 - инкассируем, 0 - нет), numpy.array
                facts: если False остатки формируются по предсказаниям, если True по фактическим
                    значениям (необходимо для backtest'ов / валидации), boolean
                kind: 'atm', 'dm', 'ipt'; str

            Возвращает:
                    if kind == 'atm':
                        lost_in_atm: строка со значениями исходящих остатков, numpy.array
                        downtime_hours: список списков [номер дня, количество простоев в часах], list
                    if kind == 'dm':
                        lost_in_dm_money: строка со значениями исходящих остатков, numpy.array
                        lost_in_dm_banknotes: строка со значениями исходящих остатков в количестве банкнот, numpy.array
                        downtime_days: список списков [номер дня, количество простоев в часах], list
                    if kind == 'ipt':
                        lost_in_ipt_money: cтрока со значениями исходящих остатков, numpy.array
                        lost_in_ipt_banknotes: строка со значениями исходящих остатков в количестве банкнот, numpy.array
                        downtime_days: cписок списков [номер дня, количество простоев в часах], list
        """

        if kind == 'atm':

            lost_in_atm = np.zeros(self._n_days)
            upload, i_hour_pred = self._row_upload(index, row, timestamp=timestamp)

            if facts:
                tmp = self._atm_facts[index, :]
            else:
                tmp = self._atm_predictions[index, :]

            # сначала разбираемся с первым днем;
            # для него надо уточниь часы простоя - для всех остальных дней это необязательно,
            # так как в динамике все равно не будет использоваться и только увеличивает сложность
            # и время работы алгоритма

            if timestamp == 'day':

                downtime_days = []
                for i_day in range(self._n_days):
                    if row[i_day] == 0 and i_day == 0:
                        lost_in_atm[i_day] = self._atm_initial_balances[index] - tmp[i_day]
                    elif row[i_day] == 0 and i_day != 0:
                        lost_in_atm[i_day] = lost_in_atm[i_day - 1] - tmp[i_day]
                    elif row[i_day] == 1:
                        lost_in_atm[i_day] = upload[i_day] - tmp[i_day]
                    if lost_in_atm[i_day] < -machine_epsilon:
                        downtime_days.append(i_day)
                        lost_in_atm[i_day] = 0

                return lost_in_atm, downtime_days

            elif timestamp == 'intraday':

                downtime_hours = []

                i_dow = pd.to_datetime(self.times[0]).weekday()
                day_dist = self._atm_intraday_distribution[index, i_dow]

                i_hour = self._get_hour(index, kind=kind, facts=facts)

                if row[0] == 1 and i_hour_pred != 24 and i_hour < i_hour_pred:
                    for h in range(i_hour, i_hour_pred):
                        downtime_hours.append([0, h])
                    lost_in_atm[0] = upload[0] - np.sum(day_dist[i_hour_pred:]) * tmp[0]
                    if lost_in_atm[0] < -machine_epsilon:
                        k = 24
                        while lost_in_atm[0] < -machine_epsilon and k >= 1:
                            k -= 1
                            lost_in_atm[0] += day_dist[k] * tmp[0]
                        if k == 0 and lost_in_atm[0] < -machine_epsilon:
                            print(index)
                            print(lost_in_atm[0])
                            print(day_dist)
                            print(i_hour)
                            print(i_hour_pred)
                            sys.exit('WARNING! case 1!')
                        for h in range(k, 24):
                            downtime_hours.append([0, h])
                        lost_in_atm[0] = 0

                elif row[0] == 1 and i_hour_pred != 24 and i_hour >= i_hour_pred:
                    lost_in_atm[0] = upload[0] - np.sum(day_dist[i_hour_pred:]) * tmp[0]
                    if lost_in_atm[0] < -machine_epsilon:
                        k = 24
                        while lost_in_atm[0] < -machine_epsilon and k >= 1:
                            k -= 1
                            lost_in_atm[0] += day_dist[k] * tmp[0]
                        if k == 0 and lost_in_atm[0] < -machine_epsilon:
                            print(index)
                            print(lost_in_atm[0])
                            print(day_dist)
                            print(i_hour)
                            print(i_hour_pred)
                            sys.exit('WARNING! case 2!')
                        for h in range(k, 24):
                            downtime_hours.append([0, h])
                        lost_in_atm[0] = 0

                elif row[0] == 1 and i_hour_pred == 24:
                    lost_in_atm[0] = upload[0] - tmp[0]
                    if lost_in_atm[0] < -machine_epsilon:
                        k = 24
                        while lost_in_atm[0] < -machine_epsilon and k >= 1:
                            k -= 1
                            lost_in_atm[0] += day_dist[k] * tmp[0]
                        if k == 0 and lost_in_atm[0] < -machine_epsilon:
                            print(index)
                            print(lost_in_atm[0])
                            print(day_dist)
                            print(i_hour)
                            print(i_hour_pred)
                            sys.exit('WARNING! case 3!')
                        for h in range(k, 24):
                            downtime_hours.append([0, h])
                        lost_in_atm[0] = 0

                elif row[0] == 0:
                    lost_in_atm[0] = self._atm_initial_balances[index] - tmp[0]
                    if lost_in_atm[0] < -machine_epsilon:
                        for h in range(i_hour, 24):
                            downtime_hours.append([0, h])
                        lost_in_atm[0] = 0

                for i_day in range(1, self._n_days):
                    if row[i_day] == 0:
                        lost_in_atm[i_day] = lost_in_atm[i_day - 1] - tmp[i_day]
                    else:
                        lost_in_atm[i_day] = upload[i_day] - tmp[i_day]
                    if lost_in_atm[i_day] < -machine_epsilon:
                        #print(lost_in_atm[i_day])
                        downtime_hours.append(i_day)   # уточняем часы простоя только для первого дня!
                        lost_in_atm[i_day] = 0

                if -machine_epsilon <= lost_in_atm[0] <= machine_epsilon:
                    lost_in_atm[0] = 0

                return lost_in_atm, downtime_hours

        if kind == 'dm':
            lost_in_dm_money = np.zeros(self._n_days)
            lost_in_dm_banknotes = np.zeros(self._n_days)
            if facts:
                tmp_money = self._dm_facts_money[index, :]
                tmp_banknotes = self._dm_facts_banknotes[index, :]
            else:
                tmp_money = self._dm_predictions_money[index, :]
                tmp_banknotes = self._dm_predictions_banknotes[index, :]

            if timestamp == 'day':

                downtime_days = []
                for i_day in range(self._n_days):
                    if row[i_day] == 0 and i_day == 0:
                        lost_in_dm_money[i_day] = self._dm_initial_balances_money[index] + tmp_money[i_day]
                        lost_in_dm_banknotes[i_day] = self._dm_initial_balances_banknotes[index] + tmp_banknotes[i_day]
                    elif row[i_day] == 0 and i_day != 0:
                        lost_in_dm_money[i_day] = lost_in_dm_money[i_day - 1] + tmp_money[i_day]
                        lost_in_dm_banknotes[i_day] = lost_in_dm_banknotes[i_day - 1] + tmp_banknotes[i_day]
                    elif row[i_day] != 0:
                        lost_in_dm_money[i_day] = tmp_money[i_day]
                        lost_in_dm_banknotes[i_day] = tmp_banknotes[i_day]

                    if lost_in_dm_banknotes[i_day] > self.dm_cassette[index] * self.dm_capacity[index]:
                        downtime_days.append(i_day)
                        lost_in_dm_money[i_day] = lost_in_dm_money[i_day - 1] + \
                                                  (self.dm_cassette[index] * self.dm_capacity[index] -
                                                   lost_in_dm_banknotes[i_day - 1]) * average_denomination
                        lost_in_dm_banknotes[i_day] = self.dm_cassette[index] * self.dm_capacity[index]
                return lost_in_dm_money, lost_in_dm_banknotes, downtime_days

            elif timestamp == 'intraday':
                pass

        if kind == 'ipt':
            lost_in_ipt_money = np.zeros(self._n_days)
            lost_in_ipt_banknotes = np.zeros(self._n_days)
            if facts:
                tmp_money = self._ipt_facts_money[index, :]
                tmp_banknotes = self._ipt_facts_banknotes[index, :]
            else:
                tmp_money = self._ipt_predictions_money[index, :]
                tmp_banknotes = self._ipt_predictions_banknotes[index, :]

            if timestamp == 'day':

                downtime_days = []
                for i_day in range(self._n_days):
                    if row[i_day] == 0 and i_day == 0:
                        lost_in_ipt_money[i_day] = self._ipt_initial_balances_money[index] + tmp_money[i_day]
                        lost_in_ipt_banknotes[i_day] = self._ipt_initial_balances_banknotes[index] + tmp_banknotes[i_day]
                    elif row[i_day] == 0 and i_day != 0:
                        lost_in_ipt_money[i_day] = lost_in_ipt_money[i_day - 1] + tmp_money[i_day]
                        lost_in_ipt_banknotes[i_day] = lost_in_ipt_banknotes[i_day - 1] + tmp_banknotes[i_day]
                    elif row[i_day] != 0:
                        lost_in_ipt_money[i_day] = tmp_money[i_day]
                        lost_in_ipt_banknotes[i_day] = tmp_banknotes[i_day]

                    if lost_in_ipt_banknotes[i_day] > self.ipt_cassette[index] * self.ipt_capacity[index]:
                        downtime_days.append(i_day)
                        lost_in_ipt_money[i_day] = lost_in_ipt_money[i_day - 1] + \
                                                   (self.ipt_cassette[index] * self.ipt_capacity[index] -
                                                    lost_in_ipt_banknotes[i_day - 1]) * average_denomination
                        lost_in_ipt_banknotes[i_day] = self.ipt_cassette[index] * self.ipt_capacity[index]

                return lost_in_ipt_money, lost_in_ipt_banknotes, downtime_days

            elif timestamp == 'intraday':
                pass

    def _row_cost(self, index, row, kind='atm', facts=False, timestamp='day'):

        """
            По строке режима инкассаций считает итоговую стоимость (фондирование + инкассация)

            Принимает:
                index: номер строки, соответствующий объекту НДО, в таблицах
                    _{kind}_intraday_distribution, _{kind}_predictions_{money/banknotes},
                    _{kind}_facts_{money/banknotes}, _{kind}_initial_balances_{money/banknotes}; int
                row: строка из нулей и единиц (1 - инкассируем, 0 - нет), numpy.array
                facts: если False остатки формируются по предсказаниям, если True по фактическим
                    значениям (необходимо для backtest'ов / валидации), boolean
                kind: 'atm', 'dm', 'ipt'; str

            Возвращает:
                суммарную стоимость, int
                downtime_hours: cписок списков [номер дня, количество простоев в часах], list

        """

        funding_costs = np.zeros(self._n_days)
        encash_costs = np.zeros(self._n_days)

        if kind == 'atm':
            lost_in_atm, downtime_timestamp = self._row_lost(index, row, kind=kind, facts=facts, timestamp=timestamp)
            for i_day in range(self._n_days):
                funding_costs[i_day] = lost_in_atm[i_day] * (self.f_rate / 365)
                if row[i_day]:
                    encash_costs[i_day] = self._atm_cost_encash[index] + self._atm_cost_kassa[index]
                else:
                    encash_costs[i_day] = 0
            return np.sum(funding_costs + encash_costs), downtime_timestamp, funding_costs, encash_costs

        if kind == 'dm':
            lost_in_dm_money, _, downtime_days = self._row_lost(index, row, facts=facts, kind=kind, timestamp=timestamp)
            for i_day in range(self._n_days):
                funding_costs[i_day] = lost_in_dm_money[i_day] * (self.f_rate / 365)
                if row[i_day] and self._dm_full_mask[index, i_day] == 1:
                    encash_costs[i_day] = self._dm_cost_kassa[index]
                elif row[i_day] and self._dm_full_mask[index, i_day] == 0:
                    encash_costs[i_day] = self._dm_cost_encash[index] + self._dm_cost_kassa[index]
                else:
                    encash_costs[i_day] = 0
            return np.sum(funding_costs + encash_costs), downtime_days

        if kind == 'ipt':
            lost_in_ipt_money, _, downtime_days = self._row_lost(index, row, facts=facts,
                                                                 kind=kind, timestamp=timestamp)
            for i_day in range(self._n_days):
                funding_costs[i_day] = lost_in_ipt_money[i_day] * (self.f_rate / 365)
                if row[i_day]:
                    encash_costs[i_day] = self._ipt_cost_encash[index] + self._ipt_cost_kassa[index]
                else:
                    encash_costs[i_day] = 0
            return np.sum(funding_costs + encash_costs), downtime_days

    def _row_check(self, index, row, kind='atm', timestamp='day'):

        """
            Проверяет строку режима инкассаций на валидность ограничениям

            Принимает:
                index: номер строки, соответствующий объекту НДО, в таблицах
                    _{kind}_intraday_distribution, _{kind}_predictions_{money/banknotes},
                    _{kind}_facts_{money/banknotes}, _{kind}_initial_balances_{money/banknotes}; int
                row: строка из нулей и единиц (1 - инкассируем, 0 - нет), numpy.array
                facts: если False ограничения сверяются по предсказаниям, если True по фактическим значениям, boolean
                kind: 'atm', 'dm', 'ipt'; str

            Возвращает:
                True/False

        """

        if kind == 'atm':
            for i_day in range(self._n_days):
                if self._atm_full_mask[index][i_day] == 1 and row[i_day] == 0:
                    # print(index)
                    # print('Error 1')
                    return False
                if np.isnan(self._atm_full_mask[index][i_day]) and row[i_day] == 1:
                    # print(index)
                    # print('Error 2')
                    return False
            if self._atm_initial_balances[index] > self.atm_capacity[index]:
                # print(index)
                # print('Error 3')
                return False
            if (self._row_upload(index, row, timestamp=timestamp)[0] > self.atm_capacity[index]).any():
                # print(index)
                # print('Error 4')
                return False
            ind_wall = np.where(self._atm_full_mask[index] == 2)[0]
            if ind_wall.size:
                ind_wall = ind_wall[0]
                if np.sum(row[:ind_wall + 1]) == 0:
                    # print(index)
                    # print('Error 5')
                    return False
            # downtime_timestamp = self._row_lost(index, row, kind, facts=False, timestamp=timestamp)[1]
            # if downtime_timestamp:
            #     print('WARNING! THERE IS(ARE) SOME ERROR(S) IN _row_lost(), BECAUSE OF DOWNTIMES: ' +
            #           str(downtime_timestamp))
            return True

        if kind == 'dm':
            for i_day in range(self._n_days):
                if np.isnan(self._dm_full_mask[index][i_day]) and row[i_day] == 1:
                    return False
                if self._dm_full_mask[index][i_day] == 1 and row[i_day] == 0:
                    return False
            if self._row_lost(index, row, kind=kind, timestamp=timestamp)[2]:
                return False
            if self._dm_initial_balances_banknotes[index] > self.dm_cassette[index] * self.dm_capacity[index]:
                return False
            ind_wall = np.where(self._dm_full_mask[index] == 2)[0]
            if ind_wall.size:
                ind_wall = ind_wall[0]
                if np.sum(row[:ind_wall + 1]) == 0:
                    return False
            return True

        if kind == 'ipt':
            for i_day in range(self._n_days):
                if np.isnan(self._ipt_full_mask[index][i_day]) and row[i_day] == 1:
                    return False
            if self._row_lost(index, row, kind=kind, timestamp=timestamp)[2]:
                return False
            if self._ipt_initial_balances_banknotes[index] > self.ipt_cassette[index] * self.ipt_capacity[index]:
                return False
            ind_wall = np.where(self._ipt_full_mask[index] == 2)[0]
            if ind_wall.size:
                ind_wall = ind_wall[0]
                if np.sum(row[:ind_wall + 1]) == 0:
                    return False
            return True

    def table_upload(self, table, timestamp='day'):
        """
               Формирует таблицу загрузок диспенсеров

               Принимает:
                   table: таблица из 0 и 1 (1 - инкассируем, 0 - нет), numpy.array
                   facts: если False загрузка формируется по предсказаниям, если True по фактическим
                       значениям (необходимо для backtest'ов / валидации), boolean
               Возвращает:
                   upload: таблица соответствующая table, где вместо единиц записаны суммы загрузок,
                       numpy.array
           """
        upload = np.zeros(shape=(self._n_atms, self._n_days))
        hours = []
        for index in range(self._n_atms):
            upload[index, :], i_hour = self._row_upload(index, table[index, :], timestamp=timestamp)
            hours.append(i_hour)
        return upload, hours

    def table_lost(self, table, kind='atm', facts=False, timestamp='day'):

        """
            Формирует таблицу исходящих остатков

             Принимает:
                table: таблица из 0 и 1 (1 - инкассируем, 0 нет), numpy.array
                facts: если False остатки формируются по предсказаниям, если True по фактическим
                    значениям (необходимо для backtest'ов / валидации), boolean
                kind: 'atm', 'dm', 'ipt'; str


             Возвращает:
                    if kind == 'atm':
                        lost: таблица со значениями исходящих остатков, numpy.array
                        downtimes: список списков [номер дня, количество простоев в часах] для каждой строки table, list
                    if kind == 'dm':
                        lost_money: таблица со значениями исходящих остатков, numpy.array
                        lost_banknotes: таблица со значениями исходящих остатков в количестве банкнот, numpy.array
                        downtimes: список списков [номер дня, количество простоев в часах] для каждой строки table, list
                    if kind == 'ipt':
                        lost_money: таблица со значениями исходящих остатков, numpy.array
                        lost_banknotes: таблица со значениями исходящих остатков в количестве банкнот, numpy.array
                        downtimes: список списков [номер дня, количество простоев в часах] для каждой строки table, list



        """

        if kind == 'atm':
            lost = np.zeros(shape=(self._n_atms, self._n_days))
            downtimes = []
            for index in range(self._n_atms):
                lost[index, :], downtime_timestamp = self._row_lost(index, table[index, :], kind=kind,
                                                                    facts=facts, timestamp=timestamp)
                downtimes.append(downtime_timestamp)
            return lost, downtimes

        if kind == 'dm':
            lost_money = np.zeros(shape=(self._n_dms, self._n_days))
            lost_banknotes = np.zeros(shape=(self._n_dms, self._n_days))
            downtimes = []
            for index in range(self._n_dms):
                lost_money[index, :], lost_banknotes[index, :], downtime_days = \
                    self._row_lost(index, table[index, :], kind=kind, facts=facts, timestamp=timestamp)
                downtimes.append(downtime_days)
            return lost_money, lost_banknotes, downtimes

        if kind == 'ipt':
            lost_money = np.zeros(shape=(self._n_ipts, self._n_days))
            lost_banknotes = np.zeros(shape=(self._n_ipts, self._n_days))
            downtimes = []
            for index in range(self._n_ipts):
                lost_money[index, :], lost_banknotes[index, :], downtime_days = \
                    self._row_lost(index, table[index, :], kind=kind, facts=facts, timestamp=timestamp)
                downtimes.append(downtime_days)
            return lost_money, lost_banknotes, downtimes

    def table_cost(self, table, kind='atm', facts=False, timestamp='day'):

        """
            По таблицы режима инкассаций считает итоговую стоимость (фондирование + инкассация)

            Принимает:
                table: таблица из 0 и 1 (1 - инкассируем, 0 нет), numpy.array
                facts: если False остатки формируются по предсказаниям, если True по фактическим
                    значениям (необходимо для backtest'ов / валидации), boolean
                kind: 'atm', 'dm', 'ipt'; str

            Возвращает:
                суммарную стоимость, int
                downtime: cписок списков [номер дня, количество простоев в часах] для каждого объекта НДО таблицы, list

        """

        #start = time.time()
        cost = 0
        downtime = []
        cost_daily_funding = np.zeros(shape=(self._n_atms, self._n_days))
        cost_daily_encash = np.zeros(shape=(self._n_atms, self._n_days))

        if kind == 'atm':
            for index in range(self._n_atms):
                cost_atm, downtime_timestamp, cf, ce = self._row_cost(index, table[index, :], kind=kind, facts=facts,
                                                              timestamp=timestamp)
                cost += cost_atm
                downtime.append(downtime_timestamp)
                cost_daily_funding[index, :] = cf
                cost_daily_encash[index, :] = ce
        if kind == 'dm':
            for index in range(self._n_dms):
                cost_dm, downtime_timestamp = self._row_cost(index, table[index, :], kind=kind,
                                                             facts=facts, timestamp=timestamp)
                cost += cost_dm
                downtime.append(downtime_timestamp)
        if kind == 'ipt':
            for i_ipt in range(self._n_ipts):
                cost_ipt, downtime_timestamp = self._row_cost(index, table[index, :], kind=kind,
                                                              facts=facts, timestamp=timestamp)
                cost += cost_ipt
                downtime.append(downtime_timestamp)

        #end = time.time()
        #print('time = ' + str(end-start))

        return cost, downtime, cost_daily_funding, cost_daily_encash

    def table_check(self, table, kind='atm', timestamp='day'):

        if kind == 'atm':
            for index in range(self._n_atms):
                if not self._row_check(index, table[index, :], kind=kind, timestamp=timestamp):
                    return False
            for i_day in range(self._n_days):
                if np.sum(table[:, i_day]) > self.power[i_day] * self.priority[kind]:
                    return False
            return True

        if kind == 'dm':
            for index in range(self._n_dms):
                if not self._row_check(index, table[index, :], kind=kind, timestamp=timestamp):
                    return False
            for i_day in range(self._n_days):
                if np.sum(table[:, i_day]) - np.sum(self._dm_atm_state_mask[:, i_day]) > \
                                self.power[i_day] * self.priority[kind]:
                    return False
            return True

        if kind == 'ipt':
            for index in range(self._n_ipts):
                if not self._row_check(index, table[index, :], kind=kind, timestamp=timestamp):
                    return False
            for i_day in range(self._n_days):
                if np.sum(table[:, i_day]) > self.power[i_day] * self.priority[kind]:
                    return False
            return True

    def _set_masks(self, kind='atm', masks_order=['first_encash', 'availability'], mode_order='rigid'):

        """
                Функция создает маски загрузок.

                Поддерживается три типа масок:

                1) маска первых инкассаций (если параметр first_encash = True, тогда маска первых инкассаций
                будет установлена и алгоритм будет учитывать ее структуру при выборе initial_state и дальнейшей
                оптимизации; если fisrt_encash=False, то соотвественно алгоритм будет игнорировать первые необходимые
                инкассации исходя из начальных балансов, правда в результате этого могут возникнуть простои).
                Маска первых инкассаций имеет такой же размер как и матрица состояния (то есть self._n_atms строк и
                self._n_days столбцов) и содержит два различных значения: 0 и 2.
                Если в маске стоит 2, то в той же строке в
                любом состоянии до этой 2 включительно должна стоять хотя бы одна загрузка. Если в маске стоит 0, то он не
                оказывает никакого эффекта на матрицы состояний.

                Устанавливается для всех устройств: 'atm', 'dm', 'ipt'.

                2) маска доступности (если параметр availability = True, тогда маска доступности
                будет установлена и алгоритм будет учитывать ее структуру при выборе initial_state и дальнейшей
                оптимизации; если availability=False, то соотвественно алгоритм будет игнорировать эту маску). Маска доступности
                имеет такой же размер как и матрица состояния (то есть self._n_atms строк и
                self._n_days столбцов) и содержит два различных значения: 0 и np.nan.
                np.nan в маске означает, что у инкассаторов в конкретный день для конкретного банкомата выходной и они не могут
                загрузить/выгрузить деньги; 0 - будний день и этот 0 не оказывает никакого эффекта на матрицы состояний.

                Устанавливается для всех устройств: 'atm', 'dm', 'ipt'.

                2) маска взаимных инкассаций dm и atm (если параметр atm_state = True, тогда взаимных инкассаций dm и atm
                будет установлена и алгоритм будет учитывать ее структуру при выборе initial_state и дальнейшей
                оптимизации; если atm_state=False, то соотвественно алгоритм будет игнорировать эту маску). Маска взаимных
                инкассаций имеет такой же размер как и матрица состояния (то есть self._n_atms строк и
                self._n_days столбцов) и содержит два различных значения: 0 и 1. Маска нужна только для того, чтобы
                рассматривать только те состояния dm, дни загрузок которых соответствуют дням загрузок соответствующего atm.
                То есть маска взаимных инкассаций просто копия состояния atm (под которое надо подогнать dm).

                Устанавливается только для одного устройства: 'dm'.


                Принимает:
                    kind - тип устройства, для которого нужно создать маски, 'atm', 'dm', 'ipt'; str
                    first_encash - нужна ли маска первых инкассаций, True/False; boolean
                    availability - нужна ли маска доступности банкоматов для инкассаторов, True/False; boolean
                    atm_state - нужна ли маска совместных инкассаций dm и atm, True/False; boolean

                Устанавливает поля в зависимости от того, какой был выбран тип устройства и какие аргументы были True:
                    или self._atm_first_encash_mask,
                        self._atm_availability_mask;
                    или self._dm_first_encash_mask,
                        self._dm_availability_mask,
                        self._dm_atm_state_mask;
                    или self._ipt_first_encash_mask,
                        self._ipt_availability_mask
        """

        if kind == 'atm':

            if 'first_encash' in masks_order:
                mask = np.zeros(shape=(self._n_atms, self._n_days))
                for i_atm in range(self._n_atms):
                    cum_sum = 0
                    for i_day in range(self._n_days):
                        cum_sum += self._atm_predictions[i_atm, i_day]
                        if cum_sum > self._atm_initial_balances[i_atm]:
                            break
                    if i_day == self._n_days - 1 and cum_sum <= self._atm_initial_balances[i_atm]:
                        i_day = self._n_days
                    for j in range(i_day):
                        if mode_order == 'rigid':
                            mask[i_atm, j] = np.nan
                        elif mode_order == 'flexible':
                            mask[i_atm, j] = 0
                    if i_day < self._n_days:
                        if mode_order == 'rigid':
                            mask[i_atm, i_day] = 1
                        elif mode_order == 'flexible':
                            mask[i_atm, i_day] = 2
                self._atm_first_encash_mask = mask

            if 'availability' in masks_order:
                mask = np.zeros(shape=(self._n_atms, self._n_days))
                for i, ID in enumerate(self.atm_ids):
                    avail_id = atm_dm_availability[atm_dm_availability.ATM_ID == ID]
                    if len(avail_id) > 0:
                        for j, index in enumerate(pd.date_range(start=self.times[0], end=self.times[1], freq='D')):
                            for k in range(7):
                                if index.weekday() == k and \
                                                avail_id.START_TIME2.iloc[k] == '0' and avail_id.END_TIME2.iloc[k] == '0':
                                    mask[i, j] = np.nan
                        if np.all(np.isnan(mask[i, :])):
                            for j in range(len(pd.date_range(start=self.times[0], end=self.times[1], freq='D'))):
                                mask[i, j] = 0
                                
                self._atm_availability_mask = mask

            if set(masks_order) - {'first_encash', 'availability'}:
                print('WARNING! THERE IS(ARE) UNKNOWN MASK(S) IN masks_order!: ' +
                      str(set(masks_order) - {'first_encash', 'availability'}))
            if {mode_order} - {'rigid', 'flexible'}:
                print('WARNING! THERE IS(ARE) UNKNOWN MODE(S) IN mode_order!: ' +
                      str(set(mode_order) - {'rigid', 'flexible'}))

            self._atm_masks_order = masks_order
            self._atm_mode_order = mode_order

        if kind == 'dm':

            if 'first_encash' in masks_order:
                mask = np.zeros(shape=(self._n_dms, self._n_days))
                for i_dm in range(self._n_dms):
                    cum_sum = self._dm_initial_balances_banknotes[i_dm]
                    for i_day in range(self._n_days):
                        cum_sum += self._dm_predictions_banknotes[i_dm, i_day]
                        if cum_sum > self.dm_capacity[i_dm] * self.dm_cassette[i_dm]:
                            break
                    if i_day == self._n_days - 1 and cum_sum <= self.dm_capacity[i_dm] * self.dm_cassette[i_dm]:
                        i_day = self._n_days
                    for j in range(i_day):
                        if mode_order == 'rigid':
                            mask[i_dm, j] = np.nan
                        elif mode_order == 'flexible':
                            mask[i_dm, j] = 0
                    if i_day < self._n_days:
                        if mode_order == 'rigid':
                            mask[i_dm, i_day] = 1
                        elif mode_order == 'flexible':
                            mask[i_dm, i_day] = 2
                self._dm_first_encash_mask = mask

            if 'availability' in masks_order:
                mask = np.zeros(shape=(self._n_dms, self._n_days))
                for i, ID in enumerate(self.dm_ids):
                    avail_id = atm_dm_availability[atm_dm_availability.ATM_ID == ID]
                    for j, index in enumerate(pd.date_range(start=self.times[0], end=self.times[1], freq='D')):
                        for k in range(7):
                            if index.weekday() == k and \
                                            avail_id.START_TIME2.iloc[k] == '0' and avail_id.END_TIME2.iloc[k] == '0':
                                mask[i, j] = np.nan
                    if np.all(np.isnan(mask[i, :])):
                        for j in range(len(pd.date_range(start=self.times[0], end=self.times[1], freq='D'))):
                            mask[i, j] = 0

                self._dm_availability_mask = mask

            if 'atm_state' in masks_order:
                index = [self.atm_ids.index(i) for i in self.dm_ids]
                self._dm_atm_state_mask = self.atm_table[index]

            if set(masks_order) - {'first_encash', 'availability', 'atm_state'}:
                print('WARNING! THERE IS(ARE) UNKNOWN MASK(S) IN masks_order!: ' +
                      str(set(masks_order) - {'first_encash', 'availability', 'atm_state'}))
            if {mode_order} - {'rigid', 'flexible'}:
                print('WARNING! THERE IS(ARE) UNKNOWN MODE(S) IN mode_order!: ' +
                      str(set(mode_order) - {'rigid', 'flexible'}))

            self._dm_masks_order = masks_order
            self._dm_mode_order = mode_order

        if kind == 'ipt':

            if 'first_encash' in masks_order:
                mask = np.zeros(shape=(self._n_ipts, self._n_days))
                for i_ipt in range(self._n_ipts):
                    cum_sum = self._ipt_initial_balances_banknotes[i_ipt]
                    for i_day in range(self._n_days):
                        cum_sum += self._ipt_predictions_banknotes[i_ipt, i_day]
                        if cum_sum > self.ipt_capacity[i_ipt] * self.ipt_cassette[i_ipt]:
                            break
                    if i_day == self._n_days - 1 and cum_sum <= self.ipt_capacity[i_ipt] * self.ipt_cassette[i_ipt]:
                        i_day = self._n_days
                    for j in range(i_day):
                        if mode_order == 'rigid':
                            mask[i_ipt, j] = np.nan
                        elif mode_order == 'flexible':
                            mask[i_ipt, j] = 0
                    if i_day < self._n_days:
                        if mode_order == 'rigid':
                            mask[i_ipt, i_day] = 1
                        elif mode_order == 'flexible':
                            mask[i_ipt, i_day] = 2
                self._ipt_first_encash_mask = mask

            if 'availability' in masks_order:
                mask = np.zeros(shape=(self._n_ipts, self._n_days))
                for i, ID in enumerate(self.ipt_ids):
                    avail_id = ipt_availability[ipt_availability.IPT_ID == ID]
                    for j, index in enumerate(pd.date_range(start=self.times[0], end=self.times[1], freq='D')):
                        for k in range(7):
                            if index.weekday() == k and \
                                            avail_id.START_TIME2.iloc[k] == '0' and avail_id.END_TIME2.iloc[k] == '0':
                                mask[i, j] = np.nan
                self._ipt_availability_mask = mask

            if set(masks_order) - {'first_encash', 'availability'}:
                print('WARNING! THERE IS(ARE) UNKNOWN MASK(S) IN masks_order!: ' +
                      str(set(masks_order) - {'first_encash', 'availability'}))
            if {mode_order} - {'rigid', 'flexible'}:
                print('WARNING! THERE IS(ARE) UNKNOWN MODE(S) IN mode_order!: ' +
                      str(set(mode_order) - {'rigid', 'flexible'}))

            self._ipt_masks_order = masks_order
            self._ipt_mode_order = mode_order


    def _synchronize_masks(self, kind='atm'):
        """
            Синхронизирует сформированные фукнцией self._set_masks() маски между собой.

            Принимает:
                kind - тип устройства, для которого надо синхронизировать маски, 'atm', 'dm', 'ipt'; str

            Устанавливает в зависимости от того, какое устройство было выбрано:
                или self._atm_full_mask - агрегированная маска для atm,
                или self._dm_full_mask - агрегированная маска для dm,
                или self._ipt_full_mask - агрегированная маска для ipt
        """

        if kind == 'atm':
            mask1 = deepcopy(self._atm_first_encash_mask)
            mask2 = deepcopy(self._atm_availability_mask)

            for i_atm in range(self._n_atms):
                if self._atm_mode_order == 'rigid':
                    i_day = np.where(mask1[i_atm, :] == 1)[0]
                else:
                    i_day = np.where(mask1[i_atm, :] == 2)[0]
                if i_day.size:
                    i_day = i_day[0].astype('int64')
                    if i_day == 0 and np.isnan(mask2[i_atm, i_day]):
                        if self._atm_mode_order == 'rigid':
                            for index in np.where(mask2[i_atm, :] == 0)[0]:
                                if index > i_day:
                                    mask1[i_atm, index] = 1
                                    break
                            for j in range(i_day, index):
                                mask1[i_atm, j] = np.nan
                        elif self._atm_mode_order == 'flexible':
                            for index in np.where(mask2[i_atm, :] == 0)[0]:
                                if index > i_day:
                                    mask1[i_atm, index] = 2
                                    break
                            for j in range(i_day, index):
                                mask1[i_atm, j] = 0
                        print('Number of ATM: ', i_atm)
                        print('with downtimes_type1: ' + str(index - i_day) + ' days: ' +
                              str(np.linspace(i_day, index-1, index-i_day)))
                    elif (0 < i_day < self._n_days) and np.isnan(mask2[i_atm, i_day]):
                        k = i_day - 1
                        while np.isnan(mask2[i_atm, k]) and k >= 0:
                            k -= 1
                        if k >= 0:
                            if self._atm_mode_order == 'rigid':
                                mask1[i_atm, k] = 1
                                for j in range(k + 1, i_day + 1):
                                    mask1[i_atm, j] = 0
                            elif self._atm_mode_order == 'flexible':
                                mask1[i_atm, k] = 2
                                for j in range(k + 1, i_day + 1):
                                    mask1[i_atm, j] = 0
                        else:
                            if self._atm_mode_order == 'rigid':
                                for index in np.where(mask2[i_atm, :] == 0)[0]:
                                    if index > i_day:
                                        mask1[i_atm, index] = 1
                                        break
                                print(i_atm)
                                print(mask2[i_atm, :])
                                for j in range(i_day, index):
                                    mask1[i_atm, j] = np.nan
                            elif self._atm_mode_order == 'flexible':
                                for index in np.where(mask2[i_atm, :] == 0)[0]:
                                    if index > i_day:
                                        mask1[i_atm, index] = 2
                                        break
                                for j in range(i_day, index):
                                    mask1[i_atm, j] = 0
                            print('Number of ATM: ', i_atm)
                            print('with downtimes_type2: ' + str(index - i_day) + ' days: ' +
                                  str(np.linspace(i_day, index-1, index-i_day)))

                self._atm_full_mask = mask1 + mask2

        if kind == 'dm':
            mask1 = deepcopy(self._dm_first_encash_mask)
            mask2 = deepcopy(self._dm_availability_mask)
            mask3 = deepcopy(self._dm_atm_state_mask)
            for i_dm in range(self._n_dms):
                i_day = np.where(mask1[i_dm, :] == 2)[0]
                if i_day.size:
                    i_day = i_day[0].astype('int64')
                    if i_day == 0 and np.isnan(mask2[i_dm, i_day]):
                        for index in np.where(mask2[i_dm, :] == 0)[0]:
                            if index > i_day:
                                mask1[i_dm, index] = 2
                                break
                        for j in range(i_day, index):
                            mask1[i_dm, j] = 0
                        print('Number of DM: ', i_dm)
                        print('with downtimes_type1: ' + str(index - i_day) + ' days')
                    elif (0 < i_day < self._n_days) and np.isnan(mask2[i_dm, i_day]):
                        k = i_day - 1
                        while np.isnan(mask2[i_dm, k]) and k >= 0:
                            k -= 1
                        if k >= 0:
                            mask1[i_dm, k] = 2
                            for j in range(k + 1, i_day + 1):
                                mask1[i_dm, j] = 0
                        else:
                            for index in np.where(mask2[i_dm, :] == 0)[0]:
                                if index > i_day:
                                    mask1[i_dm, index] = 2
                                    break
                            for j in range(i_day, index):
                                mask1[i_dm, j] = 0
                            print('Number of DM: ', i_dm)
                            print('with downtimes_type2: ' + str(index - i_day) + ' days')
                ind2 = np.where(mask1[i_dm, :] == 2)[0]
                ind1 = np.where(mask3[i_dm, :] == 1)[0]
                if ind1.size and ind2.size:
                    if ind2[0] >= ind1[0]:
                        mask1[i_dm, :] = np.where(mask1[i_dm, :] == 2, 0, 0)
            self._dm_full_mask = mask1 + mask2 + mask3

        if kind == 'ipt':
            mask1 = deepcopy(self._ipt_first_encash_mask)
            mask2 = deepcopy(self._ipt_availability_mask)
            for i_ipt in range(self._n_ipts):
                i_day = np.where(mask1[i_ipt, :] == 2)[0]
                if i_day.size:
                    i_day = i_day[0].astype('int64')
                    if i_day == 0 and np.isnan(mask2[i_ipt, i_day]):
                        for index in np.where(mask2[i_ipt, :] == 0)[0]:
                            if index > i_day:
                                mask1[i_ipt, index] = 2
                                break
                        for j in range(i_day, index):
                            mask1[i_ipt, j] = 0
                        print('Number of IPT: ', i_ipt)
                        print('with downtimes_type1: ' + str(index - i_day) + ' days')
                    elif (0 < i_day < self._n_days) and np.isnan(mask2[i_ipt, i_day]):
                        k = i_day - 1
                        while np.isnan(mask2[i_ipt, k]) and k >= 0:
                            k -= 1
                        if k >= 0:
                            mask1[i_ipt, k] = 2
                            for j in range(k + 1, i_day + 1):
                                mask1[i_ipt, j] = 0
                        else:
                            for index in np.where(mask2[i_ipt, :] == 0)[0]:
                                if index > i_day:
                                    mask1[i_ipt, index] = 2
                                    break
                            for j in range(i_day, index):
                                mask1[i_ipt, j] = 0
                            print('Number of IPT: ', i_ipt)
                            print('with downtimes_type2: ' + str(index - i_day) + ' days')
            self._ipt_full_mask = mask1 + mask2

    def _set_initial_table(self, kind='atm', timestamp='day', n_attempts=1000):

        """
            Устанавливает валидное начальное состояние.
            Валидное означает, что:
                состояние подходит под агрегированные маски;
                количество загрузок в день i_day не превосходит заданной мощности power[i_day] для любого i_day
                никогда не загружается больше, чем capacity рублей (для atm), никогда нет простоев (для dm и ipt)

            Принимает:
                kind - тип устройства, для которого надо установить начальное состояние, 'atm', 'dm', 'ipt'; str
                n_attemps - количество попыток сформировать это состояние (потому что не всегда получается
                сделать это с первого раза); int

            Устанавливает в зависимости от того, какое устройство было выбрано:
                или self._atm_table - начальное состояние для atm,
                или self._dm_table - начальное состояние для dm,
                или self._ipt_table - начальное состояние для ipt
        """

        if kind == 'atm':
            mask = self._atm_full_mask
            power = self.power * self.priority[kind]
            state = np.zeros(shape=(self._n_atms, self._n_days), dtype='int64')

            for i in range(self._n_atms):
                for j in range(self._n_days):
                    if mask[i, j] == 1:
                        state[i, j] = 1
                    if np.isnan(mask[i, j]):
                        state[i, j] = 0

            for i_att in range(n_attempts):
                print('i_att = ' + str(i_att))

                if self._atm_mode_order == 'flexible':
                    for i_att2 in range(n_attempts):
                        print('i_att2_flex = ' + str(i_att2))
                        state_dynamic = state.copy()
                        mask_dynamic = mask.copy()
                        days_good = 0
                        n_atms_0 = 0
                        for i_atm in range(self._n_atms):
                            ind_wall = np.where(mask_dynamic[i_atm, :] == 2)[0]
                            if ind_wall.size:
                                ind_wall = ind_wall[0]
                                list_0 = list(np.where(mask_dynamic[i_atm, :ind_wall] == 0)[0])
                                list_0.append(ind_wall)
                                if list_0 != [0] and n_atms_0 >= 100:
                                    list_0.remove(0)
                                ind_encash = random.sample(list_0, 1)
                                if ind_encash == [0]:
                                    n_atms_0 += 1
                                state_dynamic[i_atm, ind_encash] = 1
                                mask_dynamic[i_atm, ind_encash] = 1
                                # print(list_0)
                                # print(ind_wall)
                                print(ind_encash)
                        for i_day in range(self._n_days):
                            list_0 = list(np.where(mask_dynamic[:, i_day] == 0)[0]) + \
                                     list(np.where(mask_dynamic[:, i_day] == 2)[0])
                            list_1 = list(np.where(mask_dynamic[:, i_day] == 1)[0])
                            if power[i_day] - len(list_1) < 0:
                                print('i_day = ' + str(i_day))
                                print('power[i_day] = ' + str(power[i_day]))
                                print('len(list_1) = ' + str(len(list_1)))
                                print('\n')
                                days_good = 0
                                break
                            elif power[i_day] > len(list_1) and len(list_0) > 0:
                                n_encash = np.random.randint(1, min(len(list_0), power[i_day] - len(list_1)) + 1)
                                list_encash = random.sample(list_0, n_encash)
                                state_dynamic[list_encash, i_day] = 1
                                mask_dynamic[list_encash, i_day] = 1
                                days_good += 1
                        print('days_good = ' + str(days_good) + '\n')
                        if days_good == self._n_days:
                            break

                    if days_good == 0:
                        state_dynamic = state.copy()
                        mask_dynamic = mask.copy()
                        for i_atm in range(self._n_atms):
                            ind_wall = np.where(mask_dynamic[i_atm, :] == 2)[0]
                            if ind_wall.size:
                                ind_wall = ind_wall[0]
                                mask_dynamic[i_atm, ind_wall] = 1
                                state_dynamic[i_atm, ind_wall] = 1
                                for j in range(ind_wall):
                                    mask_dynamic[i_atm, j] = np.nan

                        for i_day in range(self._n_days):
                            list_0 = list(np.where(mask_dynamic[:, i_day] == 0)[0])
                            list_1 = list(np.where(mask_dynamic[:, i_day] == 1)[0])
                            if power[i_day] - len(list_1) < 0:
                                print('i_day = ' + str(i_day))
                                print('power[i_day] = ' + str(power[i_day]))
                                print('len(list_1) = ' + str(len(list_1)))
                                print(list_1)
                                print('Need to add ' + str(len(list_1) - power[i_day]))
                                self.power[i_day] = len(list_1)
                            elif power[i_day] > len(list_1) and len(list_0) > 0:
                                n_encash = np.random.randint(1, min(len(list_0), power[i_day] - len(list_1)) + 1)
                                list_encash = random.sample(list_0, n_encash)
                                state_dynamic[list_encash, i_day] = 1
                                mask_dynamic[list_encash, i_day] = 1

                else:
                    state_dynamic = state.copy()
                    mask_dynamic = mask.copy()

                    for i_day in range(self._n_days):
                        list_0 = list(np.where(mask_dynamic[:, i_day] == 0)[0])
                        list_1 = list(np.where(mask_dynamic[:, i_day] == 1)[0])
                        if power[i_day] - len(list_1) < 0:
                            print('i_day = ' + str(i_day))
                            print('power[i_day] = ' + str(power[i_day]))
                            print('len(list_1) = ' + str(len(list_1)))
                            print(list_1)
                            print('Need to add ' + str(len(list_1) - power[i_day]))
                            self.power[i_day] = len(list_1)
                        elif power[i_day] > len(list_1) and len(list_0) > 0:
                            n_encash = np.random.randint(1, min(len(list_0), power[i_day] - len(list_1)) + 1)
                            list_encash = random.sample(list_0, n_encash)
                            state_dynamic[list_encash, i_day] = 1
                            mask_dynamic[list_encash, i_day] = 1

                for i_atm in range(self._n_atms):
                    flag_good = 0
                    while not self._row_check(i_atm, state_dynamic[i_atm, :], kind=kind, timestamp=timestamp):
                        list_0 = list(np.where(mask_dynamic[i_atm, :] == 0)[0])
                        list_pos = list(np.where(np.sum(state_dynamic, axis=0) < power)[0])
                        list_valid = list(set(list_0).intersection(set(list_pos)))
                        if not list_valid:
                            print('I_ATM: ' + str(i_atm))
                            break
                        else:
                            cell_encash = random.sample(list_valid, 1)
                            state_dynamic[i_atm, cell_encash] = 1
                            mask_dynamic[i_atm, cell_encash] = 1
                    else:
                        flag_good = 1
                        continue
                    break
                if i_atm == self._n_atms - 1 and flag_good == 1:
                    self.atm_table = state_dynamic
                    return

            print('WARNING_ATM_STATE!')

        if kind == 'dm':
            mask = self._dm_full_mask
            power = self.power * self.priority[kind] + np.sum(self._dm_atm_state_mask, axis=0)
            state = np.zeros(shape=(self._n_dms, self._n_days), dtype='int64')

            for i in range(self._n_dms):
                for j in range(self._n_days):
                    if mask[i, j] == 1:
                        state[i, j] = 1
                    if np.isnan(mask[i, j]):
                        state[i, j] = 0

            for i_att in range(n_attempts):
                print(i_att)
                state_dynamic = state.copy()
                mask_dynamic = mask.copy()

                for i_dm in range(self._n_dms):
                    ind_wall = np.where(mask_dynamic[i_dm, :] == 2)[0]
                    if ind_wall.size:
                        ind_wall = ind_wall[0]
                        list_0 = list(np.where(mask_dynamic[i_dm, :ind_wall] == 0)[0])
                        list_0.append(ind_wall)
                        ind_encash = random.sample(list_0, 1)
                        state_dynamic[i_dm, ind_encash] = 1
                        mask_dynamic[i_dm, ind_encash] = 1
                for i_day in range(self._n_days):
                    list_0 = list(np.where(mask_dynamic[:, i_day] == 0)[0])
                    list_1 = list(np.where(mask_dynamic[:, i_day] == 1)[0])
                    if power[i_day] - len(list_1) < 0:
                        print(power[i_day])
                        print(list_1)
                        print(i_day)
                        print('Need to add ' + str(len(list_1) - power[i_day]))
                        power[i_day] = len(list_1)
                        # sys.exit('Mask is not valid')
                    elif power[i_day] > len(list_1) and len(list_0) > 0:
                        n_encash = np.random.randint(1, min(len(list_0), power[i_day] - len(list_1)) + 1)
                        list_encash = random.sample(list_0, n_encash)
                        state_dynamic[list_encash, i_day] = 1
                        mask_dynamic[list_encash, i_day] = 1
                for i_dm in range(self._n_dms):
                    flag_good = 0
                    while not self._row_check(i_dm, state_dynamic[i_dm, :], kind=kind, timestamp=timestamp):
                        list_0 = list(np.where(mask_dynamic[i_dm, :] == 0)[0])
                        list_pos = list(np.where(np.sum(state_dynamic, axis=0) < power)[0])
                        list_valid = list(set(list_0).intersection(set(list_pos)))
                        if not list_valid:
                            print('I_DM: ' + str(i_dm))
                            break
                        else:
                            cell_encash = random.sample(list_valid, 1)
                            state_dynamic[i_dm, cell_encash] = 1
                            mask_dynamic[i_dm, cell_encash] = 1
                    else:
                        flag_good = 1
                        continue
                    break
                if i_dm == self._n_dms - 1 and flag_good == 1:
                    self.dm_table = state_dynamic
                    return

            print('WARNING_DM_STATE!')

        if kind == 'ipt':
            mask = self._ipt_full_mask
            power = self.power * self.priority[kind]
            state = np.zeros(shape=(self._n_ipts, self._n_days), dtype='int64')

            for i in range(self._n_ipts):
                for j in range(self._n_days):
                    if mask[i, j] == 1:
                        state[i, j] = 1
                    if np.isnan(mask[i, j]):
                        state[i, j] = 0

            for i_att in range(n_attempts):
                print('\n')
                print(i_att)
                state_dynamic = state.copy()
                mask_dynamic = mask.copy()

                for i_ipt in range(self._n_ipts):
                    ind_wall = np.where(mask_dynamic[i_ipt, :] == 2)[0]
                    if ind_wall.size:
                        ind_wall = ind_wall[0]
                        list_0 = list(np.where(mask_dynamic[i_ipt, :ind_wall] == 0)[0])
                        list_0.append(ind_wall)
                        ind_encash = random.sample(list_0, 1)
                        state_dynamic[i_ipt, ind_encash] = 1
                        mask_dynamic[i_ipt, ind_encash] = 1
                for i_day in range(self._n_days):
                    list_0 = list(np.where(mask_dynamic[:, i_day] == 0)[0])
                    list_1 = list(np.where(mask_dynamic[:, i_day] == 1)[0])
                    if power[i_day] - len(list_1) < 0:
                        print(power[i_day])
                        print(list_1)
                        print(i_day)
                        print('Need to add ' + str(len(list_1) - power[i_day]))
                        power[i_day] = len(list_1)
                        # sys.exit('Mask is not valid')
                    elif power[i_day] > len(list_1) and len(list_0) > 0:
                        n_encash = np.random.randint(1, min(len(list_0), power[i_day] - len(list_1)) + 1)
                        list_encash = random.sample(list_0, n_encash)
                        state_dynamic[list_encash, i_day] = 1
                        mask_dynamic[list_encash, i_day] = 1
                for i_ipt in range(self._n_ipts):
                    flag_good = 0
                    while not self._row_check(i_ipt, state_dynamic[i_ipt, :], kind=kind, timestamp=timestamp):
                        list_0 = list(np.where(mask_dynamic[i_ipt, :] == 0)[0])
                        list_pos = list(np.where(np.sum(state_dynamic, axis=0) < power)[0])
                        list_valid = list(set(list_0).intersection(set(list_pos)))
                        if not list_valid:
                            print('I_IPT: ' + str(i_ipt))
                            break
                        else:
                            cell_encash = random.sample(list_valid, 1)
                            state_dynamic[i_ipt, cell_encash] = 1
                            mask_dynamic[i_ipt, cell_encash] = 1
                    else:
                        flag_good = 1
                        continue
                    break
                if i_ipt == self._n_ipts - 1 and flag_good == 1:
                    self.ipt_table = state_dynamic
                    return

            print('WARNING_IPT_STATE!')

    def set_data(self,
                 devices_order=['atm']):

        """
            Вызывает для всех устройств, указанных в order, три функции
                self.set_cash_flow
                self.set_costs
                self.set_intraday_distribution

            Принимает:
                order - список типов устройств, для которых надо вызвать фукнции; list
        """
        if set(devices_order) - set(self._devices_order):
            sys.exit('WARNING! THERE IS(ARE) UNKNOWN DEVICE(S) IN devices_order!: ' +
                     str(set(devices_order) - set(self._devices_order)))

        for kind in devices_order:
            self._set_cash_flow(kind=kind)
            self._set_costs(kind=kind)
            self._set_intraday_distribution(kind=kind)

    def set_full_masks(self,
                       devices_order=['atm'],
                       masks_order=[['first_encash', 'availability']],
                       modes_order=['rigid']):

        """
            Вызывает для всех устройств, указанных в order, три функции
                self.set_masks
                self.synchronize_masks
                self.set_initial_table

            Принимает:
                order - список типов устройств, для которых надо вызвать фукнции; list
        """
        if set(devices_order) - set(self._devices_order):
            sys.exit('WARNING! THERE IS(ARE) UNKNOWN DEVICE(S) IN devices_order!: ' +
                     str(set(devices_order) - set(self._devices_order)))

        if len(masks_order) != len(self._devices_order):
            sys.exit('WARNING! MASKS ARE NOT CORRECT!')

        if len(modes_order) != len(self._devices_order):
            sys.exit('WARNING! MASKS ARE NOT CORRECT!')

        for kind, masks, mode in \
                zip(devices_order, masks_order, modes_order):
            self._set_masks(kind=kind, masks_order=masks, mode_order=mode)
            self._synchronize_masks(kind=kind)

    def set_state(self,
                  devices_order=['atm']):
        for kind in devices_order:
            self._set_initial_table(kind=kind, timestamp='day')



# -------------------------------------------OPTIMIZATION MODULE--------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class Optimizer(object):
    """
        Класс реализующий алгоритм оптимизации

        Поля:
            center: инициализированный объект класса CashCenter

    """
    def __init__(self, cash_center):
        self.center = cash_center

    def get_random_neighbour(self, current_state, kind='atm', timestamp='day'):

        """
        Вовзращает случайную валидную таблицу инкассаций из окрестности current_state

        Принимает:
            current_state: таблица из 0 и 1 (1 - инкассируем, 0 нет), numpy.array
            kind: 'atm', 'dm', 'ipt'; str

        Возвращает:
            таблица из 0 и 1 (1 - инкассируем, 0 нет), numpy.array

        """
        power = self.center.power * self.center.priority[kind]
        next_state = current_state.copy()
        n_days = next_state.shape[1]

        j = np.random.randint(0, high=n_days)
        ind = 0

        if np.sum(current_state[:, j]) == power[j]:
            try:
                i = random.sample(list(np.where(current_state[:, j] == 1)[0]), 1)[0]
                next_state[i, j] = 0
                ind = 1
            except:
                pass
        elif np.sum(current_state[:, j]) < power[j]:
            if np.random.randint(2) == 0:
                try:
                    i = random.sample(list(np.where(current_state[:, j] == 1)[0]), 1)[0]
                    next_state[i, j] = 0
                    ind = 1
                except:
                    pass
            else:
                try:
                    i = random.sample(list(np.where(current_state[:, j] == 0)[0]), 1)[0]
                    next_state[i, j] = 1
                    ind = 1
                except:
                    pass

        if self.center.table_check(next_state, kind=kind, timestamp=timestamp) and ind == 1:
            return next_state, 'yes'
        else:
            return current_state, 'no'

    def optimize(self, kind='atm', timestamp='day', min_temp=0.1, max_temp=1.0, max_iter=6000, num_jump=1, calibration=1):

        """
            Реализация алгоритма иммитации отжига

            Принимает:
                min_temp: минимальное значение температуры, float
                max_temp: максимальное значение температуры, float
                max_iter: количество итераций за один запуск алгоритма, int
                num_jump: количество запусков алгоритма, int
                kind: 'atm', 'dm', 'ipt'; str

            Возвращает:
                current_state: таблица из 0 и 1 (1 - инкассируем, 0 нет) полученная в результате работы алгоритма,
                 numpy.array
                costs_history_list: список значений затрат для таблиц режимов инкассации, полученных в результате
                    работы алгоритма на каждой из итераций, list
                acceptance_list: список из 'initial', 0 и 1, 1 - на данной итерации произошел переход в окрестность,
                    0 - нет, 'initial' - начальное состояние, list
        """

        if kind == 'atm':
            initial_state_list = [self.center.atm_table]
            all_state_list = [self.center.atm_table]
        if kind == 'dm':
            initial_state_list = [self.center.dm_table]
            all_state_list = [self.center.dm_table]
        if kind == 'ipt':
            initial_state_list = [self.center.ipt_table]
            all_state_list = [self.center.ipt_table]

        costs_history_list = []
        acceptance_list = []

        for i_jump in range(num_jump):
            current_state = initial_state_list[i_jump]
            current_cost = self.center.table_cost(current_state, kind=kind, facts=False, timestamp=timestamp)[0] \
                           / calibration

            current_temp = max_temp
            num_iterations = 0

            costs_history = [0] * (max_iter + 1)
            costs_history[0] = current_cost * calibration
            acceptance = [0] * (max_iter + 1)
            acceptance[0] = 'initial'

            start = time.time()
            while current_cost > 0 and num_iterations < max_iter:
                neighbour, status = self.get_random_neighbour(current_state, kind=kind, timestamp=timestamp)
                if status == 'yes':
                    neighbour_cost = self.center.table_cost(neighbour, kind=kind, facts=False, timestamp=timestamp)[0] \
                                     / calibration
                else:
                    continue

                all_state_list.append(neighbour)

                cost_delta = neighbour_cost - current_cost
                num_iterations += 1
                if cost_delta <= 0 or random.random() < math.exp(-cost_delta / (100000*current_temp)):
                    current_state, current_cost = neighbour, neighbour_cost
                    acceptance[num_iterations] = 1
                costs_history[num_iterations] = current_cost * calibration

                if num_iterations % np.int(max_iter/10) == 0 and current_temp > min_temp:
                    end = time.time()
                    current_temp -= 0.1
                    print(str(np.round(current_temp, 2)) + '   ' + 'time = ' + str(end-start))
                    start = time.time()
            print('\n')
            initial_state_list.append(current_state)
            costs_history_list.append(costs_history)
            acceptance_list.append(acceptance)

        return current_state, all_state_list, costs_history_list, acceptance_list

    def optimize_light(self, kind='atm', timestamp='day', min_temp=0.1, max_temp=1.0, max_iter=6000, num_jump=1):


        if kind == 'atm':
            initial_state_list = [self.center.atm_table]
        if kind == 'dm':
            initial_state_list = [self.center.dm_table]
        if kind == 'ipt':
            initial_state_list = [self.center.ipt_table]

        for i_jump in range(num_jump):
            current_state = initial_state_list[i_jump]
            current_cost = self.center.table_cost(current_state, kind=kind, facts=False, timestamp=timestamp)[0]

            current_temp = max_temp
            num_iterations = 0

            start = time.time()
            while current_cost > 0 and num_iterations < max_iter:
                neighbour, status = self.get_random_neighbour(current_state, kind=kind, timestamp=timestamp)
                if status == 'yes':
                    neighbour_cost = self.center.table_cost(neighbour, kind=kind, facts=False, timestamp=timestamp)[0]
                else:
                    continue

                cost_delta = neighbour_cost - current_cost
                num_iterations += 1
                if cost_delta <= 0 or random.random() < math.exp(-cost_delta / (100*current_temp)):
                    current_state, current_cost = neighbour, neighbour_cost

                if num_iterations % np.int(max_iter/10) == 0 and current_temp > min_temp:
                    end = time.time()
                    current_temp -= 0.1
                    print(str(np.round(current_temp, 2)) + '   ' + 'time = ' + str(end-start))
                    start = time.time()
            print('\n')
            initial_state_list.append(current_state)

        return current_state


# ----------------------------------------------GRAPHICS MODULE---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class Plotter(object):
    def __init__(self, optimum):
        self.optimum = optimum


# ---------------------------------------------CashCenterSlice----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def indices_time(times, inner_times):
    date_list = list()
    ind_list = list()
    d1, d2 = pd.to_datetime(times[0]), pd.to_datetime(times[1])
    diff = d2 - d1

    for i in range(diff.days + 1):
        date_list.append(d1 + datetime.timedelta(i))

    for i, d in enumerate(date_list):
        if pd.to_datetime(inner_times[0]) == d:
            ind_list.append(i)
        elif pd.to_datetime(inner_times[1]) == d:
            ind_list.append(i)
    return ind_list

def get_slice(center, new_times, update):
    center = deepcopy(center)
    ind_list = indices_time(center.times, new_times)
    center_slice = CashCenter(atm_ids=center.atm_ids,
                              dm_ids=center.dm_ids,
                              ipt_ids=center.ipt_ids,
                              times=new_times,
                              power=center.power[ind_list[0]:ind_list[1] + 1],
                              priority=center.priority,
                              atm_capacity=center.atm_capacity,
                              dm_capacity=center.dm_capacity,
                              ipt_capacity=center.ipt_capacity,
                              dm_cassette=center.dm_cassette,
                              ipt_cassette=center.ipt_cassette,
                              f_rate=center.f_rate)

    if center_slice.atm_ids is not None:
        center_slice._atm_predictions = center._atm_predictions[:, ind_list[0]:ind_list[1] + 1]
        center_slice._atm_facts = center._atm_facts[:, ind_list[0]:ind_list[1] + 1]

        if ind_list[0] == 0:
            center_slice._atm_initial_balances = center._atm_initial_balances
        else:
            center_slice._atm_initial_balances = update['atm'][-1]

        center_slice._atm_cost_encash = center._atm_cost_encash
        center_slice._atm_cost_kassa = center._atm_cost_kassa

        center_slice._atm_intraday_distribution = center._atm_intraday_distribution

        center_slice._atm_availability_mask = center._atm_availability_mask[:, ind_list[0]:ind_list[1] + 1]

        center_slice._set_masks(kind='atm', masks_order=['first_encash'], mode_order='rigid')
        center_slice._synchronize_masks(kind='atm')

    if center_slice.dm_ids is not None:
        center_slice._dm_predictions_money = center._dm_predictions_money[:, ind_list[0]:ind_list[1] + 1]
        center_slice._dm_predictions_banknotes = center._dm_predictions_banknotes[:, ind_list[0]:ind_list[1] + 1]
        center_slice._dm_facts_money = center._dm_facts_money[:, ind_list[0]:ind_list[1] + 1]
        center_slice._dm_facts_banknotes = center._dm_facts_banknotes[:, ind_list[0]:ind_list[1] + 1]

        if ind_list[0] == 0:
            center_slice._dm_initial_balances_money = center._dm_initial_balances_money
            center_slice._dm_initial_balances_banknotes = center._dm_initial_balances_banknotes
        else:
            center_slice._dm_initial_balances_money = update['dm_money'][-1]
            center_slice._dm_initial_balances_banknotes = update['dm_banknotes'][-1]

        center_slice._dm_cost_encash = center._dm_cost_encash
        center_slice._dm_cost_kassa = center._dm_cost_kassa

        center_slice.dm_availability_mask = center._dm_availability_mask[:, ind_list[0]:ind_list[1] + 1]

        center_slice._set_masks(kind='dm', masks_order=['first_encash', 'atm_state'])
        center_slice._synchronize_masks(kind='dm')

    if center_slice.ipt_ids is not None:
        center_slice.ipt_predictions_money = center.ipt_predictions_money[:, ind_list[0]:ind_list[1] + 1]
        center_slice.ipt_predictions_banknotes = center.ipt_predictions_banknotes[:, ind_list[0]:ind_list[1] + 1]
        center_slice.ipt_facts_money = center.ipt_facts_money[:, ind_list[0]:ind_list[1] + 1]
        center_slice.ipt_facts_banknotes = center.ipt_facts_banknotes[:, ind_list[0]:ind_list[1] + 1]

        if ind_list[0] == 0:
            center_slice._ipt_initial_balances_money = center._dm_initial_balances_money
            center_slice._ipt_initial_balances_banknotes = center._dm_initial_balances_banknotes
        else:
            center_slice._ipt_initial_balances_money = update['ipt_money'][-1]
            center_slice._ipt_initial_balances_banknotes = update['ipt_banknotes'][-1]

        center_slice.ipt_cost_encash = center.ipt_cost_encash
        center_slice.ipt_cost_kassa = center.ipt_cost_kassa

        center_slice.ipt_availability_mask = center.ipt_availability_mask[:, ind_list[0]:ind_list[1] + 1]

        center_slice._set_masks(kind='ipt', masks_order=['first_encash'])
        center_slice._synchronize_masks(kind='ipt')

    return center_slice
