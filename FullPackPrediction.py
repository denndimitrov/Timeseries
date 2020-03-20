# encoding=utf-8
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import operator
import calendar
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from math import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression as MI
import matplotlib.gridspec as gridspec
from collections import Counter


class Evaluator(object):

    """" Класс реализующий функционал подсчёта различных ошибок,
            визуализации и сравнения предсказаний алгоритма, аналитика и фактических значений"""

    def __init__(self, y, prediction, analyst=None, days=1):
        self.y = y
        self.prediction = prediction
        self.analyst = analyst
        self.days = days

    def calculate_error(self, y=None, prediction=None, kind='mae', days=None, is_mean=False):

        if y is None:
            y = self.y
        if prediction is None:
            prediction = self.prediction
        if days is None:
            days = self.days
        
        y = np.array(y)
        prediction = np.array(prediction)

        if kind == 'mae':
            result = np.abs(y - prediction)
        elif kind == 'mape':
            result = np.abs(100 * (y - prediction) / y)
        elif kind == 'cum_err':
            err = y - prediction
            groups = np.array_split(err, round(len(err)/float(days)))
            result = np.array([])
            for array in groups:
                result = np.append(result, np.array([np.sum(array) for i in range(len(array))]))
        elif kind == 'mse':
            result = (y - prediction) ** 2
        elif kind == 'smape':
            result = 100 * np.abs((y - prediction)) / (np.abs(y) + np.abs(prediction))
        elif kind == 'sber':
            sber = np.abs(100 * (prediction - y) / prediction)
            sber[(prediction == 0) & (y < 200000)] = 0
            sber[(prediction < 100000) & (y < prediction)] = 0
            result = sber
        else:
            sys.exit('kind must be \'smape\', \'mae\', \'mape\', \'mse\', \'cum_err\', \'sber\'')

        if is_mean:
            return np.nanmean(result)
        else:
            return result

    def compare_predictions(self, y=None, prediction=None, analyst=None):

        if y is None:
            y = self.y
        if prediction is None:
            prediction = self.prediction
        if analyst is None:
            analyst = self.analyst

        mae = []
        mape = []
        cum_err = []
        mse = []
        smape = []
        sber = []

        for i in [prediction, analyst]:
            mae.append(self.calculate_error(y, i, kind='mae', is_mean=True))
            mape.append(self.calculate_error(y, i, kind='mape', is_mean=True))
            cum_err.append(self.calculate_error(y, i, kind='cum_err', is_mean=True))
            mse.append(self.calculate_error(y, i, kind='mse', is_mean=True))
            smape.append(self.calculate_error(y, i, kind='smape', is_mean=True))
            sber.append(self.calculate_error(y, i, kind='sber', is_mean=True))

        print ('Error MAE: Sberbank Vanga {} vs Analyst {}'.format(mae[0], mae[1]))
        print ('Error MAPE: Sberbank Vanga {} vs Analyst {}'.format(mape[0], mape[1]))
        print ('Error Cumulative: Sberbank Vanga {} vs Analyst {}'.format(cum_err[0], cum_err[1]))
        print ('Error MSE: Sberbank Vanga {} vs Analyst {}'.format(mse[0], mse[1]))
        print ('Error SMAPE: Sberbank Vanga {} vs Analyst {}'.format(smape[0], smape[1]))
        print ('Erorr from Cash Management Center: Sberbank Vanga {} vs Analyst {}'.format(sber[0], sber[1]))

        return [mae, mape, cum_err, mse, sber]

    def visualize_predictions(self, y=None, prediction=None, analyst=None):

        if y is None:
            y = self.y
        if prediction is None:
            prediction = self.prediction
        if analyst is None:
            analyst = self.analyst

        if analyst is not None:
            err_analyst = self.calculate_error(y, analyst, kind='cum_err', is_mean=False)
            err_analyst_mean = self.calculate_error(y, analyst, kind='mae', is_mean=True)
        err_sberbank_vanga = self.calculate_error(y, prediction, kind='cum_err', is_mean=False)
        err_sberbank_vanga_mean = self.calculate_error(y, prediction, kind='mae', is_mean=True)

        with plt.style.context('seaborn-white'):



            plt.figure(figsize=(15, 6))
            plt.subplots_adjust(wspace=0, hspace=0.2)
            gs = gridspec.GridSpec(3, 1)
            ax1 = plt.subplot(gs[:2, :])
            ax2 = plt.subplot(gs[2, :], sharex=ax1)
            if analyst is not None:
                ax1.set_title('Sberbank Vanga MAE: {:,}  Analyst MAE: {:,}'.format(int(err_sberbank_vanga_mean), int(err_analyst_mean)).replace(',', ' '))
                ax1.plot(analyst, label='Analyst', color='g')
                ax2.plot(y.index, err_analyst, color='#1e6672', label='Analyst error')
                ax2.fill_between(y.index, err_analyst, color='#34a6ba', alpha=0.7)
            else:
                ax1.set_title('Sberbank Vanga MAE: {:,}'.format(int(err_sberbank_vanga_mean)).replace(',', ' '), fontdict={'fontsize': 20})
            ax1.plot(y, label='Actual values')
            ax1.plot(prediction, label='Sberbank Vanga', color='r')
            ax2.plot(y.index, err_sberbank_vanga, color='#443f3e', label='Sberbank Vanga error')
            ax2.fill_between(y.index, err_sberbank_vanga, color='#595251')
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax1.legend()
            ax2.legend()
            ax1.grid(True)
            ax2.grid(True)
            plt.show()


class Predictor(object):

    """" Класс реализующий функционал создания необходимого признакового пространства,
         методов предсказания и кросс-валидации (в скользящем окне) в режимах постоянного подкрепления
         новыми данными (backtest) и в условиях неопределенности (realtime)"""

    def __init__(self, calendar_features, is_mean_month=True,
                 window_weekdays=3, window_days=7, lags=(1, 4),
                 auto_salary=None,
                 backward_window_size=365,
                 forward_window_size=30, re_fit=True,
                 horizon=30, model=RandomForestRegressor(random_state=2)):

        self.calendar_features = calendar_features
        self.is_mean_month = is_mean_month
        self.window_weekdays = window_weekdays
        self.window_days = window_days
        self.lags = lags
        self.auto_salary = auto_salary
        self.horizon = horizon
        self.backward_window_size = backward_window_size
        self.forward_window_size = forward_window_size
        self.re_fit = re_fit
        self.model = model

    def make_features(self, time_series, test_index,
                      calendar_features=None,
                      is_mean_month=None,
                      window_weekdays=None,
                      window_days=None, lags=None):

        if calendar_features is None:
            calendar_features = self.calendar_features
        if is_mean_month is None:
            is_mean_month = self.is_mean_month
        if window_days is None:
            window_days = self.window_days
        if window_weekdays is None:
            window_weekdays = self.window_weekdays
        if lags is None:
            lags = self.lags

        data = pd.DataFrame(time_series).copy()
        data.columns = ['y']

        # lags
        if lags is not None:
            for i in range(lags[0], lags[1]):
                data['lag_{}'.format(i)] = data.y.shift(i)

        # rolling functions
        if window_days is not None:
            data['rolling_mean'] = data.rolling(window_days).mean().y.shift(1)
            data['rolling_median'] = data.rolling(window_days).median().y.shift(1)
            data['rolling_max'] = data.rolling(window_days).max().y.shift(1)
            data['rolling_min'] = data.rolling(window_days).min().y.shift(1)
            data['rolling_std'] = data.rolling(window_days).std().y.shift(1)

        # rolling functions grouped by day of week
        if window_weekdays is not None:
            data['rolling_mean_weekday'] = data.groupby(data.index.weekday)['y'].transform(lambda x: x.rolling(window_weekdays).mean().shift(1))
            data['rolling_max_weekday'] = data.groupby(data.index.weekday)['y'].transform(lambda x: x.rolling(window_weekdays).max().shift(1))
            data['rolling_min_weekday'] = data.groupby(data.index.weekday)['y'].transform(lambda x: x.rolling(window_weekdays).min().shift(1))
            data['rolling_median_weekday'] = data.groupby(data.index.weekday)['y'].transform(lambda x: x.rolling(window_weekdays).median().shift(1))
            data['rolling_std_weekday'] = data.groupby(data.index.weekday)['y'].transform(lambda x: x.rolling(window_weekdays).std().shift(1))

        # coding mounth by its mean, max, min, median, std value
        if is_mean_month is not None:
            mmean = data['y'][:test_index].groupby(data['y'][:test_index].index.month).transform(lambda x: np.nanmean(x)).copy()
            mmax = data['y'][:test_index].groupby(data['y'][:test_index].index.month).transform(lambda x: np.nanmax(x)).copy()
            mmin = data['y'][:test_index].groupby(data['y'][:test_index].index.month).transform(lambda x: np.nanmin(x)).copy()
            mmedian = data['y'][:test_index].groupby(data['y'][:test_index].index.month).transform(lambda x: np.nanmedian(x)).copy()
            mstd = data['y'][:test_index].groupby(data['y'][:test_index].index.month).transform(lambda x: np.nanstd(x, ddof=1)).copy()

            data['month'] = data.index.month
            data['mean_month'] = list(map(dict(zip(mmean.index.month, mmean)).get, data['month']))
            data['max_month'] = list(map(dict(zip(mmax.index.month, mmax)).get, data['month']))
            data['min_month'] = list(map(dict(zip(mmin.index.month, mmin)).get, data['month']))
            data['median_month'] = list(map(dict(zip(mmedian.index.month, mmedian)).get, data['month']))
            data['std_month'] = list(map(dict(zip(mstd.index.month, mstd)).get, data['month']))
            data.drop(['month'], axis=1, inplace=True)

        # dummy variables from calendar
        if calendar_features is not None:
            data = pd.concat([data, calendar_features], axis=1, join='inner')

        return data

    def backtest(self, time_series, test_index, auto_salary=None, model=None):

        if auto_salary is None:
            auto_salary = self.auto_salary
        if model is None:
            model = self.model

        if auto_salary:
            salary = Salary(time_series[:test_index], threshold=auto_salary)
            salary_features = salary.get_features()
            calendar_features = pd.concat([self.calendar_features, salary_features], axis=1, join='inner')
        else:
            calendar_features = self.calendar_features

        data = self.make_features(time_series=time_series, test_index=test_index, calendar_features=calendar_features)
        data = data.dropna()

        x_train = data[:test_index].drop(["y"], axis=1)
        y_train = data[:test_index]["y"]
        x_test = data[test_index:].drop(["y"], axis=1)
        y_test = data[test_index:]["y"]

        model.fit(x_train, y_train)
        prediction = pd.Series(model.predict(x_test), index=y_test.index)
        prediction[prediction < 0] = 0

        return y_test, prediction

    def realtime(self, time_series, auto_salary=None, re_fit=None, horizon=None, model=None):

        timeseries = time_series.copy()
        index = []
        prediction = []

        if auto_salary is None:
            auto_salary = self.auto_salary
        if re_fit is None:
            re_fit = self.re_fit
        if horizon is None:
            horizon = self.horizon
        if model is None:
            model = self.model

        if auto_salary:
            salary = Salary(time_series, threshold=auto_salary)
            salary_features = salary.get_features()
            calendar_features = pd.concat([self.calendar_features, salary_features], axis=1, join='inner')
        else:
            calendar_features = self.calendar_features

        for day in range(horizon):

            data = self.make_features(timeseries, len(time_series), calendar_features=calendar_features)

            next_day = pd.DataFrame(index=[data.index[-1] + datetime.timedelta(days=1)])
            next_day = self.make_features(timeseries.append(next_day), len(timeseries.append(next_day)), calendar_features=calendar_features)[-1:]
            index.append(next_day.index[0])
            data = data.dropna()

            x_train = data.drop(["y"], axis=1)
            y_train = data["y"]
            x_test = next_day.drop(["y"], axis=1)

            if day == 0 or re_fit:
                model.fit(x_train, y_train)
            pred = model.predict(x_test)
            prediction.append(pred[0])
            add = pd.DataFrame({0: pred})
            add.index = [timeseries.index[-1] + datetime.timedelta(days=1)]
            timeseries = timeseries.append(add)
        prediction = pd.Series(prediction, index=pd.Index(index))
        prediction[prediction < 0] = 0
        return prediction

    def cross_validation(self, timeseries, mode='backtest', backward_window_size=None,
                         forward_window_size=None, auto_salary = None):

        if backward_window_size is None:
            backward_window_size = self.backward_window_size
        if forward_window_size is None:
            forward_window_size = self.forward_window_size

        number_folds = int((len(timeseries) - backward_window_size) / forward_window_size)
        remainder = (len(timeseries) - backward_window_size) % forward_window_size
        prediction = []
        y_total = []

        for shift in range(number_folds):
            start_index = shift * forward_window_size
            end_index = start_index + backward_window_size + forward_window_size

            if mode == 'backtest':
                y_test, pred = self.backtest(timeseries[start_index:end_index], -forward_window_size,
                                            auto_salary = auto_salary)
                prediction += list(pred)
                y_total += list(y_test)

            if mode == 'realtime':
                pred = self.realtime(timeseries[start_index:start_index + backward_window_size], horizon=forward_window_size,
                                    auto_salary = auto_salary)
                y_test = timeseries[start_index + backward_window_size:end_index]
                prediction += list(pred)
                y_total += list(y_test)

        if remainder != 0:
            if mode == 'backtest':
                y_test, pred = self.backtest(timeseries[-(backward_window_size + remainder):], -remainder, 
                                            auto_salary = auto_salary)
                prediction += list(pred)
                y_total += list(y_test)

            if mode == 'realtime':
                pred = self.realtime(timeseries[-(backward_window_size + remainder): -remainder], auto_salary = auto_salary, 
                                     horizon=remainder)
                y_test = timeseries[-remainder:]
                prediction += list(pred)
                y_total += list(y_test)

        return pd.Series(y_total, index=timeseries[backward_window_size:].index), pd.Series(prediction, index=timeseries[backward_window_size:].index)


class AnomalyDetector(object):

    """"Класс представляет собой реализацию видоизмененного алгоритма детекции аномалий CUSUM"""

    def __init__(self, backward_window_size=30, forward_window_size=14, threshold=5, drift=1.0):
        self.backward_window_size = backward_window_size
        self.forward_window_size = forward_window_size
        self.threshold = threshold
        self.drift = drift

    def one_pass(self, train_zone, prediction_zone, threshold=None, drift=None):
        if not threshold:
            threshold = self.threshold
        if not drift:
            drift = self.drift

        current_std = train_zone.std(ddof=1)
        current_mean = train_zone.mean()

        drift = drift * current_std
        threshold = threshold * current_std

        x = np.atleast_1d(prediction_zone).astype('float64')
        gp, gn = np.zeros(x.size), np.zeros(x.size)
        gp_prev = gp[0]
        gn_prev = gn[0]

        for i in range(1, x.size):
            gp[i] = gp_prev + x[i] - (current_mean + drift)
            gn[i] = gn_prev + x[i] - (current_mean - drift)

            gp_prev, gn_prev = gp[i], gn[i]

            if gp[i] < 0:
                gp_prev = 0
            if gn[i] > 0:
                gn_prev = 0

                # if gp[i] > threshold or gn[i] < -threshold:
                # gp_prev, gn_prev = 0,0

        is_fault = np.logical_or(gp > threshold, gn < -threshold)
        return is_fault

    def detect_historical(self, time_series, threshold=None, drift=None):
        detection_series = pd.Series(index=time_series.index, data=0)

        for ini_index in range(len(time_series) - (self.backward_window_size + self.forward_window_size)):
            sep_index = ini_index + self.backward_window_size
            end_index = sep_index + self.forward_window_size
            faults_indexes = self.one_pass(time_series.iloc[ini_index:sep_index],
                                           time_series.iloc[sep_index:end_index],
                                           threshold, drift)
            detection_series.iloc[sep_index:end_index][faults_indexes] = 1
        return detection_series

    def detect_and_visualize(self, time_series, threshold=None, drift=None):
        anomalies = pd.Series(np.where(self.detect_historical(time_series, threshold, drift) == 1, time_series, np.nan),
                              index=time_series.index)
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(15, 6))
            plt.title('Cusum Anomaly Detection', fontdict={'fontsize': 20})
            plt.plot(time_series, label='actual')
            plt.plot(anomalies, 'o', color='r', markersize=7, label='anomalies')
            plt.show()


class Changer(object):

    """Класс реализующий алгоритм подбора фиктивной истории после длительного простоя УС
       на основе геоданных, времени доступности, места размещения и корреляции/взаимной информации"""

    def __init__(self, ID, timeseries, timeseries_downtime, pool,
                 downtime_threshold=14, percentage_threshold=0, salary_threshold = 0.16,
                 df_address=None, df_cluster=None, df_availability=None,
                 order=('availability', 'cluster', 'address'), #, 'salary'),
                 distance_=3, kind='Correlation', normalization='minmax', salary_mode = True):

        self.timeseries = timeseries
        self.ID = ID
        self.timeseries_downtime = timeseries_downtime
        self.pool = pool
        self.downtime_threshold = downtime_threshold
        self.percentage_threshold = percentage_threshold
        self.df_address = df_address
        self.df_cluster = df_cluster
        self.df_availability = df_availability
        self.order = order
        self.distance_ = distance_
        self.kind = kind
        self.normalization = normalization
        self.salary_threshold = salary_threshold
        self.salary_mode = salary_mode
        self.lat_km_CONST = 111.3
        self.long_km_CONST = 62.25


    def detect_downtimes(self, timeseries_downtimes=None, downtime_threshold=None, percentage_threshold=None):

        if timeseries_downtimes is None:
            timeseries_downtimes = self.timeseries_downtime
        if downtime_threshold is None:
            downtime_threshold = self.downtime_threshold
        if percentage_threshold is None:
            percentage_threshold = self.percentage_threshold

        begin_dates = []
        end_dates = []

        series_mask = 1.0 * (timeseries_downtimes > percentage_threshold)
        count = 0

        for ind in series_mask.index:
            if count >= downtime_threshold and series_mask[ind] == 0:
                end_dates.append(ind - pd.Timedelta('1 days'))
                begin_dates.append(ind - pd.Timedelta(str(count) + ' days'))
            count = (count + 1) * series_mask[ind]

        if count >= downtime_threshold:
            end_dates.append(ind)
            begin_dates.append(ind - pd.Timedelta(str(count - 1) + ' days'))

        downtime_df = pd.DataFrame(np.array([begin_dates, end_dates]).T, columns=['downtime_begin', 'downtime_end'])
        return downtime_df

    def calc_distance(self, coord1, coord2):
        return np.linalg.norm((self.lat_km_CONST * (coord1[0] - coord2[0]), self.long_km_CONST * (coord1[1] - coord2[1])))

    def choose_same_atm_from_neighbourhood(self, ID=None, df_address=None, distance_=None):

        if ID is None:
            ID = self.ID
        if df_address is None:
            df_address = self.df_address
        if distance_ is None:
            distance_ = self.distance_

        pool_distances = pd.DataFrame(df_address['ATM_ID'], columns=['ATM_ID'])

        distances = []
        coord1 = tuple(df_address[df_address['ATM_ID'] == ID][['LATITUDE', 'LONGITUDE']].iloc[0])
        for i in df_address.index:
            coord2 = tuple(df_address[['LATITUDE', 'LONGITUDE']].loc[i])
            distances.append(self.calc_distance(coord1, coord2))
        pool_distances['distance'] = distances
        pool_distances = pool_distances[pool_distances.distance < distance_]
        return list(pool_distances['ATM_ID'])

    def choose_same_atm_from_availability(self, ID=None, df_availability=None):

        if ID is None:
            ID = self.ID
        if df_availability is None:
            df_availability = self.df_availability

        pool_availability = []
        for i in df_availability.index:
            if i != ID:
                if np.all(df_availability.loc[ID] == df_availability.loc[i]):
                    pool_availability.append(i)
        return pool_availability

    def choose_same_atm_from_cluster(self, ID, df_cluster):
        if ID is None:
            ID = self.ID
        if df_cluster is None:
            df_cluster = self.df_cluster
        return [x for x in list(df_cluster[df_cluster.cluster == df_cluster[df_cluster['ATM_ID'] == ID].cluster.iloc[0]]['ATM_ID']) if x != ID]

    def choose_same_atm_by_math(self, begin_index, end_index, timeseries=None, ID=None, pool=None, kind=None):

        if timeseries is None:
            timeseries = self.timeseries
        if ID is None:
            ID = self.ID
        if pool is None:
            pool = self.pool
        if kind is None:
            kind = self.kind

        ids = []
        math = []
        for i in pool.ATM_ID.unique():
            if i != ID:
                if kind == 'Correlation':
                    math.append(timeseries[begin_index:end_index].corr(
                        pool[pool['ATM_ID'] == i][begin_index:end_index]['CLIENT_OUT']))
                if kind == 'Mutual Information':
                    math.append(MI(np.array(timeseries[begin_index:end_index]).reshape(len(timeseries[begin_index:end_index]), 1),
                                   np.array(pool[pool.ATM_ID == i][begin_index:end_index]['CLIENT_OUT'])))
                ids.append(i)

        pool_math = pd.DataFrame(ids, columns=['ID'])
        pool_math['MATH'] = math
        pool_math = pool_math.sort_values(by='MATH', ascending=False)

        return list(pool_math['ID'])
    
    def choose_same_salary_atm(self, ID, pool_individual, begin_index, end_index):
        salary = Salary(self.timeseries[begin_index:end_index], threshold=self.salary_threshold)
        try:
            salary_days_id = set(salary.get_features().columns)
        except:
            return pool_individual
        pool = self.pool
        pool_salary = []
        count = 0
        for i in pool_individual:
            if i != ID:
                salary = Salary(pool[pool['ATM_ID'] == i]['CLIENT_OUT'], threshold=self.salary_threshold)
                try: 
                    cols = salary.get_features().columns
                    print(i, cols)
                except:
                    cols = []
                
                if set(cols) == salary_days_id:
                        pool_salary.append(i)
                        count = 0
                else:
                    count += 1
                
                if self.salary_mode and count > 10:
                    return pool_salary
                
        return pool_salary
                    

    def change_history(self, downtime_end = None, cold_start = False):

        if downtime_end is None:
            downtime_df = self.detect_downtimes(self.timeseries_downtime, downtime_threshold=self.downtime_threshold,
                                            percentage_threshold=self.percentage_threshold)[-1:]
            downtime_end = downtime_df[-1:]['downtime_end'].iloc[0]

        if not cold_start:
            pool_individual = self.choose_same_atm_by_math(timeseries=self.timeseries,
                                                       begin_index=downtime_end,
                                                       end_index=self.timeseries.index[-1],
                                                       ID=self.ID,
                                                       pool=self.pool,
                                                       kind=self.kind)
        else:
            pool_individual = [ATM_ID for ATM_ID in self.pool.ATM_ID.unique() if not ATM_ID == self.ID] 

        for i in self.order:
            if i == 'availability':
                if self.df_availability is not None:
                    pool_availability = self.choose_same_atm_from_availability(self.ID, self.df_availability)
                    if len(set(pool_individual).intersection(set(pool_availability))) > 0:
                        pool_individual = [x for x in pool_individual if x in pool_availability]

            if i == 'cluster':
                if self.df_cluster is not None:
                    pool_cluster = self.choose_same_atm_from_cluster(self.ID, self.df_cluster)
                    if len(set(pool_individual).intersection(set(pool_cluster))) > 0:
                        pool_individual = [x for x in pool_individual if x in pool_cluster]

            if i == 'address':
                if self.df_address is not None:
                    pool_distances = self.choose_same_atm_from_neighbourhood(self.ID, self.df_address, self.distance_)
                    if len(set(pool_individual).intersection(set(pool_distances))) > 0:
                        pool_individual = [x for x in pool_individual if x in pool_distances]
                        
            if i == 'salary':
                pool_salary = self.choose_same_salary_atm(self.ID, pool_individual, begin_index = downtime_end,
                                                              end_index=self.timeseries.index[-1])
                if len(set(pool_individual).intersection(set(pool_salary))) > 0:
                        pool_individual = [x for x in pool_individual if x in pool_salary]


        ids = pool_individual[:5]
        scaled_parts = []
        for i in ids:
            part = self.pool[self.pool.ATM_ID == i]['CLIENT_OUT'][:downtime_end]
            if self.normalization == 'minmax':
                part_minmax = (part - part.min()) / (part.max() - part.min())
                max_ts = self.timeseries[downtime_end:].max()
                min_ts = self.timeseries[downtime_end:].min()
                part_scaled = part_minmax * (max_ts - min_ts) + min_ts
            elif self.normalization == 'znorm':
                part_Z = (part - part.mean()) / part.std()
                std_ts = self.timeseries[downtime_end:].std()
                mean_ts = self.timeseries[downtime_end:].mean()
                part_scaled = (part_Z * std_ts) + mean_ts
            else:
                part_scaled = part
            scaled_parts.append(part_scaled)
        #scaled_parts[0] первая лучшая часть, прикрепленная к TS
        return scaled_parts, ids

    def visualize(self):
        scaled_parts, first, ids = self.change_history()

        with plt.style.context('seaborn-whitegrid'):
            font = {'family': 'normal',
                    'weight': 'bold',
                    'size': 40}
            matplotlib.rc('font', **font)
            plt.figure(figsize=(17, 5))
            plt.title('Before: {}'.format(self.ID), fontdict={'fontsize': 20})
            plt.plot(self.timeseries)
            for i in range(len(scaled_parts)):
                plt.figure(figsize=(17, 5))
                plt.grid(True)
                plt.title('After: {} + {}'.format(self.ID, ids[i]), fontdict={'fontsize': 20})
                plt.plot(self.timeseries[scaled_parts[i].index[-1] + pd.Timedelta('1 days'):], label='actual')
                plt.plot(scaled_parts[i], color='g', label='fictive')
                plt.legend()
                plt.show()
            plt.show()


holidays_from_prom_calendar = pd.read_csv('./data/lecture2/holidays_list.csv', index_col=0)
holidays_from_prom_calendar['holidays'] = pd.to_datetime(holidays_from_prom_calendar['holidays'])


class RussianBusinessCalendar(AbstractHolidayCalendar):
    start_date = datetime.datetime(1999, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    rules = [
        Holiday(name='Russian Day Off', year=d.year, month=d.month, day=d.day) for d in holidays_from_prom_calendar['holidays']
    ]


class Salary(object):

    def __init__(self, timeseries=None, threshold=0.15, rus_calendar=RussianBusinessCalendar(), number_iter=3,
                 params={'backward_window_size': 30, 'forward_window_size': 2, 'threshold': 1, 'drift': 1}):
        self.timeseries = timeseries
        self.threshold = threshold
        self.rus_calendar = rus_calendar
        self.number_iter = number_iter
        self.params = params

    def get_candidate_date(self, date_anomaly, threshold=None):

        if threshold is None:
            threshold = self.threshold

        anomalies_frequency = sorted(dict(Counter(list(date_anomaly[(date_anomaly == 1)].index.day))).items(),
                                     key=operator.itemgetter(1), reverse=True)

        candidate_date = []
        percentage = float(anomalies_frequency[0][1]) / sum(dict(Counter(list(date_anomaly[(date_anomaly == 1)].index.day))).values())
        if percentage > threshold:
            candidate_date.append(anomalies_frequency[0][0])
        return candidate_date

    def marking_salary(self, candidates, rus_calendar=None):

        if rus_calendar is None:
            rus_calendar = self.rus_calendar

        rusbus_day = CustomBusinessDay(calendar=rus_calendar)

        salary_dates = []
        for curr in candidates:
            for month in range(1, 13):
                for year in range(rus_calendar.start_date.year, rus_calendar.end_date.year + 1):
                    try:
                        curr_date = datetime.datetime(year, month, curr)
                    except:
                        curr_date = datetime.datetime(year, month, calendar.monthrange(year, month)[-1])
                    salary_dates.extend([curr_date + pd.Timedelta(days=1) - rusbus_day])
        salary_dates = sorted(salary_dates)
        return salary_dates

    def get_features(self, timeseries=None):

        if timeseries is None:
            timeseries = self.timeseries

        detector = AnomalyDetector(**self.params)
        tmp = timeseries.copy()
        salaries = None
        for i in range(self.number_iter):
            date_anomaly = detector.detect_historical(tmp)
            candidate_date = self.get_candidate_date(date_anomaly, threshold=self.threshold)

            if candidate_date:
                marking_salary_date = self.marking_salary(candidate_date)
                salary = pd.Series(0, index=pd.date_range(start=RussianBusinessCalendar.start_date,
                                                          end=RussianBusinessCalendar.end_date))
                salary[salary.index.isin(marking_salary_date)] = 1
                salary.name = str(candidate_date[0])
                if i == 0:
                    salaries = pd.DataFrame(salary)
                else:
                    salaries = pd.concat([salaries, salary], axis=1)
                tmp = pd.Series(np.where(salary[timeseries.index] == 1, np.nan, tmp), index=timeseries.index)
            else:
                break
        return salaries

    def visualize(self):
        salaries = self.get_features(self.timeseries)

        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(15, 6))
            plt.title('Automatic Salary days Detection')
            plt.plot(self.timeseries)
            ax = plt.axes()
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            if salaries is not None:
                for i in salaries.columns:
                    to_plot = pd.Series(np.where(salaries[i][self.timeseries.index] == 1, self.timeseries, np.nan),
                                        index=self.timeseries.index)
                    plt.plot(to_plot, "o",  markersize=7, label=salaries[i].name)
            plt.legend()
            plt.show()
            
        return salaries
    