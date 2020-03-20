# encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FuncFormatter
import seaborn as sns
from collections import Counter
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import datetime
import sys
import os

russian_calendar = pd.read_csv('./data/lecture2/holidays_list.csv', index_col=0)
russian_calendar['holidays'] = pd.to_datetime(russian_calendar['holidays'])


class RussianBusinessCalendar(AbstractHolidayCalendar):
    start_date = datetime.datetime(1999, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    rules = [
        Holiday(name='Russian Day Off', year=d.year, month=d.month, day=d.day) for d in russian_calendar['holidays']
    ]


class AnomalyDetector(object):
    """
    Class which use CUSUM anomaly detection.

    A cumulative sum (CUSUM) chart is a type of control chart used to monitor small shifts in the process mean.

    Parameters
    ----------
    backward_window_size : integer, optional, default 30
        The window size of timeseries for estimate stats (like train)

    forward_window_size : integer, optional, default 14
        The window size of timeseries for compare with backward_window_size (like test)

    threshold : float, optional, default 5.0
        The maximum(minimum, with opposite sign) value of cumulative changes

    drift : float, optional, default 1.0
        The permissible deviation of timeseries from the mean

    Attributes
    ----------
    anomalies_ : timeseries of binary value (with initial timeseries index), where 1 - anomaly, 0 - non-anomaly
    """

    def __init__(self, backward_window_size=30, forward_window_size=14, threshold=5.0, drift=1.0):
        self.backward_window_size = backward_window_size
        self.forward_window_size = forward_window_size
        self.threshold = threshold
        self.drift = drift
        self.anomalies_ = None

    def one_pass(self, train_zone, prediction_zone, threshold=None, drift=None):
        """
        Detect anomaly in one pass

        Parameters
        ----------
        train_zone : pandas.Series or pandas.DataFrame
            Train sample to calculate statistics of timeseries

        prediction_zone : pandas.Series or pandas.DataFrame
            Test sample to find anomaly variables

        threshold : float, optional, default 5.0
            See parameter in ``threshold`` in :class:`AnomalyDetector`:func:`__init__`

        drift : float, optional, default 1.0
            See parameter in ``drift`` in :class:`AnomalyDetector`:func:`__init__``

        Returns
        -------
        is_fault : binary numpy array, shape = [len(prediction_zone)]
            1 - anomaly, 0 - nonanomaly
        """

        if not threshold:
            threshold = self.threshold
        if not drift:
            drift = self.drift

        current_std = np.nanstd(train_zone, ddof=1)
        current_mean = np.nanmean(train_zone)
        drift = drift * current_std
        threshold = threshold * current_std

        x = prediction_zone.astype('float64')
        gp, gn = np.zeros(x.size), np.zeros(x.size)

        for i in range(1, x.size):
            gp[i] = max(gp[i - 1] + x[i] - current_mean - drift, 0)
            gn[i] = min(gn[i - 1] + x[i] - current_mean + drift, 0)

        is_fault = np.logical_or(gp > threshold, gn < -threshold)
        return is_fault

    def detect(self, time_series, threshold=None, drift=None, excluded_points=None):
        """
        Detect anomaly in rolling window (=forward_window_size)

        Parameters
        ----------
        time_series : pandas.Series
            Target timeseries

        threshold : float, optional, default 5.0
            See parameter in ``threshold`` in :class:`AnomalyDetector`:func:`__init__`

        drift : float, optional, default 1.0
            See parameter in ``drift`` in :class:`AnomalyDetector`:func:`__init__``

        excluded_points : pandas.Series.index
            Acquainted anomaly events. They will be removed from timeseries before anomaly detection

        Returns
        -------
        self.anomalies_ : pandas.Series, shape = [len(time_series)]
            Labeled timeseries with anomaly, where 1 - anomaly, 0 - nonanomaly
        """
        if excluded_points is not None:
            time_series[time_series.index.isin(excluded_points)] = np.nan

        ts_values = time_series.values
        ts_index = time_series.index

        detection_series = np.zeros(len(ts_values)).astype('int32')

        for ini_index in range(len(ts_values) - (self.backward_window_size + self.forward_window_size)):
            sep_index = ini_index + self.backward_window_size
            end_index = sep_index + self.forward_window_size
            faults_indexes = self.one_pass(ts_values[ini_index:sep_index],
                                           ts_values[sep_index:end_index],
                                           threshold, drift)
            detection_series[sep_index:end_index][faults_indexes] = 1
        self.anomalies_ = pd.Series(detection_series, index=ts_index)

        return self.anomalies_

    def plot(self, time_series, ax=None, figsize=(14, 7),
             xlabel='Дата', ylabel='тысяч рублей', title='Plot Cusum Anomaly Detection',
             grid=True, marketsize=5):
        """
        Plot timeseries with anomaly points

        Parameters
        ----------
        time_series : pandas.Series
            Target timeseries

        ax : matplotlib object, optional, default None
            If ax is not None, use giving axis in current subplot

        figsize : tuple, optional, default (14, 7)
            If ax is None, figsize - size of plot

        xlabel : string, optional, default 'Дата'
            Label of x axis

        ylabel : string, optional, default 'тысяч рублей'
            Label of y axis

        title : string, optional, default 'Plot Cusum Anomaly Detection'
            Title of plot

        grid : boolean, optional, default True
            If True, use grid at plot

        marketsize : float, optional, default 5
            Size of anomaly points on timeseries plot

        Returns
        -------
        Plot timeseries with anomalies
        """
        anomalies = pd.Series(np.where(self.anomalies_ == 1, time_series, np.nan),
                              index=time_series.index)
        fig, ax = AnomalyDetector._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.plot(time_series, label='actual')
        ax.plot(anomalies, 'o', color='r', markersize=marketsize, label='anomalies')
        ax.legend(loc='best')
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    def hist(self, meas='day', th=0.15, ax=None, figsize=(14, 7),
             xlabel='День месяца', ylabel='количество аномалий', title='Hist Cusum Anomaly Detection',
             grid=True):
        """
        Plot hist of anomaly points

        Parameters
        ----------
        meas : pd.datetime attribute, optional, default 'day'

        th : float, optional, default 0.15

        time_series : pandas.Series
            Target timeseries

        ax : matplotlib object, optional, default None
            If ax is not None, use giving axis in current subplot

        figsize : tuple, optional, default (14, 7)
            If ax is None, figsize - size of plot

        xlabel : string, optional, default 'День месяца'
            Label of x axis

        ylabel : string, optional, default 'количество аномалий'
            Label of y axis

        title : string, optional, default 'Plot Cusum Anomaly Detection'
            Title of plot

        grid : boolean, optional, default True
            If True, use grid at plot

        Returns
        -------
        Plot histogramm of anomalies per month
        """

        idx, anomaly_count, periodic_anomaly_idx = self.__count_anomaly(th, meas)
        simple_color = '#36b2e2'
        anomaly_gradient_colors = dict(zip(periodic_anomaly_idx,
                                           sns.color_palette("Reds", len(periodic_anomaly_idx)).as_hex()[::-1]))
        colors = [simple_color if x[1] / sum(anomaly_count) < th else anomaly_gradient_colors[x[0]]
                  for x in zip(idx, anomaly_count)]
        fig, ax = AnomalyDetector._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.set_xlim(0, max(idx))
        ax.set_ylim(0, max(anomaly_count) + 1)
        ax.bar(idx, anomaly_count, color=colors)
        handles = [(x[0], anomaly_gradient_colors[x[0]])
                   for x in zip(idx, anomaly_count) if x[1] / sum(anomaly_count) >= th]
        handles = [mpatches.Patch(color=x[1], label=x[0]) for x in handles]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1.05), fancybox=True, shadow=True)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def __count_anomaly(self, th, meas):
        anomaly_idx = getattr(self.anomalies_[self.anomalies_ == 1].index, meas)
        count_anomalies_by_idx = sorted(Counter(anomaly_idx).items(), key=lambda x: x[1], reverse=True)
        idx = [x[0] for x in count_anomalies_by_idx]
        anomaly_count = [x[1] for x in count_anomalies_by_idx]
        periodic_anomaly_idx = [x[0] for x in count_anomalies_by_idx if x[1] / len(anomaly_idx) >= th]
        return idx, anomaly_count, periodic_anomaly_idx

    @staticmethod
    def _conf_axs(ax, figsize, xlabel, ylabel, title, grid):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True)
        return fig, ax


class AnomalyDetectorDaily(AnomalyDetector):
    """
    Daily class

    Find periodic anomalies, plot and mark anomalies on calendar with holidays
    """

    def get_as_features(self, th=0.15):
        days, anomaly_count, periodic_anomaly_days = self.__count_anomaly(th)
        return periodic_anomaly_days

    def mark_features(self, day, sdate=None, edate=None, kind=0, custom_calendar=None):
        """
        Mark periodic anomaly on calendar with holidays

        Don't work with default parameters without run :class:`AnomalyDetectorDaily`:func:`detect`

        Parameters
        ----------
        day : integer
            Day which you want to mark. Try use day with the largest value from hist

        sdate : string, optional, default None
            Start date with format "yyyy-mm-dd".

        edate : string, optional, default None
            End date with format "yyyy-mm-dd".

        kind : integer, optional, default 0
            Type of shift periodic anomaly on holiday.
            kind=1 - to first working day, kind=0 - no shift, kind=-1 - to last working day

        custom_calendar : pandas object, optional, default RussianBusinessCalendar
            pandas.tseries.holiday.AbstractHolidayCalendar() - calendar with marked holidays

        Returns
        -------
        mark - pd.Series with marked periodic anomaly

        """
        if sdate is None:
            sdate = self.anomalies_.index[0]
        if edate is None:
            edate = self.anomalies_.index[-1]
        if custom_calendar is None:
            custom_calendar = RussianBusinessCalendar()
        custom_busday = CustomBusinessDay(calendar=custom_calendar)

        if kind == -1:
            mask = [datetime.datetime(year, month, day) + pd.Timedelta(days=1) - custom_busday
                    for month in range(1, 13) for year in range(sdate.year, edate.year + 1)
                    ]
        elif kind == 0:
            mask = [datetime.datetime(year, month, day)
                    for month in range(1, 13) for year in range(sdate.year, edate.year + 1)
                    ]
        elif kind == 1:
            mask = [datetime.datetime(year, month, day) - pd.Timedelta(days=1) + custom_busday
                    for month in range(1, 13) for year in range(sdate.year, edate.year + 1)
                    ]
        else:
            sys.exit("kind must be in [-1, 0, 1]")

        mask = pd.Series(self.anomalies_.index.isin(mask).astype(int), index=self.anomalies_.index)

        return mask


class OstrovPredictor(object):

    def __init__(self,
                 model=RandomForestRegressor(random_state=2),
                 add_features_table=None,
                 backward_window_size=365,
                 forward_window_size=30,
                 auto_lags=False,
                 get_features_importance=False):

        self.model = model
        self.add_features_table = add_features_table
        self.backward_window_size = backward_window_size
        self.forward_window_size = forward_window_size
        self.auto_lags = auto_lags
        self.get_features_importance = get_features_importance

    def one_pass_create_table(self, timeseries):

        data = pd.DataFrame(timeseries).copy()
        data.columns = ['y']
        
        if self.auto_lags:
            r = []
            for n in range(1, 50 + 1):
                r_current = np.corrcoef(data.y[n:], data.y.shift(n)[n:])[0, 1]
                r.append(r_current)
            all_lags = []
            b = list(zip(r, list(range(1, 51))))[:20]
            all_lags.append([x[1] for x in sorted(b, key=lambda x: np.abs(x[0]), reverse=True)[:2]])
            self._needed_lags = all_lags[0]
            for q in self._needed_lags:
                data['лаг {}'.format(q)] = data.y.shift(q)
            
        if self.add_features_table is not None:
            data = pd.concat([data, self.add_features_table], axis=1, join='inner')
        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        return data

    def one_pass_prediction(self, timeseries, test_index):

        data = self.one_pass_create_table(timeseries=timeseries)
        data = data.dropna()

        x_train = data[:test_index].drop(["y"], axis=1)
        y_train = data[:test_index]["y"]
        x_test = data[test_index:].drop(["y"], axis=1)
        y_test = data[test_index:]["y"]

        self.model.fit(x_train, y_train)
        prediction = pd.Series(self.model.predict(x_test), index=y_test.index)
        feat_imp = '-'
        if self.get_features_importance:
            feat_imp = self.model.feature_importances_
        prediction[prediction < 0] = 0

        return y_test, prediction, feat_imp

    def predict(self, timeseries, backward_window_size=None, forward_window_size=None):
        if backward_window_size is None:
            backward_window_size = self.backward_window_size
        if forward_window_size is None:
            forward_window_size = self.forward_window_size

        number_folds = int((len(timeseries) - backward_window_size) / forward_window_size)
        remainder = (len(timeseries) - backward_window_size) % forward_window_size
        prediction = []
        y_total = []
        features_importances=[]
        for shift in range(number_folds):
            start_index = shift * forward_window_size
            end_index = start_index + backward_window_size + forward_window_size
            y_test, pred, feat_imp = self.one_pass_prediction(timeseries[start_index:end_index], -forward_window_size)
            prediction += list(pred)
            y_total += list(y_test)
            features_importances.append(list(feat_imp))
            
        if remainder != 0:
            y_test, pred,feat_imp = self.one_pass_prediction(timeseries[-(backward_window_size + remainder):], -remainder)
            prediction += list(pred)
            y_total += list(y_test)
            features_importances.append(list(feat_imp))

        return pd.Series(y_total, index=timeseries[backward_window_size:].index), pd.Series(prediction, index=timeseries[backward_window_size:].index),features_importances