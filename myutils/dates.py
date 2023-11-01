# time and dates

from datetime import datetime
from myutils.process import get_list_from_csv


def get_rolling_window(dates, window_size=4):
    # Convert dates to unique and sorted datetime objects
    unique_dates = sorted(set(dates), key=lambda x: datetime.strptime(x, '%Y-%m-%d'))

    num_dates = len(unique_dates)
    if num_dates < window_size:
        return []

    rolling_window = []
    for i in range(num_dates - window_size + 1):
        window = unique_dates[i:i + window_size]
        rolling_window.append(window)

    return rolling_window


def get_months(month_input):
    month_input_is_csv = month_input.endswith('.csv')
    month_input_is_list = isinstance(month_input, list)
    if month_input_is_csv:
        months = get_list_from_csv(month_input)
    elif month_input_is_list:
        months = monthlist(month_input)
    elif month_input == 'all':
        months = monthlist(['2019-11', '2021-09'])
    elif month_input == 'reduced':
        months = monthlist(['2020-09', '2021-08'])
    else:
        months = [month_input]
    return months


def monthlist(dates):
    start, end = [datetime.strptime(_, "%Y-%m") for _ in dates]
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1, 1).strftime("%Y-%m"))
    return mlist


def total_months(dt):
    return dt.month + 12 * dt.year