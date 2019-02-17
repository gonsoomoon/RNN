
from datetime import datetime
from dateutil.relativedelta import relativedelta

class KMeanDTW():
    """
    Usage:
    KMean_DTW = KMeanDTW()

    start_time = "2018-01"
    end_time = "2018-04"
    n_learning_months = 3 # A learning period
    KMean_DTW.train_kmean_dtw(start_time, end_time, n_learning_months)
    """
    def __init__(self):
        pass

    def train_kmean_dtw(self, start_time, end_time, n_learning_period):
        """
        :param start_time: Point to a starting month
        :param end_time: Point to an ending month
        :param n_month: a learning period
        :return:
        """
        start_dt = datetime.strptime(start_time,"%Y-%m")
        end_dt = datetime.strptime(end_time, "%Y-%m")
        n_month = end_dt.month - start_dt.month
        for month in range(0, n_month+1):
            target_month_dt = start_dt + relativedelta(months= month)
            print("target time: {} ".format(target_month_dt.strftime("%Y-%m")))
            learning_months = self.get_learning_months(target_month_dt, n_learning_period)
            print("learning months: {}".format(learning_months))

    def get_learning_months(self, target_month, n_month):
        """
        :param target_month: forcasting month as datetime object
        :param n_month: # of months to be learned
        :return: a list of months to be learned
        """
        learning_months = list()
        for i in range(1, n_month+1):
            learning_month_dt = target_month - relativedelta(months= i)
            learning_month = datetime.strftime(learning_month_dt, "%Y-%m")
            #print(learning_month)
            learning_months.append(learning_month)

        return learning_months


# KMean_DTW = KMeanDTW()
#
# start_time = "2018-01"
# end_time = "2018-04"
# n_learning_months = 3 # A learning period
# KMean_DTW.train_kmean_dtw(start_time, end_time, n_learning_months)
