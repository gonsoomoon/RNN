import pickle, os
import pandas as pd
os.chdir("/Users/gonsoomoon/Documents/DeepLearning/kaggle/rossmann/rossmann_publish")
# from cluster_feature import *


class ClusterFeature():
    """
    Desc: As an input, a supervised form of feature is given and
         then generate store_index and sales of month 1 to 12
    Usage:
    PICKLE_FILE_NAME = './data/feature_train_data.pickle'
    CF = ClusterFeature(PICKLE_FILE_NAME)
    print('feature shape: {}'.format(CF.df_f_list.shape))
    print('tip of a feature:\n {}'.format(CF.df_f_list.head(3)))
    """
    def __init__(self, pickle_file_name):
        df_all = self.merge_X_y(pickle_file_name)
        g_all = df_all.groupby(['store_index', 'month']) # group by store_index and month
        m_all = g_all.sales.aggregate(['mean']) # aggregate 'mean'
        f_list = self.make_feature_list(m_all)
        col_list = ['std_idx','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']
        self.df_f_list = pd.DataFrame(f_list, columns=col_list)
        self.save_clu_feature(self.df_f_list)  # save the feature to a pickle file

    def merge_X_y(self, pickle_file_name):
        """
        :return X combined with y values as a dataframe:
        """
        f = open(pickle_file_name, 'rb')
        (X, y) = pickle.load(f)

        df_X = pd.DataFrame(X, columns=['store_open',
                                        'store_index',
                                        'day_of_week',
                                        'promo',
                                        'year',
                                        'month',
                                        'day',
                                        'state'])
        df_y = pd.DataFrame(y, columns=['sales'])

        df_all = pd.concat([df_X, df_y], axis=1)
        return df_all


    def make_feature_list(self, sales_list):
        """
        Change a format of a list to one of a feature
        :param sales_list:
        :return:
        """
        feature_list = list()
        for store_idx, store_data in sales_list.groupby(level=0):
            month_list = list()
            month_list.append(store_idx)
            for month in range(len(store_data.index)):
                #print(store_idx, ",", month, ",", store_data.iloc[month, 0])
                month_list.append(store_data.iloc[month, 0])
            feature_list.append(month_list)
        return feature_list

    def save_clu_feature(self, df_feature):
        """
        Save a foramt of a dataframe to a pickle file
        :param df_feature:
        :return:
        """
        FEATURE_CLUSTER_DATA = './data/feature_cluster_data'
        with open(FEATURE_CLUSTER_DATA, 'wb') as f:
            pickle.dump(df_feature, f, -1)


#########################################################
# Pivoting Example
#########################################################
# df = pd.DataFrame(data={'id':['id01','id02','id01'],
#                         'date':['20170911','20181109','20171002'],
#                         'power':[223,345,223]})
# df['cdate'] = pd.to_datetime(df.date, format='%Y%m%d')
# df.drop(['date'], axis=1)
# df['year_month'] = df.cdate.dt.strftime('%Y%m')
# df = df.groupby(by=['id','year_month']).agg('sum')
# df_t = df.pivot_table(index=df.index.names[0], columns=df.index.names[1], values='power')
