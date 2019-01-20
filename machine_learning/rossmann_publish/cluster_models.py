import pickle, os
os.chdir("/Users/gonsoomoon/Documents/DeepLearning/kaggle/rossmann/entity-embedding-rossmann-master")
import sys
sys.path.append("/Users/gonsoomoon/Documents/DeepLearning/kaggle/rossmann/entity-embedding-rossmann-master")
# from cluster_models import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
from sklearn import mixture
np.random.seed(1)

class ClusterModels():
    """
    Desc:
        With a cluster feature pickle file given, make KMean and GMM
        cluster models and then show performance of both.
        Using TSNE algorithm, they are visualized
    Usage:
    FEATURE_FILE_NAME = 'feature_cluster_data'
    N_CLUSTERS = 10
    models = ClusterModels(FEATURE_FILE_NAME)
    df_kmean_cluster, df_kmean_summary = models.makeKMeanCluster(N_CLUSTERS)
    df_gmm_cluster, df_gmm_summary = models.makeGMMCluster(N_CLUSTERS)

    print(models.getTotalCost(df_kmean_summary, df_kmean_cluster))
    print(models.getTotalCost(df_gmm_summary, df_gmm_cluster))

    kmean_tsne = models.makeEmbedding(models.df_kmean_cluster)
    models.showEmbedding(kmean_tsne, "KMean Cluster")

    gmm_tsne = models.makeEmbedding(models.df_gmm_cluster)
    models.showEmbedding(gmm_tsne, "GMM Cluster")

    """
    def __init__(self, file_name):
        self.df_feature = self.getCluFeature(file_name)

    def getCluFeature(self, file_name):
        """
        :return X and y values as a dataframe:df_k
        """
        f = open(file_name, 'rb')
        df_feature = pickle.load(f)
        df_feature = df_feature.iloc[:,1:]

        return df_feature

    def makeKMeanCluster(self, n_clusters):
        """
        Make a scaler, K-mean model() and cluster
        :param df_feature:
        :param n_clusters:
        :return:
        """

        df_feature_copy = self.df_feature.copy()
        scaler = StandardScaler().fit(df_feature_copy)
        standardized_data = scaler.transform(df_feature_copy)
        model = KMeans(n_clusters = n_clusters).fit(standardized_data)

        df_feature_copy['cluster'] = model.predict(standardized_data)
        summary = df_feature_copy.groupby('cluster').mean()
        summary['count'] = df_feature_copy['cluster'].value_counts()
        summary = summary.sort_values(by='count', ascending=False)
        self.df_kmean_cluster = df_feature_copy

        return df_feature_copy, summary

    def makeGMMCluster(self, n_clusters):
        """
        Make a scaler, gmm model() and cluster
        :param df_feature:
        :param n_clusters:
        :return:
        """

        df_feature_copy = self.df_feature.copy()
        scaler = StandardScaler().fit(df_feature_copy)
        standardized_data = scaler.transform(df_feature_copy)
        g = mixture.GMM(n_components= n_clusters, n_iter=200, verbose=2)
        g.fit(standardized_data)


        df_feature_copy['cluster'] = g.predict(standardized_data)
        summary = df_feature_copy.groupby('cluster').mean()
        summary['count'] = df_feature_copy['cluster'].value_counts()
        summary = summary.sort_values(by='count', ascending=False)

        self.df_gmm_cluster = df_feature_copy

        return df_feature_copy, summary

    def getTotalCost(self, df_summary, df_cluster):
        df_summary['cluster'] = df_summary.index

        # Copy Dec's Sales, cluster number
        df_cluster_eval = df_cluster.iloc[:, 11:13].copy()
        # Copy Dec's Sales, # of clusters, cluster number
        df_summary_eval = df_summary.iloc[:, 11:14].copy()
        # Join on the column 'cluster'
        df_join = df_cluster_eval.join(df_summary_eval, on='cluster', lsuffix='df_cluster_eval',
                                       rsuffix='df_summary_eval')
        df_join = df_join.iloc[:, 0:4] # Dec's sales, cluster, centroid and count
        df_join.columns = ['m12_sales', 'cluster', 'centroid', 'count']
        df_join['diff'] = (df_join['m12_sales'] - df_join['centroid']).abs()
        total_cost = df_join['diff'].sum() / len(df_join)
        return total_cost

    def makeEmbedding(self, df_cluster):
        df_feature = self.df_feature.iloc[:, 0:12]
        scaler = StandardScaler().fit(df_feature)
        std_data = scaler.transform(df_feature)
        X_embedding = TSNE(n_components=2, random_state=100).fit_transform(std_data)
        df_X_embedding = pd.DataFrame(X_embedding)
        df_tsne = pd.DataFrame()
        df_tsne['x'] = df_X_embedding.iloc[:,0]
        df_tsne['y'] = df_X_embedding.iloc[:,1]
        df_tsne['c'] = df_cluster.iloc[:,12]

        return df_tsne

    def showEmbedding(self, df_tsne, title):
        clusters = df_tsne.groupby('c').head(1).sort_values(by='c')['c'].values
        fig = plt.figure()
        plt.title(title)
        for c in clusters:
            df_c = df_tsne[df_tsne['c'] == c]
            x = df_c.iloc[:, 0]
            y = df_c.iloc[:, 1]
            plt.scatter(x=x, y=y, label=c)
            fig.legend(loc='best')
        plt.show()

# Reference:
# [1] k-means Clustering with Standardization,
#     http: // mlreference.com / k - means - standardization - sklearn




