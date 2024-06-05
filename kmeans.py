import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyKMeans:
    def __init__(self, k, df, max_iter=300):
        self.k = k
        self.df = df
        self.max_iter = max_iter
        self.total_error = 0

    """Method to initiate centroids"""
    def initiate_centroids(self):
        centroids = self.df.sample(self.k)
        return centroids

    """Method to calculate the residual sum of squares error"""
    def rsserr(self, a, b):
        return np.square(np.sum((a-b)**2))
    
    """Method to assign clusters to observations based on the nearest centroid to each observation and the error of the nearest centroid""" 
    def assign_clusters(self, centroids):
        n = self.df.shape[0]
        assignation = []
        assign_errors = []

        for obs in range(n):
            all_errors = np.array([])
            for centroid in range(self.k):
                err = self.rsserr(centroids.iloc[centroid,:], self.df.iloc[obs,:])
                all_errors = np.append(all_errors, err)

            nearest_centroid = np.where(all_errors == np.min(all_errors))[0].tolist()[0]
            nearest_centroid_error = np.amin(all_errors)

            assignation.append(nearest_centroid)
            assign_errors.append(nearest_centroid_error)

        return assignation, assign_errors
    
    """Method to fit the model to the data and return the dataframe with the assigned clusters and the centroids of the clusters"""
    def fit(self, tol=1e-4):
        workdf = self.df.copy()
        err = []
        goahead = True
        j = 0

        centroids = self.initiate_centroids()

        while goahead:
            workdf['cluster'], workdf['error'] = self.assign_clusters(centroids)
            err.append(workdf['error'].sum())

            new_centroids = workdf.groupby('cluster').agg('mean').reset_index(drop=True)

            if j > 0:
                if np.abs(err[j] - err[j-1]) < tol:
                    goahead = False

            centroids = new_centroids
            j += 1

        self.total_error = workdf['error'].sum()

        return workdf, centroids
    
    """Method to plot the data and the centroids of the clusters"""
    def plot(self, df, centroids):
        custom_cmap = plt.cm.get_cmap('viridis', self.k)
        colnames = list(df.columns[:-1])
        
        fig, ax = plt.subplots(figsize=(19, 5))
        plt.scatter(df.iloc[:,0], df.iloc[:,1], marker='o', c=df['cluster'].astype('category'), cmap=custom_cmap, s=80, alpha=0.5)
    
        plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1], marker='x', c=df['cluster'].unique(), cmap=custom_cmap, s=200, linewidths=2)
        ax.set_title('KMeans Clustering with K = {0}'.format(self.k))
        ax.set_xlabel(colnames[0])
        ax.set_ylabel(colnames[1])
        plt.show()

    def plot_3D(self, df):
        # custom_cmap = plt.cm.get_cmap('viridis', self.k)
        colnames = list(df.columns[:-1])
        
        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        for label in df['cluster'].unique():
            ax.text3D(
                df[df['cluster'] == label].iloc[:,3].mean(),
                df[df['cluster'] == label].iloc[:,0].mean(),
                df[df['cluster'] == label].iloc[:,2].mean(),
                label,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w')
            )

        ax.scatter(df.iloc[:, 3], df.iloc[:, 0], df.iloc[:, 2], c=df['cluster'], edgecolor='k', s=150, cmap='viridis')
        ax.view_init(20, -50)
        ax.set_xlabel(colnames[2])
        ax.set_ylabel(colnames[0])
        ax.set_zlabel(colnames[1])
        ax.set_title('KMeans Clustering with K = {0}'.format(self.k))
        plt.show()