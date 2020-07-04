# %% [code]
#Imports

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

# Function definitions

def elbow(data): # calculate and setup the elbow graph
    # calculate elbow curve
    wcss = []
    maxclusters = 11
    for i in range(1, maxclusters):
        kmeans = KMeans(n_clusters = i, init = 'random')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # setup elbow graph
    plt.plot(range(1, maxclusters), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Clusters')
    plt.ylabel('Within Cluster Sum of Squares')
    
    # return elbow data
    return wcss

def find_best_cluster_amount(data): # determine the ideal clustering and return it
    next_inertia = 0
    inertia = 0
    prev_inertia = 0
    current_q = 0
    best_q = 0
    best_i = 1
    elbow_data = elbow(data)
    elbow_data_len = len(elbow_data)
    elbow_data_range = range(1, elbow_data_len)
    for i in elbow_data_range:
        if i < elbow_data_len - 1:
            next_inertia = elbow_data[i + 1]
            inertia = elbow_data[i]
            prev_inertia = elbow_data[i - 1]
            
            current_q = prev_inertia - inertia + next_inertia
            
            if best_q < current_q:
                best_q = current_q
                best_i = i + 1
    return best_i


def fit_kmeans(data): # fit kmeans on the ideal clustering for the data
    kmeans = KMeans(n_clusters = find_best_cluster_amount(data), init = 'random') #random centroids
    kmeans.fit(data)
    return kmeans

def plot_with_kmeans(data, plot_column, classifier_column_label, classifier): 
    # plot figure setup
    fig = plt.figure()
    st = fig.suptitle('Forest Fires in {}'.format(classifier_column_label), fontsize='x-large')
    
    kmeans = fit_kmeans(data)
    
    # kmeans clustering graph
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 3], data[:, plot_column], s = 50, c = kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red')
    plt.title('Forest Fires Per {}'.format(classifier))
    plt.xlabel('Forest Fires')
    plt.ylabel('{}'.format(classifier))
    plt.tight_layout()
    
    st.set_y(0.95) #gap after title
    fig.subplots_adjust(top=0.80) #gap b/w y plot values

    # show result
    plt.show()

def filter_array(array, column, value): # filter an array by column and value
    filtered_array = np.array(list(filter(lambda x: x[column] == value, array)))
    return filtered_array

def array_factory(array, column, values): # filter an array by column and value for n values and return
    array_group = []
    for value in values:
        array_group.append(filter_array(array, column, value))
    return np.array(array_group)

def multiplot(plot_range, arrays, plot_column, classifier_column_labels, classifier): # plot the proper kmeans graph for each array input
    for i in plot_range:
        plot_with_kmeans(arrays[i], plot_column, classifier_column_labels[i], classifier)

# %% [code]
# load in dataset
forest_fires = pd.read_csv('amazon.csv', encoding='latin-1', sep=',')

forest_fires['state'], states = pd.factorize(forest_fires['state'])
forest_fires['month'], months = pd.factorize(forest_fires['month'])

forest_fires_array = np.array(forest_fires)
forest_fires_array = np.delete(forest_fires_array, 4, 1)

print('Pre-processed data: ')
print(forest_fires_array)

# %% [code]
# Plots for each state

plot_range = range(0, 23)
arrays = array_factory(forest_fires_array, 1, plot_range)
plot_column = 2
classifier_column_labels = states
classifier = 'Month'

multiplot(plot_range, arrays, plot_column, classifier_column_labels, classifier)

# %% [code]
# Plots for each month

plot_range = range(0, 12)
arrays = array_factory(forest_fires_array, 2, plot_range)
plot_column = 1
classifier_column_labels = months
classifier = 'State (numbered)'

multiplot(plot_range, arrays, plot_column, classifier_column_labels, classifier)