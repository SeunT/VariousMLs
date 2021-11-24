import numpy as np
from mnist import MNIST
import scipy.spatial as sci
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
def get_dist(test_data, train_data):
    dist_arr = []
    test_count = 0
    for i in test_data:
        test_count += 1
        dist = np.array([sci.distance.euclidean(i, data) for data in train_data])
        dist_arr.append(dist)
    return dist_arr


def predict(y_train, x_test, distance, K):
    predicted_val = []
    for dex, i in enumerate(x_test):
        ## grab k neirest neighbors
        neighbors = np.argsort(distance[dex])[:K]
        voting = {}
        for index in neighbors:
            label = y_train[index]
            if label in voting:
                voting[label] += 1
            else:
                voting[label] = 1
        predicted_val.append(max(voting, key=voting.get))
    return predicted_val


def dist(x1, x2):
    a = np.array(x1)
    b = np.array(x2)
    return np.sqrt(np.sum(np.square(a - b)))


# Press the green button in the gutter to run the script.
def Kmeans(data, K):
    changed = True
    cluster_centers = data.sample(n=K).values[:, :13]

    while changed:
        count = 0
        for ind, i in enumerate(data.values):
            i = i[:13]
            minimum_dis = float('inf')
            mu_class = -1
            for index, mus in enumerate(cluster_centers):
                distance = np.linalg.norm(mus - i)

                if minimum_dis > distance:
                    minimum_dis = distance
                    mu_class = index

            if data.at[ind, 'mu'] != mu_class:
                data.at[ind, 'mu'] = mu_class
                count = 1

        cluster_centers = pd.DataFrame(data).groupby(by='mu').mean().values
        if count == 1:
            changed = True
        else:
            changed = False
    return data, cluster_centers


def GMM():
    print()


if __name__ == '__main__':

    ##-------------------------------------------
    # KNN algorithm
    ##-------------------------------------------

    mndata = MNIST('data')
    train_img, train_lbl = mndata.load_training()
    test_img, test_lbl = mndata.load_testing()
    train_lbl = list(train_lbl)
    test_lbl = list(test_lbl)

    Ks = [1, 3, 5, 10, 20, 30, 40, 50, 60]
    accuracies = []
    dist_arr = get_dist(test_data=test_img, train_data=train_img)
    for k in Ks:
        pred = predict(y_train=train_lbl, x_test=test_img, distance=dist_arr, K=k)
        acc = accuracy_score(test_lbl, pred)
        accuracies.append(acc)
        print("K = " + str(k) + "; Accuracy: " + str(acc))

    plt.plot(Ks, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")

    ##-------------------------------------------
    # KMeans Algorithm
    ##-------------------------------------------

    means = []
    df = pd.read_csv("data/CSE575-HW03-Data.csv", header=None)

    #  df = df.loc[:, [0, 1]]
    df['mu'] = -1
    kVals = [2, 3, 4, 5, 6, 7, 8, 9]
    for k in kVals:
        clusters, centers = Kmeans(df, k)

        sns.scatterplot(clusters.values[:, 0], clusters.values[:, 1], hue=clusters["mu"].values)
        plt.xlabel('first')
        plt.ylabel('second')
        plt.show()
    print()

    ##-------------------------------------------
    # GMM Algorithm
    ##-------------------------------------------
