import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def app():

    dataset = ['Iris', 'Breast Cancer', 'Wine']

    dataset_name = st.selectbox('Select Dataset', dataset)

    algorithm = "KNN"

    st.write(f"## {algorithm} Algorithm on {dataset_name} Dataset")

    class KNN():
        def fit(self, X_train, Y_train):
            self.X_train = X_train
            self.Y_train = Y_train

        def predict(self, X_test):
            predictions = []
            for row in X_test:
                label = self.closest(row)
                predictions.append(label)
            return predictions

        def closest(self, row):
            best_dist = distance.euclidean(row, self.X_train[0])
            best_index = 0
            for i in range(1, len(self.X_train)):
                dist = distance.euclidean(row, self.X_train[i])
                if dist < best_dist:
                    best_dist = dist
                    best_index = i
            return self.Y_train[best_index]

    if(dataset_name == "Iris"):
        dataset = datasets.load_iris()
    elif(dataset_name == "Wine"):
        dataset = datasets.load_wine()
    else:
        dataset = datasets.load_breast_cancer()

    X = dataset.data
    y = dataset.target

    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    Y = pd.Series(dataset.target, name='response')
    df = pd.concat( [x,Y], axis=1 )

    st.write(f"A look at the {dataset_name} dataset:")
    st.write(df.head(5))
    st.write('Shape of dataset:', X.shape)
    st.write('Number of classes:', len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .75)

    classifier = KNN()

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    st.write(f'Accuracy :', acc)

    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    st.pyplot(fig)