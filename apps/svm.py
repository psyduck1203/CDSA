import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def app():

    dataset = ['Iris', 'Breast Cancer', 'Wine']

    dataset_name = st.selectbox('Select Dataset', dataset)

    algorithm = "SVM"

    st.write(f"## {algorithm} Algorithm on {dataset_name} Dataset")
 
    class SVM:
        def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
            self.lr = learning_rate
            self.lambda_param = lambda_param
            self.n_iters = n_iters
            self.w = None
            self.b = None

        def fit(self, X, y):
            n_samples, n_features = X.shape
            
            y_ = np.where(y <= 0, -1, 1)
            
            self.w = np.zeros(n_features)
            self.b = 0

            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        self.w -= self.lr * (2 * self.lambda_param * self.w)
                    else:
                        self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                        self.b -= self.lr * y_[idx]

        def predict(self, X):
            approx = np.dot(X, self.w) - self.b
            return np.sign(approx)
 
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
    
    classifier = SVM()

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