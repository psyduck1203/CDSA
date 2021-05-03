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

    algorithm = "Logistic Regression"

    st.write(f"## {algorithm} Algorithm on {dataset_name} Dataset")
 
    class LogisticRegression() :
        def __init__( self, learning_rate, iterations ) :        
            self.learning_rate = learning_rate        
            self.iterations = iterations
            
        def fit( self, X, Y ) :               
            self.m, self.n = X.shape       
            self.W = np.zeros( self.n )        
            self.b = 0        
            self.X = X        
            self.Y = Y
            
            for i in range( self.iterations ) :            
                self.update_weights()            
            return self
        
        def update_weights( self ) :           
            A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
                    
            tmp = ( A - self.Y.T )        
            tmp = np.reshape( tmp, self.m )        
            dW = np.dot( self.X.T, tmp ) / self.m         
            db = np.sum( tmp ) / self.m 
                
            self.W = self.W - self.learning_rate * dW    
            self.b = self.b - self.learning_rate * db
            
            return self
        
        def predict( self, X ) :    
            Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
            Y = np.where( Z > 0.5, 1, 0 )        
            return Y
 
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
    
    classifier = LogisticRegression(learning_rate = 0.01, iterations = 1000 )

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