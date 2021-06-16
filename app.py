import streamlit as st
from multiapp import MultiApp
from apps import home, svm, knn, logistic_regression
# decision_tree, random_forest, naive_bayes

app = MultiApp()

app.add_app("-----", home.app)
app.add_app("KNN", knn.app)
app.add_app("SVM", svm.app)
app.add_app("Logistic Regression", logistic_regression.app)
# app.add_app("Decision Tree", decision_tree.app)
# app.add_app("Random Forest", random_forest.app)
# app.add_app("Naive Bayes", naive_bayes.app)

app.run()
