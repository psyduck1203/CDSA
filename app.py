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







































































# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier

# st.write(" # Comparing Different Sklearn Algorithms ")

# classification = ['SVM', 'KNN', 'Random Forest']


# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == 'SVM':
#         C = st.sidebar.slider('C', 0.01, 10.0)
#         params['C'] = C
#     elif clf_name == 'KNN':
#         K = st.sidebar.slider('K', 1, 15)
#         params['K'] = K
#     else:
#         max_depth = st.sidebar.slider('max_depth', 2, 15)
#         params['max_depth'] = max_depth
#         n_estimators = st.sidebar.slider('n_estimators', 1, 100)
#         params['n_estimators'] = n_estimators
#     return params

# params = add_parameter_ui(classifier_name)

# def get_classifier(clf_name):
#     clf = None
#     if clf_name == 'SVM':
#         clf = SVC()
#     elif clf_name == 'KNN':
#         clf = KNeighborsClassifier()
#     else:
#         clf = LogisticRegression()
#     return clf

# clf = get_classifier(classifier_name)

# #### CLASSIFICATION ####

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test, y_pred)

# st.write(f'Classifier = {classifier_name}')
# st.write(f'Accuracy =', acc)

# #### PLOT DATASET ####
# # Project the data onto the 2 primary principal components
# pca = PCA(2)
# X_projected = pca.fit_transform(X)

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# fig = plt.figure()
# plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar()

# #plt.show()
# st.pyplot(fig)