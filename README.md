# Comparing Different Sklearn Algorithms

# Demo

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/psyduck1203/cdsa/main/app.py)

# Algorithms Implemented

- [KNN](https://sklearn.org/modules/neighbors.html#classification) - K Nearest Neighbors
- [SVM](https://sklearn.org/modules/svm.html#svm) - Support Vector Machine
- [LR](https://sklearn.org/modules/linear_model.html#logistic-regression) - Logistic Regression

# Datasets Used

- [Iris](https://sklearn.org/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)
- [Breast Cancer](https://sklearn.org/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)
- [Wine](https://sklearn.org/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *cdsaenv*
```
conda create -n cdsaenv python=3.7.9
```
Secondly, we will login to the *cdsaenv* environement
```
conda activate cdsaenv
```
### Install prerequisite libraries

Download requirements.txt file

```
https://github.com/psyduck1203/CDSA/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

Major libraries required

- [Streamlit](https://streamlit.io/)
- [Sklearn](https://sklearn.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

### Download and unzip this repo

Download this [repo](https://github.com/psyduck1203/CDSA/archive/main.zip) and unzip as your working directory.

###  Launch the app
```
streamlit run app.py
```
