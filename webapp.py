import pandas as pd
import numpy as np
import streamlit as st
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer


def main():
    selected_feature =['area error',
    'compactness error',
    'mean area',
    'mean compactness',
    'mean concave points',
    'mean concavity',
    'mean perimeter',
    'mean radius',
    'mean smoothness',
    'mean texture',
    'perimeter error',
    'radius error',
    'worst compactness',
    'worst concavity',
    'worst perimeter',
    'worst radius'
    ]
    st.write(""" # Breast Cancer Detection """)
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], 'target'))
    st.subheader("Data Information:")
    st.dataframe(df.head(20))
    X = df.drop('target',axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=111)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.subheader('Model Test Accuracy Score: '+ str(round(accuracy_score(y_test, y_pred),4)*100)+' %')


    user_input = get_user_input(df)
    prediction = rf.predict(user_input)
    st.subheader('Cancer Result: ' + ('Benign', 'Malignant')[int(prediction)])


def get_user_input(df):
    user_data = {feature : st.sidebar.slider(feature, math.floor(df[feature].min()), math.floor(df[feature].max())+1, math.floor(df[feature].mean())) for feature in df.columns[:30]}
    features = pd.DataFrame(user_data, index=[0])
    return features

if __name__ == '__main__':
    main()