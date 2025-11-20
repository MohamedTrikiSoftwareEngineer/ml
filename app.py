import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache_data
def load_data():
    # Load Titanic dataset
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    return df

@st.cache_data
def preprocess_data(df):
    # Create a copy
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Feature engineering
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                           'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'Title']
    
    return data, features

# Train model
@st.cache_resource
def train_model():
    df = load_data()
    data, features = preprocess_data(df)
    
    X = data[features]
    y = data['Survived']
    
    # Remove any remaining NaN values
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                   max_depth=5, min_samples_split=10)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cm, features, X_test, y_test

# Main application
def main():
    st.set_page_config(page_title="Titanic Survival Predictor", 
                      page_icon="🚢", layout="wide")
    
    st.title('🚢 Titanic Survival Predictor')
    st.write('Predict passenger survival using machine learning')
    
    # Sidebar
    st.sidebar.header('Passenger Information')
    
    # Train model
    model, accuracy, cm, features, X_test, y_test = train_model()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(['🔮 Prediction', '📊 Model Performance', '📈 Data Analysis'])
    
    with tab1:
        st.header('Enter Passenger Details')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pclass = st.selectbox('Passenger Class', [1, 2, 3], 
                                 help='1=First, 2=Second, 3=Third')
            sex = st.selectbox('Sex', ['Male', 'Female'])
            age = st.slider('Age', 0, 80, 30)
        
        with col2:
            fare = st.number_input('Fare', 0.0, 500.0, 50.0)
            embarked = st.selectbox('Port of Embarkation', 
                                   ['Southampton', 'Cherbourg', 'Queenstown'])
            family_size = st.slider('Family Size', 1, 11, 1)
        
        with col3:
            title = st.selectbox('Title', ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])
        
        # Prepare input
        sex_encoded = 0 if sex == 'Male' else 1
        embarked_map = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
        embarked_encoded = embarked_map[embarked]
        title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        title_encoded = title_map[title]
        is_alone = 1 if family_size == 1 else 0
        
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_encoded],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked_encoded],
            'FamilySize': [family_size],
            'IsAlone': [is_alone],
            'Title': [title_encoded]
        })
        
        if st.button('🎯 Predict Survival', type='primary'):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown('---')
            
            if prediction == 1:
                st.success(f'✅ **SURVIVED** - Probability: {probability[1]:.2%}')
            else:
                st.error(f'❌ **DID NOT SURVIVE** - Probability: {probability[0]:.2%}')
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "Survival Probability"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 50}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header('Model Performance Metrics')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric('Model Accuracy', f'{accuracy:.2%}')
            
            # Confusion Matrix
            st.subheader('Confusion Matrix')
            fig_cm = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Did Not Survive', 'Survived'],
                              y=['Did Not Survive', 'Survived'],
                              text_auto=True,
                              color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Feature importance
            st.subheader('Feature Importance')
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_imp = px.bar(feature_importance, x='importance', y='feature',
                            orientation='h',
                            title='Feature Importance Rankings')
            st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab3:
        st.header('Dataset Analysis')
        
        df = load_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival rate by class
            survival_by_class = df.groupby('Pclass')['Survived'].mean()
            fig1 = px.bar(x=survival_by_class.index, y=survival_by_class.values,
                         labels={'x': 'Passenger Class', 'y': 'Survival Rate'},
                         title='Survival Rate by Passenger Class')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Survival rate by sex
            survival_by_sex = df.groupby('Sex')['Survived'].mean()
            fig2 = px.bar(x=survival_by_sex.index, y=survival_by_sex.values,
                         labels={'x': 'Sex', 'y': 'Survival Rate'},
                         title='Survival Rate by Sex')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Age distribution
        st.subheader('Age Distribution by Survival')
        fig3 = px.histogram(df, x='Age', color='Survived', 
                           barmode='overlay',
                           labels={'Survived': 'Survived'},
                           nbins=30)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Show sample data
        st.subheader('Sample Dataset')
        st.dataframe(df.head(10), use_container_width=True)

if __name__ == '__main__':
    main()
