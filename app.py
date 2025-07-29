import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle 
import pandas as pd
import numpy as np

#Load the trained Model

model = tf.keras.models.load_model('model.h5')

#Load the encoders

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_Encoder_Geo.pkl','rb') as file:
    onehot_Encoder_Geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    Scaler = pickle.load(file)


##Streamlit app
st.title("Cusomer Churn Prediciton By Haseeb")

Geography = st.selectbox('Geography', onehot_Encoder_Geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.selectbox("Age", list(range(18, 93)))
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Car',[0,1])
Is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([Gender])[0]],
    'Age' : [age],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [Is_active_member],
    'EstimatedSalary' : [estimated_salary]


})

geo_encoded = onehot_Encoder_Geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_Encoder_Geo.get_feature_names_out(['Geography']))


#Combine one hot encoded columns with  input data
if 'Tenure' not in input_data.columns:
    input_data['Tenure'] = 0  # or any default value


input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Align columns with the scaler (to avoid feature name mismatch error)
input_data = input_data.loc[:, Scaler.feature_names_in_]

# Scale the input data
input_data_scaled = Scaler.transform(input_data)
## Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn probaility: {prediction_proba : .2f}")
if prediction_proba > 0.5:
    st.write('The customer is likely to leave the bank')
else:
    st.write('The customer is not likely to leave the bank')
