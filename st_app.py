import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import json

#Variales de Entorno
API_URL = 'https://66zl2vyt7uvasnujluqwmdfepy0pciao.lambda-url.us-east-2.on.aws/'
PREDICT_ENDPOINT = '/online_predict'
BATCH_PREDICT_ENDPOINT = '/batch_predict'

def predict_input(input_data):
    response = requests.post(API_URL + PREDICT_ENDPOINT, json = input_data)
    if response.status_code == 200:
        return response.json()['prediction']
    else:
        return None

def run():
    #Instanciar las imagenes
    from PIL import Image
    image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    # Iniciar el codigo de la app
    st.image(image, use_column_width= False)
    
    add_selectbox =  st.sidebar.selectbox(
        "Indica un metodo para realizar predicciones",
        ("Online", "Batch"))
    
    st.sidebar.info("Esta app permite predecir costos de hospitalizaicón para Pacientes")
    st.sidebar.success("")
    st.sidebar.image(image_hospital)

    if add_selectbox == "Online":
        age = st.number_input('Age', min_value=1, max_value=100, value= 25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value= 50, value=10)
        children = st.selectbox('Children', [0, 1, 2, 3, 4, 5, 6, 7, 8])
        smoker = st.selectbox('Smoker', ['yes', 'no'])
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        input_data = {'age': age, 'sex': sex, 'bmi':bmi, 'children':children, 'smoker':smoker, 'region': region}

        if st.button("Estimar"):
            output = predict_input(input_data)
            if output is None:
                st.error('Error al obtener una estimacion de la API')
            else: 
                output = '$'+str(output)
                st.success('El valor estimado es de {}'.format(output))
        
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Cargue un archivo csv para realizar estimaciones", type=['csv'])

        if file_upload is not None:
            if st.button('Estimar'):

             response = requests.post(API_URL + BATCH_PREDICT_ENDPOINT, files={'file': file_upload})
             if response.status_code == 200:
                 data = response.json()
                 df = pd.DataFrame(data)
                 st.write(df)
             else:
                 st.write("Error al realizar las estimaciones")

if __name__ == '__main__':
    run()