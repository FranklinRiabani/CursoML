import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# configurar pagina
st.set_page_config(page_title='Exploratory Data Analysis', page_icon=':bar_chart:', layout='wide')

# titulo

st.title('Curso de Machine Learning - UPDS')

# subtitulo

st.subheader('Análisis Exploratorio de Datos')

st.markdown('## Normacilización de Datos')

# agregar sidebar
st.sidebar.title('Descripcion')

# agregar una lista de modelos de regresion a la sidebar
st.sidebar.markdown('## Modelos de Regresion')
st.sidebar.markdown('- Regresion Lineal')
st.sidebar.markdown('- Regresion Logistica')
st.sidebar.markdown('- Regresion Polinomial')

# agregar un selectbox de modelos de aprendizaje supervisado
st.sidebar.markdown('## Modelos de Aprendizaje Supervisado')
model=st.sidebar.selectbox('seleccione un modelo',['Regresion Lineal','Regresion Logistica','Regresion Polinomial'])

# crear una lista de paises de sudamerica
paises=['Bolivia','Argentina','Brasil','Chile','Paraguay','Peru','Uruguay']

# agregar un select multiple de paises
st.sidebar.markdown('## Paises')
paises_seleccionados=st.sidebar.multiselect('Seleccione los paises',paises)

# agregar un slider de años desde 2000 hasta el 2024

st.sidebar.markdown('## Años')

año=st.sidebar.slider('Seleccione un año',2000,2024,2020)


# cargar el dataset con file_uploader
st.markdown('## Cargar Dataset')
file=st.file_uploader('Cargar archivo CSV',type=['csv'])

if file is not None:
    data=pd.read_csv(file)

    # agregar un subtitulo de analisis exploratorio de datos

    st.subheader('Analisis Exploratorio de Datos')

    # mostrar los datos
    st.markdown('#### Tabla de Datos')
    st.write(data)

    # mostrar la descripcion de los datos
    st.markdown('#### Descripcion de los Datos')
    st.write(data.describe())

    # mostrar los tipos de datos
    st.markdown('#### Tipos de Datos')
    st.write(data.dtypes)

    # mostrar la cantidad de datos nulos
    st.markdown('#### Datos Nulos')

    st.write(data.isnull().sum())

    # eliminar los datos nulos
    data.dropna(inplace=True)

    # correlacion de los datos
    st.markdown('#### Correlacion de los Datos')
    st.write(data.corr())

    # grafico de correlacion
    st.markdown('#### Grafico de Correlacion')
    fig=plt.figure(figsize=(10,5))
    sns.heatmap(data.corr(),annot=True)
    st.pyplot(fig)

