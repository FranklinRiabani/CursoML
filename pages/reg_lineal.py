import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# configurar pagina
st.set_page_config(page_title='Curso de Machine Learning - Regresion Lineal', page_icon=':bar_chart:', layout='wide')

# titulo

st.title('Curso de Machine Learning - UPDS')

# subtitulo

st.markdown('## Regresion Lineal')


# agregar un selectbox de modelos de aprendizaje supervisado
st.sidebar.markdown('## Regresion Lineal')



@st._cache_data
def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('.xlsx'):
            df = pd.read_excel(archivo)
        else:
            raise ValueError("Formato de archivo no soportado. Solo se aceptan archivos CSV y XLSX.")
        return df
    else:
        return None


archivo=st.sidebar.file_uploader('Cargar archivo CSV',type=['csv','xlsx'])
if archivo:
    df=cargar_datos(archivo)
    if df is not None:
        st.session_state.df=df
        fig= plt.figure(figsize=(6,6))

        plt.ylabel("Ingreso ($)")
        plt.xlabel("Promedio de horas semanales trabajadas")
        plt.scatter(df["horas"], df["ingreso"], color="pink")
        st.pyplot(fig)
        
        st.subheader('Regresión Lineal')
        
        from sklearn import linear_model
        
        regresion = linear_model.LinearRegression()

        horas = df["horas"].values.reshape((-1, 1))

        modelo = regresion.fit(horas, df["ingreso"])

        print("Intersección (b)", modelo.intercept_)
        print("Pendiente (m)", modelo.coef_)

        entrada = [[39.5], [40], [43], [43.5]]
        modelo.predict(entrada)

        plt.scatter(entrada, modelo.predict(entrada), color="red")
        plt.plot(entrada, modelo.predict(entrada), color="black")

        plt.ylabel("Ingreso ($)")
        plt.xlabel("Promedio de horas semanales trabajadas")
        plt.scatter(df["horas"], df["ingreso"], color="pink", alpha=0.55)
        st.pyplot(fig)
    
