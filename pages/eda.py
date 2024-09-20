import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# configurar pagina
st.set_page_config(page_title='Curso de Machine Learning - EDA', page_icon=':bar_chart:', layout='wide')

# titulo

st.title('Curso de Machine Learning - UPDS')

# subtitulo

st.markdown('## Normacilización de Datos')


# agregar un selectbox de modelos de aprendizaje supervisado
st.sidebar.markdown('## Modelos de Aprendizaje Supervisado')

opcion=st.sidebar.selectbox('seleccione un modelo',['Analisis Exploratorio','Correlacion','Normalizacion'])

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



if opcion=='Analisis Exploratorio':
    st.markdown('## Analisis Exploratorio de Datos')
    archivo=st.sidebar.file_uploader('Cargar archivo CSV',type=['csv','xlsx'])
    if archivo:
        df=cargar_datos(archivo)
        if df is not None:
            st.session_state.df=df
            st.write(df)
            st.markdown('## Estadisticas Descriptivas')
            st.write(df.describe())
            st.markdown('## Informacion del Dataset')
            st.write(df.info())
            st.markdown('## Valores Nulos')
            st.write(df.isnull().sum())
            st.markdown('## Graficos')
            st.write(sns.pairplot(df))
            fig=plt.figure()
            sns.pairplot(df)
            st.pyplot(fig)
elif opcion=='Correlacion':
    if 'df' not in st.session_state:
        st.warning("Por favor, carga un archivo primero en la sección 'Cargar Datos'.")
    else:
        df = st.session_state.df
        st.markdown('## Correlacion')
        st.subheader("Primeras filas del dataset")
        st.markdown('## Matriz de Correlacion')
        st.write(df.corr())
        fig=plt.figure()
        sns.heatmap(df.corr(),annot=True)
        st.pyplot(fig)
elif opcion=='Normalizacion':
    if 'df' not in st.session_state:
        st.warning("Por favor, carga un archivo primero en la sección 'Cargar Datos'.")
    else:
        df = st.session_state.df
        st.markdown('## Normalizacion')
        st.subheader("Primeras filas del dataset")
        st.write(df.head())
        st.markdown('## Normalizacion de Datos')
        st.write(df)
        st.markdown('## Normalizacion Min-Max')
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        df_minmax = min_max_scaler.fit_transform(df)
        df_minmax = pd.DataFrame(df_minmax, columns=df.columns)
        st.write(df_minmax)
        st.markdown('## Normalizacion Z-Score')
        from sklearn.preprocessing import StandardScaler
        z_score_scaler = StandardScaler()
        df_zscore = z_score_scaler.fit_transform(df)
        df_zscore = pd.DataFrame(df_zscore, columns=df.columns)
        st.write(df_zscore)
        st.markdown('## Normalizacion standarScaler')
        from sklearn.preprocessing import StandardScaler
        standar_scaler = StandardScaler()
        df_standar = standar_scaler.fit_transform(df)
        df_standar = pd.DataFrame(df_standar, columns=df.columns)
        st.write(df_standar)

        fig=plt.figure(figsize=(10,5))
        ax1=fig.add_subplot(1,2,1)
        ax2=fig.add_subplot(1,2,2)
        
        ax1.set_title('Antes de la normalización', fontsize=20)
        ax1.scatter(df["ingreso"], df['horas'],
                    marker='8',s=500,c='purple',alpha=0.5)
        ax1.set_xlabel('Ingreso')
        ax1.set_ylabel('Horas Trabajadas')

        ax2.set_title('Después de la normalización', fontsize=20)
        ax2.scatter(df_standar["ingreso"], df_standar['horas'],
                    marker='8',s=500,c='red',alpha=0.5)
        ax2.set_xlabel('Ingreso')
        ax2.set_ylabel('Horas Trabajadas')
        st.pyplot(fig)





