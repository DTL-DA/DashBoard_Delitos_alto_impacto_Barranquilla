import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="Dashboard Analitico", page_icon="📊", layout="wide")

def colores_pastel():
    return {
        'azul_pastel': '#A7C7E7',
        'verde_pastel': '#B5EAD7',
        'rosa_pastel': '#F7C6D9',
        'amarillo_pastel': '#FFF2B2',
        'morado_pastel': '#D7BDE2',
        'fondo': '#F8F9FA',
        'texto': '#2C3E50'
    }

st.title("Dashboard Analitico con Metodologia QUEST")
st.markdown("Este dashboard sigue la metodologia QUEST para un analisis estructurado y profesional.")

# 🔹 Ruta fija de Colab
RUTA_ARCHIVO = "https://github.com/DTL-DA/ActClass4/blob/DTL-DA-patch-1/Comparativo_de_delitos_de_alto_impacto_en_la_ciudad_de_Barranquilla_20260221.csv"
@st.cache_data
def cargar_datos():
    try:
        datos = pd.read_csv(RUTA_ARCHIVO)
        return datos
    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")
        return pd.DataFrame()

datos = cargar_datos()

if not datos.empty:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estructura de Datos")
        st.dataframe(datos.head(), use_container_width=True)

    with col2:
        st.subheader("Informacion General")
        buffer = io.StringIO()
        datos.info(buf=buffer)
        st.text(buffer.getvalue())
        st.metric("Filas Totales", len(datos))
        st.metric("Columnas", len(datos.columns))

    numericas = datos.select_dtypes(include=['number']).columns.tolist()
    categoricas = datos.select_dtypes(include=['object']).columns.tolist()

    # Exploración
    st.subheader("Exploracion de Datos")

    if numericas:
        col_hist = st.selectbox("Seleccionar columna numerica (Histograma)", numericas)
        fig_hist = px.histogram(
            datos,
            x=col_hist,
            nbins=20,
            template='plotly_white',
            color_discrete_sequence=[colores_pastel()['azul_pastel']]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Resumen estadístico
    if numericas:
        st.subheader("Resumen Estadistico")
        st.dataframe(datos[numericas].describe(), use_container_width=True)

    st.markdown("Dashboard generado con colores pastel para una presentacion profesional.")

else:
    st.info("No se pudo cargar el dataset.")

   
                           
              
