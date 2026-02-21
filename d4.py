import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Dashboard Delitos Barranquilla", page_icon="📊", layout="wide")

def colores_pastel():
    return {
        'azul_pastel': '#A7C7E7',
        'verde_pastel': '#B5EAD7',
        'rosa_pastel': '#F7C6D9',
        'amarillo_pastel': '#FFF2B2',
        'morado_pastel': '#D7BDE2',
        'naranja_pastel': '#FFE4B5',
        'fondo': '#F8F9FA',
        'texto': '#2C3E50'
    }

st.title("Dashboard Analitico Delitos Alto Impacto Barranquilla")
st.markdown("Analisis comparativo utilizando Metodologia QUEST sobre el archivo proporcionado.")

ruta_archivo = '/content/Comparativo_de_delitos_de_alto_impacto_en_la_ciudad_de_Barranquilla_20260221.csv'

@st.cache_data
def cargar_datos(ruta):
    try:
        datos = pd.read_csv(ruta, encoding='utf-8', sep=',')
        st.success("Archivo cargado exitosamente.")
        return datos
    except Exception as e:
        st.error(f"Error al cargar: {str(e)}. Verifique la ruta y suba el archivo en Colab.")
        return pd.DataFrame()

datos = cargar_datos(ruta_archivo)

if not datos.empty:
    # Q: Question - Preguntas clave: Evolucion de delitos, tipos mas comunes, tendencias
    st.subheader("Q: Preguntas Clave del Analisis")
    st.write("Cuales son los delitos mas frecuentes. Como evolucionan en el tiempo. Diferencias por area.")

    # U: Understand - Estructura
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros Totales", f"{len(datos):,}")
    with col2:
        st.metric("Columnas", len(datos.columns))
    with col3:
        st.metric("Valores Unicos Totales", datos.nunique().sum())

    st.subheader("U: Entender Estructura")
    st.dataframe(datos.head(10), use_container_width=True)
    
    buffer = io.StringIO()
    datos.info(buf=buffer)
    st.text(buffer.getvalue())

    # E: Explore - Exploracion
    st.subheader("E: Exploracion Inicial")
    col1, col2 = st.columns(2)
    with col1:
        if 'Tipo_Delito' in datos.columns or datos.select_dtypes(include='object').columns.any():
            cols_obj = datos.select_dtypes(include='object').columns
            col_obj = cols_obj[0] if len(cols_obj)>0 else None
            if col_obj:
                fig_count = px.bar(datos[col_obj].value_counts().head(10).reset_index(),
                                   x=col_obj, y=col_obj, orientation='h',
                                   color_discrete_sequence=[colores_pastel()['rosa_pastel']],
                                   template='plotly_white')
                st.plotly_chart(fig_count, use_container_width=True)
    with col2:
        numericas = datos.select_dtypes(include=['number']).columns
        if len(numericas)>0:
            fig_corr = px.imshow(datos[numericas].corr(), color_continuous_scale='RdBu_r',
                                 template='plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)

    # S: Summarize - Resumen
    st.subheader("S: Resumen Estadistico")
    st.dataframe(datos.describe(), use_container_width=True)

    # T: Transform - Visualizaciones clave para delitos
    st.subheader("T: Visualizaciones Transformadas")
    colores = list(colores_pastel().values())[1:6]

    if 'Fecha' in datos.columns or 'Mes' in datos.columns or datos.select_dtypes(include='datetime').columns.any():
        fecha_col = next((col for col in datos.columns if 'fecha' in col.lower() or 'mes' in col.lower()), None)
        if fecha_col:
            fig_line = px.line(datos.groupby(fecha_col).size().reset_index(),
                               x=fecha_col, y=0, title="Evolucion Temporal Delitos",
                               color_discrete_sequence=[colores_pastel()['azul_pastel']],
                               template='plotly_white')
            st.plotly_chart(fig_line, use_container_width=True)

    # Grafico de pastel o dona para tipos de delito
    tipo_delito = next((col for col in datos.columns if 'delito' in col.lower() or 'tipo' in col.lower()), None)
    if tipo_delito:
        top_tipos = datos[tipo_delito].value_counts().head(8)
        fig_pie = px.pie(values=top_tipos.values, names=top_tipos.index,
                         color_discrete_sequence=colores,
                         template='plotly_white')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Mapa si hay coordenadas, sino barras por categoria
    cols_num = datos.select_dtypes(include=['number']).columns.tolist()
    if len(cols_num) >= 2:
        fig_scatter_delitos = px.scatter(datos, x=cols_num[0], y=cols_num[1],
                                         color=tipo_delito if tipo_delito else None,
                                         color_discrete_sequence=colores,
                                         template='plotly_white', title="Relacion Variables Numericas")
        st.plotly_chart(fig_scatter_delitos, use_container_width=True)

    st.markdown("Este dashboard se adapta a las columnas del archivo. Colores pastel para visualizacion profesional.")
else:
    st.info("Suba el archivo CSV a /content/ en Google Colab o ajuste la ruta_archivo al nombre exacto.")



   
  
   

   
