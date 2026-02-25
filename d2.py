import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Serie Temporal Delitos Barranquilla", layout="wide")

st.title("Serie Temporal y Pronóstico de Delitos de Alto Impacto")

# =========================
# CARGA
# =========================

df = pd.read_csv("Delitos_de_alto_impacto_en_Barranquilla.csv")
df.columns = df.columns.str.strip()

# =========================
# LIMPIEZA NUMÉRICA
# =========================

df["Casos/denuncias último periodo"] = (
    df["Casos/denuncias último periodo"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["Casos/denuncias último periodo"] = pd.to_numeric(
    df["Casos/denuncias último periodo"],
    errors="coerce"
)

df = df.dropna(subset=["Casos/denuncias último periodo"])

# =========================
# EXTRAER AÑO FINAL DEL RANGO
# =========================

df["Año"] = df["Años comparados"].str.split("-").str[1]

# =========================
# CREAR FECHA
# =========================

df["Fecha"] = pd.to_datetime(
    df["Periodo meses comparado"] + " " + df["Año"],
    errors="coerce"
)

df = df.dropna(subset=["Fecha"])

# =========================
# FILTRAR SOLO ALTO IMPACTO
# =========================

df = df[df["Delito"].str.contains("alto", case=False, na=False)]

# =========================
# CREAR SERIE TEMPORAL
# =========================

df = df.sort_values("Fecha")
df = df.set_index("Fecha")

serie = df["Casos/denuncias último periodo"]

# Agrupar por mes por seguridad
serie = serie.resample("M").sum()

st.subheader("Resumen de la serie")
st.write("Fecha inicial:", serie.index.min())
st.write("Fecha final:", serie.index.max())
st.write("Cantidad de meses:", len(serie))

if len(serie) < 24:
    st.warning("La serie tiene pocos datos. El pronóstico puede ser inestable.")

# =========================
# MODELO HOLT (TENDENCIA)
# =========================

modelo = ExponentialSmoothing(
    serie,
    trend="add",
    seasonal=None
).fit()

# =========================
# PRONÓSTICO HASTA DICIEMBRE 2025
# =========================

ultima_fecha = serie.index[-1]
meses_proyeccion = (2025 - ultima_fecha.year) * 12 + (12 - ultima_fecha.month)

pronostico = modelo.forecast(meses_proyeccion)

# =========================
# GRÁFICO
# =========================

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(serie.index, serie.values, label="Histórico")
ax.plot(pronostico.index, pronostico.values, linestyle="--", label="Pronóstico")

ax.set_xlabel("Fecha")
ax.set_ylabel("Casos")
ax.legend()

st.pyplot(fig)

# =========================
# TABLA PRONÓSTICO
# =========================

df_pronostico = pronostico.reset_index()
df_pronostico.columns = ["Fecha", "Pronóstico"]

st.subheader("Pronóstico hasta diciembre 2025")
st.dataframe(df_pronostico)
