import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Serie Temporal Delitos Barranquilla", layout="wide")

st.title("Serie Temporal y Pronóstico de Delitos")

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
# EXTRAER AÑO FINAL
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
# CREAR SERIE AGREGADA
# =========================

df = df.sort_values("Fecha")

serie = (
    df.groupby("Fecha")["Casos/denuncias último periodo"]
    .sum()
    .sort_index()
)

# Forzar frecuencia mensual
serie = serie.asfreq("M")

st.subheader("Resumen de la serie")
st.write("Fecha inicial:", serie.index.min())
st.write("Fecha final:", serie.index.max())
st.write("Cantidad de meses:", len(serie))

# Verificación crítica
if serie.empty:
    st.error("La serie quedó vacía. Revisar estructura del archivo.")
    st.stop()

if len(serie) < 12:
    st.warning("Hay pocos datos. El modelo puede ser inestable.")

# =========================
# MODELO
# =========================

modelo = ExponentialSmoothing(
    serie,
    trend="add",
    seasonal=None
).fit()

# =========================
# PRONÓSTICO
# =========================

ultima_fecha = serie.index[-1]
meses_proyeccion = (2025 - ultima_fecha.year) * 12 + (12 - ultima_fecha.month)

if meses_proyeccion > 0:
    pronostico = modelo.forecast(meses_proyeccion)
else:
    pronostico = pd.Series()

# =========================
# GRÁFICO
# =========================

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(serie.index, serie.values, label="Histórico")

if not pronostico.empty:
    ax.plot(pronostico.index, pronostico.values, linestyle="--", label="Pronóstico")

ax.legend()
ax.set_xlabel("Fecha")
ax.set_ylabel("Casos")

st.pyplot(fig)

# =========================
# TABLA
# =========================

if not pronostico.empty:
    st.subheader("Pronóstico hasta diciembre 2025")
    st.dataframe(pronostico.reset_index().rename(columns={0: "Pronóstico"}))
