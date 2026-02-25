# app.py
# Tablero analítico (Streamlit) para "Delitos_de_alto_impacto_en_Barranquilla.csv"
#
# Ejecuta:
#   pip install -r requirements.txt
#   streamlit run app.py

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing


def to_number(x):
    if pd.isna(x):
        return np.nan

    v = str(x).strip()  # evita NameError
    if v == "" or v.lower() in ("na", "n/a", "none", "nan"):
        return np.nan

    v = re.sub(r"[^\d\-,\.]", "", v)

    if "," in v and "." in v:
        if v.rfind(",") > v.rfind("."):
            v = v.replace(".", "")
            v = v.replace(",", ".")
        else:
            v = v.replace(",", "")
    elif "," in v and "." not in v:
        v = v.replace(",", ".")

    try:
        return float(v)
    except ValueError:
        return np.nan


def to_percent(x):
    if pd.isna(x):
        return np.nan
    v = str(x).strip()
    if v == "" or v.lower() in ("na", "n/a", "none", "nan"):
        return np.nan
    v = v.replace("%", "").strip()
    v = re.sub(r"[^\d\-,\.]", "", v).replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return np.nan


MESES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def extraer_mes(texto: str):
    if pd.isna(texto):
        return None
    t = str(texto).strip().lower()
    for mes, n in MESES.items():
        if mes in t:
            return n
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c: re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns}

    def find_col(all_terms):
        for orig, low in cols_lower.items():
            if all(term in low for term in all_terms):
                return orig
        return None

    c_periodo = find_col(["periodo"]) or find_col(["período"])
    c_years = find_col(["años"]) or find_col(["anos"]) or find_col(["comparados"])
    c_delito = find_col(["delito"])
    c_prev = find_col(["anterior", "periodo"]) or find_col(["anterior", "período"])
    c_last = find_col(["último", "periodo"]) or find_col(["ultimo", "periodo"])
    c_var_pct = find_col(["variación", "%"]) or find_col(["variacion", "%"])
    c_var_abs = find_col(["variación", "absoluta"]) or find_col(["variacion", "absoluta"])
    c_fuente = find_col(["fuente"])

    rename = {}
    if c_periodo:
        rename[c_periodo] = "periodo"
    if c_years:
        rename[c_years] = "anios_comparados"
    if c_delito:
        rename[c_delito] = "delito"
    if c_prev:
        rename[c_prev] = "casos_prev"
    if c_last:
        rename[c_last] = "casos_last"
    if c_var_pct:
        rename[c_var_pct] = "var_pct"
    if c_var_abs:
        rename[c_var_abs] = "var_abs"
    if c_fuente:
        rename[c_fuente] = "fuente"

    df = df.rename(columns=rename)

    if "casos_prev" in df.columns:
        df["casos_prev"] = df["casos_prev"].apply(to_number)
    if "casos_last" in df.columns:
        df["casos_last"] = df["casos_last"].apply(to_number)
    if "var_abs" in df.columns:
        df["var_abs"] = df["var_abs"].apply(to_number)
    if "var_pct" in df.columns:
        df["var_pct"] = df["var_pct"].apply(to_percent)

    if "casos_prev" in df.columns and "casos_last" in df.columns:
        if "var_abs" not in df.columns:
            df["var_abs"] = df["casos_last"] - df["casos_prev"]
        else:
            m = df["var_abs"].isna()
            df.loc[m, "var_abs"] = df.loc[m, "casos_last"] - df.loc[m, "casos_prev"]

        if "var_pct" not in df.columns:
            df["var_pct"] = np.where(
                df["casos_prev"].fillna(0) == 0,
                np.nan,
                (df["casos_last"] - df["casos_prev"]) / df["casos_prev"] * 100,
            )
        else:
            m = df["var_pct"].isna()
            df.loc[m, "var_pct"] = np.where(
                df.loc[m, "casos_prev"].fillna(0) == 0,
                np.nan,
                (df.loc[m, "casos_last"] - df.loc[m, "casos_prev"]) / df.loc[m, "casos_prev"] * 100,
            )

    if "periodo" in df.columns:
        df["mes"] = df["periodo"].apply(extraer_mes)

    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return standardize_columns(df)


def fmt_int(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.1f}%"


def construir_serie_mensual(df_std: pd.DataFrame) -> pd.DataFrame:
    """
    El CSV suele venir como acumulados (enero a fin de mes) para dos años comparados.
    Se reconstruye una serie mensual por diferenciación del acumulado, para ambos años.
    """
    req = {"anios_comparados", "delito", "mes", "casos_prev", "casos_last"}
    if not req.issubset(set(df_std.columns)):
        return pd.DataFrame()

    rows = []
    base = df_std.dropna(subset=["anios_comparados", "delito", "mes"]).copy()

    for _, r in base.iterrows():
        anios = str(r["anios_comparados"])
        parts = re.split(r"\s*-\s*", anios)
        if len(parts) != 2:
            continue
        try:
            y_prev = int(parts[0])
            y_last = int(parts[1])
        except ValueError:
            continue

        m = int(r["mes"])
        rows.append({"anio": y_prev, "mes": m, "delito": r["delito"], "acumulado": r["casos_prev"]})
        rows.append({"anio": y_last, "mes": m, "delito": r["delito"], "acumulado": r["casos_last"]})

    long = pd.DataFrame(rows).dropna(subset=["acumulado"])
    if long.empty:
        return long

    long = long.sort_values(["delito", "anio", "mes"])
    long["mensual"] = long.groupby(["delito", "anio"])["acumulado"].diff()
    long["mensual"] = long["mensual"].fillna(long["acumulado"])
    long["mensual"] = long["mensual"].clip(lower=0)

    long["fecha"] = pd.to_datetime(dict(year=long["anio"].astype(int), month=long["mes"].astype(int), day=1))
    return long[["fecha", "anio", "mes", "delito", "mensual"]]


def preparar_total_mensual(serie_mensual: pd.DataFrame) -> pd.Series:
    """
    Agrega todos los delitos a un total mensual y devuelve una serie con índice mensual (MS).
    """
    if serie_mensual.empty:
        return pd.Series(dtype=float)

    total = (
        serie_mensual.groupby("fecha", as_index=True)["mensual"]
        .sum()
        .sort_index()
    )
    total = total.asfreq("MS")
    total = total.fillna(0.0)
    return total


def fit_exponential_smoothing(y: pd.Series):
    """
    Ajusta un modelo de suavizado exponencial usando statsmodels.
    Selección automática según longitud:
      - >= 24 puntos: Holt-Winters (tendencia + estacionalidad 12)
      - 12 a 23 puntos: Holt (tendencia)
      - < 12 puntos: Simple Exponential Smoothing
    """
    y = y.astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(y)
    if n < 3:
        return None, "No hay datos suficientes para ajustar un modelo."

    if n >= 24:
        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        return fit, "Holt-Winters aditivo (tendencia y estacionalidad, 12 meses)."

    if n >= 12:
        model = Holt(y, initialization_method="estimated")
        fit = model.fit(optimized=True)
        return fit, "Holt (tendencia)."

    model = SimpleExpSmoothing(y, initialization_method="estimated")
    fit = model.fit(optimized=True)
    return fit, "Suavizado exponencial simple."


def forecast_to_dec_2025(y: pd.Series, fit, start_date="2023-01-01", end_date="2025-12-01"):
    """
    Genera pronóstico hasta diciembre de 2025.
    Devuelve un DataFrame con serie real (desde 2023-01) y pronóstico.
    """
    y = y.asfreq("MS")
    y = y.fillna(0.0)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    y_train = y[y.index <= min(y.index.max(), end)]
    y_train = y_train[y_train.index >= start]

    # Cantidad de pasos a pronosticar desde el último dato disponible
    last_obs = y_train.index.max()
    if pd.isna(last_obs):
        return pd.DataFrame()

    future_index = pd.date_range(start=last_obs + pd.offsets.MonthBegin(1), end=end, freq="MS")
    steps = len(future_index)

    if steps <= 0:
        # Ya hay datos hasta (o más allá de) dic-2025
        out = pd.DataFrame({"fecha": y_train.index, "real": y_train.values})
        out["pronostico"] = np.nan
        return out.reset_index(drop=True)

    fc = fit.forecast(steps)
    fc = pd.Series(fc.values, index=future_index).clip(lower=0)

    out = pd.DataFrame({"fecha": y_train.index, "real": y_train.values})
    out_fc = pd.DataFrame({"fecha": fc.index, "real": np.nan, "pronostico": fc.values})
    out["pronostico"] = np.nan

    out = pd.concat([out, out_fc], ignore_index=True)
    return out


st.set_page_config(page_title="Tablero analítico de delitos", layout="wide")
st.title("Tablero analítico de delitos de alto impacto en Barranquilla")

DEFAULT_FILE = "Delitos_de_alto_impacto_en_Barranquilla.csv"

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("Sube el CSV (opcional)", type=["csv"])

    if uploaded is not None:
        df = standardize_columns(pd.read_csv(uploaded))
    else:
        if not os.path.exists(DEFAULT_FILE):
            st.warning(
                f"No se encontró el archivo '{DEFAULT_FILE}' en el repositorio. "
                "Súbelo desde aquí o agrégalo a la raíz del proyecto."
            )
            st.stop()
        df = load_csv(DEFAULT_FILE)

    st.divider()
    st.header("Filtros")

    anios = sorted(df["anios_comparados"].dropna().unique().tolist()) if "anios_comparados" in df.columns else []
    periodos = sorted(df["periodo"].dropna().unique().tolist()) if "periodo" in df.columns else []
    delitos = sorted(df["delito"].dropna().unique().tolist()) if "delito" in df.columns else []

    sel_anios = st.multiselect("Años comparados", anios, default=anios[:1] if anios else [])
    sel_periodos = st.multiselect("Período", periodos, default=periodos[:1] if periodos else [])
    sel_delitos = st.multiselect("Delito", delitos, default=[])

df_f = df.copy()
if sel_anios and "anios_comparados" in df_f.columns:
    df_f = df_f[df_f["anios_comparados"].isin(sel_anios)]
if sel_periodos and "periodo" in df_f.columns:
    df_f = df_f[df_f["periodo"].isin(sel_periodos)]
if sel_delitos and "delito" in df_f.columns:
    df_f = df_f[df_f["delito"].isin(sel_delitos)]

needed = {"delito", "casos_prev", "casos_last", "var_abs", "var_pct"}
missing = [c for c in needed if c not in df_f.columns]
if missing:
    st.error(
        "Faltan columnas necesarias para el tablero: " + ", ".join(missing) + ". "
        "Revisa el CSV o ajusta el mapeo en standardize_columns()."
    )
    st.stop()

total_prev = df_f["casos_prev"].sum(skipna=True)
total_last = df_f["casos_last"].sum(skipna=True)
total_abs = total_last - total_prev
total_pct = np.nan if (pd.isna(total_prev) or total_prev == 0) else (total_abs / total_prev) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total de casos (período anterior)", fmt_int(total_prev))
k2.metric("Total de casos (último período)", fmt_int(total_last))
k3.metric("Variación absoluta (total)", fmt_int(total_abs))
k4.metric("Variación porcentual (total)", fmt_pct(total_pct))

st.divider()

agg = (
    df_f.groupby("delito", as_index=False)
    .agg(
        casos_prev=("casos_prev", "sum"),
        casos_last=("casos_last", "sum"),
        var_abs=("var_abs", "sum"),
    )
)
agg["var_pct"] = np.where(
    agg["casos_prev"].fillna(0) == 0,
    np.nan,
    (agg["casos_last"] - agg["casos_prev"]) / agg["casos_prev"] * 100,
)
agg = agg.sort_values("casos_last", ascending=False)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Comparativo por delito: período anterior vs último período")

    long = agg.melt(
        id_vars=["delito"],
        value_vars=["casos_prev", "casos_last"],
        var_name="periodo_tipo",
        value_name="casos",
    )
    long["periodo_tipo"] = long["periodo_tipo"].map({"casos_prev": "Anterior", "casos_last": "Último"})

    fig_bar = px.bar(
        long,
        x="delito",
        y="casos",
        color="periodo_tipo",
        barmode="group",
        labels={"delito": "Delito", "casos": "Casos o denuncias", "periodo_tipo": "Período"},
    )
    fig_bar.update_layout(xaxis_tickangle=-25, height=520, legend_title_text="Período")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Variación absoluta por delito")

    fig_abs = px.bar(
        agg.sort_values("var_abs", ascending=True),
        x="var_abs",
        y="delito",
        orientation="h",
        labels={"var_abs": "Variación absoluta", "delito": "Delito"},
    )
    fig_abs.update_layout(height=520)
    st.plotly_chart(fig_abs, use_container_width=True)

st.divider()

st.subheader("Pronóstico con suavizado exponencial (statsmodels) hasta diciembre de 2025")

serie_m = construir_serie_mensual(df)
if serie_m.empty:
    st.info("No se pudo construir la serie mensual. Verifica que existan 'Años comparados' y 'Período' con meses.")
else:
    total_m = preparar_total_mensual(serie_m)

    # Se ajusta usando datos desde enero de 2023 en adelante (si existen)
    y_train = total_m[total_m.index >= "2023-01-01"].copy()
    if len(y_train) < 3:
        st.info("No hay suficientes datos desde 2023 para ajustar el modelo.")
    else:
        fit, descripcion = fit_exponential_smoothing(y_train)
        if fit is None:
            st.info(descripcion)
        else:
            # Pronóstico hasta 2025-12 (si faltan meses por observar)
            out = forecast_to_dec_2025(y_train, fit, start_date="2023-01-01", end_date="2025-12-01")

            fig_ts = px.line(
                out,
                x="fecha",
                y=["real", "pronostico"],
                labels={"value": "Total mensual", "fecha": "Fecha", "variable": "Serie"},
            )
            fig_ts.update_layout(height=520)
            st.plotly_chart(fig_ts, use_container_width=True)

            st.caption(f"Modelo utilizado: {descripcion}")

            with st.expander("Ver tabla de serie real y pronóstico"):
                st.dataframe(out, use_container_width=True, height=360)

st.divider()

st.subheader("Tabla resumen por delito")
view = agg[["delito", "casos_prev", "casos_last", "var_abs", "var_pct"]].copy()
view["casos_prev"] = view["casos_prev"].round(0).astype("Int64")
view["casos_last"] = view["casos_last"].round(0).astype("Int64")
view["var_abs"] = view["var_abs"].round(0).astype("Int64")
view["var_pct"] = view["var_pct"].round(2)
st.dataframe(view, use_container_width=True, height=420)

csv_export = view.to_csv(index=False).encode("utf-8")
with st.sidebar:
    st.divider()
    st.header("Exportación")
    st.download_button(
        "Descargar resumen (CSV)",
        data=csv_export,
        file_name="resumen_delitos_por_delito.csv",
        mime="text/csv",
    )
