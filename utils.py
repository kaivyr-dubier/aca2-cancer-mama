"""Funciones auxiliares de la app ACA 2: modelo ONNX, DuckDB, historial SQLite y nombres."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
RUTA_ONNX = BASE / "models" / "ft_transformer_cancer_mama.onnx"
RUTA_META = BASE / "models" / "metadata.json"
RUTA_CSV = BASE / "data" / "breast_cancer_wisconsin.csv"
RUTA_EJEMPLO = BASE / "data" / "ejemplo_pacientes.csv"
RUTA_HISTORIAL = BASE / "data" / "historial_predicciones.db"

C_BEN, C_MAL, C_OSC = "#2a9d8f", "#e76f51", "#264653"

RASGOS = {
    "radius": ("Radio", "Distancia media del centro al borde del núcleo"),
    "texture": ("Textura", "Desviación estándar de la escala de grises"),
    "perimeter": ("Perímetro", "Longitud del contorno del núcleo"),
    "area": ("Área", "Superficie del núcleo (píxeles²)"),
    "smoothness": ("Suavidad", "Variación local en la longitud del radio"),
    "compactness": ("Compacidad", "Perímetro² / área − 1"),
    "concavity": ("Concavidad", "Severidad de las partes cóncavas del contorno"),
    "concave points": ("Puntos cóncavos", "Número de partes cóncavas del contorno"),
    "symmetry": ("Simetría", "Simetría del núcleo"),
    "fractal dimension": ("Dimensión fractal", "Complejidad del borde"),
}
GRUPOS = {"mean": "Media", "error": "Error estándar", "worst": "Peor valor"}


def grupo_de(f: str) -> str:
    return "mean" if f.startswith("mean ") else "worst" if f.startswith("worst ") else "error"


def rasgo_de(f: str) -> str:
    return f.replace("mean ", "").replace("worst ", "").replace(" error", "")


def nombre_es(f: str) -> str:
    return f"{RASGOS[rasgo_de(f)][0]} ({GRUPOS[grupo_de(f)].lower()})"


# ---------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando la red neuronal…")
def cargar_sesion() -> ort.InferenceSession:
    return ort.InferenceSession(str(RUTA_ONNX), providers=["CPUExecutionProvider"])


@st.cache_data
def cargar_meta() -> dict:
    return json.loads(RUTA_META.read_text(encoding="utf-8"))


def predecir(sesion, X: pd.DataFrame, features: list[str], umbral: float = 0.5):
    """Devuelve probabilidades, etiquetas y la atención [CLS]→variable de cada paciente."""
    entrada = X[features].to_numpy(dtype=np.float32)
    proba, atencion = sesion.run(None, {"variables": entrada})
    etiqueta = np.where(proba >= umbral, "Maligno", "Benigno")
    return proba, etiqueta, atencion


# ---------------------------------------------------------------- datos / DuckDB
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(RUTA_CSV)
    df["Diagnóstico"] = df["diagnostico"].map({0: "Benigno", 1: "Maligno"})
    return df


@st.cache_resource
def conexion_duckdb() -> duckdb.DuckDBPyConnection:
    """Base DuckDB en memoria con la tabla 'pacientes' (columnas sin espacios)."""
    df = cargar_datos().drop(columns="Diagnóstico").copy()
    df.insert(0, "paciente_id", range(1, len(df) + 1))
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["diagnostico_texto"] = np.where(df["diagnostico"] == 1, "maligno", "benigno")
    con = duckdb.connect(":memory:")
    con.register("df_tmp", df)
    con.execute("CREATE TABLE pacientes AS SELECT * FROM df_tmp")
    con.unregister("df_tmp")
    return con


def consulta_segura(con, sql: str, limite: int = 500) -> pd.DataFrame:
    """Solo permite consultas de lectura (SELECT / WITH) y limita las filas."""
    limpio = sql.strip().rstrip(";")
    if not limpio.lower().startswith(("select", "with")) or ";" in limpio:
        raise ValueError("Solo se permiten consultas de lectura (SELECT o WITH), una a la vez.")
    return con.execute(f"SELECT * FROM ({limpio}) AS q LIMIT {int(limite)}").fetchdf()


# ---------------------------------------------------------------- historial / SQLite
def _con_historial() -> sqlite3.Connection:
    con = sqlite3.connect(RUTA_HISTORIAL, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS predicciones (
                     id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, origen TEXT,
                     probabilidad REAL, umbral REAL, prediccion TEXT, variable_mas_atendida TEXT)""")
    return con


def guardar_prediccion(origen: str, proba: float, umbral: float, pred: str, top_var: str) -> None:
    with _con_historial() as con:
        con.execute("INSERT INTO predicciones (fecha, origen, probabilidad, umbral, prediccion, variable_mas_atendida) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), origen, round(float(proba), 4),
                     float(umbral), pred, top_var))


def leer_historial() -> pd.DataFrame:
    with _con_historial() as con:
        return pd.read_sql_query("SELECT * FROM predicciones ORDER BY id DESC", con)


def borrar_historial() -> None:
    with _con_historial() as con:
        con.execute("DELETE FROM predicciones")
