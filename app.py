"""
App Streamlit – ACA 2: Diagnóstico de cáncer de mama con un Transformer tabular (FT-Transformer)
Modelo exportado a ONNX desde PyTorch · DuckDB para consultas SQL · SQLite para el historial.
Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (C_BEN, C_MAL, C_OSC, GRUPOS, RASGOS, RUTA_EJEMPLO, borrar_historial, cargar_datos,
                   cargar_meta, cargar_sesion, conexion_duckdb, consulta_segura, grupo_de, guardar_prediccion,
                   leer_historial, nombre_es, predecir, rasgo_de)

st.set_page_config(page_title="Transformer · Cáncer de mama", page_icon="🧠", layout="wide")

sesion, meta, df = cargar_sesion(), cargar_meta(), cargar_datos()
FEATURES, RANGOS = meta["features"], meta["rangos"]
PALETA = {"Benigno": C_BEN, "Maligno": C_MAL}
M = meta["metricas_test"]
HP = meta["hiperparametros"]


# ------------------------------------------------------------ estado de los controles
def fijar_valores(fuente: str):
    if fuente == "aleatorio":
        fila = df.sample(1).iloc[0]
        st.session_state["caso_real"] = fila["Diagnóstico"]
        valores = {c: float(fila[c]) for c in FEATURES}
    else:
        st.session_state["caso_real"] = None
        clave = {"general": "mediana", "benigno": "mediana_benigno", "maligno": "mediana_maligno"}[fuente]
        valores = {c: RANGOS[c][clave] for c in FEATURES}
    for c, v in valores.items():
        st.session_state[f"in_{c}"] = float(np.clip(v, RANGOS[c]["min"], RANGOS[c]["max"]))


if f"in_{FEATURES[0]}" not in st.session_state:
    fijar_valores("general")


def paso(c):
    return float(10 ** np.floor(np.log10((RANGOS[c]["max"] - RANGOS[c]["min"]) / 200)))


def formato(c):
    return f"%.{max(0, int(-np.log10(paso(c))))}f"


def barra_atencion(aten: np.ndarray, titulo: str, n: int = 10):
    media = np.array([meta["atencion"]["Global"][f] for f in FEATURES])
    orden = np.argsort(aten)[::-1][:n]
    d = pd.DataFrame({"Variable": [nombre_es(FEATURES[i]) for i in orden],
                      "Este paciente": aten[orden], "Promedio (prueba)": media[orden]})
    d = d.melt(id_vars="Variable", var_name="Serie", value_name="Atención")
    fig = px.bar(d, y="Variable", x="Atención", color="Serie", barmode="group", orientation="h",
                 color_discrete_map={"Este paciente": C_MAL, "Promedio (prueba)": "#9fb7c4"}, title=titulo)
    fig.update_layout(height=420, yaxis={"autorange": "reversed"}, legend_title="", margin=dict(t=50))
    return fig


# ------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown("## 🧠 Transformer tabular")
    st.caption("Cáncer de mama · Deep Learning · ACA 2")
    pagina = st.radio("Navegación", [
        "🏠 Inicio", "🧪 Predicción individual", "📂 Predicción por lotes", "🗄️ Datos y SQL",
        "🗂️ Historial (SQLite)", "🏗️ Arquitectura y entrenamiento", "📊 Desempeño", "ℹ️ Acerca de"],
        label_visibility="collapsed")
    st.divider()
    umbral = st.slider("Umbral de decisión (maligno si prob. ≥ umbral)", 0.05, 0.95, 0.50, 0.05,
                       help="Bajarlo detecta más malignos a cambio de más falsas alarmas.")
    st.divider()
    st.caption(f"Modelo: **FT-Transformer** ({meta['parametros']:,} parámetros) · ONNX  \n"
               f"F1 prueba: **{M['F1-score']:.3f}** · AUC: **{M['ROC-AUC']:.3f}**")
    st.caption("CUN · Especialización en IA")

# ============================================================ INICIO
if pagina == "🏠 Inicio":
    st.title("Diagnóstico asistido de cáncer de mama con un Transformer")
    st.markdown(
        "Esta aplicación usa una **red neuronal Transformer para datos tabulares (FT-Transformer)**. "
        "Cada una de las 30 mediciones del núcleo celular se convierte en un *token* y la red aplica "
        "**autoatención** para aprender cómo se relacionan entre sí. Además de predecir, muestra **en qué "
        "variables se fijó** para cada paciente.")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{M['Accuracy']:.1%}")
    c[1].metric("Precision", f"{M['Precision']:.1%}")
    c[2].metric("Recall", f"{M['Recall']:.1%}")
    c[3].metric("F1-score", f"{M['F1-score']:.3f}")
    c[4].metric("ROC-AUC", f"{M['ROC-AUC']:.3f}")
    svm = meta["comparacion"]["SVM (ACA 1)"]
    st.caption(f"Conjunto de prueba: {meta['n_test']} pacientes nunca vistos, la **misma partición de la ACA 1**. "
               f"SVM de la ACA 1: F1 = {svm['F1-score']:.3f} con {int(svm['FN'])} malignos sin detectar; "
               f"Transformer: F1 = {M['F1-score']:.3f} con {int(M['FN'])}.")

    a, b = st.columns(2)
    with a:
        st.subheader("🎯 Objetivo")
        st.write("Clasificar tumores como benignos o malignos con Deep Learning, integrando SQL, aumento de datos, "
                 "entrenamiento, evaluación y despliegue interactivo.")
        st.subheader("❓ Pregunta orientadora")
        st.write("¿Cómo aprovechar la autoatención de un Transformer para extraer patrones complejos de los datos "
                 "de biopsia, mejorar el rendimiento del modelo clásico y ofrecer predicciones interpretables?")
    with b:
        st.subheader("🛠 Ruta metodológica")
        for t, d in [("1. Datos y SQL", "DuckDB: almacenamiento, inserción y consultas del EDA"),
                     ("2. Pipeline", f"Escalado, tokenización por variable y aumento de datos "
                                     f"(+{meta['aumento']['sinteticos']} pacientes sintéticos con GMM y ruido gaussiano)"),
                     ("3. Modelado", "FT-Transformer vs. MLP vs. CNN 1D en PyTorch"),
                     ("4. Evaluación", "Loss, Accuracy, F1, matriz de confusión, curvas y ablación"),
                     ("5. Despliegue", "Exportación a ONNX y esta app en Streamlit")]:
            st.markdown(f"**{t}** — {d}")
    st.warning("⚠️ Herramienta académica de apoyo. No reemplaza el diagnóstico histopatológico ni el criterio médico.")

# ============================================================ PREDICCIÓN INDIVIDUAL
elif pagina == "🧪 Predicción individual":
    st.title("🧪 Predicción individual")
    st.write("Ajusta las mediciones o carga un caso de referencia. La red recalcula la probabilidad y la atención.")
    b = st.columns(4)
    b[0].button("Caso típico benigno", on_click=fijar_valores, args=("benigno",), width="stretch")
    b[1].button("Caso típico maligno", on_click=fijar_valores, args=("maligno",), width="stretch")
    b[2].button("🎲 Paciente real aleatorio", on_click=fijar_valores, args=("aleatorio",), width="stretch")
    b[3].button("↺ Restablecer", on_click=fijar_valores, args=("general",), width="stretch")

    col_in, col_out = st.columns([3, 2], gap="large")
    with col_in:
        for tab, clave in zip(st.tabs(list(GRUPOS.values())), GRUPOS):
            with tab:
                cols = st.columns(2)
                for i, c in enumerate([f for f in FEATURES if grupo_de(f) == clave]):
                    r = RANGOS[c]
                    cols[i % 2].slider(RASGOS[rasgo_de(c)][0], float(r["min"]), float(r["max"]), step=paso(c),
                                       format=formato(c), key=f"in_{c}", help=RASGOS[rasgo_de(c)][1])

    X_in = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in FEATURES}])
    proba, etiqueta, aten = predecir(sesion, X_in, FEATURES, umbral)
    p, pred, aten = float(proba[0]), etiqueta[0], aten[0]
    top_var = nombre_es(FEATURES[int(np.argmax(aten))])

    with col_out:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": " %", "font": {"size": 44}},
            title={"text": "Probabilidad de malignidad"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": C_MAL if pred == "Maligno" else C_BEN},
                   "steps": [{"range": [0, umbral * 100], "color": "#e8f5f3"},
                             {"range": [umbral * 100, 100], "color": "#fdece7"}],
                   "threshold": {"line": {"color": C_OSC, "width": 4}, "thickness": 0.8, "value": umbral * 100}}))
        fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(fig, width="stretch")
        if pred == "Maligno":
            st.error(f"### 🔴 Predicción: MALIGNO\nProbabilidad {p:.1%} ≥ umbral {umbral:.0%}")
        else:
            st.success(f"### 🟢 Predicción: BENIGNO\nProbabilidad {p:.1%} < umbral {umbral:.0%}")
        if abs(p - umbral) < 0.15:
            st.warning("Caso **cercano al umbral**: se recomienda revisión adicional.")
        if st.session_state.get("caso_real"):
            ok = st.session_state["caso_real"] == pred
            st.info(f"Diagnóstico real: **{st.session_state['caso_real']}** {'✅ coincide' if ok else '❌ no coincide'}")
        if st.button("💾 Guardar esta predicción en el historial", width="stretch"):
            guardar_prediccion("individual", p, umbral, pred, top_var)
            st.toast("Predicción guardada en SQLite", icon="💾")

    st.subheader("🔍 ¿En qué variables se fijó la red?")
    st.plotly_chart(barra_atencion(aten, "Atención del token [CLS] hacia cada variable (top 10)"), width="stretch")
    st.caption("La atención muestra **dónde mira** el Transformer para este paciente. Las barras rosadas por encima "
               "de las grises son variables a las que la red da más peso que en un paciente promedio. "
               "Es una ayuda de interpretación, no una medida causal.")

# ============================================================ POR LOTES
elif pagina == "📂 Predicción por lotes":
    st.title("📂 Predicción por lotes")
    st.write("Sube un CSV con las 30 columnas del dataset (nombres en inglés) y obtén la predicción de cada paciente.")
    with open(RUTA_EJEMPLO, "rb") as f:
        st.download_button("⬇️ Descargar archivo de ejemplo (10 pacientes)", f, "ejemplo_pacientes.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}"); st.stop()
        faltan = [c for c in FEATURES if c not in datos.columns]
        if faltan:
            st.error(f"Faltan {len(faltan)} columnas, por ejemplo: {', '.join(faltan[:5])}"); st.stop()
        X = datos[FEATURES].apply(pd.to_numeric, errors="coerce")
        invalidas = X.isnull().any(axis=1)
        if invalidas.any():
            st.warning(f"{int(invalidas.sum())} fila(s) con valores vacíos o no numéricos fueron omitidas.")
            X = X[~invalidas]
        if X.empty:
            st.error("No quedan filas válidas."); st.stop()
        proba, etiqueta, aten = predecir(sesion, X, FEATURES, umbral)
        salida = pd.DataFrame({"Probabilidad maligno": proba.round(4), "Predicción": etiqueta,
                               "Variable más atendida": [nombre_es(FEATURES[i]) for i in aten.argmax(1)]},
                              index=X.index)
        c = st.columns(3)
        c[0].metric("Pacientes", len(salida))
        c[1].metric("Predichos malignos", int((salida["Predicción"] == "Maligno").sum()))
        c[2].metric("Predichos benignos", int((salida["Predicción"] == "Benigno").sum()))
        st.dataframe(salida.style.map(lambda v: f"color: {C_MAL}; font-weight: bold" if v == "Maligno"
                                      else (f"color: {C_BEN}" if v == "Benigno" else ""), subset=["Predicción"]),
                     width="stretch")
        a, b = st.columns(2)
        a.download_button("⬇️ Descargar resultados", pd.concat([salida, X], axis=1).to_csv(index=False),
                          "predicciones.csv", "text/csv", type="primary", width="stretch")
        if b.button("💾 Guardar todas en el historial", width="stretch"):
            for (_, fila) in salida.iterrows():
                guardar_prediccion("lote", fila["Probabilidad maligno"], umbral, fila["Predicción"],
                                   fila["Variable más atendida"])
            st.toast(f"{len(salida)} predicciones guardadas en SQLite", icon="💾")

# ============================================================ DATOS Y SQL
elif pagina == "🗄️ Datos y SQL":
    st.title("🗄️ Datos y consultas SQL (DuckDB)")
    st.write("El dataset está cargado en una base **DuckDB** en la tabla `pacientes` "
             f"({len(df)} filas, columnas sin espacios: `mean_radius`, `worst_area`, …, `diagnostico_texto`).")
    con = conexion_duckdb()
    ejemplos = {
        "Distribución del diagnóstico": "SELECT diagnostico_texto, COUNT(*) AS casos,\n"
            "       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS porcentaje\n"
            "FROM pacientes GROUP BY diagnostico_texto",
        "Promedios por diagnóstico": "SELECT diagnostico_texto,\n       ROUND(AVG(mean_radius), 2) AS radio,\n"
            "       ROUND(AVG(mean_texture), 2) AS textura,\n       ROUND(AVG(worst_area), 1) AS peor_area,\n"
            "       ROUND(AVG(worst_concave_points), 4) AS peor_puntos_concavos\n"
            "FROM pacientes GROUP BY diagnostico_texto",
        "Pacientes con área extrema (> p95)": "WITH u AS (SELECT QUANTILE_CONT(worst_area, 0.95) AS p95 FROM pacientes)\n"
            "SELECT paciente_id, worst_area, diagnostico_texto\nFROM pacientes, u WHERE worst_area > u.p95\n"
            "ORDER BY worst_area DESC",
        "Top 10 radios más grandes": "SELECT paciente_id, mean_radius, mean_concavity, diagnostico_texto\n"
            "FROM pacientes ORDER BY mean_radius DESC LIMIT 10",
    }
    eleccion = st.selectbox("Consulta de ejemplo", list(ejemplos))
    sql = st.text_area("Consulta SQL (solo lectura: SELECT / WITH)", ejemplos[eleccion], height=150)
    if st.button("▶️ Ejecutar consulta", type="primary"):
        try:
            res = consulta_segura(con, sql)
            st.success(f"{len(res)} fila(s)")
            st.dataframe(res, width="stretch")
            num = res.select_dtypes("number").columns.tolist()
            if "diagnostico_texto" in res.columns and len(res) <= 5 and num:
                st.plotly_chart(px.bar(res, x="diagnostico_texto", y=num[0], color="diagnostico_texto",
                                       color_discrete_map={"benigno": C_BEN, "maligno": C_MAL}), width="stretch")
        except Exception as e:
            st.error(f"Error en la consulta: {e}")

    st.subheader("Exploración rápida")
    opciones = {nombre_es(c): c for c in FEATURES}
    var = opciones[st.selectbox("Variable", list(opciones), index=FEATURES.index("worst concave points"))]
    fig = px.histogram(df, x=var, color="Diagnóstico", barmode="overlay", opacity=0.65, nbins=40,
                       color_discrete_map=PALETA, title=nombre_es(var))
    st.plotly_chart(fig, width="stretch")

# ============================================================ HISTORIAL
elif pagina == "🗂️ Historial (SQLite)":
    st.title("🗂️ Historial de predicciones (SQLite)")
    st.write("Cada vez que guardas una predicción, la app ejecuta un `INSERT` en la tabla `predicciones` "
             "de una base **SQLite** (`data/historial_predicciones.db`).")
    hist = leer_historial()
    if hist.empty:
        st.info("Todavía no hay predicciones guardadas. Ve a **Predicción individual** y pulsa **💾 Guardar**.")
    else:
        c = st.columns(3)
        c[0].metric("Registros", len(hist))
        c[1].metric("Malignos", int((hist["prediccion"] == "Maligno").sum()))
        c[2].metric("Probabilidad media", f"{hist['probabilidad'].mean():.1%}")
        st.dataframe(hist, width="stretch", hide_index=True)
        a, b = st.columns(2)
        a.download_button("⬇️ Descargar historial", hist.to_csv(index=False), "historial.csv", "text/csv",
                          width="stretch")
        if b.button("🗑️ Borrar historial", width="stretch"):
            borrar_historial(); st.rerun()
    st.caption("En Streamlit Community Cloud el almacenamiento es temporal: el historial se reinicia cuando la app "
               "se reinicia.")

# ============================================================ ARQUITECTURA
elif pagina == "🏗️ Arquitectura y entrenamiento":
    st.title("🏗️ Arquitectura y entrenamiento")
    st.markdown(f"""
**FT-Transformer** (Gorishniy et al., 2021), un Transformer adaptado a datos tabulares:

1. **Tokenizador de variables:** cada variable $x_j$ se convierte en un vector de {HP['d_token']} dimensiones,
   $\\text{{token}}_j = x_j \\cdot w_j + b_j$. Se añade un token especial **[CLS]**, y el paciente queda como una
   secuencia de **31 tokens**.
2. **{HP['capas']} bloques Transformer:** LayerNorm → **atención multi-cabeza ({HP['cabezas']} cabezas)** →
   conexión residual → LayerNorm → red feed-forward ({HP['d_ff']} neuronas, GELU) → conexión residual.
3. **Cabeza de clasificación:** la salida del token [CLS] pasa por una capa lineal y una sigmoide, que da la
   probabilidad de malignidad.

**Entrenamiento:** {HP['optimizador']} (lr = {HP['lr']}, weight decay = {HP['weight_decay']}), lotes de {HP['lote']},
pérdida BCE, parada temprana con paciencia {HP['early_stopping']}, ruido gaussiano σ = {meta['aumento']['ruido_sigma']}
y {meta['aumento']['sinteticos']} pacientes sintéticos generados con un modelo de mezcla gaussiana (GMM).
Total: **{meta['parametros']:,} parámetros**.
""")
    st.subheader("Búsqueda de hiperparámetros (sobre validación)")
    st.dataframe(pd.DataFrame(meta["busqueda"]), width="stretch", hide_index=True)
    st.subheader("Curvas de entrenamiento y validación")
    modelo_sel = st.radio("Modelo", ["FT-Transformer", "MLP", "CNN 1D", "FT-Transformer (sin aumento)"], horizontal=True)
    cur = meta["curvas"][modelo_sel]
    a, b = st.columns(2)
    for col, clave, titulo in [(a, "loss", "Pérdida (BCE)"), (b, "acc", "Exactitud")]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cur[f"{clave}_train"], name="entrenamiento", line=dict(color=C_OSC)))
        fig.add_trace(go.Scatter(y=cur[f"{clave}_val"], name="validación", line=dict(color=C_MAL)))
        fig.add_vline(x=cur["mejor_epoca"] - 1, line_dash="dash", line_color="gray",
                      annotation_text=f"mejor época ({cur['mejor_epoca']})")
        fig.update_layout(title=titulo, xaxis_title="época", height=360)
        col.plotly_chart(fig, width="stretch")
    st.caption("La parada temprana conserva los pesos de la mejor época (línea punteada), antes de que la pérdida "
               "de validación empiece a subir por sobreajuste.")

# ============================================================ DESEMPEÑO
elif pagina == "📊 Desempeño":
    st.title("📊 Desempeño en prueba")
    c = st.columns(6)
    for col, k in zip(c, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Especificidad"]):
        col.metric(k, f"{M[k]:.3f}")
    t1, t2, t3, t4 = st.tabs(["Matriz y ROC", "Comparación de modelos", "Aumento de datos", "Umbral y atención"])
    with t1:
        a, b = st.columns(2)
        cm = meta["matriz_confusion"]
        fig = px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], x=["Pred. Benigno", "Pred. Maligno"],
                        y=["Real Benigno", "Real Maligno"], text_auto=True, color_continuous_scale="Blues")
        fig.update_traces(textfont_size=22)
        fig.update_layout(title="Matriz de confusión (umbral 0,5)", coloraxis_showscale=False, height=400)
        a.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Scatter(x=meta["roc"]["fpr"], y=meta["roc"]["tpr"], fill="tozeroy",
                                   line=dict(color=C_MAL, width=3), name=f"AUC = {M['ROC-AUC']:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Aleatorio"))
        fig.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Recall", height=400)
        b.plotly_chart(fig, width="stretch")
    with t2:
        comp = pd.DataFrame(meta["comparacion"]).T
        comp.insert(0, "Parámetros", [f"{meta['parametros_modelos'][n]:,}" if n in meta["parametros_modelos"]
                                      else "—" for n in comp.index])
        st.dataframe(comp.style.format({k: "{:.4f}" for k in ["Accuracy", "Precision", "Recall", "F1-score",
                                                              "ROC-AUC", "Especificidad"]}), width="stretch")
        d = comp.reset_index().melt(id_vars="index", value_vars=["Accuracy", "F1-score", "Recall"])
        fig = px.bar(d, x="index", y="value", color="variable", barmode="group",
                     color_discrete_sequence=[C_OSC, C_MAL, "#e9c46a"])
        fig.update_layout(yaxis_range=[0.85, 1], xaxis_title="", yaxis_title="", legend_title="", height=400)
        st.plotly_chart(fig, width="stretch")
        st.caption("Mismos 114 pacientes de prueba que la ACA 1. Con un conjunto de prueba tan pequeño, "
                   "diferencias de 1 o 2 pacientes se deben interpretar con prudencia.")
    with t3:
        comp = pd.DataFrame(meta["comparacion"]).T
        abl = comp.loc[["FT-Transformer (sin aumento)", "FT-Transformer"], ["Accuracy", "Recall", "F1-score", "ROC-AUC", "FN"]]
        abl.index = ["Sin aumento", "Con aumento (GMM + ruido)"]
        st.dataframe(abl.style.format("{:.4f}", subset=["Accuracy", "Recall", "F1-score", "ROC-AUC"]), width="stretch")
        au = meta["aumento"]
        st.write(f"Entrenamiento real: **{au['real']}** pacientes · sintéticos: **{au['sinteticos']}** · "
                 f"total: **{au['total']}** (clases equilibradas 50/50). Ruido gaussiano σ = {au['ruido_sigma']}.")
    with t4:
        st.dataframe(pd.DataFrame(meta["umbral"]).set_index("Umbral"), width="stretch")
        aten = pd.DataFrame(meta["atencion"]).sort_values("Global", ascending=False).head(12)
        aten.index = [nombre_es(f) for f in aten.index]
        fig = px.bar(aten[["Benigno", "Maligno"]].iloc[::-1], orientation="h", barmode="group",
                     color_discrete_map={"Benigno": C_BEN, "Maligno": C_MAL},
                     title="Atención media del token [CLS] en prueba (top 12)")
        fig.update_layout(height=460, legend_title="", yaxis_title="", xaxis_title="atención")
        st.plotly_chart(fig, width="stretch")

# ============================================================ ACERCA DE
else:
    st.title("ℹ️ Acerca del proyecto")
    st.markdown("""
**Corporación Unificada Nacional de Educación Superior (CUN)** · Escuela de Ingeniería
Especialización en Inteligencia Artificial · APRENDIZAJE AUTOMÁTICO/54ES1/SEGUNDO BLOQUE/26ES4
Formador: Hamilton Rivera · Actividad: ACA N° 2 · Fecha: 8 de noviembre de 2026

**Integrantes:** Duvan Daniel Gómez Alían, Jeisson Jesith Ortega Lozano, Dubier Alberto Quintero Gómez y Duberney Arboleda Luján

**Stack:** Python · PyTorch (entrenamiento) · ONNX Runtime (inferencia) · DuckDB · SQLite · Streamlit · Plotly

#### Referencias (APA 7)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems, 34*, 18932–18943.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008.
- Wolberg, W., Street, N., & Mangasarian, O. (1995). *UCI Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
""")
    st.warning("Herramienta con fines académicos. No debe usarse para decisiones clínicas reales.")
