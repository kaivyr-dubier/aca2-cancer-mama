# ACA 2 – Diagnóstico de cáncer de mama con un Transformer tabular (FT-Transformer)

**Asignatura:** APRENDIZAJE AUTOMÁTICO/54ES1/SEGUNDO BLOQUE/26ES4 · CUN · Especialización en IA
**Formador:** Hamilton Rivera · **Fecha de entrega:** 8 de noviembre de 2026
**Integrantes:** Duvan Daniel Gómez Alían, Jeisson Jesith Ortega Lozano, Dubier Alberto Quintero Gómez y Duberney Arboleda Luján
**Pilar:** Salud · **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)

Continuación de la ACA 1 (SVM clásica). Aquí se usa Deep Learning: un **FT-Transformer** con autoatención sobre las 30 variables del núcleo celular, comparado con un MLP, una CNN 1D y la SVM de la ACA 1, sobre **la misma partición de prueba**.

| Modelo (114 pacientes de prueba) | Accuracy | F1 | ROC-AUC | Malignos sin detectar |
|---|---|---|---|---|
| **FT-Transformer (desplegado)** | **0.9825** | **0.9762** | **0.9947** | **1** |
| MLP | 0.9649 | 0.9524 | 0.9940 | 2 |
| CNN 1D | 0.9474 | 0.9302 | 0.9937 | 2 |
| SVM (ACA 1) | 0.9737 | 0.9630 | 0.9947 | 3 |

## Estructura
```
├── ACA2_Transformer_CancerMama.ipynb  ← notebook: SQL, pipeline, entrenamiento, curvas, evaluación
├── app.py                             ← app Streamlit
├── utils.py                           ← modelo ONNX, DuckDB, historial SQLite
├── requirements.txt                   ← dependencias de la app (sin PyTorch)
├── requirements-notebook.txt          ← dependencias para reentrenar
├── models/
│   ├── ft_transformer_cancer_mama.onnx  ← red exportada (incluye el escalado)
│   ├── ft_transformer_pesos.pt          ← pesos PyTorch
│   └── metadata.json                    ← métricas, curvas, atención
├── data/  (CSV del dataset, base DuckDB y ejemplo para lotes)
├── figures/                           ← gráficas generadas por el notebook
└── .github/workflows/mantener-app-activa.yml  ← evita que la app se duerma
```

## Cómo se carga el modelo en la app
El notebook entrena la red en **PyTorch** y la exporta a **ONNX** con el escalador incluido. La app la carga con `onnxruntime`:
```python
sesion = onnxruntime.InferenceSession("models/ft_transformer_cancer_mama.onnx")
probabilidad, atencion = sesion.run(None, {"variables": X.astype("float32")})  # X: (n, 30) sin escalar
```
El orden de las 30 columnas está en `models/metadata.json → features`.

## Ejecutar
```bash
pip install -r requirements.txt
streamlit run app.py
# Para reentrenar:
pip install -r requirements-notebook.txt
jupyter notebook ACA2_Transformer_CancerMama.ipynb
```
El notebook también corre en Google Colab: súbelo y ejecútalo completo, porque instala lo que falte y tarda unos 2 minutos en CPU.

## Publicar en Streamlit Community Cloud
1. Sube este contenido a un repositorio público de GitHub.
2. En https://share.streamlit.io elige **Create app** y llena: repositorio, rama `main`, archivo `app.py`, App URL `aca2-cancer-mama`. En **Advanced settings** elige Python **3.13**.
3. El workflow `.github/workflows/mantener-app-activa.yml` visita la app cada 6 horas para que no se duerma. Si cambias la URL, actualízala en ese archivo.

## Referencias (APA 7)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems, 34*, 18932–18943. https://arxiv.org/abs/2106.11959
- Raasveldt, M., & Mühleisen, H. (2019). DuckDB: An embeddable analytical database. En *Proceedings of the 2019 International Conference on Management of Data* (pp. 1981–1984). ACM. https://doi.org/10.1145/3299869.3320212
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008.
- Wolberg, W., Street, N., & Mangasarian, O. (1995). *UCI Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
