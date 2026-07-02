# ⚽ Sistema de Predicción e Inteligencia Deportiva — FIFA World Cup 2026

Proyecto final del curso **Introducción a Deep Learning**
Docente: **Ing. Jairo Pinedo Taquia**

Sistema de inteligencia artificial interactivo que permite a analistas y periodistas
deportivos **simular escenarios del Mundial 2026**, predecir resultados de partidos y
explorar cómo cambia el campeón probable al reorganizar los grupos. Combina dos modelos
de Deep Learning con un dashboard funcional que cualquier usuario no técnico puede operar.

---

## 👥 Equipo de desarrollo

| Código | Nombre |
|---|---|---|
| U202210644 | Nathaly Eliane Anaya Vadillo |
| U202220990 | Marsi Valeria Figueroa Larragán |
| U20221E167 | Liam Mikael Quino Neff |
| U202115571 | Loana Colleen Rodriguez Matos |

---

## 📋 De qué trata el proyecto

El sistema tiene **dos componentes integrados**:

### 1. Modelo predictivo de partidos
Predice el resultado de un partido entre dos selecciones a partir de:
- **Ranking FIFA** de ambas selecciones
- **Forma reciente** (promedio de goles anotados/recibidos en los últimos 10 partidos)
- **Historial de enfrentamientos directos** (victorias, empates, derrotas, diferencia de goles)
- **Atributos FIFA** de la plantilla (promedio del top-23: overall, pace, shooting, defending, physical)

Y produce: **P(victoria A) · P(empate) · P(victoria B)** y el **marcador esperado** (goles A–B).

Se entrenan y comparan **dos arquitecturas** sobre el mismo test set (Mundial 2022):

| Arquitectura | Descripción | F1 macro | F1 ponderado |
|---|---|---|---|
| **MLP (Adam)** ⭐ | Red densa multi-salida (4 capas ocultas ReLU, L2 + Dropout) | **0.435** | **0.502** |
| MLP (SGD+Nesterov) | Misma red, optimizador SGD con momentum | 0.422 | 0.489 |
| LSTM + Dense | Rama recurrente siamesa sobre la secuencia de 10 partidos | 0.386 | 0.450 |
| GRU + Dense | Igual que LSTM, capa GRU (más rápida, menos parámetros) | 0.396 | 0.461 |

Se documenta además el problema de ***vanishing gradient*** con la ecuación de BPTT y evidencia
empírica (SimpleRNN vs LSTM). El **MLP con Adam** es el modelo elegido para el simulador por su
mayor F1 y su inferencia más simple y rápida (ideal para el recálculo en tiempo real).

### 2. Simulador del torneo (dashboard)
Dashboard interactivo en español (Streamlit) con tres vistas, todas recalculando con el modelo real:
- **Fase de grupos** — 12 grupos editables; tabla con puntos proyectados, goles esperados y % de clasificación.
- **Eliminación directa** — bracket completo (32avos → final) con el favorito de cada cruce, las tres
  probabilidades por partido (victoria A · empate · victoria B) y opción de **forzar** el avance de una selección.
- **Campeones probables** — top 10 con probabilidad de título y contexto histórico de mundiales ganados.

---

## 📊 Datasets (públicos, Kaggle)

| Dataset | Uso |
|---|---|
| International football results 1872–2024 | Historial de partidos y goles |
| FIFA World Rankings | Ranking por selección |
| FIFA Players 15–24 | Atributos agregados de plantilla |
| World Cup historical stats | Referencia del test set + contexto de campeones |

Se descargan automáticamente al ejecutar el notebook (requiere `kaggle.json`).

---

## 📁 Estructura del repositorio

```
.
├── tb2_idl_TF.ipynb          # Notebook principal (ejecutado de inicio a fin)
├── app.py                    # Dashboard interactivo (Componente 2)
├── requirements.txt          # Dependencias
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml           # Tema oscuro del dashboard
└── outputs/                  # Artefactos generados por el notebook
    ├── models/               # mlp_model_best.keras · lstm_hybrid_model.keras
    ├── scalers/              # normalizadores (.pkl)
    ├── metadata/             # orden de features + clases (.json)
    ├── tables/               # model_df_features.csv (lo usa el dashboard) y comparativas
    └── figures/              # gráficos de evaluación (.png)
```

> `data/` (datasets crudos) y `.venv/` no se versionan: ver `.gitignore`.

---

## 🚀 Guía de ejecución

### Requisitos
- Python 3.10+
- Un `kaggle.json` válido (Kaggle → *Settings* → *API* → *Create New Token*) **solo si vas a
  reejecutar el notebook**. El dashboard NO lo necesita: usa los artefactos ya guardados en `outputs/`.

### Opción A — Local (recomendada)

```bash
# 1. Crear e instalar el entorno
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. (Opcional) Reentrenar: abrir tb2_idl_TF.ipynb y ejecutar de inicio a fin.
#    Regenera outputs/ y reescribe app.py. Requiere kaggle.json.

# 3. Lanzar el dashboard (usa el modelo ya entrenado en outputs/)
streamlit run app.py
```

El dashboard abre en `http://localhost:8501`.

### Opción B — Google Colab

1. Sube el notebook `tb2_idl_TF.ipynb` a Colab.
2. Ejecuta las celdas de inicio a fin (la primera celda de datos te pedirá subir tu `kaggle.json`).
3. La última celda levanta el dashboard con un túnel público de Cloudflare e imprime una URL
   `https://…trycloudflare.com` para abrirlo desde el navegador.

---

## ✅ Notas de la entrega

- El notebook está **ejecutado de inicio a fin** (todas las celdas con salida) e incluye teoría en
  Markdown y gráficos de evaluación etiquetados.
- El dashboard **carga el modelo real** (`mlp_model_best.keras`) y reconstruye las 28 features con
  el mismo `scaler` del entrenamiento, de modo que notebook y dashboard cuentan la misma historia.
