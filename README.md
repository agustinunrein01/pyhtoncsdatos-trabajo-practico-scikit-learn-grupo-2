# Trabajo práctico de Scikit-learn — Grupo 2

Ejercicios de aprendizaje supervisado con Python y Scikit-learn sobre datos de
anuncios de alojamientos. El proyecto compara una regresión lineal simple con
un modelo de regresión k-NN.

## Estructura

```text
.
├── data/
│   ├── README.md
│   └── raw/
│       └── listings.csv.gz
    └── processed/
         ├── ejercicio_2.csv
│        └── ejercicio_5.csv
├── src/
│   ├── ejercicio_2.py
│   └── ejercicio_5.py
├── reports/
│   ├── figures
│       ├── ejercicio_2.png
│       └── ejercicio_5.png
├── .gitignore
├── README.md
└── requirements.txt
```

## Requisitos

- Python 3.10 o posterior
- Las dependencias listadas en [`requirements.txt`](requirements.txt)


## Instalación

Desde la carpeta raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, crear y activar el entorno con:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecución

Desde la carpeta raíz del proyecto, ejecutar:

```bash
python src/ejercicio_2.py
python src/ejercicio_5.py
```

Los programas cargan automáticamente el dataset ubicado en `data/raw/listings.csv.gz`.

Ambos scripts imprimen en la terminal las métricas, predicciones y resultados solicitados, y generan los gráficos correspondientes.

## Ejercicios

- **Ejercicio 2:** utiliza Regresión Lineal Simple para estimar el precio nocturno
  a partir de la cantidad de dormitorios. Se elige este modelo porque la variable
  objetivo es numérica continua y se busca representar una relación lineal simple
  y fácil de interpretar. Cuando `room_type` está disponible, el análisis se limita
  a alojamientos de tipo `Entire home/apt`. El programa muestra métricas, una
  estimación para dos dormitorios y gráficos.
- **Ejercicio 5:** utiliza k-NN para regresión con latitud, longitud, capacidad
  y dormitorios como predictores. Se elige este modelo porque permite estimar
  un precio numérico continuo comparando cada alojamiento con otros de ubicación
  y características físicas similares. Las variables se estandarizan mediante
  un `Pipeline` con `StandardScaler`, y el programa muestra los vecinos más
  cercanos, sus distancias estandarizadas y la comparación entre valores reales
  y estimados.

## Datos y limitaciones

El dataset comprimido se encuentra en [`data/raw/listings.csv.gz`](data/raw/listings.csv.gz).
Su descripción, procedencia y consideraciones de distribución están en
[`data/README.md`](data/README.md).

Los scripts eliminan filas incompletas, descartan registros imposibles y recortan
los precios fuera de los percentiles 1 y 99 antes de separar entrenamiento y prueba. 
El precio se extrae de la columna `price`; antes de
interpretar o comparar métricas, verifica la moneda y el período del dataset.
El modelo lineal utiliza únicamente la cantidad de dormitorios como predictor.
En el modelo k-NN, las distancias se calculan sobre las variables estandarizadas,
por lo que no deben interpretarse como distancias físicas en kilómetros.
