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
├── scripts/
│   ├── ejercicio_2.py
│   └── ejercicio_5.py
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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows, activa el entorno con:

```powershell
.venv\Scripts\Activate.ps1
```

## Ejecución

Los scripts buscan el dataset dentro de `data/raw/`, sin depender de la carpeta
desde la que se invoquen:

```bash
python scripts/ejercicio_2.py
python scripts/ejercicio_5.py
```

Ambos programas imprimen métricas y resultados en la terminal. También abren
gráficos interactivos; para verlos, ejecútalos en un entorno con interfaz
gráfica.

## Ejercicios

- **Ejercicio 2:** regresión lineal del precio en función de dormitorios,
  limitada a alojamientos de tipo `Entire home/apt` cuando esa columna existe.
  Muestra métricas, una estimación para dos dormitorios y gráficos.
- **Ejercicio 5:** regresión k-NN con latitud, longitud, capacidad y dormitorios.
  Estandariza las variables mediante un `Pipeline` y muestra los vecinos más
  cercanos y la comparación entre valores reales y estimados.

## Datos y limitaciones

El dataset comprimido se encuentra en [`data/raw/listings.csv.gz`](data/raw/listings.csv.gz).
Su descripción, procedencia y consideraciones de distribución están en
[`data/README.md`](data/README.md).

Los scripts eliminan filas incompletas y recortan los precios fuera de los
percentiles 1 y 99 antes de separar entrenamiento y prueba. Por tanto, los
percentiles se calculan sobre el conjunto completo; esta limitación se imprime
al ejecutar el análisis. El precio se extrae de la columna `price`; antes de
interpretar o comparar métricas, verifica la moneda y el período del dataset.
El modelo lineal usa solo dormitorios, mientras que la distancia del modelo
k-NN combina dos variables geográficas y dos características del alojamiento,
por lo que no representa una distancia física.
