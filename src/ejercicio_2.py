"""
Ejercicio 2 - Regresión Lineal Simple

Estima el precio nocturno de alojamientos de Buenos Aires utilizando
la cantidad de dormitorios como único predictor.

Fuente de datos: Inside Airbnb.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RUTA = Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.csv.gz"

def cargar_datos():
    # Carga el CSV comprimido y crea una versión numérica del precio.
    try:
        df = pd.read_csv(RUTA, compression="gzip")
    except FileNotFoundError:
        print(f"Aviso: No se encontró el archivo en {RUTA}. Se requiere para la ejecución.")
        return None
    
    if "price" not in df.columns:
        raise ValueError("No se encontró la columna price.")
    df["price_num"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
        errors="coerce"
    )
    return df

def preparar(df, predictores):
    # Convierte columnas, elimina faltantes y recorta el 1% inferior/superior.
    # Comprobación de columnas requeridas
    faltan = set(predictores) - set(df.columns)
    assert not faltan, f"Faltan columnas: {faltan}"

    columnas = predictores + ["price_num"]
    datos = df[columnas].copy()
    
    # Revisión de faltantes en bedrooms
    if "bedrooms" in predictores:
        faltantes = datos["bedrooms"].isna().sum()
        print(f"Valores faltantes en 'bedrooms' antes de limpiar: {faltantes}")

    for col in predictores:
        datos[col] = pd.to_numeric(datos[col], errors="coerce")
    datos = datos.dropna()
    datos = datos[datos["bedrooms"] >= 0]
    datos = datos[datos["price_num"] > 0]
    
    # Filtro del 1% al 99%
    q01, q99 = datos["price_num"].quantile([0.01, 0.99])
    datos = datos[datos["price_num"].between(q01, q99)]
    
    print("Nota: Se eliminan precios atípicos (el 1% inferior y el 1% superior).")
    print("Justificación: Se eliminan precios atípicos (errores de carga o propiedades de lujo) que sesgarían la pendiente del modelo.")
    
    return datos

def evaluar(y_real, y_pred):
    # Calcula y muestra métricas de regresión.
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    return mae, rmse, r2

def graficar_resultados(datos, modelo):
    """Genera el gráfico de regresión y el boxplot de precios por dormitorios."""

    muestra = datos.sample(
        min(1500, len(datos)),
        random_state=42
    ).sort_values("bedrooms")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(
        muestra["bedrooms"],
        muestra["price_num"],
        alpha=0.25,
        label="Anuncios"
    )

    plt.plot(
        muestra["bedrooms"],
        modelo.predict(muestra[["bedrooms"]]),
        color="crimson",
        linewidth=2,
        label="Recta estimada"
    )

    plt.xlabel("bedrooms")
    plt.ylabel("Precio nocturno")
    plt.title("Dormitorios vs Precio")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(
        x="bedrooms",
        y="price_num",
        data=muestra,
        color="lightblue"
    )

    plt.title("Distribución (Boxplot)")
    plt.xlabel("bedrooms")
    plt.ylabel("Precio nocturno")

    plt.tight_layout()
    RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "reports" / "figures"
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        RUTA_FIGURAS / "ejercicio_2.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    df = cargar_datos()

    if df is not None:
        if "room_type" in df.columns:
            df_ej = df[df["room_type"] == "Entire home/apt"].copy()
        else:
            df_ej = df.copy()

        datos = preparar(df_ej, ["bedrooms"])

        RUTA_PROCESADOS = Path(__file__).resolve().parents[1] / "data" / "processed"
        RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)

        datos.to_csv(
            RUTA_PROCESADOS / "ejercicio_2.csv",
            index=False
        )

        X = datos[["bedrooms"]]
        y = datos["price_num"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        print("\n--- MODELO SELECCIONADO ---")
        print(
        "Se utiliza Regresión Lineal Simple porque se busca estimar una variable "
        "numérica continua (precio) a partir de un único predictor (dormitorios), "
        "mediante una relación lineal fácil de interpretar."
        )

        
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        print("\n--- MÉTRICAS ---")
        evaluar(y_test, y_pred)
        print(f"Intercepto: {modelo.intercept_:.2f}")
        print(f"Pendiente: {modelo.coef_[0]:.2f}")
        
        nuevo = pd.DataFrame({"bedrooms": [2]})
        print(f"Estimación del nuevo caso (2 dormitorios): {modelo.predict(nuevo)[0]:.2f}")
        
        comparacion = pd.DataFrame({"Real": y_test, "Predicho": y_pred}).round(0)
        print("\n--- REAL VS PREDICHO ---")
        print(comparacion.head(10))
        
        print("\n--- CONCLUSIÓN ---")
        print("La regresión lineal refleja que, en promedio, añadir un dormitorio incrementa el precio nocturno según lo marcado por la pendiente.")
        print("Si bien el modelo identifica la tendencia, El R² obtenido es bajo, lo que indica que el número de dormitorios por sí solo no basta para predecir con exactitud el precio.")
        print("Es importante verificar la moneda de origen de los datos para interpretar adecuadamente el MAE y el RMSE en el contexto real.")
        
        # Gráficos (Scatter y Boxplot)
        graficar_resultados(datos, modelo)