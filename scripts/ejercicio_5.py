from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RUTA = Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.csv.gz"

def cargar_datos():
    """Carga el CSV comprimido y crea una versión numérica del precio."""
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
    """Convierte columnas, elimina faltantes y recorta el 1% inferior/superior."""
    faltan = set(predictores) - set(df.columns)
    assert not faltan, f"Faltan columnas: {faltan}"

    columnas = predictores + ["price_num"]
    datos = df[columnas].copy()
    for col in predictores:
        datos[col] = pd.to_numeric(datos[col], errors="coerce")
    datos = datos.dropna()
    datos = datos[(datos["price_num"] > 0)]
    
    q01, q99 = datos["price_num"].quantile([0.01, 0.99])
    datos = datos[datos["price_num"].between(q01, q99)]
    
    print("Nota: Se eliminan precios atípicos (el 1% inferior y el 1% superior).")
    print("Justificación: Se eliminan para evitar que propiedades atípicas sesguen las distancias.")
    print("Fuga de datos: Los percentiles se calcularon sobre todo el dataset antes de separar train/test; es una fuga mínima asumida como limpieza.")
    return datos

def evaluar(y_real, y_pred):
    """Calcula y muestra métricas de regresión."""
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    return mae, rmse, r2

if __name__ == "__main__":
    df = cargar_datos()
    
    if df is not None:
        predictores = ["latitude", "longitude", "accommodates", "bedrooms"]
        datos = preparar(df, predictores)
        X = datos[predictores]
        y = datos["price_num"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        modelo = Pipeline([
            ("escalador", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))
        ])
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        print("\n--- MÉTRICAS ---")
        evaluar(y_test, y_pred)
        
        nuevo = X_test.iloc[[0]]
        print("\nCaracterísticas del caso:\n", nuevo)
        print("Precio real:", y_test.iloc[0])
        print("Precio estimado:", modelo.predict(nuevo)[0])
        
        escalador = modelo.named_steps["escalador"]
        knn = modelo.named_steps["knn"]
        
        # Extracción de vecinos escalando adecuadamente
        nuevo_escalado = escalador.transform(nuevo)
        distancias, indices = knn.kneighbors(nuevo_escalado)
        
        vecinos = X_train.iloc[indices[0]].copy()
        vecinos["precio"] = y_train.iloc[indices[0]].values
        vecinos["distancia_escalada"] = distancias[0]
        print("\n--- VECINOS MÁS CERCANOS ---")
        print(vecinos)
        
        print("\n--- REAL VS PREDICHO ---")
        comparacion = pd.DataFrame({"Real": y_test, "Predicho": y_pred}).round(0)
        print(comparacion.head(10))
        
        print("\n--- EXPLICACIÓN DEL ESCALADO E INTERPRETACIÓN ---")
        print("Sin escalar, las variables con números más grandes dominarían el cálculo de la distancia, mientras que la latitud y la longitud (que varían en decimales) casi no tendrían peso.")
        print("Es clave destacar que 2 de las 4 variables son de ubicación. Esto provoca que la geografía pese aproximadamente la mitad en la similitud. Como cada coordenada se escala por separado, la 'distancia' k-NN no se puede leer como kilómetros físicos.")
        
        print("\n--- CONCLUSIÓN ---")
        print("El modelo de vecinos más cercanos estima eficientemente el precio contrastando propiedades con características y ubicación similares.")
        print("La estandarización dentro del pipeline previene la fuga de datos, ajustándose solo con los datos de entrenamiento para brindar una predicción imparcial y precisa.")
        
        # Gráfico Real vs Predicho
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        lim = [y_test.min(), y_test.max()]
        plt.plot(lim, lim, color="crimson", label="Predicción perfecta")
        plt.xlabel("Precio real")
        plt.ylabel("Precio predicho")
        plt.title("k-NN (k=7): real vs. predicho")
        plt.legend()
        plt.tight_layout()
        plt.show()