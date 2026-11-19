# Dataset

El proyecto utiliza el archivo `listings.csv.gz`, correspondiente a datos detallados de anuncios de alojamientos de Buenos Aires publicados por **Inside Airbnb**.

El archivo se conserva comprimido en formato gzip y los scripts lo leen directamente con pandas.

## Fuente

Fuente: Inside Airbnb — Get the Data  
Ciudad: Buenos Aires  
Archivo: `listings.csv.gz`  
Tipo de dataset: Detailed Listings data

## Ubicación

El archivo debe encontrarse en:

```text
data/raw/listings.csv.gz
```

## Columnas utilizadas

Los ejercicios utilizan las siguientes columnas del dataset:

- `price`: precio nocturno publicado.
- `bedrooms`: cantidad de dormitorios.
- `room_type`: tipo de alojamiento.
- `latitude`: latitud del alojamiento.
- `longitude`: longitud del alojamiento.
- `accommodates`: cantidad de huéspedes admitidos.

## Lectura

Los scripts leen el archivo comprimido directamente mediante pandas:

```python
pd.read_csv("data/raw/listings.csv.gz", compression="gzip")
```

## Consideraciones

Inside Airbnb publica archivos por ciudad y los datos pueden variar según la fecha de extracción. Por este motivo, los programas verifican las columnas utilizadas y realizan tareas de limpieza antes del entrenamiento de los modelos.

El campo `price` corresponde al precio publicado de alojamientos de corta estancia y no al valor de compraventa de una propiedad.