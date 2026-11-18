# Dataset

`raw/listings.csv.gz` es un archivo CSV comprimido con registros de anuncios de
alojamientos. Se conserva en formato comprimido y los scripts lo leen
directamente con pandas.

## Uso esperado

- Ruta: `data/raw/listings.csv.gz`
- Formato: CSV comprimido con gzip
- Columnas usadas por los ejercicios: `price`, `bedrooms`, `room_type`,
  `latitude`, `longitude` y `accommodates`

No se incluyeron metadatos sobre la fuente original, el método de recolección ni
la licencia del dataset. El archivo contiene campos descriptivos de anuncios y
anfitriones; confirma que su redistribución pública está permitida antes de
publicar este repositorio. La licencia MIT en la raíz cubre el código, no los
datos.
