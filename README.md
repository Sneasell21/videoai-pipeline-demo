# VideoAI · Pipeline de Procesamiento

Demo de Streamlit para presentaciones: simula un sistema de procesamiento de video con subida a S3, pipeline animado paso a paso y resultados mock. Diseñado para nunca fallar en vivo.

## Correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opcional: crear `.streamlit/secrets.toml` copiando el ejemplo y completando tus credenciales. Sin ese archivo la app activa **Modo demo** automáticamente.

## Deploy en Streamlit Community Cloud

1. Subir este repositorio a GitHub.
2. Ir a [share.streamlit.io](https://share.streamlit.io) → **New app** → elegir el repo y `app.py`.
3. En **Advanced settings → Secrets** pegar el contenido de `.streamlit/secrets.toml.example` con tus valores reales.
4. Hacer deploy.

## Configurar secrets (S3 / compatible)

En Streamlit Cloud (Settings → Secrets):

```toml
S3_ENDPOINT = "https://s3.amazonaws.com"
S3_KEY      = "TU_ACCESS_KEY"
S3_SECRET   = "TU_SECRET_KEY"
S3_BUCKET   = "nombre-del-bucket"
S3_REGION   = "us-east-1"
```

Sin estos valores la app funciona igual en **Modo demo** (subida simulada, pipeline real).

## Adaptar el pipeline

Editar la constante `STEPS` al inicio de `app.py`. Ejemplo para agregar un paso de compresión:

```python
STEPS = [
    {"nombre": "Recepción en bucket",   "descripcion": "...", "dur_rel": 0.8},
    {"nombre": "Compresión adaptativa", "descripcion": "Recodificando a H.265", "dur_rel": 1.0},
    # ...
]
```

- `dur_rel` es la duración relativa del paso (multiplicada por el slider **Segundos por paso**).
- Agregar, quitar o renombrar pasos no requiere tocar ninguna otra parte del código.

## Nota importante para presentaciones

> Las apps gratuitas de Streamlit Cloud se duermen tras 7 días de inactividad.
> **Abrí la app al menos 5 minutos antes de presentar** y recargá la página para despertarla.
> Una vez activa, no se duerme mientras tenga visitas.

Para garantizar disponibilidad, activar **Modo demo** en la barra lateral desacopla el demo de cualquier dependencia externa.
