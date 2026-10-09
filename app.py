"""
VideoAI · Pipeline de Procesamiento
Demo Streamlit — listo para Streamlit Community Cloud
"""

import io
import json
import random
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ── Configuración de página (debe ser la primera llamada) ─────────────────────
st.set_page_config(
    page_title="VideoAI · Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PASOS DEL PIPELINE — editar aquí para adaptar el demo.
# Cambiar nombres, descripciones o dur_rel no requiere tocar otra parte del código.
# ══════════════════════════════════════════════════════════════════════════════
STEPS = [
    {
        "nombre": "Recepción en bucket",
        "descripcion": "Verificando integridad y persistencia del archivo",
        "dur_rel": 0.8,
    },
    {
        "nombre": "Validación e indexado",
        "descripcion": "Extrayendo metadatos y validando codec / contenedor",
        "dur_rel": 0.6,
    },
    {
        "nombre": "Extracción de frames",
        "descripcion": "Descomprimiendo y muestreando frames clave",
        "dur_rel": 1.4,
    },
    {
        "nombre": "Detección (modelo IA)",
        "descripcion": "Inferencia con modelo de detección de objetos YOLOv9",
        "dur_rel": 2.0,
    },
    {
        "nombre": "Clasificación / identificación",
        "descripcion": "Agrupando y etiquetando detecciones por clase",
        "dur_rel": 1.2,
    },
    {
        "nombre": "Generación de reporte",
        "descripcion": "Consolidando resultados y construyendo artefactos",
        "dur_rel": 0.5,
    },
]

CLASES = ["persona", "vehículo", "objeto", "animal", "señal vial", "edificio"]

CONTENT_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
}

# ── CSS mínimo ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Scrollbar en el log */
.stCode > div { max-height: 230px; overflow-y: auto; }
/* Separador suave en métricas */
[data-testid="metric-container"] { border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="metric-container"]:last-child { border-right: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "videos_procesados": 0,
    "ultimo_estado": "—",
    "ultima_duracion": "—",
    "objetos_detectados": 0,
    "historial": [],
    "resultado": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS S3
# ══════════════════════════════════════════════════════════════════════════════

def _leer_secrets():
    """Lee configuración S3 desde st.secrets. Retorna dict o None."""
    try:
        s = st.secrets
        return {
            "endpoint_url": s["S3_ENDPOINT"],
            "aws_access_key_id": s["S3_KEY"],
            "aws_secret_access_key": s["S3_SECRET"],
            "region_name": s.get("S3_REGION", "us-east-1"),
            "bucket": s["S3_BUCKET"],
        }
    except Exception:
        return None


_CFG = _leer_secrets()
BUCKET_DISPONIBLE = _CFG is not None


def _cliente_s3():
    """Crea cliente boto3. Retorna (client, bucket) o (None, None)."""
    if _CFG is None:
        return None, None
    try:
        import boto3  # import tardío para no fallar si boto3 no está disponible
        client = boto3.client(
            "s3",
            endpoint_url=_CFG["endpoint_url"],
            aws_access_key_id=_CFG["aws_access_key_id"],
            aws_secret_access_key=_CFG["aws_secret_access_key"],
            region_name=_CFG["region_name"],
        )
        return client, _CFG["bucket"]
    except Exception:
        return None, None


def subir_archivo(data: bytes, nombre: str, content_type: str):
    """
    Sube el archivo al bucket S3.
    Retorna: (True, clave) si OK · (False, clave) si error boto3 · (None, clave) si no configurado
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clave = f"uploads/{ts}_{nombre}"
    client, bucket = _cliente_s3()
    if client is None:
        return None, clave
    try:
        client.upload_fileobj(
            io.BytesIO(data),
            bucket,
            clave,
            ExtraArgs={"ContentType": content_type},
        )
        return True, clave
    except Exception:
        return False, clave


def url_firmada(clave: str):
    """Genera URL prefirmada con validez de 1 hora."""
    client, bucket = _cliente_s3()
    if client is None:
        return None
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": clave},
            ExpiresIn=3600,
        )
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE DATOS MOCK
# ══════════════════════════════════════════════════════════════════════════════

def _generar_detecciones(seed: int):
    rng = random.Random(seed)
    n = rng.randint(5, 12)
    rows = []
    for _ in range(n):
        mm, ss, ms = rng.randint(0, 4), rng.randint(0, 59), rng.randint(0, 9)
        x1, y1 = rng.randint(0, 700), rng.randint(0, 400)
        x2, y2 = x1 + rng.randint(50, 250), y1 + rng.randint(30, 180)
        rows.append(
            {
                "Timestamp video": f"{mm:02d}:{ss:02d}.{ms}",
                "Clase": rng.choice(CLASES),
                "Confianza": round(rng.uniform(0.62, 0.99), 3),
                "Bounding box": f"[{x1}, {y1}, {x2}, {y2}]",
            }
        )
    return rows


def _linea_log(rng, paso_idx: int, step: dict, frame: int, total: int) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    if paso_idx == 2:
        kf = "sí" if rng.random() > 0.6 else "no"
        return f"[{ts}] Frame {frame}/{total} extraído · keyframe={kf}"
    if paso_idx == 3:
        dets = rng.randint(0, 5)
        lat = rng.randint(22, 145)
        return f"[{ts}] Frame {frame}/{total} analizado · {dets} det · {lat} ms"
    genericos = [
        f"[{ts}] {step['nombre']} · chunk {rng.randint(1, 8)}/8 listo",
        f"[{ts}] Checksum verificado · OK",
        f"[{ts}] Buffer {rng.randint(10, 90)}% ocupado",
        f"[{ts}] Worker #{rng.randint(0, 3)} · carga {rng.randint(20, 95)}%",
        f"[{ts}] Escritura en disco · {rng.randint(5, 50)} MB/s",
    ]
    return rng.choice(genericos)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_pipeline(
    nombre: str,
    tamaño: int,
    clave: str,
    real: bool,
    vel: float,
    step_phs: list,
    log_ph,
) -> dict:
    """Anima el pipeline paso a paso. Retorna dict con todos los resultados."""
    seed = hash(nombre) & 0xFFFFFF
    rng = random.Random(seed)
    total_frames = rng.randint(300, 1200)
    tiempos = []
    log_lines = []

    def render_log():
        log_ph.code("\n".join(log_lines[-18:]), language=None)

    t0 = time.time()

    for i, step in enumerate(STEPS):
        dur = vel * step["dur_rel"]
        ticks = max(6, int(dur / 0.18))
        t_paso = time.time()

        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Iniciando: {step['nombre']}")
        render_log()

        for tick in range(ticks):
            pct = int((tick + 1) / ticks * 100)
            frame_n = int((i + (tick + 1) / ticks) / len(STEPS) * total_frames)
            barra = "▰" * (pct // 10) + "▱" * (10 - pct // 10)

            step_phs[i].markdown(
                f"🔵 **{step['nombre']}** `{barra}` {pct}%  \n"
                f"*{step['descripcion']}*"
            )

            if tick % max(1, ticks // 4) == 0:
                log_lines.append(_linea_log(rng, i, step, frame_n, total_frames))
                render_log()

            time.sleep(dur / ticks)

        elapsed = round(time.time() - t_paso, 2)
        tiempos.append(elapsed)
        step_phs[i].markdown(f"✅ **{step['nombre']}** — {elapsed:.2f}s")
        log_lines.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ✔ {step['nombre']} completado ({elapsed:.2f}s)"
        )
        render_log()

    tiempo_total = round(time.time() - t0, 1)
    dets = _generar_detecciones(seed)
    n_dets = len(dets)
    conf_avg = round(sum(d["Confianza"] for d in dets) / n_dets, 3)

    return {
        "nombre": nombre,
        "tamaño_bytes": tamaño,
        "clave": clave,
        "real": real,
        "frames": total_frames,
        "n_dets": n_dets,
        "conf_avg": conf_avg,
        "tiempo_total": tiempo_total,
        "tiempos_paso": tiempos,
        "detecciones": dets,
        "log": log_lines,
        "ts": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 VideoAI")
    st.markdown("**Pipeline de Procesamiento**")
    st.divider()

    if BUCKET_DISPONIBLE:
        st.markdown("**Bucket S3:** 🟢 Conectado")
    else:
        st.markdown("**Bucket S3:** 🔴 No configurado")

    st.divider()

    modo_demo = st.toggle(
        "🧪 Modo demo",
        value=not BUCKET_DISPONIBLE,
        help="El pipeline corre aunque la subida al bucket falle.",
    )
    st.caption("En modo demo la subida es simulada.")

    velocidad = st.slider(
        "⏱ Segundos por paso (base)",
        min_value=0.3,
        max_value=4.0,
        value=1.5,
        step=0.1,
        help="Multiplicado por la duración relativa de cada paso.",
    )

    st.divider()

    if st.button("🔄 Reiniciar sesión", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CABECERA — métricas de sesión
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎬 VideoAI · Pipeline de Procesamiento")

m1, m2, m3, m4 = st.columns(4)
m1.metric("📹 Videos procesados", st.session_state.videos_procesados)
m2.metric("📊 Último estado", st.session_state.ultimo_estado)
m3.metric("⏱ Duración del último proceso", st.session_state.ultima_duracion)
m4.metric("🔍 Objetos detectados", st.session_state.objetos_detectados)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL — dos columnas
# ══════════════════════════════════════════════════════════════════════════════
col_izq, col_der = st.columns([1, 1.2])

# ── Columna izquierda: carga del archivo ──────────────────────────────────────
with col_izq:
    with st.container(border=True):
        st.markdown("##### 📤 Entrada")
        archivo = st.file_uploader(
            "Seleccioná un video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            label_visibility="collapsed",
        )
        if archivo:
            mb = archivo.size / 1024 / 1024
            st.markdown(f"📄 **{archivo.name}** · {mb:.2f} MB")
            procesar = st.button(
                "🚀 Subir y procesar", type="primary", use_container_width=True
            )
        else:
            st.info("Subí un video para comenzar el pipeline.")
            procesar = False

    if archivo:
        st.markdown("**Vista previa**")
        st.video(archivo)


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DEL PIPELINE (solo cuando se hace clic en el botón)
# ══════════════════════════════════════════════════════════════════════════════
if procesar and archivo:
    # Construir el panel de pipeline en col_der ANTES de ejecutar
    with col_der:
        with st.container(border=True):
            st.markdown("##### ⚙️ Pipeline en ejecución")
            step_phs = [st.empty() for _ in STEPS]

        with st.container(border=True):
            st.markdown("##### 📋 Log en vivo")
            log_ph = st.empty()

    # Inicializar pasos como pendientes
    for i, step in enumerate(STEPS):
        step_phs[i].markdown(f"⚪ **{step['nombre']}**  \n*{step['descripcion']}*")
    log_ph.code("", language=None)

    # ── Subida al bucket ─────────────────────────────────────────────────────
    data = archivo.getvalue()
    nombre = archivo.name
    ext = nombre.rsplit(".", 1)[-1].lower()
    ct = CONTENT_TYPES.get(ext, "video/mp4")

    ok, clave = subir_archivo(data, nombre, ct)

    if ok is None:
        # Bucket no configurado
        st.warning("ℹ️ Bucket no configurado — procesando en modo demo.")
        subida_real = False
    elif ok is False:
        # Error de boto3
        if not modo_demo:
            st.error(
                "❌ Error al subir al bucket. "
                "Activá **Modo demo** en la barra lateral para continuar sin bucket."
            )
            st.stop()
        else:
            st.warning("⚠️ Subida simulada (error en bucket) — modo demo activo.")
            subida_real = False
    else:
        st.success(f"✅ Archivo subido al bucket → `{clave}`")
        subida_real = True

    # ── Ejecutar pipeline ────────────────────────────────────────────────────
    resultado = ejecutar_pipeline(
        nombre=nombre,
        tamaño=len(data),
        clave=clave,
        real=subida_real,
        vel=velocidad,
        step_phs=step_phs,
        log_ph=log_ph,
    )

    # Actualizar estado de sesión
    st.session_state.resultado = resultado
    st.session_state.videos_procesados += 1
    st.session_state.ultimo_estado = "✅ Completado"
    st.session_state.ultima_duracion = f"{resultado['tiempo_total']}s"
    st.session_state.objetos_detectados = resultado["n_dets"]
    st.session_state.historial.append(
        {
            "Nombre": nombre,
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Estado": "✅ Completado",
            "Detecciones": resultado["n_dets"],
        }
    )

    st.rerun()

else:
    # ── Col derecha: estado estático (idle o procesamiento anterior) ──────────
    with col_der:
        with st.container(border=True):
            st.markdown("##### ⚙️ Pipeline de procesamiento")
            res = st.session_state.resultado
            if res and res.get("tiempos_paso"):
                for step, t in zip(STEPS, res["tiempos_paso"]):
                    st.markdown(f"✅ **{step['nombre']}** — {t:.2f}s")
            else:
                for step in STEPS:
                    st.markdown(f"⚪ **{step['nombre']}**  \n*{step['descripcion']}*")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTADOS (persisten entre reruns via session_state)
# ══════════════════════════════════════════════════════════════════════════════
res = st.session_state.resultado
if res:
    st.divider()
    st.markdown("### 📊 Resultados")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Resumen", "🔍 Detecciones", "📈 Timeline", "📁 Archivo"]
    )

    # ── Tab 1: Resumen ────────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames analizados", f"{res['frames']:,}")
        c2.metric("Detecciones totales", res["n_dets"])
        c3.metric("Confianza promedio", f"{res['conf_avg']:.1%}")
        c4.metric("Tiempo total", f"{res['tiempo_total']}s")
        st.success(f"✅ Procesamiento completado exitosamente · `{res['nombre']}`")

    # ── Tab 2: Detecciones ────────────────────────────────────────────────────
    with tab2:
        df = pd.DataFrame(res["detecciones"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("**Confianza por detección**")
        st.line_chart(df[["Confianza"]], use_container_width=True)

    # ── Tab 3: Timeline del pipeline ─────────────────────────────────────────
    with tab3:
        df_tl = pd.DataFrame(
            {
                "Paso": [s["nombre"] for s in STEPS],
                "Duración (s)": res["tiempos_paso"],
            }
        )
        st.bar_chart(df_tl.set_index("Paso"), use_container_width=True)

    # ── Tab 4: Info del archivo ───────────────────────────────────────────────
    with tab4:
        st.markdown(f"**Clave del objeto en bucket:**")
        st.code(res["clave"])
        kb = res["tamaño_bytes"] / 1024
        mb_val = kb / 1024
        st.markdown(f"**Tamaño:** {kb:.1f} KB ({mb_val:.2f} MB)")

        if res["real"]:
            url = url_firmada(res["clave"])
            if url:
                st.markdown(f"**URL firmada (válida 1h):** [🔗 Abrir video en bucket]({url})")
            else:
                st.warning("No se pudo generar la URL firmada.")
        else:
            st.info("Subida simulada — no hay URL de bucket disponible.")

        if res.get("log"):
            with st.expander("📋 Log completo del proceso"):
                st.code("\n".join(res["log"]), language=None)

    # ── Botón de descarga del reporte ─────────────────────────────────────────
    reporte_json = json.dumps(res, indent=2, ensure_ascii=False, default=str)
    st.download_button(
        label="⬇️ Descargar reporte (JSON)",
        data=reporte_json,
        file_name=f"reporte_{res['nombre'].rsplit('.', 1)[0]}.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE LA SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.historial:
    n = len(st.session_state.historial)
    with st.expander(f"📂 Procesamientos anteriores en esta sesión ({n} video{'s' if n > 1 else ''})"):
        st.dataframe(
            pd.DataFrame(st.session_state.historial),
            use_container_width=True,
            hide_index=True,
        )
