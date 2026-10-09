"""
Pantera Lab Video · Pipeline de Procesamiento
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
    page_title="Pantera Lab Video · Pipeline",
    page_icon="🐆",
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
.stCode > div { max-height: 230px; overflow-y: auto; }
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
    "video_pendiente": None,  # nombre de video en Supabase esperando ser procesado
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS S3
# ══════════════════════════════════════════════════════════════════════════════

def _leer_secrets_s3():
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


_CFG_S3 = _leer_secrets_s3()
BUCKET_DISPONIBLE = _CFG_S3 is not None


def _cliente_s3():
    if _CFG_S3 is None:
        return None, None
    try:
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=_CFG_S3["endpoint_url"],
            aws_access_key_id=_CFG_S3["aws_access_key_id"],
            aws_secret_access_key=_CFG_S3["aws_secret_access_key"],
            region_name=_CFG_S3["region_name"],
        )
        return client, _CFG_S3["bucket"]
    except Exception:
        return None, None


def subir_archivo(data: bytes, nombre: str, content_type: str):
    """
    Sube al bucket S3.
    Retorna: (True, clave) OK · (False, clave) error boto3 · (None, clave) sin config
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clave = f"uploads/{ts}_{nombre}"
    client, bucket = _cliente_s3()
    if client is None:
        return None, clave
    try:
        client.upload_fileobj(
            io.BytesIO(data), bucket, clave,
            ExtraArgs={"ContentType": content_type},
        )
        return True, clave
    except Exception:
        return False, clave


def url_firmada(clave: str):
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
# HELPERS SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

SUPABASE_BUCKET = "videos"  # nombre del bucket en Supabase Storage


def _leer_secrets_supabase():
    try:
        return st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
    except Exception:
        return None, None


_SB_URL, _SB_KEY = _leer_secrets_supabase()
SUPABASE_DISPONIBLE = _SB_URL is not None


def _sb():
    """Retorna cliente Supabase o None."""
    if not SUPABASE_DISPONIBLE:
        return None
    try:
        from supabase import create_client
        return create_client(_SB_URL, _SB_KEY)
    except Exception:
        return None


def sb_listar_videos():
    """Lista videos en el bucket de Supabase. Retorna (lista, error_str|None)."""
    client = _sb()
    if client is None:
        return [], "No se pudo crear el cliente Supabase — verificá SUPABASE_URL y SUPABASE_KEY."
    try:
        archivos = client.storage.from_(SUPABASE_BUCKET).list()
        videos = [
            f for f in archivos
            if isinstance(f, dict) and f.get("name", "").split(".")[-1].lower() in CONTENT_TYPES
        ]
        return videos, None
    except Exception as e:
        return [], str(e)


def sb_descargar_video(nombre: str):
    """Descarga un video de Supabase. Retorna bytes o None."""
    client = _sb()
    if client is None:
        return None
    try:
        return client.storage.from_(SUPABASE_BUCKET).download(nombre)
    except Exception:
        return None


def sb_url_video(nombre: str):
    """URL firmada de Supabase (1 hora) para reproducir un video."""
    client = _sb()
    if client is None:
        return None
    try:
        res = client.storage.from_(SUPABASE_BUCKET).create_signed_url(nombre, 3600)
        if isinstance(res, dict):
            return (
                res.get("signedURL")
                or res.get("signed_url")
                or (res.get("data") or {}).get("signedUrl")
            )
        return None
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
        rows.append({
            "Timestamp video": f"{mm:02d}:{ss:02d}.{ms}",
            "Clase": rng.choice(CLASES),
            "Confianza": round(rng.uniform(0.62, 0.99), 3),
            "Bounding box": f"[{x1}, {y1}, {x2}, {y2}]",
        })
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
# MONITOREO DE SUBIDA (métricas animadas)
# ══════════════════════════════════════════════════════════════════════════════

def _animar_subida(nombre: str, tamaño_bytes: int):
    """Muestra métricas animadas de subida con rangos realistas."""
    rng = random.Random(hash(nombre) ^ 0xABCD)
    total_mb = max(tamaño_bytes / 1024 / 1024, 0.01)

    with st.container(border=True):
        st.markdown("##### 📡 Monitoreo de subida")
        pb = st.progress(0, text="Preparando transferencia...")
        cols = st.columns(4)
        ph_vel   = cols[0].empty()
        ph_pico  = cols[1].empty()
        ph_trans = cols[2].empty()
        ph_eta   = cols[3].empty()

        pasos = 22
        historial_vel = []
        pico = 0.0

        for i in range(pasos + 1):
            pct = i / pasos

            # velocidad con variación en rango [6.2, 48.5] MB/s
            vel_base = rng.uniform(18.0, 38.0)
            ruido    = rng.uniform(-12.0, 10.0)
            vel      = max(6.2, min(48.5, vel_base + ruido))
            historial_vel.append(vel)
            pico = max(pico, vel)

            vel_prom   = sum(historial_vel) / len(historial_vel)
            transferido = total_mb * pct
            eta         = (total_mb - transferido) / vel if vel > 0 and pct < 1 else 0.0

            delta_vel = f"{vel - historial_vel[-2]:+.1f}" if len(historial_vel) > 1 else None

            texto_barra = f"Subiendo... {pct:.0%}  —  {vel:.1f} MB/s" if pct < 1 else "✅ Transferencia completada"
            pb.progress(pct, text=texto_barra)

            ph_vel.metric("⚡ Velocidad actual", f"{vel:.1f} MB/s", delta=delta_vel)
            ph_pico.metric("🏔 Pico / Promedio", f"{pico:.1f} / {vel_prom:.1f} MB/s")
            ph_trans.metric("📦 Transferido", f"{transferido:.2f} / {total_mb:.2f} MB")
            ph_eta.metric("⏳ ETA", f"{eta:.1f}s" if pct < 1 else "—")

            time.sleep(0.11)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_pipeline(nombre, tamaño, clave, real, vel, step_phs, log_ph):
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


def _guardar_resultado(resultado, nombre):
    st.session_state.resultado = resultado
    st.session_state.videos_procesados += 1
    st.session_state.ultimo_estado = "✅ Completado"
    st.session_state.ultima_duracion = f"{resultado['tiempo_total']}s"
    st.session_state.objetos_detectados = resultado["n_dets"]
    st.session_state.historial.append({
        "Nombre": nombre,
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Estado": "✅ Completado",
        "Detecciones": resultado["n_dets"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🐆 Pantera Lab Video")
    st.markdown("**Pipeline de Procesamiento**")
    st.divider()

    st.markdown(
        f"**Bucket S3:** {'🟢 Conectado' if BUCKET_DISPONIBLE else '🔴 No configurado'}"
    )
    st.markdown(
        f"**Supabase:** {'🟢 Conectado' if SUPABASE_DISPONIBLE else '🔴 No configurado'}"
    )

    st.divider()

    modo_demo = st.toggle(
        "🧪 Modo demo",
        value=not BUCKET_DISPONIBLE,
        help="El pipeline corre aunque la subida al bucket falle.",
    )
    st.caption("En modo demo la subida es simulada.")

    velocidad = st.slider(
        "⏱ Segundos por paso (base)",
        min_value=0.3, max_value=4.0, value=1.5, step=0.1,
    )

    st.divider()
    if st.button("🔄 Reiniciar sesión", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CABECERA
# ══════════════════════════════════════════════════════════════════════════════
st.title("🐆 Pantera Lab Video · Pipeline de Procesamiento")

m1, m2, m3, m4 = st.columns(4)
m1.metric("📹 Videos procesados", st.session_state.videos_procesados)
m2.metric("📊 Último estado", st.session_state.ultimo_estado)
m3.metric("⏱ Duración del último proceso", st.session_state.ultima_duracion)
m4.metric("🔍 Objetos detectados", st.session_state.objetos_detectados)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
col_izq, col_der = st.columns([1, 1.2])

with col_izq:
    with st.container(border=True):
        st.markdown("##### 📤 Entrada manual")
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
            st.info("Subí un video manualmente, o procesá uno de los **Videos entrantes** (abajo).")
            procesar = False

    if archivo:
        st.markdown("**Vista previa**")
        st.video(archivo)


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN — unifica flujo manual (procesar) y desde Supabase (video_pendiente)
# ══════════════════════════════════════════════════════════════════════════════
_desde_supabase = bool(st.session_state.video_pendiente)
_ejecutar = (procesar and archivo) or _desde_supabase

if _ejecutar:
    # Armar col_der con pipeline activo
    with col_der:
        with st.container(border=True):
            st.markdown("##### ⚙️ Pipeline en ejecución")
            step_phs = [st.empty() for _ in STEPS]
        with st.container(border=True):
            st.markdown("##### 📋 Log en vivo")
            log_ph = st.empty()

    for i, step in enumerate(STEPS):
        step_phs[i].markdown(f"⚪ **{step['nombre']}**  \n*{step['descripcion']}*")
    log_ph.code("", language=None)

    # Obtener datos del video según origen
    if _desde_supabase:
        nombre_sb = st.session_state.video_pendiente
        st.session_state.video_pendiente = None

        with col_izq:
            with st.spinner(f"Descargando `{nombre_sb}` de Supabase..."):
                data = sb_descargar_video(nombre_sb)

        if data is None:
            st.error(f"❌ No se pudo descargar `{nombre_sb}` de Supabase.")
            st.stop()

        nombre = nombre_sb
        with col_izq:
            st.markdown("**Vista previa**")
            st.video(data)
    else:
        data = archivo.getvalue()
        nombre = archivo.name

    ext = nombre.rsplit(".", 1)[-1].lower()
    ct = CONTENT_TYPES.get(ext, "video/mp4")

    # Métricas animadas de subida
    with col_izq:
        _animar_subida(nombre, len(data))

    # Subir al bucket S3
    ok, clave = subir_archivo(data, nombre, ct)
    if ok is None:
        st.warning("ℹ️ Bucket S3 no configurado — procesando en modo demo.")
        subida_real = False
    elif ok is False:
        if not modo_demo:
            st.error("❌ Error al subir al bucket. Activá **Modo demo** para continuar.")
            st.stop()
        else:
            st.warning("⚠️ Subida simulada (error en bucket) — modo demo activo.")
            subida_real = False
    else:
        st.success(f"✅ Archivo subido al bucket S3 → `{clave}`")
        subida_real = True

    # Correr pipeline
    resultado = ejecutar_pipeline(
        nombre=nombre, tamaño=len(data), clave=clave,
        real=subida_real, vel=velocidad,
        step_phs=step_phs, log_ph=log_ph,
    )

    _guardar_resultado(resultado, nombre)
    st.rerun()

else:
    # Col derecha: estado estático
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
# RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
res = st.session_state.resultado
if res:
    st.divider()
    st.markdown("### 📊 Resultados")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Resumen", "🔍 Detecciones", "📈 Timeline", "📁 Archivo"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames analizados", f"{res['frames']:,}")
        c2.metric("Detecciones totales", res["n_dets"])
        c3.metric("Confianza promedio", f"{res['conf_avg']:.1%}")
        c4.metric("Tiempo total", f"{res['tiempo_total']}s")
        st.success(f"✅ Procesamiento completado · `{res['nombre']}`")

    with tab2:
        df = pd.DataFrame(res["detecciones"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("**Confianza por detección**")
        st.line_chart(df[["Confianza"]], use_container_width=True)

    with tab3:
        df_tl = pd.DataFrame({
            "Paso": [s["nombre"] for s in STEPS],
            "Duración (s)": res["tiempos_paso"],
        })
        st.bar_chart(df_tl.set_index("Paso"), use_container_width=True)

    with tab4:
        st.markdown("**Clave del objeto en bucket:**")
        st.code(res["clave"])
        kb = res["tamaño_bytes"] / 1024
        st.markdown(f"**Tamaño:** {kb:.1f} KB ({kb/1024:.2f} MB)")
        if res["real"]:
            url = url_firmada(res["clave"])
            if url:
                st.markdown(f"**URL firmada (1h):** [🔗 Abrir video]({url})")
            else:
                st.warning("No se pudo generar la URL firmada.")
        else:
            st.info("Subida simulada — no hay URL de bucket disponible.")
        if res.get("log"):
            with st.expander("📋 Log completo"):
                st.code("\n".join(res["log"]), language=None)

    reporte_json = json.dumps(res, indent=2, ensure_ascii=False, default=str)
    st.download_button(
        "⬇️ Descargar reporte (JSON)",
        data=reporte_json,
        file_name=f"reporte_{res['nombre'].rsplit('.', 1)[0]}.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.historial:
    n = len(st.session_state.historial)
    with st.expander(f"📂 Procesamientos anteriores en esta sesión ({n})"):
        st.dataframe(
            pd.DataFrame(st.session_state.historial),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VIDEOS ENTRANTES — se refresca automáticamente cada 15 segundos
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=15)
def seccion_videos_entrantes():
    st.divider()
    c_titulo, c_badge = st.columns([5, 1])
    c_titulo.markdown("### 📥 Videos entrantes")
    c_badge.caption("↻ cada 15 s")

    if not SUPABASE_DISPONIBLE:
        st.info(
            "Configurá `SUPABASE_URL` y `SUPABASE_KEY` en Secrets para activar "
            "la recepción de videos del equipo externo."
        )
        return

    videos, sb_error = sb_listar_videos()

    if sb_error:
        st.error(f"❌ Error al leer Supabase: `{sb_error}`")
        return

    if not videos:
        st.info("⏳ Sin videos entrantes todavía. El equipo externo puede subir videos al bucket `videos` de Supabase.")
        return

    # Ordenar más reciente primero
    videos = sorted(videos, key=lambda v: v.get("created_at") or "", reverse=True)

    ya_procesados = {h["Nombre"] for h in st.session_state.historial}
    st.caption(f"{len(videos)} video{'s' if len(videos) > 1 else ''} disponible{'s' if len(videos) > 1 else ''}")

    for i in range(0, len(videos), 2):
        cols = st.columns(2)
        for col, v in zip(cols, videos[i : i + 2]):
            nombre = v["name"]
            size_mb = (v.get("metadata") or {}).get("size", 0) / 1024 / 1024
            fecha = (v.get("created_at") or "")[:10]
            procesado = nombre in ya_procesados

            with col:
                with st.container(border=True):
                    # Header: nombre + badge
                    h1, h2 = st.columns([3, 1])
                    h1.markdown(f"**{nombre}**")
                    h2.markdown(
                        f"<div style='text-align:right'>{'🟢' if procesado else '🟡'}</div>",
                        unsafe_allow_html=True,
                    )

                    # Player inline
                    url = sb_url_video(nombre)
                    if url:
                        st.video(url)
                    else:
                        st.caption("⚠️ Preview no disponible.")

                    # Metadata
                    meta = []
                    if size_mb > 0:
                        meta.append(f"📦 {size_mb:.1f} MB")
                    if fecha:
                        meta.append(f"📅 {fecha}")
                    if meta:
                        st.caption(" · ".join(meta))

                    # Acción
                    if not procesado:
                        if st.button(
                            "🚀 Procesar este video",
                            key=f"sb_{nombre}",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.video_pendiente = nombre
                            st.rerun()
                    else:
                        st.button(
                            "✅ Ya procesado",
                            key=f"sb_{nombre}",
                            disabled=True,
                            use_container_width=True,
                        )


seccion_videos_entrantes()
