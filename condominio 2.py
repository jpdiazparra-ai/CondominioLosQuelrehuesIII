"""
Condominio - Control de Cobranza GC
Análisis técnico-profesional desde Google Sheets (CSV publicado)

Requisitos:
    pip install pandas numpy matplotlib openpyxl jinja2 python-dateutil

Uso:
    python analisis_gc.py

Salida:
    /outputs/
        - quality_report.xlsx
        - kpis_resumen.xlsx
        - morosidad_por_parcela.xlsx (si hay tarifario o cargos)
        - anomalies.xlsx
        - report.html
"""

from __future__ import annotations

import os
import re
import math
import json
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# Opcional (si quieres gráficos)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================
# CONFIG
# ==========================

GOOGLE_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/pub?output=csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Si conoces tu rango real de parcelas, ajústalo aquí (ej. 17-36)
VALID_PARCELAS = set(range(17, 37))

# Umbrales de control
MIN_MONTO = -50_000_000  # CLP
MAX_MONTO =  50_000_000  # CLP
ANOMALY_ZSCORE = 4.0

# Si el CSV incluye cargos (GC) y pagos mezclados:
# - detecta "tipo" por columna tipo/descripcion, o por signo del monto.
# - si en tu data "costos" vienen negativos y "pagos" positivos, funciona bien.
DEFAULT_SIGN_RULE = True

# ==========================
# UTILIDADES
# ==========================

def _normalize_colname(c: str) -> str:
    c = c.strip()
    c = c.replace("\n", " ").replace("\t", " ")
    c = re.sub(r"\s+", "_", c)
    c = c.lower()
    # normalizaciones comunes
    c = c.replace("n°", "n").replace("nº", "n")
    c = c.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    return c

def _coerce_parcela(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"[^\d]", "", s)
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        return None

def _coerce_monto(x) -> Optional[float]:
    """
    Convierte CLP con formato chileno:
      "1.234.567" -> 1234567
      "1,234,567" -> 1234567
      "$ 12.345"  -> 12345
      "(12.345)"  -> -12345
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None

    neg = False
    # paréntesis contables
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    s = s.replace("$", "").replace("CLP", "").replace(" ", "")
    # elimina separadores miles y estandariza decimal
    # si hay coma y punto, asumimos . miles y , decimal (raro en CLP); para GC suele ser entero
    if "," in s and "." in s:
        # intentamos inferir
        # ej: "1.234,56" -> "1234.56"
        s = s.replace(".", "").replace(",", ".")
    else:
        # ej: "1.234.567" -> "1234567" o "1,234,567" -> "1234567"
        s = s.replace(".", "").replace(",", "")

    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ("", "-", ".", "-."):
        return None

    try:
        val = float(s)
        if neg:
            val = -abs(val)
        return val
    except Exception:
        return None

def _coerce_fecha(x) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    # intenta dd-mm-aaaa / dd/mm/aaaa y variantes
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    except Exception:
        return None

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, None) and not pd.isna(b) else np.nan

def build_google_csv_url(spreadsheet_url: str, sheet_gid: Optional[str]) -> str:
    """
    Construye URL CSV para una hoja específica usando gid.
    Si no se puede inferir, retorna el input original.
    """
    if not spreadsheet_url:
        return spreadsheet_url
    if "output=csv" in spreadsheet_url:
        return spreadsheet_url

    # Extrae ID del spreadsheet
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", spreadsheet_url)
    if not m:
        return spreadsheet_url
    sheet_id = m.group(1)
    gid = sheet_gid.strip() if sheet_gid else ""

    if gid:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

# ==========================
# DETECCIÓN DE COLUMNAS
# ==========================

@dataclass
class ColumnMap:
    fecha: str
    parcela: str
    monto: str
    medio: Optional[str] = None
    glosa: Optional[str] = None
    tipo: Optional[str] = None
    categoria: Optional[str] = None
    subcategoria: Optional[str] = None

def infer_columns(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)

    def pick(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in cols:
                return cand
        return None

    fecha = pick(["fecha", "date", "fecha_pago", "fecha_movimiento", "fec"])
    parcela = pick(["parcela", "lote", "unidad", "sitio", "n_parcela", "numero_parcela"])
    monto = pick(["monto", "importe", "valor", "amount", "clp", "pesos", "total", "abono", "ingreso", "ingresos"])

    if not (fecha and parcela and monto):
        raise ValueError(
            "No pude inferir columnas mínimas (fecha/parcela/monto). "
            f"Columnas encontradas: {cols}"
        )

    return ColumnMap(
        fecha=fecha,
        parcela=parcela,
        monto=monto,
        medio=pick(["medio", "forma_pago", "metodo", "canal"]),
        glosa=pick(["glosa", "descripcion", "detalle", "concepto", "observacion"]),
        tipo=pick(["tipo", "movimiento", "ingreso_egreso", "categoria_tipo"]),
        categoria=pick(["cc", "categoria", "rubro"]),
        subcategoria=pick(["ccc", "subcategoria", "sub_rubro"])
    )

# ==========================
# CARGA Y NORMALIZACIÓN
# ==========================

def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    # Si el CSV trae encabezados vacios ("Unnamed"), intentar inferir desde la primera fila con datos
    if all(str(c).strip().lower().startswith("unnamed") for c in df.columns):
        df_raw = pd.read_csv(url, header=None)
        header_row = None
        for i, row in df_raw.iterrows():
            non_null = row.notna().sum()
            if non_null >= 2:
                header_row = i
                break
        if header_row is not None:
            new_header = df_raw.iloc[header_row].astype(str).tolist()
            df = df_raw.iloc[header_row + 1 :].copy()
            df.columns = new_header

    # normaliza nombres y elimina columnas totalmente vacias
    df.columns = [_normalize_colname(c) for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df

def normalize(df_raw: pd.DataFrame, cmap_override: Optional[ColumnMap] = None) -> Tuple[pd.DataFrame, ColumnMap]:
    cmap = cmap_override or infer_columns(df_raw)

    df = df_raw.copy()

    # Normaliza tipos principales
    df["fecha_norm"] = df[cmap.fecha].apply(_coerce_fecha)
    df["parcela_norm"] = df[cmap.parcela].apply(_coerce_parcela)
    df["monto_norm"] = df[cmap.monto].apply(_coerce_monto)

    # Limpieza strings opcionales
    for opt in [cmap.medio, cmap.glosa, cmap.tipo, cmap.categoria, cmap.subcategoria]:
        if opt and opt in df.columns:
            df[opt] = df[opt].astype(str).str.strip()

    # Derivadas de fecha
    df["anio"] = df["fecha_norm"].dt.year
    df["mes"] = df["fecha_norm"].dt.month
    df["periodo"] = df["fecha_norm"].dt.to_period("M").astype(str)

    # Regla de signo / tipo
    if cmap.tipo and cmap.tipo in df.columns:
        # Normaliza el tipo si viene como texto
        df["tipo_norm"] = (
            df[cmap.tipo].astype(str).str.lower()
            .str.replace("á", "a").str.replace("é", "e").str.replace("í", "i").str.replace("ó", "o").str.replace("ú", "u")
        )
    else:
        df["tipo_norm"] = None

    if DEFAULT_SIGN_RULE:
        # Si no hay tipo, inferimos: monto>0 = pago/ingreso, monto<0 = cargo/egreso
        df["signo"] = np.sign(df["monto_norm"].fillna(0.0))
        df["es_pago_inferido"] = df["monto_norm"] > 0
        df["es_cargo_inferido"] = df["monto_norm"] < 0

    return df, cmap

# ==========================
# CALIDAD DE DATOS
# ==========================

def quality_report(df: pd.DataFrame) -> Dict:
    rep = {}

    rep["rows"] = int(len(df))
    rep["cols"] = int(df.shape[1])

    rep["missing_fecha"] = int(df["fecha_norm"].isna().sum())
    rep["missing_parcela"] = int(df["parcela_norm"].isna().sum())
    rep["missing_monto"] = int(df["monto_norm"].isna().sum())

    rep["invalid_parcela_fuera_rango"] = int(
        df["parcela_norm"].dropna().apply(lambda x: x not in VALID_PARCELAS).sum()
    )

    rep["monto_fuera_umbral"] = int(
        df["monto_norm"].dropna().apply(lambda x: (x < MIN_MONTO) or (x > MAX_MONTO)).sum()
    )

    # Duplicados (heurística): misma fecha, parcela y monto
    rep["dup_tripleta_fecha_parcela_monto"] = int(
        df.duplicated(subset=["fecha_norm", "parcela_norm", "monto_norm"]).sum()
    )

    # Cobertura temporal
    rep["min_fecha"] = str(df["fecha_norm"].min())
    rep["max_fecha"] = str(df["fecha_norm"].max())

    # Integridad por parcela
    per_parcela = df.groupby("parcela_norm")["monto_norm"].agg(["count", "sum"]).reset_index()
    rep["parcelas_con_movimientos"] = int(per_parcela["parcela_norm"].notna().sum())
    rep["parcelas_sin_movimientos_est"] = int(len(VALID_PARCELAS) - rep["parcelas_con_movimientos"])

    return rep

# ==========================
# KPIs & MOROSIDAD
# ==========================

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs mensuales: pagos, cargos, neto, #transacciones, ticket promedio, etc.
    Funciona incluso si hay mezcla pagos/cargos con signo.
    """
    base = df.dropna(subset=["fecha_norm", "parcela_norm", "monto_norm"]).copy()

    base["pagos"] = base["monto_norm"].where(base["monto_norm"] > 0, 0.0)
    base["cargos"] = base["monto_norm"].where(base["monto_norm"] < 0, 0.0)

    k = (
        base.groupby("periodo")
        .agg(
            n_tx=("monto_norm", "size"),
            pagos_total=("pagos", "sum"),
            cargos_total=("cargos", "sum"),
            neto=("monto_norm", "sum"),
            ticket_promedio=("monto_norm", "mean"),
            parcelas_activas=("parcela_norm", "nunique"),
        )
        .reset_index()
        .sort_values("periodo")
    )

    # métricas adicionales
    k["pagos_total"] = k["pagos_total"].round(0)
    k["cargos_total"] = k["cargos_total"].round(0)
    k["neto"] = k["neto"].round(0)
    k["ticket_promedio"] = k["ticket_promedio"].round(0)

    return k

def saldo_por_parcela(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saldo simple por parcela = sum(montos).
    Si tus cargos GC están como negativos y los pagos positivos, el saldo final:
      saldo > 0  => superávit a favor del condominio (raro)
      saldo < 0  => faltante / deuda neta (cargos > pagos)
    Ojo: depende de tu convención.
    """
    base = df.dropna(subset=["parcela_norm", "monto_norm"]).copy()
    s = (
        base.groupby("parcela_norm")
        .agg(
            n_mov=("monto_norm", "size"),
            total=("monto_norm", "sum"),
            pagos=("monto_norm", lambda x: float(x[x > 0].sum())),
            cargos=("monto_norm", lambda x: float(x[x < 0].sum())),
            ultima_fecha=("fecha_norm", "max"),
        )
        .reset_index()
        .sort_values("parcela_norm")
    )
    s["total"] = s["total"].round(0)
    s["pagos"] = s["pagos"].round(0)
    s["cargos"] = s["cargos"].round(0)

    # deuda_neta (asumiendo cargos negativos)
    s["deuda_neta_clp"] = (-s["total"]).round(0)

    return s

# ==========================
# ANOMALÍAS / FRAUD CHECKS
# ==========================

def anomalies(df: pd.DataFrame) -> pd.DataFrame:
    base = df.dropna(subset=["fecha_norm", "parcela_norm", "monto_norm"]).copy()

    # Z-score por monto absoluto (outliers)
    base["abs_monto"] = base["monto_norm"].abs()
    mu = base["abs_monto"].mean()
    sd = base["abs_monto"].std(ddof=0) if base["abs_monto"].std(ddof=0) else 0.0
    if sd > 0:
        base["z_abs_monto"] = (base["abs_monto"] - mu) / sd
    else:
        base["z_abs_monto"] = 0.0

    # Flags
    base["flag_parcela_invalida"] = ~base["parcela_norm"].isin(list(VALID_PARCELAS))
    base["flag_monto_fuera_umbral"] = (base["monto_norm"] < MIN_MONTO) | (base["monto_norm"] > MAX_MONTO)
    base["flag_outlier_z"] = base["z_abs_monto"].abs() >= ANOMALY_ZSCORE
    base["flag_fecha_nula"] = base["fecha_norm"].isna()

    # Duplicados sospechosos
    base["flag_dup_tripleta"] = base.duplicated(subset=["fecha_norm", "parcela_norm", "monto_norm"], keep=False)

    flags = ["flag_parcela_invalida", "flag_monto_fuera_umbral", "flag_outlier_z", "flag_dup_tripleta"]
    an = base.loc[base[flags].any(axis=1)].copy()

    # Orden para revisión
    an = an.sort_values(["flag_monto_fuera_umbral", "flag_outlier_z", "flag_dup_tripleta"], ascending=False)

    # Limpia columnas de apoyo
    keep_cols = ["fecha_norm", "parcela_norm", "monto_norm", "abs_monto", "z_abs_monto"] + flags
    # Agrega glosa/medio si existiesen en df original
    for c in df.columns:
        if c in ("glosa", "descripcion", "detalle", "concepto", "observacion", "medio", "forma_pago", "metodo", "canal"):
            if c in an.columns and c not in keep_cols:
                keep_cols.append(c)

    return an[keep_cols]

# ==========================
# EXPORTES
# ==========================

def export_excel(dfs: Dict[str, pd.DataFrame], filepath: str) -> None:
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for name, d in dfs.items():
            # limita nombre de hoja a 31 chars
            sheet = name[:31]
            d.to_excel(writer, sheet_name=sheet, index=False)

def export_html_report(quality: Dict, kpis: pd.DataFrame, saldo: pd.DataFrame, anom: pd.DataFrame, filepath: str) -> None:
    # Reporte HTML minimalista y útil
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Reporte GC</title></head><body>")
    html.append("<h1>Reporte Técnico - Gastos Comunes</h1>")

    html.append("<h2>1) Calidad de datos</h2>")
    html.append("<pre>" + json.dumps(quality, indent=2, ensure_ascii=False) + "</pre>")

    html.append("<h2>2) KPIs mensuales</h2>")
    html.append(kpis.tail(24).to_html(index=False))

    html.append("<h2>3) Saldo por parcela</h2>")
    html.append(saldo.to_html(index=False))

    html.append("<h2>4) Anomalías / casos a revisar</h2>")
    html.append(anom.head(200).to_html(index=False))

    html.append("</body></html>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

# ==========================
# STREAMLIT
# ==========================

def _is_streamlit_run() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def run_streamlit():
    global GOOGLE_CSV_URL, VALID_PARCELAS, MIN_MONTO, MAX_MONTO, ANOMALY_ZSCORE
    try:
        import streamlit as st
    except Exception as e:
        raise SystemExit("Streamlit no está instalado. Ejecuta: pip install streamlit") from e

    st.set_page_config(page_title="Condominio - Control de Cobranza GC", layout="wide")
    st.title("Condominio - Control de Cobranza GC")
    st.caption("Análisis técnico-profesional desde Google Sheets (CSV publicado)")
    st.info("App cargada. Configura la fuente en el panel izquierdo y presiona 'Ejecutar análisis'.")
    st.caption("Para el gid: abre la pestaña Ingresos V2.3 y copia el número que aparece en la URL después de 'gid='.")

    try:
        with st.sidebar:
            st.header("Configuración")
            st.markdown("Pega la URL del Google Sheet y el gid de la pestaña (Ingresos V2.3).")
            sheet_url = st.text_input("Google Sheets URL (no CSV)", value="")
            sheet_gid = st.text_input("Sheet gid (pestaña)", value="")
            csv_url = st.text_input("CSV URL (si ya lo tienes)", value=GOOGLE_CSV_URL)
            parc_min = st.number_input("Parcela mínima", min_value=1, value=min(VALID_PARCELAS))
            parc_max = st.number_input("Parcela máxima", min_value=1, value=max(VALID_PARCELAS))
            min_monto = st.number_input("Monto mínimo (CLP)", value=int(MIN_MONTO))
            max_monto = st.number_input("Monto máximo (CLP)", value=int(MAX_MONTO))
            zscore = st.number_input("Z-Score anomalía", min_value=1.0, value=float(ANOMALY_ZSCORE))
            ejecutar = st.button("Ejecutar análisis", type="primary")
            if sheet_url:
                st.caption("CSV generado desde URL + gid:")
                st.code(build_google_csv_url(sheet_url, sheet_gid))
    except Exception as e:
        st.error("Error construyendo el panel de configuración.")
        st.exception(e)
        return

    if parc_min > parc_max:
        st.error("Rango de parcelas inválido: la mínima no puede ser mayor que la máxima.")
        return

    # Actualiza globals para reutilizar la lógica existente
    effective_url = build_google_csv_url(sheet_url, sheet_gid) if sheet_url else csv_url
    GOOGLE_CSV_URL = effective_url
    VALID_PARCELAS = set(range(int(parc_min), int(parc_max) + 1))
    MIN_MONTO = float(min_monto)
    MAX_MONTO = float(max_monto)
    ANOMALY_ZSCORE = float(zscore)

    @st.cache_data(show_spinner=False)
    def _run_pipeline(url_value: str, cmap_override: Optional[Dict[str, str]] = None):
        df_raw = load_data(url_value)
        cmap = ColumnMap(**cmap_override) if cmap_override else None
        df, _ = normalize(df_raw, cmap_override=cmap)
        q = quality_report(df)
        k = compute_kpis(df)
        s = saldo_por_parcela(df)
        a = anomalies(df)
        return df, q, k, s, a

    def _render_results(df, q, k, s, a):
        st.success("Listo. Resultados generados.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Calidad de datos")
            st.json(q)
        with c2:
            st.subheader("Vista previa (normalizada)")
            st.dataframe(df.head(50), use_container_width=True)

        st.subheader("KPIs mensuales")
        st.dataframe(k, use_container_width=True)

        st.subheader("Saldo por parcela")
        st.dataframe(s, use_container_width=True)

        st.subheader("Anomalías / casos a revisar")
        st.dataframe(a, use_container_width=True)

        if st.button("Exportar reportes a /outputs"):
            export_excel(
                {
                    "data_normalizada": df,
                    "kpis_mensuales": k,
                    "saldo_por_parcela": s,
                    "anomalies": a,
                },
                os.path.join(OUTPUT_DIR, "kpis_resumen.xlsx")
            )
            export_excel(
                {
                    "quality_dict": pd.DataFrame([q]),
                },
                os.path.join(OUTPUT_DIR, "quality_report.xlsx")
            )
            export_html_report(
                quality=q,
                kpis=k,
                saldo=s,
                anom=a,
                filepath=os.path.join(OUTPUT_DIR, "report.html")
            )
            st.success(f"Archivos generados en: {os.path.abspath(OUTPUT_DIR)}")

    if "needs_mapping" not in st.session_state:
        st.session_state["needs_mapping"] = False

    if ejecutar:
        with st.spinner("Procesando..."):
            try:
                df, q, k, s, a = _run_pipeline(GOOGLE_CSV_URL)
                st.session_state["needs_mapping"] = False
                _render_results(df, q, k, s, a)
            except ValueError as e:
                if "No pude inferir columnas mínimas" in str(e):
                    st.session_state["needs_mapping"] = True
                    st.session_state["df_raw"] = load_data(GOOGLE_CSV_URL)
                    st.error(
                        "No se pudieron inferir columnas. "
                        "Selecciona manualmente las columnas mínimas."
                    )
                else:
                    raise
    else:
        st.warning("Configura el CSV y presiona 'Ejecutar análisis' para ver resultados.")

    if st.session_state.get("needs_mapping") and "df_raw" in st.session_state:
        df_raw = st.session_state["df_raw"]
        cols = list(df_raw.columns)
        st.subheader("Mapeo manual de columnas")
        col_fecha = st.selectbox("Columna fecha", cols, key="map_fecha")
        col_parcela = st.selectbox("Columna parcela", cols, key="map_parcela")
        col_monto = st.selectbox("Columna monto", cols, key="map_monto")

        optional_cols = ["(ninguna)"] + cols
        col_medio = st.selectbox("Columna medio (opcional)", optional_cols, key="map_medio")
        col_glosa = st.selectbox("Columna glosa/descripcion (opcional)", optional_cols, key="map_glosa")
        col_tipo = st.selectbox("Columna tipo (opcional)", optional_cols, key="map_tipo")
        col_categoria = st.selectbox("Columna categoria (opcional)", optional_cols, key="map_categoria")
        col_subcat = st.selectbox("Columna subcategoria (opcional)", optional_cols, key="map_subcategoria")

        if st.button("Procesar con mapeo"):
            cmap_override = {
                "fecha": col_fecha,
                "parcela": col_parcela,
                "monto": col_monto,
                "medio": None if col_medio == "(ninguna)" else col_medio,
                "glosa": None if col_glosa == "(ninguna)" else col_glosa,
                "tipo": None if col_tipo == "(ninguna)" else col_tipo,
                "categoria": None if col_categoria == "(ninguna)" else col_categoria,
                "subcategoria": None if col_subcat == "(ninguna)" else col_subcat,
            }
            with st.spinner("Procesando con mapeo..."):
                df, q, k, s, a = _run_pipeline(GOOGLE_CSV_URL, cmap_override)
            st.session_state["needs_mapping"] = False
            _render_results(df, q, k, s, a)

# ==========================
# MAIN
# ==========================

def main():
    print("Cargando CSV desde Google Sheets...")
    df_raw = load_data(GOOGLE_CSV_URL)

    print("Normalizando columnas/valores...")
    try:
        df, cmap = normalize(df_raw)
    except ValueError as e:
        if "No pude inferir columnas mínimas" in str(e):
            print("No se pudieron inferir columnas mínimas. Verifica que el CSV sea transaccional.")
            print("Sugerencia: exporta la hoja con movimientos (fecha/parcela/monto).")
            return
        raise

    print("Generando reporte de calidad...")
    q = quality_report(df)

    print("Calculando KPIs...")
    k = compute_kpis(df)

    print("Calculando saldo por parcela...")
    s = saldo_por_parcela(df)

    print("Detectando anomalías...")
    a = anomalies(df)

    # Exportes
    print("Exportando reportes...")
    export_excel(
        {
            "data_normalizada": df,
            "kpis_mensuales": k,
            "saldo_por_parcela": s,
            "anomalies": a,
        },
        os.path.join(OUTPUT_DIR, "kpis_resumen.xlsx")
    )

    export_excel(
        {
            "quality_dict": pd.DataFrame([q]),
        },
        os.path.join(OUTPUT_DIR, "quality_report.xlsx")
    )

    export_html_report(
        quality=q,
        kpis=k,
        saldo=s,
        anom=a,
        filepath=os.path.join(OUTPUT_DIR, "report.html")
    )

    # Gráfico simple (opcional)
    if plt is not None:
        try:
            k2 = k.copy()
            k2["periodo_dt"] = pd.to_datetime(k2["periodo"] + "-01")
            plt.figure()
            plt.plot(k2["periodo_dt"], k2["pagos_total"])
            plt.title("Pagos totales por mes")
            plt.xlabel("Periodo")
            plt.ylabel("CLP")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "pagos_mensuales.png"), dpi=150)
        except Exception as e:
            print("No se pudo generar gráfico:", e)
    else:
        print("matplotlib no está instalado; se omite el gráfico.")

    print("\nOK. Archivos generados en:", os.path.abspath(OUTPUT_DIR))
    print("Column mapping inferido:", cmap)

if __name__ == "__main__":
    if _is_streamlit_run():
        run_streamlit()
    else:
        main()
