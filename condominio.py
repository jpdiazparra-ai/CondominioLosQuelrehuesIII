"""
Condominio Los Queltehues III - Dashboard (Control de obligaciones + Ingresos + Costos)
"""

from __future__ import annotations

import html
import re
from typing import Optional

import pandas as pd


from condominio_data import PUBLICATION_ID, PUBLICATION_URL, SHEETS, URLS, load_snapshot, parse_money
from condominio_traceability import money as _money_clp

CACHE_VERSION = 4
INGRESOS_CSV_URL = URLS["ingresos"]
COSTOS_CSV_URL = URLS["costos"]
OBLIGACIONES_CSV_URL = URLS["obligaciones"]
PROPIETARIOS_CSV_URL = URLS["propietarios"]
TD23_CSV_URL = URLS["proyectos"]
MANTENCION_CSV_URL = URLS["mantencion"]
POR_PAGAR_CSV_URL = URLS["proveedores"]


def _normalize_colname(c: str) -> str:
    c = str(c).strip()
    c = c.replace("\n", " ").replace("\t", " ")
    c = re.sub(r"\s+", "_", c)
    c = c.lower()
    c = c.replace("n°", "n").replace("nº", "n")
    c = c.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    c = re.sub(r"[^a-z0-9_]", "", c)
    return c


def _pick_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _parse_monto_series(series: pd.Series) -> pd.Series:
    return parse_money(series)


def _df_to_pdf_bytes(df: pd.DataFrame, title: str, revision: str = "", captured_at: str = "") -> bytes:
    from control_obligaciones_pdf import build_table_pdf
    return build_table_pdf(df, title, revision, captured_at, PUBLICATION_URL)


def build_series_mensual_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    col_fecha = _pick_col(cols, ["fecha"])
    col_monto = next((c for c in cols if c.startswith("abono")), None)
    if not col_fecha or not col_monto:
        return pd.DataFrame(columns=["periodo", "ingresos"])

    base = df.copy()
    base["fecha_norm"] = pd.to_datetime(base[col_fecha], dayfirst=True, errors="coerce")
    base["monto_norm"] = _parse_monto_series(base[col_monto])
    base = base.dropna(subset=["fecha_norm", "monto_norm"])
    base["periodo"] = base["fecha_norm"].dt.to_period("M").astype(str)
    return (
        base.groupby("periodo", as_index=False)["monto_norm"]
        .sum()
        .rename(columns={"monto_norm": "ingresos"})
        .sort_values("periodo")
    )


def build_series_mensual_costos(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    col_fecha = _pick_col(cols, ["d", "fecha"])
    col_monto = _pick_col(cols, ["monto", "total", "importe", "valor"])
    if not col_fecha or not col_monto:
        return pd.DataFrame(columns=["periodo", "costos"])

    base = df.copy()
    if 'estado_pago' in base:
        base = base[base.estado_pago.eq('Pagado')]
    base["fecha_norm"] = pd.to_datetime(base[col_fecha], dayfirst=True, errors="coerce")
    base["monto_norm"] = _parse_monto_series(base[col_monto])
    base = base.dropna(subset=["fecha_norm", "monto_norm"])
    base["periodo"] = base["fecha_norm"].dt.to_period("M").astype(str)
    return (
        base.groupby("periodo", as_index=False)["monto_norm"]
        .sum()
        .rename(columns={"monto_norm": "costos"})
        .sort_values("periodo")
    )


def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="Condominio Los Queltehues III · Administración",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    def _sparkline_svg(color: str, points: str = "0,28 20,18 38,21 56,10 76,22 96,4 116,15 136,10 156,21") -> str:
        return (
            "<svg class='dash-sparkline' viewBox='0 0 156 34' preserveAspectRatio='none' aria-hidden='true'>"
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3' "
            "stroke-linecap='round' stroke-linejoin='round'/>"
            f"<g fill='{color}'>"
            + "".join(
                f"<circle cx='{p.split(',')[0]}' cy='{p.split(',')[1]}' r='2.6'/>"
                for p in points.split()
            )
            + "</g></svg>"
        )

    def _kpi_tile(icon: str, title: str, value: str, subtitle: str, tone: str, spark_points: str = "") -> str:
        color_map = {
            "green": "#00a86b",
            "red": "#ff2d2d",
            "violet": "#8a2be2",
            "teal": "#008b78",
            "blue": "#0f6bff",
        }
        color = color_map.get(tone, "#0f6bff")
        spark = ""  # No se dibujan tendencias sin una serie real de origen.
        return (
            f'<div class="dash-kpi-card dash-{tone}">'
            f'<div class="dash-icon">{icon}</div>'
            f'<div class="dash-kpi-title">{html.escape(title)}</div>'
            f'<div class="dash-kpi-value">{html.escape(value)}</div>'
            f'<div class="dash-kpi-sub">{html.escape(subtitle)}</div>'
            f"{spark}"
            "</div>"
        )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        :root {
          --dash-ink: #071326;
          --dash-muted: #64748b;
          --dash-line: #e5edf6;
          --dash-blue: #0f6bff;
          --dash-navy: #002b41;
          --dash-navy-2: #004b5f;
        }
        html, body, [class*="css"] {
          font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
          background:
            radial-gradient(circle at 78% 8%, rgba(15, 107, 255, 0.05), transparent 30%),
            linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
          color: var(--dash-ink);
        }
        .block-container {
          padding-top: 3.25rem;
          padding-left: 2rem;
          padding-right: 2rem;
          width: 100%;
          max-width: none;
        }
        [data-testid="stSidebar"] {
          background: rgba(255, 255, 255, 0.93);
          border-right: 1px solid #e6edf5;
          box-shadow: 10px 0 28px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
          min-width: 176px !important;
          max-width: 176px !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
          width: 0 !important;
          min-width: 0 !important;
          max-width: 0 !important;
          flex-basis: 0 !important;
          margin: 0 !important;
          padding: 0 !important;
          border: 0 !important;
          box-shadow: none;
          overflow: hidden;
        }
        [data-testid="stMain"] {
          flex: 1 1 0%;
          min-width: 0;
        }
        [data-testid="stSidebar"] > div:first-child {
          padding: 1.7rem 1rem 1rem 1rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] hr {
          display: none;
        }
        .side-logo {
          width: 58px;
          height: 58px;
          margin: 0 auto 28px auto;
          border-radius: 10px;
          display: grid;
          place-items: center;
          color: #eaffff;
          font-size: 30px;
          background: linear-gradient(145deg, #002b41 0%, #00475b 100%);
          box-shadow: 0 12px 24px rgba(0, 43, 65, 0.22);
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
          display: grid;
          gap: 14px;
        }
        [data-testid="stSidebar"] .stButton {
          margin-top: 14px;
        }
        [data-testid="stSidebar"] .stButton > button {
          width: 100%;
          min-height: 42px;
          border-radius: 8px;
          border: 1px solid #cfd9e6;
          background: #ffffff;
          color: #071326;
          box-shadow: 0 6px 14px rgba(15,23,42,0.08);
          font-size: 12px;
          font-weight: 900;
          line-height: 1.2;
          white-space: normal;
          padding: 0.45rem 0.65rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
          display: none;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
          min-height: 86px;
          border-radius: 10px;
          display: grid;
          place-items: center;
          align-content: center;
          gap: 8px;
          padding: 8px 4px;
          color: #4b596c;
          font-size: 13px;
          font-weight: 700;
          text-align: center;
          line-height: 1.2;
          transition: background 0.15s ease, color 0.15s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
          background: #f5f8fc;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
          background: #eef5ff;
          color: #0066ff;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
          display: none;
        }
        [data-testid="stSidebar"] [role="radiogroup"] p {
          font-size: 13px;
          font-weight: 700;
          line-height: 1.2;
          margin: 0;
          white-space: pre-line;
        }
        .side-about {
          position: static;
          margin-top: 24px;
          width: 100%;
          display: grid;
          place-items: center;
          gap: 6px;
          color: #526174;
          font-size: 13px;
          font-weight: 500;
        }
        .top-shell {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 24px;
          margin-bottom: 8px;
        }
        .general-page-head {
          display:flex;
          justify-content:space-between;
          align-items:center;
          gap:22px;
          margin:0 0 12px 0;
        }
        .general-title {
          color:#071326;
          font-size:20px;
          line-height:1.2;
          font-weight:900;
          margin:0 !important;
          padding:0 !important;
          overflow-wrap:anywhere;
        }
        .general-subtitle {
          color:#52647f;
          font-size:12px;
          line-height:1.3;
          font-weight:600;
          margin-top:5px;
        }
        .general-actions {
          display:flex;
          align-items:center;
          justify-content:flex-end;
          gap:22px;
          color:#071326;
          font-size:22px;
          line-height:1;
          white-space:nowrap;
        }
        .general-report-btn {
          height:36px;
          min-width:216px;
          border:1px solid #d8e0eb;
          border-radius:8px;
          background:#fff;
          box-shadow:0 6px 14px rgba(15,23,42,.06);
          display:flex;
          align-items:center;
          justify-content:center;
          gap:10px;
          color:#071326;
          font-size:12px;
          font-weight:900;
        }
        .general-icons {
          display:flex;
          align-items:center;
          gap:22px;
          font-size:22px;
          font-weight:900;
          color:#071326;
        }
        .app-title {
          margin: 0;
          color: var(--dash-ink);
          font-size: clamp(1.85rem, 3vw, 2.65rem);
          line-height: 1.12;
          letter-spacing: 0;
          font-weight: 900;
          overflow-wrap: anywhere;
        }
        .top-actions {
          display: flex;
          justify-content: flex-end;
          align-items: center;
          gap: 22px;
          color: #35445c;
          font-size: 25px;
          line-height: 1;
          margin-top: 0.1rem;
          min-width: 230px;
        }
        div.stDownloadButton > button {
          width: 100%;
          min-height: 48px;
          border-radius: 8px;
          border: 1px solid #d8e0eb;
          background: #ffffff;
          color: #101828;
          box-shadow: 0 8px 18px rgba(15,23,42,0.12);
          font-weight: 800;
          line-height: 1.2;
          padding: 0.55rem 1.55rem;
          white-space: normal;
        }
        div.stDownloadButton > button:hover {
          border-color: #c5cfdd;
          background: #fbfdff;
          color: #0f172a;
        }
        .block-container div.stButton > button {
          min-height: 44px;
          border-radius: 8px;
          border: 1px solid #d8e0eb;
          background: #ffffff;
          color: #101828;
          box-shadow: 0 6px 14px rgba(15,23,42,0.08);
          font-size: 13px;
          font-weight: 900;
          padding: 0 1.35rem;
        }
        .block-container div.stButton > button:hover {
          border-color:#c5cfdd;
          background:#fbfdff;
          color:#0f172a;
        }
        .dash-hero {
          min-height: 112px;
          display: grid;
          grid-template-columns: minmax(0, 1fr) 1px minmax(0, 1fr) 280px;
          gap: 22px;
          align-items: center;
          margin: 0 0 10px 0;
          padding: 16px 32px;
          border-radius: 12px;
          position: relative;
          overflow: hidden;
          color: #ffffff;
          background:
            radial-gradient(circle at 89% 18%, rgba(122, 231, 230, 0.16), transparent 18%),
            radial-gradient(circle at 84% 76%, rgba(122, 231, 230, 0.14), transparent 28%),
            linear-gradient(130deg, #002b41 0%, #003a50 45%, #00465a 100%);
          box-shadow: 0 13px 24px rgba(15, 23, 42, 0.18);
        }
        .dash-hero::before {
          content: "";
          position: absolute;
          inset: auto -6% -18% 16%;
          height: 105px;
          background: rgba(22, 92, 116, 0.25);
          border-radius: 55% 45% 0 0;
        }
        .dash-hero-eyebrow,
        .dash-hero-label {
          position: relative;
          z-index: 1;
          font-size: 0.62rem;
          letter-spacing: 0.1em;
          font-weight: 900;
          text-transform: uppercase;
        }
        .dash-hero-value {
          position: relative;
          z-index: 1;
          margin-top: 6px;
          font-size: clamp(1.65rem, 2.25vw, 2.25rem);
          line-height: 0.98;
          font-weight: 900;
          text-shadow: 0 3px 0 rgba(255, 255, 255, 0.12), 0 7px 18px rgba(0, 0, 0, 0.25);
        }
        .dash-hero-note,
        .dash-hero-sub {
          position: relative;
          z-index: 1;
          margin-top: 6px;
          color: #edf7fb;
          font-size: 0.72rem;
          font-weight: 700;
        }
        .dash-hero-rule {
          display:none;
        }
        .dash-hero-divider {
          position: relative;
          z-index: 1;
          width:1px;
          height:64px;
          background: rgba(223, 244, 248, 0.45);
          justify-self:center;
          margin:0;
        }
        .dash-hero-secondary-value {
          position: relative;
          z-index: 1;
          margin-top: 6px;
          color:#6ee0a4;
          font-size: clamp(1.55rem, 2.05vw, 2.05rem);
          font-weight: 900;
          line-height: 1;
        }
        .bank-art {
          position: relative;
          z-index: 1;
          height: 90px;
          display: grid;
          place-items: end center;
          opacity: 0.88;
        }
        .bank-art .bank {
          position: relative;
          width: 112px;
          height: 76px;
        }
        .bank-roof {
          position: absolute;
          left: 9px;
          top: 3px;
          width: 96px;
          height: 25px;
          background: #86dce1;
          clip-path: polygon(50% 0%, 100% 72%, 95% 100%, 5% 100%, 0% 72%);
          box-shadow: inset 0 -6px 0 rgba(14, 84, 98, 0.18);
        }
        .bank-row {
          position: absolute;
          left: 13px;
          top: 31px;
          width: 87px;
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 10px;
        }
        .bank-col {
          height: 41px;
          border-radius: 8px 8px 2px 2px;
          background: linear-gradient(180deg, #9be8e9, #5dc5cb);
          box-shadow: inset 7px 0 0 rgba(255,255,255,0.18);
        }
        .bank-base {
          position: absolute;
          left: 5px;
          bottom: 0;
          width: 106px;
          height: 11px;
          border-radius: 3px;
          background: #a8edf0;
          box-shadow: 0 -9px 0 #73d4d8;
        }
        .bank-coin {
          position: absolute;
          left: 38px;
          top: 51px;
          width: 34px;
          height: 34px;
          border-radius: 50%;
          display: grid;
          place-items: center;
          background: #ecffff;
          color: #0c6671;
          font-size: 18px;
          font-weight: 900;
          box-shadow: 0 0 0 6px rgba(129, 218, 224, 0.74);
        }
        .dash-kpi-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 9px;
          margin: 0 0 10px 0;
        }
        .dash-kpi-card {
          min-height: 92px;
          background: #ffffff;
          border: 1px solid #e2eaf4;
          border-top: 4px solid var(--tone);
          border-radius: 12px;
          padding: 9px 11px 6px 11px;
          box-shadow: 0 9px 20px rgba(15,23,42,0.08);
          overflow: hidden;
        }
        .dash-green { --tone: #00a86b; }
        .dash-red { --tone: #ff2d2d; }
        .dash-violet { --tone: #8a2be2; }
        .dash-teal { --tone: #008b78; }
        .dash-blue { --tone: #0f6bff; }
        .dash-icon {
          height: 30px;
          width: 30px;
          border-radius: 50%;
          display: grid;
          place-items: center;
          color: var(--tone);
          background: color-mix(in srgb, var(--tone) 15%, white);
          font-size: 14px;
          font-weight: 900;
          margin-bottom: 6px;
        }
        .dash-kpi-title {
          color: #3d485c;
          font-size: 0.48rem;
          font-weight: 900;
          text-transform: uppercase;
          line-height: 1.25;
        }
        .dash-kpi-value {
          margin-top: 4px;
          color: #071326;
          font-size: clamp(0.9rem, 1.08vw, 1.12rem);
          line-height: 1.05;
          font-weight: 900;
        }
        .dash-kpi-sub {
          margin-top: 5px;
          color: var(--tone);
          font-size: 0.52rem;
          font-weight: 700;
          min-height: 16px;
        }
        .dash-sparkline {
          width: 100%;
          height: 17px;
          margin-top: 5px;
        }
        @media (max-width: 1300px) {
          .dash-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .dash-hero { grid-template-columns: 1fr 210px; }
        }
        @media (max-width: 780px) {
          .block-container { padding-top: 2.4rem; padding-left: 1.1rem; padding-right: 1.1rem; }
          [data-testid="stSidebar"][aria-expanded="true"] { min-width: 138px !important; max-width: 138px !important; }
          .side-about { width: 138px; }
          .top-shell { display: block; }
          .top-actions { justify-content: flex-start; margin-top: 20px; }
          .dash-hero { grid-template-columns: 1fr; padding: 20px 22px; }
          .bank-art { display: none; }
          .dash-kpi-grid { grid-template-columns: 1fr; gap: 16px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("data_publication") != PUBLICATION_ID:
        st.session_state["data_publication"] = PUBLICATION_ID
        st.session_state["data_cache_version"] = CACHE_VERSION

    if "data_cache_version" not in st.session_state:
        st.session_state["data_cache_version"] = CACHE_VERSION

    with st.sidebar:
        st.markdown(
            """
            <div class="side-logo">🏢</div>
            """,
            unsafe_allow_html=True,
        )
        section_options = ["Control de obligaciones", "Ingresos V2.3", "Costos", "Fuentes y control"]
        if st.session_state.get("selected_section") not in section_options:
            st.session_state["selected_section"] = section_options[0]
        section_labels = {
            "Ingresos V2.3": "↗\nIngresos",
            "Fuentes y control": "✓\nFuentes y control",
            "Costos": "$\nCostos",
            "Control de obligaciones": "◎\nControl de obligaciones",
        }
        selected_section = st.radio(
            "Navegación",
            section_options,
            index=0,
            format_func=lambda option: section_labels[option],
            label_visibility="collapsed",
            key="selected_section",
        )
        if st.button("Actualizar data del URL", key="refresh_url_data"):
            st.session_state["data_cache_version"] += 1
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            """
            <div class="side-about">
              <div style="font-size:22px;line-height:1;">ⓘ</div>
              <div>Acerca de</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    data_cache_version = st.session_state["data_cache_version"]

    @st.cache_data(show_spinner=False, ttl=300)
    def _get_snapshot(publication: str, version: int):
        return load_snapshot()

    try:
        with st.spinner("Validando las fuentes y conciliando la nueva planilla…"):
            snapshot = _get_snapshot(PUBLICATION_ID, data_cache_version)
    except Exception as exc:
        st.error(f"No se pudo cargar la publicación activa: {exc}")
        st.info("Comprueba la planilla publicada y vuelve a pulsar Actualizar data del URL.")
        st.stop()

    def _load(url_value: str, cache_version: int, expected_cols=None) -> pd.DataFrame:
        if expected_cols in ({"parcela", "propietario"}, {"n_parcela", "nombre"}):
            return snapshot.frames["propietarios"].copy()
        key = next((key for key, value in URLS.items() if value == url_value and key != "propietarios"), None)
        if key is None or key not in snapshot.frames:
            raise ValueError("La fuente solicitada no pertenece a la publicación activa.")
        return snapshot.frames[key].copy()

    def _load_td23(url_value: str, cache_version: int) -> pd.DataFrame:
        return snapshot.frames["proyecto_config"].copy()

    def _load_mantencion(url_value: str, cache_version: int) -> pd.DataFrame:
        frame = snapshot.frames["mantencion"].copy()
        return pd.DataFrame({"parcela": pd.to_numeric(frame.parcela, errors="coerce"), "mantencion": parse_money(frame.monto)})

    def _canonical():
        annual = snapshot.annual.rename(columns={"Año": "anio", "Cargo por parcela": "gc_total"})
        table = snapshot.legacy_balances()
        return annual.copy(), table[["parcela", "pagado", "gc_total", "pendiente"]].copy()

    from condominio_traceability import render_source_bar, render_sources
    render_source_bar(st, snapshot)
    if selected_section == "Fuentes y control":
        render_sources(st, snapshot)
        return



    if selected_section in ("Ingresos V2.3", "Costos"):
        from condominio_traceability import render_transactions
        render_transactions(st, snapshot, "ingresos" if selected_section == "Ingresos V2.3" else "costos")

    if selected_section == "Control de obligaciones":
        from control_obligaciones import render_control

        render_control(st, _load, _load_td23, {
            "obligaciones": OBLIGACIONES_CSV_URL,
            "ingresos": INGRESOS_CSV_URL,
            "propietarios": PROPIETARIOS_CSV_URL,
            "mantencion": MANTENCION_CSV_URL,
            "proyectos": TD23_CSV_URL,
        }, data_cache_version, _parse_monto_series, prepared=snapshot)





if __name__ == "__main__":
    run_streamlit()
