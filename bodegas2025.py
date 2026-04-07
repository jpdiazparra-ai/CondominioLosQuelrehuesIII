import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import base64
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

# =========================
# Configuración base
# =========================
st.set_page_config(page_title="Análisis Financiero - OK-DTA V2", layout="wide")

# --- Logo -> data URI ---
def data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""  # si no está, ocultamos el <img>
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lower().replace(".", "") or "png"
    return f"data:image/{ext};base64,{b64}"

LOGO_URI = data_uri("Logo_balmaceda.png")  # el archivo está en la misma carpeta del .py
HERO_URI = data_uri("IMG_7331.jpeg")

# =======================
# HERO HEADER
# =======================
st.markdown("""
<style>
/* Quitar espacio superior global */
main {
    padding-top: 0rem !important;
}
.block-container {
    padding-top: 0.8rem !important;
}
.dashboard-hero {
    position: relative;
    overflow: hidden;
    min-height: 305px;
    margin: 0 0 12px 0;
    border-radius: 34px;
    border: 1px solid #d8e1ed;
    background:
        linear-gradient(180deg, rgba(248,250,252,0.30) 0%, rgba(248,250,252,0.58) 100%),
        linear-gradient(90deg, rgba(255,255,255,0.88) 0%, rgba(255,255,255,0.70) 34%, rgba(255,255,255,0.18) 100%),
        url('""" + HERO_URI + """');
    background-size: cover;
    background-position: center 58%;
    box-shadow: 0 22px 44px rgba(15, 23, 42, 0.10);
}
.dashboard-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.08) 45%, rgba(15,23,42,0.10) 100%);
    pointer-events: none;
}
.dashboard-hero-inner {
    position: relative;
    z-index: 1;
    max-width: 1280px;
    padding: 40px 46px 34px 46px;
}
.dashboard-hero-badge {
    width: 78px;
    height: 78px;
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.52);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(203,213,225,0.95);
    box-shadow: 0 10px 20px rgba(148,163,184,0.18);
}
.dashboard-hero-badge img {
    width: 54px;
    height: 54px;
    object-fit: contain;
}
.dashboard-hero-title {
    max-width: 1400px;
    margin: 18px 0 10px 0;
    font-size: clamp(2.5rem, 4vw, 4.45rem);
    line-height: 0.98;
    letter-spacing: -0.06em;
    font-weight: 900;
    color: #0f172a;
}
.dashboard-hero-subtitle {
    max-width: 1500px;
    font-size: clamp(1rem, 1.5vw, 1.18rem);
    line-height: 1.55;
    color: rgba(51,65,85,0.86);
    font-weight: 500;
}
.dashboard-hero-meta {
    margin-top: 14px;
    font-size: 0.92rem;
    color: rgba(71,85,105,0.88);
    font-weight: 600;
}
@media (max-width: 960px) {
    .dashboard-hero {
        min-height: 260px;
        border-radius: 26px;
    }
    .dashboard-hero-inner {
        padding: 28px 24px 24px 24px;
    }
    .dashboard-hero-badge {
        width: 66px;
        height: 66px;
        border-radius: 20px;
    }
    .dashboard-hero-badge img {
        width: 46px;
        height: 46px;
    }
}
@media (max-width: 640px) {
    .dashboard-hero {
        min-height: 230px;
    }
    .dashboard-hero-title {
        line-height: 1.02;
    }
}
</style>
<div class="dashboard-hero">
    <div class="dashboard-hero-inner">
        <div class="dashboard-hero-badge">
            <img src='""" + LOGO_URI + """' alt='Logo'>
        </div>
        <div class="dashboard-hero-title">Arquitectura Financiera de Bodegas</div>
        <div class="dashboard-hero-subtitle">
            Panel interactivo para analizar caja, cobranzas, canon, egresos y liquidación eléctrica del proyecto,
            con foco en lectura ejecutiva y seguimiento financiero consolidado.
        </div>
        <div class="dashboard-hero-meta">Fuente: Google Sheets (CSV) · Agrupaciones dinámicas y visualizaciones interactivas</div>
    </div>
</div>
""", unsafe_allow_html=True)



# =========================
# Carga de datos
# =========================
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSuoX_V5rYls-pBu7F3_VP2APS3FL7-eYbn9uDWUGJQZbxNfQTm9gRlyDlE69wWJjsDQpDzi2lt31Ak/pub?gid=1154929321&single=true&output=csv"

st.sidebar.header("⚙️ Controles")
if st.sidebar.button("🔄 Actualizar datos (limpiar caché)"):
    st.cache_data.clear()

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Normalización y tipos
    df["Fecha"] = pd.to_datetime(df.get("Fecha"), errors="coerce")
    df["Monto"] = pd.to_numeric(
        df["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce"
    )
    # Columnas esperadas
    for c in ["Obs", "CC1", "Sit", "Responsable", "Año", "Esp", "CC"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalización fuerte de CC y Sit
    df["CC"]  = df["CC"].astype(str).str.strip().str.upper()
    df["Sit"] = df["Sit"].astype(str).str.strip().str.upper()
    df["Obs"] = df["Obs"].astype(str)
    df["CC1"] = df["CC1"].astype(str)

    df = df.dropna(subset=["Monto", "CC"])
    return df

df = load_data(CSV_URL)

# SIN filtros (por ahora): usar todo el dataset
df_f = df.copy()

# =========================
# Carga de Electricidad (Excel local o URL XLSX)
# =========================
ELECTRICIDAD_XLSX = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSm1GzFOATXqiYiGSKUZWZ5C2dE9nXv7bTtR3XmchtWhC9ZRMRK5OGDQfZhb624dA/pub?output=xlsx"

@st.cache_data
def load_electricidad(path_or_url: str) -> dict[str, pd.DataFrame]:
    if path_or_url.startswith(("http://", "https://")):
        sheets = pd.read_excel(path_or_url, sheet_name=None)
    else:
        p = Path(path_or_url)
        if not p.exists():
            return {}
        sheets = pd.read_excel(p, sheet_name=None)
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df_sheet in sheets.items():
        if df_sheet is None:
            continue
        df_sheet = df_sheet.copy()
        if df_sheet.shape[1] == 0:
            continue
        df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
        cleaned[name] = df_sheet
    return cleaned


@st.cache_data(show_spinner=False)
def load_electricidad_parsed(path_or_url: str) -> dict[str, dict[str, pd.DataFrame]]:
    sheets = load_electricidad(path_or_url)
    parsed: dict[str, dict[str, pd.DataFrame]] = {}
    for name, df_sheet in sheets.items():
        try:
            parsed_sheet = _parse_mes_sheet(df_sheet)
        except Exception:
            continue
        if parsed_sheet["inputs_bodega"].empty and parsed_sheet["liquidacion"].empty:
            continue
        parsed[name] = parsed_sheet
    return parsed


def _infer_date_column(df_in: pd.DataFrame) -> str | None:
    best_col = None
    best_ratio = 0.0
    for col in df_in.columns:
        s = pd.to_datetime(df_in[col], errors="coerce")
        ratio = s.notna().mean()
        if ratio > best_ratio and ratio >= 0.5:
            best_ratio = ratio
            best_col = col
    return best_col

def _pick_primary_numeric(df_in: pd.DataFrame) -> str | None:
    preferred = ["TOTAL", "MONTO", "VALOR", "COSTO", "KW", "KWH", "CONSUMO", "ENERGIA"]
    cols = [str(c) for c in df_in.columns]
    for p in preferred:
        for c in cols:
            if p in c.upper():
                return c
    num_df = df_in.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in num_df.columns if num_df[c].notna().any()]
    return num_cols[0] if num_cols else None

def _clean_header_list(row_vals: list) -> list[str]:
    out = []
    for v in row_vals:
        if pd.isna(v):
            break
        out.append(str(v).strip())
    return out

def _dedupe_cols(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def _fmt_cell(v, is_pct=False, is_currency=False, is_int=False):
    try:
        if pd.isna(v):
            return ""
    except Exception:
        if v is None:
            return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if is_int:
        try:
            return f"{float(v):,.0f}"
        except Exception:
            return str(v)
    if is_pct:
        try:
            return f"{float(v):.2%}"
        except Exception:
            return str(v)
    if is_currency:
        try:
            return f"${float(v):,.0f}"
        except Exception:
            return str(v)
    if hasattr(v, "strftime"):
        try:
            return v.strftime("%H:%M")
        except Exception:
            return str(v)
    return str(v)

def _render_table(df_in: pd.DataFrame, header_bg="#1f4e78", header_fg="white",
                  row_alt="#f8f4e8", compact=True, int_cols=None, cur_cols_extra=None) -> None:
    columns = list(df_in.columns)
    pct_cols = {c for c in columns if "%" in str(c)}
    cur_cols = {c for c in columns if "$" in str(c) or "TOTAL" in str(c).upper()}
    int_cols = set(int_cols or [])
    cur_cols |= set(cur_cols_extra or [])

    rows_html = []
    for i, row in enumerate(df_in.itertuples(index=False, name=None)):
        tds = []
        for c, value in zip(columns, row):
            v = _fmt_cell(
                value,
                is_pct=(c in pct_cols),
                is_currency=(c in cur_cols),
                is_int=(c in int_cols),
            )
            tds.append(f"<td>{v}</td>")
        cls = "alt" if i % 2 == 0 else ""
        rows_html.append(f"<tr class='{cls}'>" + "".join(tds) + "</tr>")

    ths = "".join([f"<th>{c}</th>" for c in columns])
    table_html = f"""
    <div class="elec-table-wrap">
      <table class="elec-table">
        <thead><tr>{ths}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """

    st.markdown(
        f"""
        <style>
        .elec-table-wrap {{ overflow-x: auto; }}
        .elec-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: { "12px" if compact else "13px" };
        }}
        .elec-table th {{
            background: {header_bg};
            color: {header_fg};
            text-align: center;
            padding: 6px 8px;
            border: 1px solid #d5d5d5;
            font-weight: 700;
        }}
        .elec-table td {{
            padding: 6px 8px;
            border: 1px solid #d5d5d5;
            text-align: center;
            white-space: nowrap;
        }}
        .elec-table tr.alt td {{
            background: {row_alt};
        }}
        </style>
        {table_html}
        """,
        unsafe_allow_html=True,
    )

def _parse_mes_sheet(df_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df_raw is None or df_raw.empty or df_raw.shape[1] == 0:
        empty_ig = pd.DataFrame(columns=["Parámetro", "Valor", "Unidad"])
        empty_bc = pd.DataFrame(columns=["Concepto", "Monto $"])
        return {
            "inputs_generales": empty_ig,
            "boleta_cge": empty_bc,
            "inputs_bodega": pd.DataFrame(),
            "liquidacion": pd.DataFrame(),
        }

    def _slice_cols(start: int, stop: int) -> pd.DataFrame:
        start = max(0, start)
        stop = min(df_raw.shape[1], stop)
        if start >= stop:
            return pd.DataFrame()
        return df_raw.iloc[:, start:stop].copy()

    def _pad_or_trim_columns(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df_out = df_in.copy()
        target = len(cols)
        if df_out.shape[1] < target:
            for i in range(target - df_out.shape[1]):
                df_out[f"__pad_{i}"] = pd.NA
        df_out = df_out.iloc[:, :target]
        df_out.columns = cols
        return df_out

    def _find_row_any(label: str) -> int | None:
        txt = df_raw.astype(str).apply(lambda s: s.str.strip())
        hits = txt.apply(lambda row: row.str.contains(label, case=False, na=False).any(), axis=1)
        if hits.any():
            return int(hits.idxmax())
        return None

    def _slice_until_blank(start_row: int, col_idx: int, col_slice: slice) -> pd.DataFrame:
        end = len(df_raw)
        for i in range(start_row, len(df_raw)):
            v = df_raw.iloc[i, col_idx] if col_idx < df_raw.shape[1] else None
            if pd.isna(v) or str(v).strip() == "":
                end = i
                break
            txt = str(v).upper()
            if "INPUTS POR BODEGA" in txt or "LIQUIDACIÓN POR BODEGA" in txt:
                end = i
                break
        return df_raw.iloc[start_row:end, col_slice].copy()

    # INPUTS GENERALES
    row_ig = _find_row_any("INPUTS GENERALES")
    if row_ig is not None:
        ig = _slice_until_blank(row_ig + 1, 0, slice(0, 3))
    else:
        ig = _slice_cols(0, 3).iloc[3:8].copy()
    ig = _pad_or_trim_columns(ig, ["Parámetro", "Valor", "Unidad"])
    ig = ig.dropna(how="all")

    # BOLETA CGE
    row_bc = _find_row_any("BOLETA CGE")
    if row_bc is not None:
        # +2 porque la fila inmediatamente inferior suele ser encabezado Concepto/Monto
        bc = _slice_until_blank(row_bc + 2, 5, slice(5, 7))
    else:
        bc = _slice_cols(5, 7).iloc[4:9].copy()
    bc = _pad_or_trim_columns(bc, ["Concepto", "Monto $"])
    bc = bc.dropna(how="all")

    def _find_row(label: str) -> int | None:
        col0 = df_raw.iloc[:, 0].astype(str)
        hits = col0.str.contains(label, case=False, na=False)
        if hits.any():
            return int(hits.idxmax())
        return None

    def _slice_section(start_row: int) -> pd.DataFrame:
        end = len(df_raw)
        col0 = df_raw.iloc[:, 0]
        for i in range(start_row, len(df_raw)):
            v = col0.iloc[i]
            if pd.isna(v):
                end = i
                break
            if isinstance(v, str) and ("LIQUIDACIÓN POR BODEGA" in v.upper() or "INPUTS POR BODEGA" in v.upper()):
                if i != start_row:
                    end = i
                    break
        return df_raw.iloc[start_row:end]

    # INPUTS POR BODEGA
    row_inputs = _find_row("INPUTS POR BODEGA")
    if row_inputs is not None:
        hdr_row = row_inputs + 1
        data_row = row_inputs + 2
        hdr_in = _clean_header_list(df_raw.iloc[hdr_row, 0:15].tolist())
        hdr_in = _dedupe_cols(hdr_in)
        if hdr_in:
            data_in = _slice_section(data_row).iloc[:, 0:len(hdr_in)].copy()
            data_in.columns = hdr_in
        else:
            data_in = pd.DataFrame()
    else:
        hdr_in = _clean_header_list(df_raw.iloc[11, 0:9].tolist()) if len(df_raw) > 11 else []
        hdr_in = _dedupe_cols(hdr_in)
        if hdr_in:
            data_in = df_raw.iloc[12:20, 0:len(hdr_in)].copy()
            data_in.columns = hdr_in
        else:
            data_in = pd.DataFrame()

    # LIQUIDACIÓN POR BODEGA
    row_liq = _find_row("LIQUIDACIÓN POR BODEGA")
    if row_liq is not None:
        hdr_row = row_liq + 1
        data_row = row_liq + 2
        hdr_liq = _clean_header_list(df_raw.iloc[hdr_row, 0:20].tolist())
        hdr_liq = _dedupe_cols(hdr_liq)
        if hdr_liq:
            data_liq = _slice_section(data_row).iloc[:, 0:len(hdr_liq)].copy()
            data_liq.columns = hdr_liq
        else:
            data_liq = pd.DataFrame()
    else:
        hdr_liq = _clean_header_list(df_raw.iloc[22, 0:15].tolist()) if len(df_raw) > 22 else []
        hdr_liq = _dedupe_cols(hdr_liq)
        if hdr_liq:
            data_liq = df_raw.iloc[23:31, 0:len(hdr_liq)].copy()
            data_liq.columns = hdr_liq
        else:
            data_liq = pd.DataFrame()

    return {
        "inputs_generales": ig,
        "boleta_cge": bc,
        "inputs_bodega": data_in,
        "liquidacion": data_liq,
    }

def _df_to_rl_table(
    df_in: pd.DataFrame,
    header_bg=colors.HexColor("#0F2942"),
    header_fg=colors.white,
    row_alt: colors.Color | None = None,
    grid_color=colors.HexColor("#D9E1EA"),
    max_width=520,
    font_size=7,
    min_col_width=42,
    max_col_width=165,
    cell_padding=4,
) -> Table:
    data = [[str(c) for c in df_in.columns]]
    for _, row in df_in.fillna("").astype(str).iterrows():
        data.append([str(v) for v in row.tolist()])

    ncols = len(df_in.columns)
    # Ajuste de ancho por contenido para evitar celdas sobredimensionadas.
    sample_rows = df_in.fillna("").astype(str).head(80)
    char_counts = []
    for c in df_in.columns:
        header_len = len(str(c))
        content_len = sample_rows[c].map(len).quantile(0.85) if not sample_rows.empty else 0
        est = max(header_len, int(content_len))
        char_counts.append(max(5, min(30, est)))
    raw_widths = [min(max_col_width, max(min_col_width, 5.2 * cc)) for cc in char_counts]
    total_raw = sum(raw_widths) or 1
    if total_raw > max_width:
        scale = max_width / total_raw
        col_widths = [max(min_col_width, w * scale) for w in raw_widths]
        # Si al aplicar mínimos se excede, se distribuye uniforme como fallback.
        if sum(col_widths) > max_width:
            col_widths = [max_width / max(1, ncols)] * ncols
    else:
        col_widths = raw_widths

    tbl = Table(data, hAlign="LEFT", colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_fg),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.35, grid_color),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), cell_padding),
        ("TOPPADDING", (0, 0), (-1, -1), cell_padding),
        ("LEFTPADDING", (0, 0), (-1, -1), cell_padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), cell_padding),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Alineación por tipo de columna para una lectura más ejecutiva.
    for idx, c in enumerate(df_in.columns):
        cu = str(c).upper()
        if any(k in cu for k in ["$", "TOTAL", "IVA", "NETO", "KWH", "%"]):
            style_cmds.append(("ALIGN", (idx, 1), (idx, -1), "RIGHT"))
        else:
            style_cmds.append(("ALIGN", (idx, 1), (idx, -1), "LEFT"))
    if row_alt is not None:
        for r in range(1, len(data)):
            if (r - 1) % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, r), (-1, r), row_alt))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _compact_pdf_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    alias = {
        "Bodega": "Bod.",
        "Remarcador": "Rem.",
        "Hora inicio jornada (Input)": "Hora ini. jor.",
        "Hora inicio horario especial": "Hora ini. esp.",
        "Factor horario especial (>18:00)": "Factor esp. >18",
        "Hora salida (Input)": "Hora salida",
        "Carga inductiva (SI/NO)": "Carga ind.",
        "Pago con atraso SI/NO": "Pago atraso",
        "% post-18 (calc)": "% post-18",
        "kWh post-18 (calc)": "kWh post-18",
        "kWh día (calc)": "kWh día",
        "$ Cargos Fijos": "$ C. Fijos",
        "% Cargos Fijos": "% C. Fijos",
        "TOTAL NETO $": "Total neto",
        "TOTAL c/IVA $": "Total c/IVA",
        "Intereses/Mora $": "Interés/Mora $",
    }
    renamed = {}
    for c in df.columns:
        c_txt = str(c)
        if c_txt in alias:
            renamed[c] = alias[c_txt]
            continue
        c_txt = (
            c_txt.replace(" (calc)", "")
            .replace(" (Input)", "")
            .replace(" (SI/NO)", "")
            .replace("REMARCADOR", "Rem.")
            .replace("REMARCADOR", "Rem.")
            .replace("  ", " ")
            .strip()
        )
        renamed[c] = c_txt
    return df.rename(columns=renamed)


def _pdf_section_banner(text: str, width: float, bg_hex="#0F2942", fg=colors.white) -> Table:
    banner = Table([[text]], colWidths=[width])
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_hex)),
                ("TEXTCOLOR", (0, 0), (-1, -1), fg),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return banner

def _format_pdf_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for c in df.columns:
        cu = str(c).upper()
        if "HORA" in cu:
            df[c] = df[c].apply(lambda v: v.strftime("%H:%M") if hasattr(v, "strftime") else str(v))
            continue

        is_pct = "%" in cu
        is_cur = "$" in cu or "TOTAL" in cu or "IVA" in cu or "NETO" in cu
        is_kwh = "KWH" in cu

        if is_pct:
            def fmt_pct(v):
                try:
                    v = float(v)
                    if abs(v) <= 1.5:
                        v = v * 100.0
                    return f"{v:.2f}%"
                except Exception:
                    return str(v)
            df[c] = df[c].apply(fmt_pct)
        elif is_cur:
            def fmt_cur(v):
                try:
                    v = float(v)
                    return f"${v:,.0f}"
                except Exception:
                    return str(v)
            df[c] = df[c].apply(fmt_cur)
        elif is_kwh:
            def fmt_int(v):
                try:
                    return f"{float(v):,.0f}"
                except Exception:
                    return str(v)
            df[c] = df[c].apply(fmt_int)
        else:
            df[c] = df[c].apply(lambda v: str(v))
    return df

@st.cache_data(show_spinner=False)
def build_electricidad_pdf(
    title: str,
    sel_months: list[str],
    sel_bodega: str,
    inputs_generales: pd.DataFrame,
    boleta_avg: pd.DataFrame,
    inputs_bodega: pd.DataFrame,
    liquidacion: pd.DataFrame,
    charts: list[dict],
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        title=title,
        leftMargin=20,
        rightMargin=20,
        topMargin=20,
        bottomMargin=20,
    )
    styles = getSampleStyleSheet()
    story = []
    content_w = doc.width
    generated_at = pd.Timestamp.now().strftime("%d-%m-%Y %H:%M")

    title_style = ParagraphStyle(
        "ElecPdfTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=24,
        textColor=colors.HexColor("#0B1F33"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "ElecPdfMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        alignment=TA_LEFT,
    )
    section_title_style = ParagraphStyle(
        "ElecPdfSection",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#0B1F33"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    def _draw_footer(canv, d):
        canv.saveState()
        y = 12
        canv.setStrokeColor(colors.HexColor("#D9E1EA"))
        canv.line(d.leftMargin, y + 8, A4[0] - d.rightMargin, y + 8)
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#64748B"))
        canv.drawString(d.leftMargin, y, f"Generado: {generated_at}")
        canv.drawRightString(A4[0] - d.rightMargin, y, f"Página {canv.getPageNumber()}")
        canv.restoreState()

    main_title = title or "Liquidación Eléctrica por Bodega"
    story.append(Paragraph(main_title, title_style))
    story.append(Paragraph(f"Meses: {', '.join(sel_months)} &nbsp;&nbsp;|&nbsp;&nbsp; Bodega: {sel_bodega}", meta_style))
    story.append(Spacer(1, 8))

    # Primer bloque en dos columnas: Inputs Generales (izq) + Boleta CGE (der)
    split_w = (content_w - 8) / 2
    left_block = [
        _pdf_section_banner("INPUTS GENERALES", split_w),
        Spacer(1, 3),
        Paragraph("Parámetros base para liquidación eléctrica.", section_title_style),
        _df_to_rl_table(
            _format_pdf_df(_compact_pdf_columns(inputs_generales)),
            header_bg=colors.HexColor("#123A5A"),
            row_alt=colors.HexColor("#F8FBFF"),
            max_width=split_w,
            font_size=7.2,
            cell_padding=3,
        ),
    ]
    right_block = [
        _pdf_section_banner("BOLETA CGE (PROMEDIO SEGÚN MESES)", split_w),
        Spacer(1, 3),
        Paragraph("Resumen consolidado por concepto de facturación.", section_title_style),
        _df_to_rl_table(
            _format_pdf_df(_compact_pdf_columns(boleta_avg)),
            header_bg=colors.HexColor("#123A5A"),
            row_alt=colors.HexColor("#F8FBFF"),
            max_width=split_w,
            font_size=7.2,
            cell_padding=3,
        ),
    ]
    intro_grid = Table([[left_block, right_block]], colWidths=[split_w, split_w], hAlign="LEFT")
    intro_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(intro_grid)
    story.append(Spacer(1, 8))

    story.append(_pdf_section_banner("INPUTS POR BODEGA (REMARCADOR + HORARIO EFECTIVO)", content_w))
    story.append(Spacer(1, 3))
    story.append(
        _df_to_rl_table(
            _format_pdf_df(_compact_pdf_columns(inputs_bodega)),
            header_bg=colors.HexColor("#123A5A"),
            row_alt=colors.HexColor("#FFFDF7"),
            max_width=content_w,
            font_size=6.8,
            min_col_width=36,
            max_col_width=120,
            cell_padding=2.5,
        )
    )
    story.append(Spacer(1, 6))

    story.append(_pdf_section_banner("LIQUIDACIÓN POR BODEGA (ASIGNACIÓN DE COSTOS + CRITERIO HORARIO)", content_w))
    story.append(Spacer(1, 3))
    story.append(
        _df_to_rl_table(
            _format_pdf_df(_compact_pdf_columns(liquidacion)),
            header_bg=colors.HexColor("#123A5A"),
            row_alt=colors.HexColor("#F6FCF8"),
            max_width=content_w,
            font_size=6.8,
            min_col_width=36,
            max_col_width=120,
            cell_padding=2.5,
        )
    )
    story.append(Spacer(1, 8))

    if charts:
        import matplotlib.pyplot as plt
        story.append(_pdf_section_banner("GRÁFICO — LIQUIDACIÓN POR BODEGA (DISTRIBUCIÓN DE COSTOS)", content_w))
        story.append(Spacer(1, 4))
        for ch in charts:
            try:
                df_plot = ch["df"].copy()
                x_col = ch["x_col"]
                title = ch.get("title", "")
                cols_costos = ch["cols_costos"]
                col_total = ch["col_total"]
                palette = ch["palette"]

                chart_type = ch.get("chart_type", "stacked")
                if chart_type == "donut":
                    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
                    vals = pd.to_numeric(df_plot[cols_costos].sum(numeric_only=True), errors="coerce").fillna(0)
                    raw_values = [float(v) for v in vals.tolist()]
                    # Matplotlib pie no admite negativos; se recortan para evitar fallas.
                    values = [max(0.0, v) for v in raw_values]
                    labels = cols_costos
                    pie_colors = [palette.get(c, "#94a3b8") for c in labels]
                    total_vals = pd.to_numeric(df_plot[col_total], errors="coerce").fillna(0)
                    total_num = float(total_vals.sum())
                    bodega_label = str(df_plot.iloc[0, 1]) if not df_plot.empty else ""
                    mes_label = str(df_plot["Mes"].iloc[0]) if ("Mes" in df_plot.columns and not df_plot.empty) else ""
                    if sum(values) > 0:
                        ax.pie(
                            values,
                            labels=None,
                            colors=pie_colors,
                            startangle=90,
                            counterclock=False,
                            wedgeprops=dict(width=0.44, edgecolor="white"),
                            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                            pctdistance=0.8,
                        )
                        center_txt = f"{bodega_label}\n{mes_label}\nTotal c/IVA\n${total_num:,.0f}"
                        ax.text(0, 0, center_txt, ha="center", va="center", fontsize=11, weight="bold", color="#111827")
                        ax.axis("equal")
                        ax.legend(labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, fontsize=7, frameon=False)
                        if title:
                            ax.set_title(title)
                        plt.tight_layout()
                    else:
                        # Sin base positiva para dona: fallback robusto a barras.
                        y_pos = np.arange(len(labels))
                        ax.barh(y_pos, raw_values, color=pie_colors)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(labels, fontsize=8)
                        ax.axvline(0, color="#94a3b8", linewidth=1)
                        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
                        center_txt = f"{bodega_label} · {mes_label} · Total c/IVA: ${total_num:,.0f}"
                        ax.set_title(center_txt, fontsize=10, color="#111827")
                        plt.tight_layout()
                else:
                    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
                    bottom = None
                    for c in cols_costos:
                        vals = pd.to_numeric(df_plot[c], errors="coerce").fillna(0)
                        ax1.bar(df_plot[x_col], vals, label=c, bottom=bottom, color=palette.get(c, "#94a3b8"))
                        bottom = vals if bottom is None else bottom + vals

                    ax1.set_ylabel("Costo (CLP)")
                    ax1.tick_params(axis="x", rotation=0)

                    ax2 = ax1.twinx()
                    total_vals = pd.to_numeric(df_plot[col_total], errors="coerce").fillna(0)
                    ax2.plot(df_plot[x_col], total_vals, color="#111827", marker="o", linewidth=2, label="TOTAL c/IVA $")
                    ax2.set_ylabel("Total c/IVA (CLP)")

                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center",
                               bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=7, frameon=False)
                    if title:
                        ax1.set_title(title)
                    plt.tight_layout()

                img_buf = BytesIO()
                fig.savefig(img_buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                img_buf.seek(0)
                story.append(RLImage(img_buf, width=content_w, height=270))
                story.append(Spacer(1, 8))
            except Exception as e:
                raise RuntimeError(f"No fue posible renderizar el gráfico en el PDF: {e}")

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    pdf = buf.getvalue()
    buf.close()
    return pdf

# =========================
# Utilidades de formato
# =========================
def fmt_clp_largo(v: float) -> str:
    return f"${v:,.0f}"

def fmt_short(v: float) -> str:
    v = float(v)
    av = abs(v)
    if av >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:,.0f}"


def card_finanza(titulo, valor, color_hex, subtitulo="Indicador financiero clave", etiqueta="Indicador", size="md"):
    size_map = {
        "lg": ("42px", "kpi-card kpi-card-lg"),
        "md": ("34px", "kpi-card kpi-card-md"),
        "sm": ("28px", "kpi-card kpi-card-sm"),
        "stack": ("32px", "kpi-card kpi-card-stack"),
        "top": ("30px", "kpi-card kpi-card-top"),
    }
    value_size, card_class = size_map.get(size, size_map["md"])
    subtitle_html = f'<div class="kpi-sub">{subtitulo}</div>' if subtitulo else ""
    eyebrow_html = f'<div class="kpi-eyebrow">{etiqueta}</div>' if etiqueta else ""
    return f"""
    <div class="{card_class}" style="--accent:{color_hex};">
        {eyebrow_html}
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value" style="font-size:{value_size};">{valor}</div>
        {subtitle_html}
    </div>
    """


def kpi_resumen_panel(titulo, subtitulo, items):
    rows = []
    for item in items:
        rows.append(
            (
                f'<div class="kpi-summary-row">'
                f'<div class="kpi-summary-label">{item["label"]}</div>'
                f'<div class="kpi-summary-meta">{item["meta"]}</div>'
                f'<div class="kpi-summary-value">{item["value"]}</div>'
                f'</div>'
            )
        )
    return (
        f'<div class="kpi-summary-card">'
        f'<div class="kpi-summary-eyebrow">{titulo}</div>'
        f'<div class="kpi-summary-title">{subtitulo}</div>'
        f'<div class="kpi-summary-head">'
        f'<div>Indicador</div>'
        f'<div>Descripción</div>'
        f'<div>Valor</div>'
        f'</div>'
        f'<div class="kpi-summary-list">{"".join(rows)}</div>'
        f'</div>'
    )


def section_heading(icono: str, titulo: str, subtitulo: str = "", weight_class: str = "section-heading-title") -> str:
    subtitle_html = f'<div class="section-heading-sub">{subtitulo}</div>' if subtitulo else ""
    return (
        f'<div class="section-heading-wrap">'
        f'<div class="{weight_class}">{icono} {titulo}</div>'
        f'{subtitle_html}'
        f'</div>'
    )

# =========================
# KPI base (cálculos comunes)
# =========================
CAPEX = 151_834_571

# Canon arriendo
mask_canon = (
    (df_f["CC"] == "INGRESO") &
    (
        df_f["CC1"].str.contains("arriendo", case=False, na=False) |
        df_f["Obs"].str.contains("canon", case=False, na=False)
    )
)
ingresos_canon = df_f.loc[mask_canon, "Monto"].sum()
cobertura_capex = ingresos_canon / CAPEX if CAPEX else 0

# Saldo cuenta = PAGADO + ABONOS (según signos en CSV)
total_pagado = df_f.loc[df_f["Sit"] == "PAGADO", "Monto"].sum()
mask_abono_obs = df_f["Obs"].str.contains(r"\babono_*\b", case=False, na=False)
total_abonos = df_f.loc[mask_abono_obs, "Monto"].sum()
saldo_cuenta = total_pagado + total_abonos

# Ingresos / Egresos KPIs
mask_ingreso = df_f["CC"].eq("INGRESO")
mask_egreso  = df_f["CC"].eq("EGRESO")
sit_up = df_f["Sit"]  # ya normalizado

mask_sit_pagado = sit_up.eq("PAGADO")
mask_sit_abono  = sit_up.str.startswith("ABONO")
mask_sueldo_accionista = df_f["Obs"].str.contains(
    r"sueldos?\s+accion(?:ista|ita)s?",
    case=False,
    na=False,
    regex=True,
)

ingresos_kpi = df_f.loc[mask_ingreso & (mask_sit_pagado | mask_sit_abono), "Monto"].sum()
egresos_kpi  = df_f.loc[mask_egreso  &  mask_sit_pagado, "Monto"].sum()
utilidad_operativa = ingresos_kpi + egresos_kpi
margen_neto = (utilidad_operativa / ingresos_kpi) if ingresos_kpi else 0.0
total_sueldos_accionistas = df_f.loc[mask_sueldo_accionista, "Monto"].sum()
utilidad_sobre_capex = (abs(total_sueldos_accionistas) / CAPEX) if CAPEX else 0.0
balance_kpi = saldo_cuenta  # interpretación: saldo en cuenta BCI

df_egresos_mes = df_f.loc[mask_egreso & mask_sit_pagado, ["Monto", "Año", "Mes", "Fecha"]].copy()
df_egresos_mes["Año_calc"] = pd.to_numeric(df_egresos_mes.get("Año"), errors="coerce")
mes_raw_kpi = df_egresos_mes.get("Mes")
mes_num_kpi = mes_raw_kpi.astype(str).str.extract(r"(\d{1,2})", expand=False) if mes_raw_kpi is not None else None
df_egresos_mes["Mes_calc"] = pd.to_numeric(mes_num_kpi, errors="coerce")
df_egresos_mes["Fecha"] = pd.to_datetime(df_egresos_mes.get("Fecha"), errors="coerce")
df_egresos_mes["Año_calc"] = df_egresos_mes["Año_calc"].fillna(df_egresos_mes["Fecha"].dt.year)
df_egresos_mes["Mes_calc"] = df_egresos_mes["Mes_calc"].fillna(df_egresos_mes["Fecha"].dt.month)
df_egresos_mes = df_egresos_mes.dropna(subset=["Año_calc", "Mes_calc"])

if not df_egresos_mes.empty:
    df_egresos_mes["Periodo"] = pd.to_datetime(
        dict(
            year=df_egresos_mes["Año_calc"].astype(int),
            month=df_egresos_mes["Mes_calc"].astype(int),
            day=1,
        ),
        errors="coerce",
    )
    egreso_mensual_promedio = (
        df_egresos_mes.dropna(subset=["Periodo"])
        .groupby("Periodo", as_index=False)["Monto"]
        .sum()["Monto"]
        .abs()
        .mean()
    )
else:
    egreso_mensual_promedio = 0.0

cobertura_egresos = (balance_kpi / egreso_mensual_promedio) if egreso_mensual_promedio else 0.0

# Cuentas por cobrar & egresos por pagar
sit_norm = df_f["Sit"]
cc_up    = df_f["CC"]

no_pagado_total  = df_f.loc[cc_up.eq("INGRESO") & sit_norm.eq("NO PAGADO"), "Monto"].sum()
abonos_total     = df_f.loc[cc_up.eq("INGRESO") & sit_norm.str.startswith("ABONO"), "Monto"].sum()
if abonos_total == 0:
    abonos_total = df_f.loc[
        cc_up.eq("INGRESO") &
        df_f["Obs"].str.contains(r"\babono\b", case=False, na=False),
        "Monto"
    ].sum()

pct_cobranza = (abonos_total / no_pagado_total) if no_pagado_total else 0.0
cuentas_por_cobrar_neto = no_pagado_total - abonos_total

total_egresos_por_pagar = df_f.loc[
    cc_up.eq("EGRESO") & sit_norm.eq("NO PAGADO"), "Monto"
].sum()

posicion_neta = cuentas_por_cobrar_neto + total_egresos_por_pagar + balance_kpi

# =========================
# Estilos HTML para KPIs
# =========================
st.markdown("""
    <style>
    .kpi-card {
        position: relative;
        min-height: 162px;
        padding: 18px 20px 18px 22px;
        border-radius: 28px;
        border: 1px solid color-mix(in srgb, var(--accent, #d7dfeb) 28%, #d7dfeb);
        background:
            radial-gradient(circle at top right, color-mix(in srgb, var(--accent, #f8fafc) 12%, transparent) 0%, transparent 34%),
            linear-gradient(135deg, color-mix(in srgb, var(--accent, #f7f9fc) 14%, #f7f9fc) 0%, #ffffff 55%, color-mix(in srgb, var(--accent, #f4f7fb) 10%, #f4f7fb) 100%);
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 8px;
        background: linear-gradient(180deg, var(--accent, #d5ddeb) 0%, color-mix(in srgb, var(--accent, #d5ddeb) 28%, #eef3f9) 100%);
        border-radius: 28px 0 0 28px;
    }
    .kpi-card::after {
        content: "";
        position: absolute;
        top: 18px;
        right: 18px;
        width: 12px;
        height: 12px;
        border-radius: 999px;
        background: color-mix(in srgb, var(--accent, #cbd5e1) 60%, white);
        opacity: 0.55;
    }
    .kpi-card-lg {
        min-height: 176px;
    }
    .kpi-card-md {
        min-height: 138px;
        padding: 16px 18px 16px 20px;
    }
    .kpi-card-sm {
        min-height: 112px;
        padding: 14px 16px 14px 18px;
    }
    .kpi-card-stack {
        min-height: 170px;
        padding: 16px 18px 18px 20px;
    }
    .kpi-card-top {
        min-height: 250px;
        padding: 16px 18px 18px 20px;
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
    }
    .kpi-card-top .kpi-title {
        min-height: 78px;
    }
    .kpi-card-top .kpi-sub {
        min-height: 84px;
    }
    .kpi-eyebrow {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 12px;
        opacity: 0.95;
        color: color-mix(in srgb, var(--accent, #64748b) 74%, #475569);
    }
    .kpi-title {
        font-size: 17px;
        line-height: 1.2;
        color: #1f2937;
        font-weight: 800;
        margin-bottom: 12px;
        max-width: 82%;
    }
    .kpi-value {
        display: inline-block;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
        margin-bottom: 10px;
        color: color-mix(in srgb, var(--accent, #0f172a) 60%, #0f172a);
        padding: 6px 0 2px 0;
    }
    .kpi-sub {
        font-size: 14px;
        color: #667085;
        font-weight: 500;
        max-width: 88%;
    }
    .kpi-card-md .kpi-title, .kpi-card-sm .kpi-title, .kpi-card-stack .kpi-title {
        margin-bottom: 8px;
    }
    .kpi-card-sm .kpi-eyebrow, .kpi-card-stack .kpi-eyebrow {
        margin-bottom: 8px;
    }
    .kpi-card-sm .kpi-sub, .kpi-card-stack .kpi-sub {
        font-size: 12.5px;
        line-height: 1.25;
    }
    .kpi-summary-card {
        min-height: 100%;
        padding: 20px 24px;
        border-radius: 28px;
        border: 1px solid #d7dfeb;
        background:
            radial-gradient(circle at top right, rgba(127,166,162,0.08) 0%, transparent 28%),
            linear-gradient(135deg, #f7f9fc 0%, #ffffff 55%, #f4f7fb 100%);
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
    }
    .kpi-summary-eyebrow {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-summary-title {
        font-size: 14px;
        line-height: 1.2;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 18px;
    }
    .kpi-summary-head {
        display: grid;
        grid-template-columns: 1.15fr 1fr auto;
        gap: 12px;
        padding: 0 0 12px 0;
        border-top: 1px solid #e6edf5;
        border-bottom: 1px solid #e6edf5;
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .kpi-summary-list {
        padding-top: 2px;
    }
    .kpi-summary-row {
        display: grid;
        grid-template-columns: 1.15fr 1fr auto;
        gap: 12px;
        align-items: center;
        padding: 17px 0;
        border-bottom: 1px solid #e6edf5;
    }
    .kpi-summary-row:last-child {
        border-bottom: none;
        padding-bottom: 2px;
    }
    .kpi-summary-label {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
    }
    .kpi-summary-meta {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        line-height: 1.2;
    }
    .kpi-summary-value {
        font-size: 15px;
        font-weight: 800;
        color: #0f172a;
        text-align: right;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        padding-left: 14px;
    }
    .section-heading-wrap {
        margin: 6px 0 14px 0;
    }
    .section-heading-title {
        font-size: clamp(2rem, 2.2vw, 2.75rem);
        line-height: 1.02;
        letter-spacing: -0.045em;
        font-weight: 900;
        color: #0f172a;
    }
    .section-heading-title-soft {
        font-size: 18px;
        line-height: 1.25;
        letter-spacing: 0;
        font-weight: 500;
        color: #0f172a;
    }
    .section-heading-sub {
        margin-top: 8px;
        font-size: 1rem;
        line-height: 1.45;
        color: #64748b;
        font-weight: 500;
    }
    div[role="radiogroup"][aria-label="Sección"] {
        display: grid !important;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        width: 100%;
    }
    div[role="radiogroup"][aria-label="Sección"] > label {
        position: relative;
        min-width: 0;
        width: 100%;
        padding: 10px 14px;
        border-radius: 20px;
        border: 1px solid #d9e2ec;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        transition: all 0.15s ease;
        align-items: center !important;
    }
    div[role="radiogroup"][aria-label="Sección"] > label:hover {
        border-color: #cfd8e6;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.09);
        transform: translateY(-1px);
    }
    div[role="radiogroup"][aria-label="Sección"] > label > div:last-child {
        font-size: 14px;
        font-weight: 700;
        color: #344054;
        line-height: 1.15;
    }
    div[role="radiogroup"][aria-label="Sección"] > label:has(input:checked) {
        border-color: #c9d5e5;
        background: linear-gradient(135deg, #eef3f9 0%, #f8fbff 50%, #ffffff 100%);
        box-shadow: 0 16px 30px rgba(109, 132, 164, 0.16);
    }
    div[role="radiogroup"][aria-label="Sección"] > label:has(input:checked)::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 6px;
        background: linear-gradient(180deg, #b9c7da 0%, #e3ebf5 100%);
        border-radius: 20px 0 0 20px;
    }
    div[role="radiogroup"][aria-label="Sección"] > label:has(input:checked) > div:last-child {
        color: #1f2937;
    }
    @media (max-width: 1200px) {
        div[role="radiogroup"][aria-label="Sección"] {
            grid-template-columns: repeat(2, minmax(220px, 1fr));
        }
        div[role="radiogroup"][aria-label="Sección"] > label {
            min-width: 220px;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# NAVEGACIÓN PRINCIPAL
# =========================
section_options = [
    "🏠 Visión general",
    "⚠️ Riesgos & cobranzas",
    "🏢 Canon anual / mensual",
    "📈 Ingresos & egresos",
    "⚡ Electricidad",
]
active_section = st.radio(
    "Sección",
    options=section_options,
    horizontal=True,
    label_visibility="collapsed",
)

# =========================================================
# 🏠 TAB 1: VISIÓN GENERAL
# =========================================================
if active_section == "🏠 Visión general":
    import plotly.graph_objects as go

    st.markdown(
        section_heading(
            "📊",
            "Estado general del proyecto",
            weight_class="section-heading-title-soft",
        ),
        unsafe_allow_html=True,
    )

    # --- KPIs principales en una sola línea ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.markdown(
            card_finanza(
                "🏦 Caja Banco BCI",
                fmt_clp_largo(balance_kpi),
                "#D85E5D",
                subtitulo="Saldo disponible en cuenta corriente para operación",
                etiqueta="Bloque 1",
                size="top",
            ),
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            card_finanza(
                "💼 CAPEX",
                fmt_clp_largo(CAPEX),
                "#DCAA67",
                subtitulo="Inversión total del proyecto",
                etiqueta="Bloque 2",
                size="top",
            ),
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            card_finanza(
                "💹 Utilidad / CAPEX",
                f"{utilidad_sobre_capex:.2f}x",
                "#7FA6A2",
                subtitulo="Sueldos accionistas acumulados relativos a la inversión total",
                etiqueta="Bloque 3",
                size="top",
            ),
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            card_finanza(
                "📈 Margen neto",
                f"{margen_neto:.1%}",
                "#4B5563",
                subtitulo="Resultado operativo sobre ingresos acumulados",
                etiqueta="Bloque 4",
                size="top",
            ),
            unsafe_allow_html=True,
        )

    with kpi5:
        st.markdown(
            card_finanza(
                "💳 Caja / egreso mensual promedio",
                f"{cobertura_egresos:.2f}x",
                "#A8A8A8",
                subtitulo="Caja disponible relativa al gasto mensual promedio",
                etiqueta="Bloque 5",
                size="top",
            ),
            unsafe_allow_html=True,
        )

    st.caption("Cobertura CAPEX = Ingresos canon acumulados / CAPEX total invertido.")
    st.markdown("")

    hero_col, side_col = st.columns(2)
    resumen_items = [
        {"label": "Ingresos canon arriendo", "meta": "Acumulado histórico de canon", "value": fmt_clp_largo(ingresos_canon)},
        {"label": "Total ingresos", "meta": "Flujo acumulado", "value": fmt_clp_largo(ingresos_kpi)},
        {"label": "Total egresos", "meta": "Compromisos pagados", "value": fmt_clp_largo(egresos_kpi)},
        {"label": "Cuentas por cobrar", "meta": "Neto de abonos", "value": fmt_clp_largo(cuentas_por_cobrar_neto)},
        {"label": "Egresos por pagar", "meta": "Pendientes de salida", "value": fmt_clp_largo(total_egresos_por_pagar)},
    ]

    with hero_col:
        max_ref = max(abs(balance_kpi), abs(posicion_neta), 1)
        metric_df = pd.DataFrame(
            {
                "Métrica": ["Caja Banco BCI", "Posición Neta"],
                "Valor": [float(balance_kpi), float(posicion_neta)],
                "Detalle": [
                    "Saldo disponible en cuenta corriente para operación.",
                    "Caja + CxC + EPP",
                ],
                "Color": ["#4B5563", "#D85E5D"],
            }
        )

        fig_metrics = go.Figure()
        fig_metrics.add_vrect(
            x0=-max_ref * 1.28,
            x1=0,
            fillcolor="rgba(148,163,184,0.08)",
            line_width=0,
            layer="below",
        )
        fig_metrics.add_vline(x=0, line_width=1.5, line_dash="dot", line_color="rgba(71,85,105,0.50)")

        for _, row in metric_df.iterrows():
            fig_metrics.add_trace(
                go.Bar(
                    y=[row["Métrica"]],
                    x=[row["Valor"]],
                    orientation="h",
                    marker=dict(
                        color=row["Color"],
                        line=dict(color="rgba(255,255,255,0.75)", width=1),
                    ),
                    width=0.46,
                    text=[f"<b>{row['Métrica']}</b><br>{fmt_clp_largo(row['Valor'])}"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=15, color="#ffffff", family="Arial Black, Arial, sans-serif"),
                    customdata=[[row["Detalle"]]],
                    hovertemplate="<b>%{y}</b><br>Valor: %{text}<br>%{customdata[0]}<extra></extra>",
                    showlegend=False,
                    cliponaxis=False,
                )
            )

        fig_metrics.update_layout(
            template="plotly_white",
            height=430,
            margin=dict(l=28, r=28, t=52, b=26),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fbfcfe",
            xaxis=dict(
                range=[-max_ref * 1.28, max_ref * 1.18],
                showgrid=True,
                gridcolor="rgba(203,213,225,0.28)",
                zeroline=False,
                tickformat=",.0f",
                title="CLP",
                title_font=dict(size=13, color="#64748b"),
                tickfont=dict(size=12, color="#64748b"),
                linecolor="rgba(148,163,184,0.20)",
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                linecolor="rgba(148,163,184,0.20)",
            ),
            bargap=0.52,
            annotations=[
                dict(
                    x=0,
                    y=1.18,
                    xref="paper",
                    yref="paper",
                    text="Pulso financiero consolidado",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=12, color="#64748b"),
                ),
                dict(
                    x=0,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text="<b>Liquidez y posición neta</b>",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=19, color="#0f172a"),
                )
            ],
        )
        st.plotly_chart(
            fig_metrics,
            use_container_width=True,
            config={"displaylogo": False, "displayModeBar": False},
        )

    with side_col:
        st.markdown(
            kpi_resumen_panel(
                "Resumen ejecutivo",
                "Indicadores secundarios del estado financiero actual",
                resumen_items,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # =========================
    # Gráfico PRO: Ingresos por Canon de Arriendo por año + MA-3
    # =========================
    # =========================
    # 🏢 Ingresos por canon de arriendo — por año (MA-3)
    # =========================
    st.markdown(
        section_heading(
            "📊",
            "Ingresos por Canon de Arriendo",
            weight_class="section-heading-title-soft",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 8px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Evolución anual de canon de arriendo · Indicador estratégico de ingresos
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Usa df_f si existen filtros aplicados; si no, usa df completo
    data_src = df_f if "df_f" in locals() else df

    # Filtro canon: CC == INGRESO y (CC1 contiene "arriendo" o Obs contiene "canon")
    mask_canon = (
        (data_src["CC"].astype(str) == "INGRESO") &
        (
            data_src["CC1"].astype(str).str.contains("arriendo", case=False, na=False) |
            data_src["Obs"].astype(str).str.contains("canon", case=False, na=False)
        )
    )

    # Agregado anual
    df_canon = (
        data_src.loc[mask_canon, ["Año", "Monto"]]
        .copy()
    )

    # Tipos
    df_canon["Año"] = pd.to_numeric(df_canon["Año"], errors="coerce")
    df_canon["Monto"] = pd.to_numeric(
        df_canon["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce"
    )
    df_canon = df_canon.dropna(subset=["Año", "Monto"])

    if df_canon.empty:
        st.info("No hay datos de canon de arriendo para construir la serie anual.")
    else:
        df_canon = (
            df_canon
            .groupby("Año", as_index=False)["Monto"]
            .sum()
            .sort_values("Año")
            .rename(columns={"Monto": "Canon_anual_CLP"})
        )

        # Promedio móvil 3 años
        df_canon["MA3"] = (
            df_canon["Canon_anual_CLP"]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        # ---------- KPI resumen ----------
        ultimo_anio = int(df_canon["Año"].max())
        ultimo_valor = float(
            df_canon.loc[df_canon["Año"] == ultimo_anio, "Canon_anual_CLP"].iloc[0]
        )

        anio_prev = ultimo_anio - 1
        if (df_canon["Año"] == anio_prev).any():
            valor_prev = float(
                df_canon.loc[df_canon["Año"] == anio_prev, "Canon_anual_CLP"].iloc[0]
            )
            var_yoy = (ultimo_valor - valor_prev) / valor_prev if valor_prev != 0 else 0
        else:
            valor_prev = None
            var_yoy = 0.0

        color_yoy = "#10B981" if var_yoy >= 0 else "#EF4444"

        st.markdown(
            f"""
            <div style='padding:12px 18px; border-radius:12px;
                 background:linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
                 border:1px solid #D0D5DD; width: fit-content; margin-bottom:-8px;
                 box-shadow:0 6px 16px rgba(15,45,82,0.08);'>
                <span style='font-size:13px; color:#344054; font-weight:600;'>Último año ({ultimo_anio}):</span>
                <span style='font-size:20px; font-weight:800; color:#0F2D52;'>
                    ${ultimo_valor:,.0f}
                </span>
                <span style='font-size:13px; font-weight:700; color:{color_yoy};'>
                    ({var_yoy:+.1%} YoY)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- GRÁFICO ----------
        fig = go.Figure()

        # Barras: canon anual
        fig.add_trace(
            go.Bar(
                x=df_canon["Año"],
                y=df_canon["Canon_anual_CLP"],
                name="Canon anual (CLP)",
                marker=dict(
                    color="rgba(127, 166, 162, 0.92)",
                    line=dict(width=1.0, color="rgba(255,255,255,0.6)"),
                ),
                hovertemplate="<b>Año %{x}</b><br>Canon: $%{y:,.0f}<extra></extra>",
            )
        )

        # Línea: MA-3 (glow + línea principal)
        fig.add_trace(
            go.Scatter(
                x=df_canon["Año"],
                y=df_canon["MA3"],
                name="Promedio móvil (MA-3)",
                mode="lines",
                line=dict(color="rgba(216,94,93,0.24)", width=8),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_canon["Año"],
                y=df_canon["MA3"],
                name="Promedio móvil (MA-3)",
                mode="lines+markers",
                line=dict(color="#D85E5D", width=3),
                marker=dict(color="#D85E5D", size=6, line=dict(color="white", width=1)),
                hovertemplate="<b>Año %{x}</b><br>MA-3: $%{y:,.0f}<extra></extra>",
            )
        )

        # Sombreado técnico bajo la línea
        x_list = df_canon["Año"].tolist()
        y_ma = df_canon["MA3"].astype(float)
        y_lower = (y_ma * 0.97).tolist()
        y_upper = (y_ma * 1.03).tolist()

        fig.add_trace(
            go.Scatter(
                x=x_list + x_list[::-1],
                y=y_lower + y_upper[::-1],
                fill="toself",
                fillcolor="rgba(216,94,93,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=56, b=20),
            title=dict(
                text="📊 Ingresos por Canon de Arriendo (Anual)",
                x=0.01,
                xanchor="left",
                font=dict(size=18, color="#0F2D52"),
            ),
            legend=dict(
                orientation="h",
                y=1.12,
                x=0.5,
                xanchor="center",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(15,45,82,0.15)",
                borderwidth=1,
            ),
            xaxis=dict(
                title="Año",
                tickmode="linear",
                showgrid=False,
                linecolor="rgba(15,45,82,0.25)",
                tickfont=dict(size=12, color="#334155"),
            ),
            yaxis=dict(
                title="Monto (CLP)",
                tickformat=",.0f",
                gridcolor="rgba(15,45,82,0.10)",
                zeroline=False,
                ticks="outside",
                ticklen=6,
                tickfont=dict(size=12, color="#334155"),
                linecolor="rgba(15,45,82,0.20)",
            ),
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#F8FAFC",
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_color="#111",
                bordercolor="#E5E7EB",
            ),
            bargap=0.22,
        )


        st.plotly_chart(fig, use_container_width=True)

        # Descargar dataset agregado
        csv_canon = df_canon.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar ingresos por canon (CSV)",
            data=csv_canon,
            file_name="ingresos_canon_por_anio.csv",
            mime="text/csv",
        )

        st.markdown(
            section_heading(
                "📑",
                "Tabla de KPIs Financieros",
                weight_class="section-heading-title-soft",
            ),
            unsafe_allow_html=True,
        )

        cartera_vencida_df = df_f.loc[mask_ingreso & sit_norm.eq("NO PAGADO")].copy()
        cartera_vencida_df["Responsable"] = cartera_vencida_df["Responsable"].astype(str).str.strip()
        deuda_top3 = (
            cartera_vencida_df.groupby("Responsable")["Monto"].sum().abs().sort_values(ascending=False)
            if not cartera_vencida_df.empty
            else pd.Series(dtype=float)
        )
        concentracion_top3 = (deuda_top3.head(3).sum() / deuda_top3.sum()) if not deuda_top3.empty and deuda_top3.sum() else 0.0
        ticket_promedio_deuda = (
            cartera_vencida_df["Monto"].abs().sum() / len(cartera_vencida_df)
            if not cartera_vencida_df.empty
            else 0.0
        )

        abonos_resp_df = df_f.loc[mask_ingreso & sit_norm.str.startswith("ABONO")].copy()
        abonos_resp_df["Responsable"] = abonos_resp_df["Responsable"].astype(str).str.strip()
        responsables_con_abono = abonos_resp_df["Responsable"].replace("", pd.NA).dropna().nunique()
        abono_promedio_responsable = (
            abonos_resp_df["Monto"].abs().sum() / responsables_con_abono
            if responsables_con_abono
            else 0.0
        )

        cartera_edad_df = cartera_vencida_df.copy()
        cartera_edad_df["Fecha"] = pd.to_datetime(cartera_edad_df.get("Fecha"), errors="coerce")
        cartera_edad_df = cartera_edad_df.dropna(subset=["Fecha"])
        if not cartera_edad_df.empty:
            cartera_edad_df["Dias_vencidos"] = (
                pd.Timestamp.today().normalize() - cartera_edad_df["Fecha"].dt.normalize()
            ).dt.days.clip(lower=0)
            edad_promedio_cartera = float(cartera_edad_df["Dias_vencidos"].mean())
            edad_promedio_cartera_fmt = f"{edad_promedio_cartera:.0f} días"
        else:
            edad_promedio_cartera = None
            edad_promedio_cartera_fmt = "N/D"

        cobertura_caja_cartera = (
            balance_kpi / abs(cuentas_por_cobrar_neto)
            if cuentas_por_cobrar_neto
            else 0.0
        )

        canon_period_df = data_src.loc[mask_canon, ["Monto", "Año", "Mes", "Fecha", "Esp"]].copy()
        canon_period_df["Año_calc"] = pd.to_numeric(canon_period_df.get("Año"), errors="coerce")
        canon_mes_raw = canon_period_df.get("Mes")
        canon_mes_num = canon_mes_raw.astype(str).str.extract(r"(\d{1,2})", expand=False) if canon_mes_raw is not None else None
        canon_period_df["Mes_calc"] = pd.to_numeric(canon_mes_num, errors="coerce")
        canon_period_df["Fecha"] = pd.to_datetime(canon_period_df.get("Fecha"), errors="coerce")
        canon_period_df["Año_calc"] = canon_period_df["Año_calc"].fillna(canon_period_df["Fecha"].dt.year)
        canon_period_df["Mes_calc"] = canon_period_df["Mes_calc"].fillna(canon_period_df["Fecha"].dt.month)
        canon_period_df = canon_period_df.dropna(subset=["Monto"])

        if not canon_period_df.empty:
            canon_period_df["Periodo"] = pd.to_datetime(
                dict(
                    year=canon_period_df["Año_calc"].fillna(1).astype(int),
                    month=canon_period_df["Mes_calc"].fillna(1).astype(int),
                    day=1,
                ),
                errors="coerce",
            )
            canon_mensual_promedio = (
                canon_period_df.dropna(subset=["Periodo"])
                .groupby("Periodo", as_index=False)["Monto"]
                .sum()["Monto"]
                .mean()
            )
        else:
            canon_mensual_promedio = 0.0

        ingresos_esp_df = df_f.loc[mask_ingreso & (mask_sit_pagado | mask_sit_abono), ["Monto", "Esp"]].copy()
        ingresos_esp_df["Esp_num"] = pd.to_numeric(
            ingresos_esp_df["Esp"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        ingresos_esp_df = ingresos_esp_df[ingresos_esp_df["Esp_num"].between(1, 7, inclusive="both")]
        espacios_con_ingreso = ingresos_esp_df["Esp_num"].nunique()
        ingreso_promedio_espacio = (
            ingresos_esp_df["Monto"].sum() / espacios_con_ingreso
            if espacios_con_ingreso
            else 0.0
        )

        m2_map_fin = {1: 120, 2: 72, 3: 72, 4: 72, 5: 180, 6: 130, 7: 60}
        canon_m2_df = canon_period_df.copy()
        canon_m2_df["Esp_num"] = pd.to_numeric(canon_m2_df["Esp"], errors="coerce")
        canon_m2_df["m2"] = canon_m2_df["Esp_num"].map(m2_map_fin)
        canon_m2_df = canon_m2_df.dropna(subset=["m2"])
        canon_m2_promedio = (
            (canon_m2_df["Monto"] / canon_m2_df["m2"]).mean()
            if not canon_m2_df.empty
            else 0.0
        )

        electricidad_mask = (
            mask_egreso & mask_sit_pagado &
            (
                df_f["Obs"].str.contains(r"\bcge\b|electricidad", case=False, na=False, regex=True) |
                df_f["CC1"].str.contains(r"\bcge\b|electricidad", case=False, na=False, regex=True)
            )
        )
        egreso_electricidad = df_f.loc[electricidad_mask, "Monto"].abs().sum()
        pct_electricidad_egresos = (
            egreso_electricidad / abs(egresos_kpi)
            if egresos_kpi
            else 0.0
        )

        kpi_fin_df = pd.DataFrame(
            [
                {
                    "Indicador": "Caja disponible",
                    "Valor": fmt_clp_largo(balance_kpi),
                    "Lectura": "Saldo operativo disponible en cuenta corriente.",
                },
                {
                    "Indicador": "Posición neta",
                    "Valor": fmt_clp_largo(posicion_neta),
                    "Lectura": "Caja + cuentas por cobrar netas + egresos por pagar.",
                },
                {
                    "Indicador": "CxC neto",
                    "Valor": fmt_clp_largo(cuentas_por_cobrar_neto),
                    "Lectura": "Cuentas por cobrar descontando abonos registrados.",
                },
                {
                    "Indicador": "Resultado operativo",
                    "Valor": fmt_clp_largo(utilidad_operativa),
                    "Lectura": "Ingresos menos egresos pagados acumulados.",
                },
                {
                    "Indicador": "Margen neto",
                    "Valor": f"{margen_neto:.1%}",
                    "Lectura": "Resultado operativo sobre ingresos acumulados.",
                },
                {
                    "Indicador": "Avance de cobranza sobre cartera vencida",
                    "Valor": f"{pct_cobranza:.1%}",
                    "Lectura": "Abonos registrados sobre cartera acumulada en estado no pagado.",
                },
                {
                    "Indicador": "Caja / egreso mensual promedio",
                    "Valor": f"{cobertura_egresos:.2f}x",
                    "Lectura": "Caja disponible relativa al promedio mensual de egresos pagados.",
                },
                {
                    "Indicador": "Cobertura CAPEX",
                    "Valor": f"{cobertura_capex:.1%}",
                    "Lectura": "Canon acumulado respecto de la inversión total.",
                },
                {
                    "Indicador": "Concentración de deuda top 3",
                    "Valor": f"{concentracion_top3:.1%}",
                    "Lectura": "Participación de los 3 mayores responsables sobre la cartera vencida.",
                },
                {
                    "Indicador": "Ticket promedio de deuda",
                    "Valor": fmt_clp_largo(ticket_promedio_deuda),
                    "Lectura": "Monto promedio por transacción en estado no pagado.",
                },
                {
                    "Indicador": "Abono promedio por responsable",
                    "Valor": fmt_clp_largo(abono_promedio_responsable),
                    "Lectura": "Abono promedio entre responsables con registros de abono.",
                },
                {
                    "Indicador": "Edad promedio de cartera vencida",
                    "Valor": edad_promedio_cartera_fmt,
                    "Lectura": "Antigüedad promedio de documentos no pagados con fecha válida.",
                },
                {
                    "Indicador": "Cobertura caja / cartera vencida neta",
                    "Valor": f"{cobertura_caja_cartera:.2f}x",
                    "Lectura": "Caja Banco BCI relativa a cuentas por cobrar netas.",
                },
                {
                    "Indicador": "Canon mensual promedio",
                    "Valor": fmt_clp_largo(canon_mensual_promedio),
                    "Lectura": "Promedio mensual de canon de arriendo sobre períodos con datos.",
                },
                {
                    "Indicador": "Variación YoY del canon",
                    "Valor": f"{var_yoy:+.1%}",
                    "Lectura": f"Variación del canon anual entre {anio_prev if valor_prev is not None else 'N/D'} y {ultimo_anio}.",
                },
                {
                    "Indicador": "Ingreso promedio por espacio",
                    "Valor": fmt_clp_largo(ingreso_promedio_espacio),
                    "Lectura": "Ingreso acumulado promedio entre espacios con ingresos registrados.",
                },
                {
                    "Indicador": "Canon por m² promedio",
                    "Valor": f"${canon_m2_promedio:,.0f}/m²",
                    "Lectura": "Canon promedio por metro cuadrado a partir de los espacios 1 al 7.",
                },
                {
                    "Indicador": "% electricidad sobre egresos",
                    "Valor": f"{pct_electricidad_egresos:.1%}",
                    "Lectura": "Peso de egresos asociados a electricidad/CGE sobre egresos pagados.",
                },
            ]
        )

        _render_table(
            kpi_fin_df,
            header_bg="#163A5F",
            header_fg="white",
            row_alt="#F8FAFC",
            compact=False,
        )

# =========================================================
# ⚠️ TAB 2: RIESGOS & COBRANZAS
# =========================================================
if active_section == "⚠️ Riesgos & cobranzas":
              # ---------- Resumen por Responsable (NO PAGADO vs Abonos) ----------
    st.markdown(
        section_heading("📋", "Cuentas por Cobrar / Pagar", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )

    df_cob = df_f.copy()

    # --- Cálculos base ---
    df_np = df_cob[df_cob["Sit"] == "NO PAGADO"]
    df_abonos = df_cob[df_cob["Obs"].astype(str).str.contains("abono", case=False, na=False)]

    no_pagado_grouped = (
        df_np.groupby("Responsable")["Monto"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Monto NO PAGADO", "count": "Transacciones NO PAGADO"})
    )
    abonos_grouped = (
        df_abonos.groupby("Responsable")["Monto"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Monto Abonos", "count": "Cantidad Abonos"})
    )

    resumen = no_pagado_grouped.join(abonos_grouped, how="outer").fillna(0)
    resumen["Deuda"] = resumen["Monto NO PAGADO"] - resumen["Monto Abonos"]
    resumen["% Abonado"] = (
        resumen["Monto Abonos"] / resumen["Monto NO PAGADO"]
    ).replace([pd.NA, pd.NaT], 0).fillna(0)
    resumen["Progreso"] = resumen["% Abonado"].clip(lower=0, upper=1)

    def badge_pct(p):
        if p >= 1:
            return "🟢 En curso"
        if p >= 0.5:
            return "🟠 En curso"
        return "🔴 Bajo"

    resumen["Estado"] = resumen["% Abonado"].apply(badge_pct)

    # Ordenar por deuda y preparar tabla
    resumen = resumen.sort_values("Deuda", ascending=False)
    tabla = resumen.reset_index()

    cols_order = [
        "Responsable",
        "Monto NO PAGADO",
        "Monto Abonos",
        "Deuda",
        "% Abonado",
        "Transacciones NO PAGADO",
        "Cantidad Abonos",
        "Estado",
        "Progreso",  # queda al final para la barra
    ]
    tabla = tabla[cols_order]

    # ---------- KPIs en formato “card” ----------
    total_deuda_neta = tabla["Deuda"].sum() if not tabla.empty else 0
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    kpi_titulo_deuda = (
        "DEUDA AL FIN DEL EJERCICIO"
        if total_deuda_neta < 0
        else "MONTO A FAVOR FIN DE EJERCICIO"
    )
    deuda_color = "#EF4444" if total_deuda_neta < 0 else "#10B981"
    bci_color = "#10B981" if balance_kpi >= 0 else "#EF4444"
    pos_neta_color = "#10B981" if posicion_neta >= 0 else "#EF4444"

    with col_kpi1:
        st.markdown(
            card_finanza(
                kpi_titulo_deuda,
                f"${total_deuda_neta:,.0f}",
                deuda_color,
                subtitulo="Resultado neto de cobros menos abonos",
                etiqueta="Cobranzas",
            ),
            unsafe_allow_html=True,
        )

    with col_kpi2:
        st.markdown(
            card_finanza(
                "CAJA BANCO BCI",
                f"${balance_kpi:,.0f}",
                bci_color,
                subtitulo="Saldo operacional consolidado",
                etiqueta="Liquidez",
            ),
            unsafe_allow_html=True,
        )

    with col_kpi3:
        st.markdown(
            card_finanza(
                "POSICIÓN NETA (CXC + EPP + BN)",
                f"${posicion_neta:,.0f}",
                pos_neta_color,
                subtitulo="Lectura global de exposición financiera",
                etiqueta="Resumen",
            ),
            unsafe_allow_html=True,
        )

    with col_kpi4:
        st.markdown(
            card_finanza(
                "AVANCE DE COBRANZA",
                f"{pct_cobranza:.1%}",
                "#A8A8A8",
                subtitulo="Abonos registrados sobre cartera vencida acumulada",
                etiqueta="Resumen",
                size="md",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 6px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Estado de cuentas por cobrar/pagar · Resumen por responsable
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Estilo visual de la tabla ----
    deuda_cmap = LinearSegmentedColormap.from_list(
        "deuda_palette",
        ["#A8A8A8", "#DCAA67", "#D85E5D", "#4B5563"],
    )
    styler = (
        tabla.style
        .format({
            "Monto NO PAGADO": "${:,.0f}",
            "Monto Abonos": "${:,.0f}",
            "Deuda": "${:,.0f}",
            "% Abonado": "{:.1%}",
            "Transacciones NO PAGADO": "{:,.0f}",
            "Cantidad Abonos": "{:,.0f}",
            "Progreso": "{:.0%}",   # ahora se ve 94% en vez de 0.94
        })
        .hide(axis="index")
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#163A5F"),
                    ("color", "white"),
                    ("font-weight", "700"),
                    ("font-size", "13px"),
                    ("border-bottom", "1px solid #0F2740"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("font-size", "12px"),
                    ("padding", "7px 8px"),
                    ("border-bottom", "1px solid #E5E7EB"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#F8FAFC")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#EEF2F7")],
            },
        ])
        .set_properties(subset=["Responsable"], **{"text-align": "left", "font-weight": "400"})
        .set_properties(subset=["Deuda"], **{"font-weight": "400", "color": "#B42318"})
        .set_properties(subset=["Monto NO PAGADO"], **{"font-weight": "400", "color": "#7A271A"})
        .set_properties(subset=["Monto Abonos"], **{"font-weight": "400", "color": "#027A48"})
        .background_gradient(subset=["Deuda"], cmap=deuda_cmap)
        .bar(subset=["Progreso"], color="#10B981")
    )

    st.dataframe(styler, use_container_width=True)

    st.markdown(
        section_heading("🧾", "Monto a cancelar por espacio (1 al 7)", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )
    st.caption("Detalle por concepto según Año y Mes, para Esp 1..7.")

    df_cancel = df_f.copy()
    df_cancel = df_cancel.dropna(subset=["Monto"])

    df_cancel["Año_sel"] = pd.to_numeric(df_cancel.get("Año"), errors="coerce")
    mes_raw_cancel = df_cancel.get("Mes")
    mes_num_cancel = mes_raw_cancel.astype(str).str.extract(r"(\d{1,2})", expand=False)
    df_cancel["Mes_sel"] = pd.to_numeric(mes_num_cancel, errors="coerce")

    # Respaldo con fecha cuando Año/Mes no vienen informados
    if df_cancel["Año_sel"].isna().all() or df_cancel["Mes_sel"].isna().all():
        df_cancel["Fecha"] = pd.to_datetime(df_cancel.get("Fecha"), errors="coerce")
        if df_cancel["Año_sel"].isna().all():
            df_cancel["Año_sel"] = df_cancel["Fecha"].dt.year
        if df_cancel["Mes_sel"].isna().all():
            df_cancel["Mes_sel"] = df_cancel["Fecha"].dt.month

    years_cancel = sorted(df_cancel["Año_sel"].dropna().astype(int).unique().tolist())
    year_opts_cancel = ["Todos"] + years_cancel

    c_can1, c_can2, c_can4, c_can5 = st.columns([1, 1, 1, 1.3])
    with c_can1:
        sel_year_cancel = st.selectbox(
            "Año (cancelación)",
            year_opts_cancel,
            index=0,
            key="year_cancel_esp",
        )
    with c_can2:
        sel_month_cancel = st.selectbox(
            "Mes (cancelación)",
            ["Todos"] + list(range(1, 13)),
            index=0,
            key="month_cancel_esp",
        )
    with c_can4:
        sel_esp_cancel = st.selectbox(
            "Espacio",
            ["Todos"] + list(range(1, 8)),
            index=0,
            key="esp_cancel_esp",
        )
    df_resp_opts = df_f.copy()
    df_resp_opts["Esp_num"] = pd.to_numeric(
        df_resp_opts["Esp"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    df_resp_opts = df_resp_opts[df_resp_opts["Esp_num"].between(1, 7, inclusive="both")]
    if sel_esp_cancel != "Todos":
        df_resp_opts = df_resp_opts[df_resp_opts["Esp_num"] == int(sel_esp_cancel)]
    responsables_opts_cancel = ["Todos"] + sorted(
        df_resp_opts["Responsable"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    with c_can5:
        sel_resp_cancel = st.selectbox(
            "Responsable",
            responsables_opts_cancel,
            index=0,
            key="resp_cancel_esp",
        )

    if sel_year_cancel != "Todos":
        df_cancel = df_cancel[df_cancel["Año_sel"] == sel_year_cancel]
    if sel_month_cancel != "Todos":
        df_cancel = df_cancel[df_cancel["Mes_sel"] == sel_month_cancel]

    df_cancel["Sit"] = df_cancel["Sit"].astype(str).str.strip().str.upper()
    df_cancel = df_cancel[df_cancel["Sit"].isin(["PAGADO", "NO PAGADO"])]

    df_cancel["Esp_num"] = pd.to_numeric(
        df_cancel["Esp"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    df_cancel = df_cancel[df_cancel["Esp_num"].between(1, 7, inclusive="both")]
    df_cancel["Esp_num"] = df_cancel["Esp_num"].astype(int)
    if sel_esp_cancel != "Todos":
        df_cancel = df_cancel[df_cancel["Esp_num"] == int(sel_esp_cancel)]
    if sel_resp_cancel != "Todos":
        df_cancel = df_cancel[df_cancel["Responsable"].astype(str).str.strip() == sel_resp_cancel]
    df_cancel_scope = df_cancel.copy()

    txt_cancel = (
        df_cancel["CC1"].astype(str).fillna("")
        + " "
        + df_cancel["Obs"].astype(str).fillna("")
    ).str.lower()

    df_cancel["Concepto"] = np.select(
        [
            txt_cancel.str.contains(r"canon\s*mensual", regex=True, na=False),
            txt_cancel.str.contains(r"gastos?\s*comunes?", regex=True, na=False),
            txt_cancel.str.contains(r"\bcge\b|boleta\s*cge|electricidad", regex=True, na=False),
            txt_cancel.str.contains(r"verisure", regex=True, na=False),
            txt_cancel.str.contains(r"administrativ[oa]", regex=True, na=False),
        ],
        ["Canon mensual", "Gastos comunes", "CGE", "Verisure", "Administrativo"],
        default="Otros",
    )

    conceptos_objetivo = ["Canon mensual", "Gastos comunes", "CGE", "Verisure", "Administrativo"]
    df_cancel = df_cancel[df_cancel["Concepto"].isin(conceptos_objetivo)].copy()
    df_cancel["Monto_abs"] = df_cancel["Monto"].abs()

    # Deuda por responsable: NO PAGADO - ABONO, excluyendo el período seleccionado.
    df_deuda = df_f.copy().dropna(subset=["Monto"])
    df_deuda["Sit"] = df_deuda["Sit"].astype(str).str.strip().str.upper()
    df_deuda["Obs"] = df_deuda["Obs"].astype(str)
    df_deuda["Responsable_clean"] = df_deuda["Responsable"].astype(str).str.strip()
    df_deuda = df_deuda[df_deuda["Responsable_clean"] != ""]

    df_deuda["Año_sel"] = pd.to_numeric(df_deuda.get("Año"), errors="coerce")
    mes_raw_deuda = df_deuda.get("Mes")
    mes_num_deuda = mes_raw_deuda.astype(str).str.extract(r"(\d{1,2})", expand=False)
    df_deuda["Mes_sel"] = pd.to_numeric(mes_num_deuda, errors="coerce")
    df_deuda["Fecha"] = pd.to_datetime(df_deuda.get("Fecha"), errors="coerce")
    df_deuda["Año_sel"] = df_deuda["Año_sel"].fillna(df_deuda["Fecha"].dt.year)
    df_deuda["Mes_sel"] = df_deuda["Mes_sel"].fillna(df_deuda["Fecha"].dt.month)

    df_deuda = df_deuda.dropna(subset=["Año_sel", "Mes_sel"])
    df_deuda["Periodo_ref"] = pd.to_datetime(
        dict(
            year=df_deuda["Año_sel"].astype(int),
            month=df_deuda["Mes_sel"].astype(int),
            day=1,
        ),
        errors="coerce",
    )
    df_deuda = df_deuda.dropna(subset=["Periodo_ref"])

    corte_periodo = None
    if sel_year_cancel != "Todos":
        if sel_month_cancel != "Todos":
            corte_periodo = pd.Timestamp(int(sel_year_cancel), int(sel_month_cancel), 1)
        else:
            corte_periodo = pd.Timestamp(int(sel_year_cancel), 1, 1)
    if corte_periodo is not None:
        df_deuda = df_deuda[df_deuda["Periodo_ref"] < corte_periodo]

    # Responsables válidos según el período seleccionado y filtros activos en la vista
    responsables_periodo = (
        df_cancel_scope["Responsable"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    if sel_resp_cancel != "Todos":
        responsables_periodo = [sel_resp_cancel]
    if responsables_periodo:
        df_deuda = df_deuda[df_deuda["Responsable_clean"].isin(responsables_periodo)]
    else:
        df_deuda = df_deuda.iloc[0:0]

    deuda_np_por_resp = (
        df_deuda[df_deuda["Sit"] == "NO PAGADO"]
        .groupby("Responsable_clean")["Monto"]
        .sum()
    )
    abonos_por_resp = (
        df_deuda[df_deuda["Obs"].str.contains("abono", case=False, na=False)]
        .groupby("Responsable_clean")["Monto"]
        .sum()
    )
    deuda_por_resp = deuda_np_por_resp - abonos_por_resp

    if sel_esp_cancel != "Todos":
        idx_esp = [int(sel_esp_cancel)]
    elif sel_resp_cancel != "Todos":
        idx_esp = sorted(
            df_f.loc[
                df_f["Responsable"].astype(str).str.strip() == sel_resp_cancel,
                "Esp"
            ]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .dropna()
            .astype(int)
            .loc[lambda s: s.between(1, 7)]
            .unique()
            .tolist()
        )
        if not idx_esp:
            idx_esp = list(range(1, 8))
    else:
        idx_esp = list(range(1, 8))

    if df_cancel.empty:
        tabla_cancel = pd.DataFrame(index=idx_esp)
    else:
        tabla_cancel = (
            df_cancel.groupby(["Esp_num", "Concepto"], as_index=False)["Monto_abs"]
            .sum()
            .pivot(index="Esp_num", columns="Concepto", values="Monto_abs")
            .fillna(0)
        )

    tabla_cancel = tabla_cancel.reindex(index=idx_esp, fill_value=0)
    for c in conceptos_objetivo:
        if c not in tabla_cancel.columns:
            tabla_cancel[c] = 0

    responsables_por_esp = (
        df_cancel_scope.assign(
            Responsable_clean=df_cancel_scope["Responsable"].astype(str).str.strip()
        )
        .loc[lambda d: d["Responsable_clean"] != ""]
        .groupby("Esp_num")["Responsable_clean"]
        .apply(lambda s: ", ".join(sorted(s.dropna().unique().tolist())))
    )
    responsables_lista_por_esp = (
        df_cancel_scope.assign(
            Responsable_clean=df_cancel_scope["Responsable"].astype(str).str.strip()
        )
        .loc[lambda d: d["Responsable_clean"] != ""]
        .groupby("Esp_num")["Responsable_clean"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
    )

    tabla_cancel = tabla_cancel[conceptos_objetivo]
    tabla_cancel.insert(
        0,
        "Responsable",
        tabla_cancel.index.to_series().map(responsables_por_esp).fillna("-"),
    )
    tabla_cancel["Deuda"] = tabla_cancel.index.to_series().map(
        lambda esp: sum(deuda_por_resp.get(r, 0) for r in responsables_lista_por_esp.get(esp, []))
    ).fillna(0)
    tabla_cancel["Total a cancelar"] = tabla_cancel[conceptos_objetivo + ["Deuda"]].sum(axis=1)
    tabla_cancel.index = [f"Esp {i}" for i in tabla_cancel.index]

    if df_cancel.empty:
        st.info("En el período seleccionado no hay cargos de conceptos, se muestra deuda acumulada previa.")

    periodo_lbl = (
        f"{int(sel_year_cancel)}-{int(sel_month_cancel):02d}"
        if sel_year_cancel != "Todos" and sel_month_cancel != "Todos"
        else (f"Año {int(sel_year_cancel)}" if sel_year_cancel != "Todos" else "Todos los períodos")
    )
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 6px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Estado de cobro arrendatarios · Periodo: {periodo_lbl}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabla_cancel_view = tabla_cancel.rename(columns={"Administrativo": "Interes/ otros"})
    cols_monto_cancel = [c for c in tabla_cancel_view.columns if c != "Responsable"]
    styler_cancel = (
        tabla_cancel_view.style
        .format("${:,.0f}", subset=cols_monto_cancel)
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#163A5F"),
                    ("color", "white"),
                    ("font-weight", "700"),
                    ("font-size", "13px"),
                    ("border-bottom", "1px solid #0F2740"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("font-size", "12px"),
                    ("padding", "7px 8px"),
                    ("border-bottom", "1px solid #E5E7EB"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#F8FAFC")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#EEF2F7")],
            },
        ])
        .set_properties(subset=["Responsable"], **{"text-align": "left", "font-weight": "600"})
        .set_properties(subset=["Deuda"], **{"font-weight": "700", "color": "#B42318"})
        .set_properties(subset=["Total a cancelar"], **{"font-weight": "800", "color": "#0F2D52"})
        .background_gradient(subset=["Total a cancelar"], cmap="Blues")
    )
    st.dataframe(
        styler_cancel,
        use_container_width=True,
    )

    esp_lbl_chart = str(sel_esp_cancel)
    resp_lbl_chart = str(sel_resp_cancel)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 10px 0 8px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Composición de cobro por espacio · Espacio: {esp_lbl_chart} · Responsable: {resp_lbl_chart}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Barras apiladas por concepto + deuda, con línea de total a cancelar.")

    import plotly.graph_objects as go

    chart_cols = conceptos_objetivo + ["Deuda"]
    chart_df = tabla_cancel.reset_index().rename(columns={"index": "Espacio"}).copy()
    single_space_view = len(chart_df) == 1
    single_month_one_space = (
        single_space_view
        and sel_year_cancel != "Todos"
        and sel_month_cancel != "Todos"
        and str(sel_esp_cancel).strip().lower() not in {"todos", "todas"}
    )

    fig_cancel = go.Figure()
    color_map = {
        "Canon mensual": "#4B5563",
        "Gastos comunes": "#7FA6A2",
        "CGE": "#DCAA67",
        "Verisure": "#A8A8A8",
        "Administrativo": "#D85E5D",
        "Deuda": "#D85E5D",
    }

    if single_month_one_space:
        row = chart_df.iloc[0]
        raw_vals = pd.Series({c: float(pd.to_numeric(row[c], errors="coerce") or 0) for c in chart_cols})
        pos_vals = raw_vals[raw_vals > 0].sort_values(ascending=False)
        neg_total = float(raw_vals[raw_vals < 0].sum())
        total_single = float(pd.to_numeric(row["Total a cancelar"], errors="coerce") or 0)

        if not pos_vals.empty:
            fig_cancel.add_trace(
                go.Pie(
                    labels=pos_vals.index.tolist(),
                    values=pos_vals.values.tolist(),
                    hole=0.62,
                    sort=False,
                    marker=dict(
                        colors=[color_map.get(c, "#64748B") for c in pos_vals.index.tolist()],
                        line=dict(color="white", width=1),
                    ),
                    textinfo="percent",
                    textfont=dict(size=13, color="#FFFFFF"),
                    hovertemplate="<b>%{label}</b><br>Monto: $%{value:,.0f}<br>Participación: %{percent}<extra></extra>",
                )
            )
        else:
            # Sin componentes positivos: fallback simple para evitar gráfico vacío.
            fig_cancel.add_trace(
                go.Bar(
                    y=[chart_df["Espacio"].iloc[0]],
                    x=[total_single],
                    orientation="h",
                    name="Total a cancelar",
                    marker_color="#1D4ED8",
                    hovertemplate="<b>%{y}</b><br>Total: $%{x:,.0f}<extra></extra>",
                )
            )

        center_text = (
            f"<b>{chart_df['Espacio'].iloc[0]}</b><br>"
            f"{periodo_lbl}<br>"
            f"<span style='font-size:13px'>Total a cancelar</span><br>"
            f"<b>${total_single:,.0f}</b>"
        )
        fig_cancel.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=center_text,
            showarrow=False,
            align="center",
            font=dict(size=16, color="#0F172A"),
        )
        if neg_total < 0:
            fig_cancel.add_annotation(
                x=0.5,
                y=0.06,
                xref="paper",
                yref="paper",
                text=f"Ajustes negativos considerados: ${neg_total:,.0f}",
                showarrow=False,
                font=dict(size=11, color="#B42318"),
            )
    elif single_space_view:
        for c in chart_cols:
            fig_cancel.add_trace(
                go.Bar(
                    y=chart_df["Espacio"],
                    x=chart_df[c],
                    orientation="h",
                    name=c,
                    marker_color=color_map.get(c, "#64748B"),
                    hovertemplate="<b>%{y}</b><br>" + c + ": $%{x:,.0f}<extra></extra>",
                )
            )
        total_single = float(chart_df["Total a cancelar"].iloc[0])
        fig_cancel.add_annotation(
            x=total_single,
            y=chart_df["Espacio"].iloc[0],
            text=f"Total: ${total_single:,.0f}",
            showarrow=False,
            xshift=14,
            font=dict(size=12, color="#0F172A"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(15,45,82,0.20)",
            borderwidth=1,
        )
    else:
        for c in chart_cols:
            fig_cancel.add_trace(
                go.Bar(
                    x=chart_df["Espacio"],
                    y=chart_df[c],
                    name=c,
                    marker_color=color_map.get(c, "#64748B"),
                    hovertemplate="<b>%{x}</b><br>" + c + ": $%{y:,.0f}<extra></extra>",
                )
            )
        fig_cancel.add_trace(
            go.Scatter(
                x=chart_df["Espacio"],
                y=chart_df["Total a cancelar"],
                mode="lines+markers+text",
                name="Total a cancelar",
                line=dict(color="#4B5563", width=3),
                marker=dict(size=8, color="#4B5563"),
                text=[f"${v:,.0f}" for v in chart_df["Total a cancelar"]],
                textposition="top center",
                hovertemplate="<b>%{x}</b><br>Total: $%{y:,.0f}<extra></extra>",
            )
        )

    fig_cancel.update_layout(
        barmode="stack",
        template="plotly_white",
        height=430 if single_month_one_space else (340 if single_space_view else 500),
        margin=dict(l=20, r=20, t=140, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            x=0.5 if single_month_one_space else 0.01,
            xanchor="center" if single_month_one_space else "left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(15,45,82,0.15)",
            borderwidth=1,
        ),
        title=dict(
            text="📊 Composición de Cobro por Espacio",
            x=0.01,
            xanchor="left",
            font=dict(size=18, color="#0F2D52"),
            pad=dict(b=14),
        ),
        xaxis_title=("Monto (CLP)" if single_space_view else "Espacio"),
        yaxis_title=("" if single_space_view else "Monto (CLP)"),
        hovermode="closest" if single_month_one_space else ("y unified" if single_space_view else "x unified"),
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#FFFFFF",
    )
    if single_month_one_space:
        fig_cancel.update_xaxes(visible=False, showgrid=False, zeroline=False)
        fig_cancel.update_yaxes(visible=False, showgrid=False, zeroline=False)
    elif single_space_view:
        fig_cancel.update_xaxes(
            tickformat=",.0f",
            gridcolor="rgba(15,45,82,0.10)",
            zeroline=False,
            linecolor="rgba(15,45,82,0.20)",
        )
        fig_cancel.update_yaxes(showgrid=False, linecolor="rgba(15,45,82,0.25)")
    else:
        fig_cancel.update_xaxes(showgrid=False, linecolor="rgba(15,45,82,0.25)")
        fig_cancel.update_yaxes(
            tickformat=",.0f",
            gridcolor="rgba(15,45,82,0.10)",
            zeroline=False,
            linecolor="rgba(15,45,82,0.20)",
        )

    st.plotly_chart(
        fig_cancel,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["toImage"],
        },
    )

    # ---- Exportar PDF / CSV ----
    from reportlab.lib.pagesizes import A4, A3, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import cm

    def export_resumen_pdf(df_in: pd.DataFrame, filename: str = "resumen_no_pagado_abonos.pdf"):
        short_names = {
            "Monto NO PAGADO": "Monto NO PAG.",
            "Transacciones NO PAGADO": "Transacc. NO PAG.",
            "Cantidad Abonos": "Cant. Abonos",
            "Monto Abonos": "Monto Abonos",
            "% Abonado": "% Abonado",
        }
        df_local = df_in.rename(columns={c: short_names.get(c, c) for c in df_in.columns})

        ncols = len(df_local.columns)
        page_size = landscape(A4) if ncols <= 8 else landscape(A3)

        doc = SimpleDocTemplate(
            filename,
            pagesize=page_size,
            leftMargin=1*cm,
            rightMargin=1*cm,
            topMargin=1*cm,
            bottomMargin=1*cm,
        )

        base = ParagraphStyle(
            "base", fontSize=8.5, leading=10,
            spaceBefore=0, spaceAfter=0,
            wordWrap="CJK", splitLongWords=True,
        )
        sty_h = ParagraphStyle("head", parent=base, alignment=TA_CENTER, fontSize=9, leading=11)
        sty_l = ParagraphStyle("left", parent=base, alignment=TA_LEFT)
        sty_r = ParagraphStyle("right", parent=base, alignment=TA_RIGHT)

        story = [
            Paragraph("<b>Resumen por Responsable (NO PAGADO vs Abonos)</b>",
                      getSampleStyleSheet()["Title"]),
            Spacer(1, 0.4*cm),
        ]

        text_cols = {"Responsable", "Estado"}
        header = [Paragraph(col, sty_h) for col in df_local.columns]
        rows = []
        for _, r in df_local.iterrows():
            row = []
            for col, val in zip(df_local.columns, r):
                if pd.isna(val):
                    s = ""
                elif col in {"% Abonado"}:
                    s = f"{float(val):,.1f} %"
                elif isinstance(val, (int, float)) and col not in text_cols:
                    s = f"{val:,.0f}"
                else:
                    s = str(val)
                row.append(Paragraph(s, sty_l if col in text_cols else sty_r))
            rows.append(row)
        data = [header] + rows

        page_width = doc.width
        weights = []
        for c in df_local.columns:
            if c.lower().startswith("responsable"):
                weights.append(3.8)
            elif "Transacc" in c or "Cant." in c or c == "Estado":
                weights.append(1.2)
            else:
                weights.append(2.0)
        total_w = sum(weights)
        col_widths = [max(1.6*cm, page_width*(w/total_w)) for w in weights]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.Color(0.98,0.98,0.98), colors.white]),
        ]))

        story.append(tbl)
        doc.build(story)
    st.markdown("---")



    try:
        pdf_path = "resumen_no_pagado_abonos.pdf"
        export_resumen_pdf(tabla, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Descargar PDF",
                data=f.read(),
                file_name=pdf_path,
                mime="application/pdf",
            )
    except Exception as e:
        st.warning(f"No se pudo generar PDF. Te dejo el CSV como alternativa. Detalle: {e}")
        st.download_button(
            "⬇️ Descargar CSV",
            data=tabla.to_csv(index=False).encode("utf-8"),
            file_name="resumen_no_pagado_abonos.csv",
            mime="text/csv",
        )

# =========================================================
# 🏢 TAB 3: CANON ANUAL / MENSUAL
# =========================================================
if active_section == "🏢 Canon anual / mensual":
    st.markdown(
        section_heading("🏢", "Canon mensual por Año y por Esp", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )

    import plotly.graph_objects as go

    _data_src = df_f
    esps_tab_canon = sorted(
        pd.to_numeric(_data_src.get("Esp"), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    sel_esps_tab_canon = st.multiselect(
        "Espacios (aplica a todos los gráficos de esta pestaña)",
        options=esps_tab_canon,
        default=esps_tab_canon,
        key="sel_esps_tab_canon",
        placeholder="Selecciona uno o más espacios",
    )

    mask_canon_mensual = (
        (_data_src["CC"] == "INGRESO") &
        (
            _data_src["CC1"].str.contains("canon mensual", case=False, na=False) |
            _data_src["Obs"].str.contains("canon mensual", case=False, na=False)
        )
    )
    dm = _data_src.loc[mask_canon_mensual].copy()

    dm["Año"] = pd.to_numeric(dm["Año"], errors="coerce")
    dm["Esp"] = pd.to_numeric(dm["Esp"], errors="coerce")
    dm["Monto"] = pd.to_numeric(
        dm["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce"
    )
    mes_raw_canon = dm.get("Mes")
    mes_num_canon = mes_raw_canon.astype(str).str.extract(r"(\d{1,2})", expand=False)
    dm["Mes_num"] = pd.to_numeric(mes_num_canon, errors="coerce")
    dm["Fecha"] = pd.to_datetime(dm.get("Fecha"), errors="coerce")
    dm["Año_eff"] = dm["Año"].fillna(dm["Fecha"].dt.year)
    dm["Mes_eff"] = dm["Mes_num"].fillna(dm["Fecha"].dt.month)
    dm = dm.dropna(subset=["Esp"])
    dm["Esp"] = dm["Esp"].astype(int)
    dm = dm.dropna(subset=["Año_eff", "Mes_eff"])
    dm["Periodo"] = pd.to_datetime(
        dict(
            year=dm["Año_eff"].astype(int),
            month=dm["Mes_eff"].astype(int),
            day=1,
        ),
        errors="coerce",
    )
    dm = dm.dropna(subset=["Periodo"])

    agg = (
        dm.groupby(["Periodo","Esp"], as_index=False)["Monto"]
          .sum()
          .sort_values(["Periodo","Esp"])
    )

    all_periods = (
        pd.date_range(agg["Periodo"].min(), agg["Periodo"].max(), freq="MS")
        if not agg.empty else []
    )
    all_esps  = sorted(agg["Esp"].dropna().unique())
    grid = pd.MultiIndex.from_product([all_periods, all_esps], names=["Periodo","Esp"])
    agg_full = (
        agg.set_index(["Periodo","Esp"])
           .reindex(grid, fill_value=0)
           .reset_index()
    )

    st.caption("El gráfico muestra el canon mensual total por mes y por espacio (Esp).")

    if len(all_esps) == 0:
        st.info("No hay datos de 'canon mensual' para mostrar.")
    else:
        esps_validos = [e for e in sel_esps_tab_canon if e in all_esps]
        if esps_validos:
            agg_full = agg_full[agg_full["Esp"].isin(esps_validos)]
            esp_sel_lbl = ", ".join([str(e) for e in esps_validos])
        else:
            agg_full = agg_full.iloc[0:0]
            esp_sel_lbl = "Ninguno"
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
                border-radius: 10px;
                padding: 10px 14px;
                margin: 8px 0 10px 0;
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 600;">
                Evolución de canon mensual · Espacios seleccionados: {esp_sel_lbl}
            </div>
            """,
            unsafe_allow_html=True,
        )

        palette = [
            "#1D4ED8","#15803D","#B45309","#B42318","#475467",
            "#0E7490","#334155","#0369A1","#854D0E","#166534",
            "#9333EA","#BE123C"
        ]

        fig_line = go.Figure()
        end_labels = []

        def _fmt_label_clp(v: float) -> str:
            v = float(v)
            s = f"{v:,.0f}".replace(",", ".")
            return f"${s} CLP"

        for i, esp in enumerate(sorted(agg_full["Esp"].unique())):
            df_e = agg_full[agg_full["Esp"] == esp]
            color_esp = palette[i % len(palette)]
            fig_line.add_trace(go.Scatter(
                x=df_e["Periodo"], y=df_e["Monto"],
                mode="lines+markers",
                name=f"Esp {esp}",
                line=dict(width=2.4, color=color_esp),
                marker=dict(
                    size=4,
                    opacity=0.75,
                    color=color_esp,
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x|%b %Y}</b><br>Esp: "+str(esp)+"<br>Monto: $%{y:,.0f}<extra></extra>",
            ))

            if not df_e.empty:
                last_row = df_e.iloc[-1]
                end_labels.append(
                    {
                        "x": last_row["Periodo"],
                        "y": float(last_row["Monto"]),
                        "text": f"Esp {esp}  {_fmt_label_clp(last_row['Monto'])}",
                        "color": color_esp,
                    }
                )

        # Etiquetas de cierre en columna derecha (fuera del plot), ordenadas mayor -> menor.
        if end_labels:
            labels_sorted = sorted(end_labels, key=lambda d: d["y"], reverse=True)
            n = len(labels_sorted)
            # Reparto vertical homogéneo en coordenadas del "paper" para lectura estable.
            y_top = 0.88
            y_bot = 0.22
            if n == 1:
                y_slots = [0.55]
            else:
                y_slots = list(np.linspace(y_top, y_bot, n))

            for lbl, y_slot in zip(labels_sorted, y_slots):
                fig_line.add_annotation(
                    x=1.01,
                    xref="paper",
                    y=y_slot,
                    yref="paper",
                    text=lbl["text"],
                    showarrow=False,
                    xanchor="left",
                    align="left",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=lbl["color"],
                    borderwidth=1,
                    font=dict(color="#0F172A", size=11),
                )

        visibility_all = [True] * len(fig_line.data)
        visibility_none = [False] * len(fig_line.data)
        fig_line.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1, xanchor="right", y=1.15, yanchor="top",
                    buttons=[
                        dict(label="Mostrar todo", method="update", args=[{"visible": visibility_all}]),
                        dict(label="Ocultar todo", method="update", args=[{"visible": visibility_none}]),
                    ]
                )
            ]
        )

        fig_line.update_layout(
            title=dict(
                text="🏢 Canon mensual por Mes y Esp",
                x=0.02,
                xanchor="left",
                font=dict(size=18, color="#0F2D52"),
            ),
            xaxis_title="Período (Mes)",
            yaxis_title="Monto (CLP)",
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=20, r=180, t=78, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(15,45,82,0.15)",
                borderwidth=1,
            ),
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
        )
        fig_line.update_xaxes(
            showgrid=False,
            linecolor="rgba(15,45,82,0.25)",
            tickformat="%Y",
            tickformatstops=[
                dict(dtickrange=[None, "M12"], value="%b %Y"),
                dict(dtickrange=["M12", None], value="%Y"),
            ],
        )
        fig_line.update_yaxes(
            showgrid=True,
            gridcolor="rgba(15,45,82,0.10)",
            zeroline=False,
            tickformat=",.0f",
            linecolor="rgba(15,45,82,0.20)",
        )
        fig_line.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

        st.plotly_chart(fig_line, use_container_width=True, config={
            "displaylogo": False,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["toImage","drawline","drawrect","eraseshape"]
        })

# =========================================================
# 🧩 CANON POR M² (integrado en TAB CANON)
# =========================================================
if active_section == "🏢 Canon anual / mensual":
    st.markdown(
        section_heading("🧩", "Canon por m² — Canon Mensual (por Año y Esp)", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 8px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Control de valor por m² · Canon mensual por espacio y año
        </div>
        """,
        unsafe_allow_html=True,
    )

    import numpy as np
    import plotly.graph_objects as go
    from io import BytesIO

    c1, c2 = st.columns([1,1])
    with c1:
        escala_m2 = st.selectbox(
            "Escala",
            ["Mensual", "Diario"],
            index=0,
            key="escala_m2_final",
        )
    with c2:
        uf_url = st.text_input("URL CSV UF (opcional: Año,UF_promedio)", value="", key="uf_url_m2_final",
                               placeholder="https://.../uf_promedio_anual.csv")

    _df = df_f

    canon_mask = (
        (_df["CC"] == "INGRESO") &
        (
            _df["CC1"].str.contains(r"canon\s*mensual|canon.*arriendo|arriendo.*mensual", case=False, na=False) |
            _df["Obs"].str.contains(r"canon\s*mensual|canon.*arriendo|arriendo.*mensual", case=False, na=False)
        )
    )
    dm = _df.loc[canon_mask].copy()

    dm["Año"] = pd.to_numeric(dm["Año"], errors="coerce")
    dm["Esp"] = pd.to_numeric(dm["Esp"], errors="coerce")
    dm["Monto"] = pd.to_numeric(dm["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True), errors="coerce")
    dm = dm.dropna(subset=["Año","Esp","Monto"])
    dm["Año"] = dm["Año"].astype(int)
    dm["Esp"] = dm["Esp"].astype(int)

    M2_MAP = {1:120, 2:72, 3:72, 4:72, 5:180, 6:130, 7:60}
    dm["m2"] = dm["Esp"].map(M2_MAP)
    dm = dm[dm["m2"].notna()]

    dm["Canon_m2_mes"] = dm["Monto"] / dm["m2"]
    dm["Canon_m2_dia"] = dm["Canon_m2_mes"] / 30.0

    agg = (
        dm.groupby(["Año","Esp"], as_index=False)[["Canon_m2_mes","Canon_m2_dia"]]
          .mean()
          .sort_values(["Año","Esp"])
          .copy()
    )

    years_all = (
        pd.to_numeric(_df["Año"], errors="coerce")
          .dropna()
          .astype(int)
          .sort_values()
          .unique()
          .tolist()
    )
    if not years_all:
        years_all = sorted(agg["Año"].unique().tolist())

    valor_col = "Canon_m2_mes" if escala_m2 == "Mensual" else "Canon_m2_dia"
    agg["valor_m2_clp"] = agg[valor_col]

    agg["UF_promedio"] = np.nan
    if uf_url.strip():
        try:
            uf_df = pd.read_csv(uf_url)
            uf_df = uf_df.rename(columns={c: c.strip() for c in uf_df.columns})
            if {"Año","UF_promedio"}.issubset(uf_df.columns):
                uf_df["Año"] = pd.to_numeric(uf_df["Año"], errors="coerce").astype("Int64")
                uf_df["UF_promedio"] = pd.to_numeric(uf_df["UF_promedio"], errors="coerce")
                agg = agg.merge(
                    uf_df.dropna(subset=["Año","UF_promedio"]).astype({"Año":"int"}),
                    on="Año", how="left"
                )
            else:
                st.warning("El CSV de UF debe tener columnas: Año,UF_promedio. Se omite UF.")
        except Exception as e:
            st.warning(f"No se pudo leer UF desde la URL: {e}")

    monedas = ["CLP"] + (["UF"] if agg["UF_promedio"].notna().any() else [])
    moneda_m2 = st.selectbox(
        "Moneda",
        monedas,
        index=0,
        key="moneda_m2_final",
    )

    if moneda_m2 == "UF":
        agg["valor_m2"] = agg["valor_m2_clp"] / agg["UF_promedio"]
    else:
        agg["valor_m2"] = agg["valor_m2_clp"]

    todos_esps = sorted(agg["Esp"].unique().tolist())
    sel_esps = [e for e in sel_esps_tab_canon if e in todos_esps]
    agg = agg[agg["Esp"].isin(sel_esps)] if sel_esps else agg.iloc[0:0]

    plot_df = (
        agg.pivot_table(index="Año", columns="Esp", values="valor_m2", aggfunc="mean")
           .reindex(years_all)
           .fillna(0)
    )

    if plot_df.empty:
        st.info("No hay datos para el filtro actual de 'canon mensual' o para los Esp seleccionados.")
    else:
        fig = go.Figure()
        x_years = plot_df.index.astype(int).values
        end_labels_m2 = []

        palette = [
            "#1D4ED8","#15803D","#B45309","#B42318","#475467",
            "#0E7490","#334155","#0369A1","#854D0E","#166534",
            "#9333EA","#BE123C","#0F766E","#7C3AED"
        ]

        for i, esp in enumerate(plot_df.columns):
            y_series = plot_df[esp].values
            color_esp = palette[i % len(palette)]
            custom = np.where(y_series == 0, "⚠️ Sin registros de ‘canon mensual’", " ")
            fig.add_trace(go.Scatter(
                x=x_years, y=y_series, customdata=custom,
                mode="lines+markers",
                name=f"Esp {esp}",
                line=dict(width=2.4, color=color_esp),
                marker=dict(size=4, opacity=0.75, color=color_esp, line=dict(width=0)),
                hovertemplate="<b>Año %{x}</b><br>Esp: "+str(esp)+
                              "<br>Valor: %{y:,.2f}"+(" UF/m²" if moneda_m2=="UF" else " CLP/m²")+
                              "<br>%{customdata}<extra></extra>"
            ))

            if len(y_series) and pd.notna(y_series[-1]):
                end_labels_m2.append(
                    {
                        "y": float(y_series[-1]),
                        "text": f"Esp {esp}  {y_series[-1]:,.2f}{' UF/m²' if moneda_m2=='UF' else ' CLP/m²'}",
                        "color": color_esp,
                    }
                )

        if end_labels_m2:
            labels_sorted = sorted(end_labels_m2, key=lambda d: d["y"], reverse=True)
            n = len(labels_sorted)
            y_top = 0.88
            y_bot = 0.22
            y_slots = [0.55] if n == 1 else list(np.linspace(y_top, y_bot, n))
            for lbl, y_slot in zip(labels_sorted, y_slots):
                fig.add_annotation(
                    x=1.01,
                    xref="paper",
                    y=y_slot,
                    yref="paper",
                    text=lbl["text"],
                    showarrow=False,
                    xanchor="left",
                    align="left",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=lbl["color"],
                    borderwidth=1,
                    font=dict(color="#0F172A", size=11),
                )

        titulo_y = "UF/m²" if moneda_m2 == "UF" else "CLP/m²"
        titulo_esc = "mensual" if escala_m2 == "Mensual" else "diario"
        fig.update_layout(
            title=dict(
                text=f"🧩 Canon por m² ({titulo_esc}) — {moneda_m2} · por Año y Esp",
                x=0.02,
                xanchor="left",
                font=dict(size=18, color="#0F2D52"),
            ),
            xaxis_title="Año",
            yaxis_title=titulo_y,
            template="plotly_white",
            hovermode="x",
            margin=dict(l=20, r=220, t=78, b=20),
            legend=dict(
                orientation="h",
                y=1.02,
                x=0.02,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(15,45,82,0.15)",
                borderwidth=1,
            ),
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
            updatemenus=[dict(type="buttons", direction="right", x=1, xanchor="right", y=1.15, yanchor="top",
                              buttons=[dict(label="Mostrar todo", method="update", args=[{"visible":[True]*len(plot_df.columns)}]),
                                       dict(label="Ocultar todo", method="update", args=[{"visible":[False]*len(plot_df.columns)}])])]
        )

        if len(x_years):
            fig.update_xaxes(
                type="linear",
                tickmode="linear",
                tick0=int(x_years.min()),
                dtick=1,
                range=[int(x_years.min()) - 0.5, int(x_years.max()) + 0.5],
                showgrid=False,
                linecolor="rgba(15,45,82,0.25)"
            )

        y_min = 0
        y_max = plot_df.replace(0, np.nan).max().max()
        y_max = float(y_max) * 1.1 if pd.notna(y_max) else 1.0
        fig.update_yaxes(
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="rgba(15,45,82,0.10)",
            zeroline=False,
            linecolor="rgba(15,45,82,0.20)",
        )

        st.plotly_chart(
            fig, use_container_width=True,
            config={"displaylogo": False, "displayModeBar": True,
                    "modeBarButtonsToAdd": ["toImage","drawline","drawrect","eraseshape"]}
        )

    # --- Tabla + Excel ---
    escala_lbl = "Mensual" if escala_m2 == "Mensual" else "Diario"
    st.markdown(
        section_heading("📄", f"Dataset agregado (Canon/m² — Año x Esp · {escala_lbl})", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )

    df_x = plot_df.reset_index().rename(columns={"index": "Año"}).copy()
    esp_cols_x = [c for c in df_x.columns if c != "Año"]
    df_display = df_x.copy()

    def miles_punto(n):
        s = f"{n:,.0f}"
        return s.replace(",", ".")

    def uf_chileno(n):
        s = f"{n:,.2f}"
        s = s.replace(",", "§")
        s = s.replace(".", ",")
        s = s.replace("§", ".")
        return s

    if moneda_m2 == "CLP":
        for c in esp_cols_x:
            df_display[c] = df_display[c].round(0).apply(lambda v: f"${miles_punto(v)}")
    else:
        for c in esp_cols_x:
            df_display[c] = df_display[c].round(2).apply(uf_chileno)

    if moneda_m2 == "CLP":
        df_display = df_display.rename(
            columns={c: f"Esp {c} (CLP/m² · {escala_lbl})" for c in esp_cols_x}
        )
    else:
        df_display = df_display.rename(
            columns={c: f"Esp {c} (UF/m² · {escala_lbl})" for c in esp_cols_x}
        )

    # Quitar años sin datos (todo 0) antes de insertar la fila m²
    if not df_x.empty and esp_cols_x:
        mask_has_data = (df_x[esp_cols_x].fillna(0).abs().sum(axis=1) > 0)
        df_x = df_x.loc[mask_has_data].copy()
        df_display = df_display.loc[mask_has_data.values].copy()

    # Fila m² justo debajo de la primera fila del dataset
    cols_esp_display = [c for c in df_display.columns if c != "Año"]
    fila_m2 = {"Año": "m²"}
    for col_name in cols_esp_display:
        esp_txt = str(col_name)
        esp_num = pd.to_numeric(esp_txt.replace("Esp ", "").split(" ")[0], errors="coerce")
        if pd.notna(esp_num):
            m2_val = M2_MAP.get(int(esp_num))
            fila_m2[col_name] = f"{int(m2_val):,} m²".replace(",", ".") if m2_val is not None else "-"
        else:
            fila_m2[col_name] = "-"

    if not df_display.empty:
        df_display = pd.concat(
            [pd.DataFrame([fila_m2]), df_display],
            ignore_index=True,
        )
    else:
        df_display = pd.DataFrame([fila_m2], columns=df_display.columns)

    # Tabla estilo Electricidad
    _render_table(
        df_display,
        header_bg="#1f4e78",
        header_fg="white",
        row_alt="#eef3fb",
        compact=False,
    )

    excel_buffer = BytesIO()
    df_x_export = df_x.copy()
    fila_m2_export = {"Año": "m²"}
    for c in esp_cols_x:
        esp_num = pd.to_numeric(str(c), errors="coerce")
        if pd.notna(esp_num):
            m2_val = M2_MAP.get(int(esp_num))
            fila_m2_export[c] = m2_val if m2_val is not None else ""
        else:
            fila_m2_export[c] = ""
    if not df_x_export.empty:
        df_x_export = pd.concat(
            [pd.DataFrame([fila_m2_export]), df_x_export],
            ignore_index=True,
        )
    else:
        df_x_export = pd.DataFrame([fila_m2_export], columns=df_x_export.columns)
    try:
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_x_export.to_excel(writer, index=False, sheet_name="canon_m2")
            wb = writer.book
            ws = writer.sheets["canon_m2"]

            head_fmt = wb.add_format({"bold": True})
            ws.set_row(0, None, head_fmt)

            ws.set_column(0, 0, 10)
            ws.set_column(1, 1, 14)
            if moneda_m2 == "CLP":
                numfmt = wb.add_format({"num_format": "$#,##0"})
                for j, c in enumerate(esp_cols_x, start=1):
                    ws.write(0, j, f"Esp {c} (CLP/m²)", head_fmt)
            else:
                numfmt = wb.add_format({"num_format": '#,##0.00'})
                for j, c in enumerate(esp_cols_x, start=1):
                    ws.write(0, j, f"Esp {c} (UF/m²)", head_fmt)

            ws.set_column(1, len(df_x_export.columns)-1, 14, numfmt)
    except ModuleNotFoundError:
        export_cols = ["Año"] + [
            f"Esp {c} ({'CLP/m²' if moneda_m2 == 'CLP' else 'UF/m²'})" for c in esp_cols_x
        ]
        df_x_export_fallback = df_x_export.copy()
        df_x_export_fallback.columns = export_cols
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_x_export_fallback.to_excel(writer, index=False, sheet_name="canon_m2")

    excel_buffer.seek(0)
    st.download_button(
        "⬇️ Descargar Excel (Canon/m² — Año x Esp)",
        data=excel_buffer.getvalue(),
        file_name=f"canon_m2_{'mensual' if escala_m2=='Mensual' else 'diario'}_{moneda_m2}_por_anio_y_esp.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# =========================================================
# 📈 TAB 5: INGRESOS & EGRESOS (Mensual / Anual)
# =========================================================
if active_section == "📈 Ingresos & egresos":
    st.markdown(
        section_heading("📈", "Ingresos vs Egresos — Totales por período", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 8px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Desempeño financiero por período · Control de ingresos, egresos y resultado neto
        </div>
        """,
        unsafe_allow_html=True,
    )

    import plotly.graph_objects as go

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        periodo = st.radio("Periodo", ["Mensual", "Anual"], horizontal=True, index=0, key="periodo_ing_eg")
    with c2:
        st.caption("Se usan montos desde la columna F (Monto). Ingresos y egresos según columna CC.")
    with c3:
        pass

    _df = df_f.copy()
    _df = _df.dropna(subset=["Monto", "CC"])
    _df["CC"] = _df["CC"].astype(str).str.strip().str.upper()
    _df = _df[_df["CC"].isin(["INGRESO", "EGRESO"])]

    _df["Sit"] = _df["Sit"].astype(str).str.strip().str.upper()
    _df = _df[_df["Sit"].isin(["PAGADO", "NO PAGADO"])]

    # Usar columnas de la base: "Año" y "Mes"
    _df["Año_sel"] = pd.to_numeric(_df.get("Año"), errors="coerce")
    mes_raw = _df.get("Mes")
    mes_num = (
        mes_raw.astype(str)
        .str.extract(r"(\d{1,2})", expand=False)
    )
    _df["Mes_sel"] = pd.to_numeric(mes_num, errors="coerce")

    # Fallback si la columna Mes no trae datos
    if _df["Mes_sel"].isna().all():
        st.warning("La columna 'Mes' no tiene datos. Usando la columna 'Fecha' como respaldo.")
        _df["Fecha"] = pd.to_datetime(_df.get("Fecha"), errors="coerce")
        _df["Año_sel"] = _df["Fecha"].dt.year
        _df["Mes_sel"] = _df["Fecha"].dt.month

    _df = _df.dropna(subset=["Año_sel"])

    years = sorted(_df["Año_sel"].dropna().astype(int).unique().tolist())
    year_opts = ["Todos"] + years

    c_year, c_month = st.columns([1, 1])
    with c_year:
        sel_year = st.selectbox("Año", year_opts, index=0, key="year_ing_eg")
    with c_month:
        month_opts = ["Todos"] + list(range(1, 13))
        sel_month = st.selectbox("Mes", month_opts, index=0, key="month_ing_eg",
                                 disabled=(periodo == "Anual"))

    if sel_year != "Todos":
        _df = _df[_df["Año_sel"] == sel_year]
    if sel_month != "Todos" and periodo == "Mensual":
        _df = _df[_df["Mes_sel"] == sel_month]

    # Construir Periodo usando Año/Mes
    if periodo == "Mensual":
        _df = _df.dropna(subset=["Mes_sel"])
        _df["Periodo"] = pd.to_datetime(
            dict(year=_df["Año_sel"].astype(int), month=_df["Mes_sel"].astype(int), day=1),
            errors="coerce"
        )
        label_x = "Mes"
        x_hover = "%b %Y"
    else:
        _df["Periodo"] = pd.to_datetime(
            dict(year=_df["Año_sel"].astype(int), month=1, day=1),
            errors="coerce"
        )
        label_x = "Año"
        x_hover = "%Y"

    _df = _df.dropna(subset=["Periodo"])
    _df["Periodo"] = pd.to_datetime(_df["Periodo"], errors="coerce")
    _df = _df.dropna(subset=["Periodo"])

    agg_ie = (
        _df.groupby(["Periodo", "CC"], as_index=False)["Monto"]
           .sum()
           .sort_values("Periodo")
    )

    ingresos = agg_ie[agg_ie["CC"] == "INGRESO"].rename(columns={"Monto": "Ingresos"})
    egresos = agg_ie[agg_ie["CC"] == "EGRESO"].rename(columns={"Monto": "Egresos"})

    base = pd.DataFrame({"Periodo": sorted(agg_ie["Periodo"].dropna().unique())})
    base["Periodo"] = pd.to_datetime(base["Periodo"], errors="coerce")
    base = base.merge(ingresos[["Periodo", "Ingresos"]], on="Periodo", how="left")
    base = base.merge(egresos[["Periodo", "Egresos"]], on="Periodo", how="left")
    base = base.fillna(0)

    if base.empty:
        st.info("No hay datos suficientes para calcular ingresos y egresos por período.")
    else:
        base["Egresos_abs"] = base["Egresos"].abs()
        base["Neto"] = base["Ingresos"] - base["Egresos_abs"]

        total_ing = float(base["Ingresos"].sum())
        total_egr = float(base["Egresos_abs"].sum())
        total_neto = float(base["Neto"].sum())

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                card_finanza("TOTAL INGRESO", fmt_clp_largo(total_ing), "#10B981"),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                card_finanza("TOTAL EGRESO", fmt_clp_largo(total_egr), "#EF4444"),
                unsafe_allow_html=True,
            )
        with k3:
            net_color = "#10B981" if total_neto >= 0 else "#EF4444"
            st.markdown(
                card_finanza("TOTAL NETO", fmt_clp_largo(total_neto), net_color),
                unsafe_allow_html=True,
            )

        fig_ie = go.Figure()

        fig_ie.add_trace(
            go.Scatter(
                x=base["Periodo"],
                y=base["Ingresos"],
                mode="lines+markers",
                name="Ingresos",
                line=dict(color="#10B981", width=3),
                marker=dict(size=6),
                hovertemplate=f"<b>{label_x} %{x_hover}</b><br>Ingresos: $%{{y:,.0f}}<extra></extra>",
            )
        )
        fig_ie.add_trace(
            go.Scatter(
                x=base["Periodo"],
                y=base["Egresos_abs"],
                mode="lines+markers",
                name="Egresos",
                line=dict(color="#EF4444", width=3),
                marker=dict(size=6),
                hovertemplate=f"<b>{label_x} %{x_hover}</b><br>Egresos: $%{{y:,.0f}}<extra></extra>",
            )
        )

        fig_ie.add_trace(
            go.Bar(
                x=base["Periodo"],
                y=base["Neto"],
                name="Neto",
                marker_color="#2563EB",
                opacity=0.25,
                hovertemplate=f"<b>{label_x} %{x_hover}</b><br>Neto: $%{{y:,.0f}}<extra></extra>",
            )
        )

        fig_ie.update_layout(
            title=dict(
                text=f"📈 Ingresos y Egresos — {periodo}",
                x=0.02,
                xanchor="left",
                font=dict(size=18, color="#0F2D52"),
            ),
            xaxis_title=label_x,
            yaxis_title="Monto (CLP)",
            template="plotly_white",
            height=520,
            margin=dict(l=20, r=20, t=78, b=20),
            legend=dict(
                orientation="h",
                y=1.02,
                x=0.02,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(15,45,82,0.15)",
                borderwidth=1,
            ),
            hovermode="x unified",
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
        )
        fig_ie.update_xaxes(showgrid=False, linecolor="rgba(15,45,82,0.25)")
        fig_ie.update_yaxes(
            showgrid=True,
            gridcolor="rgba(15,45,82,0.10)",
            zeroline=False,
            tickformat=",.0f",
            linecolor="rgba(15,45,82,0.20)",
        )

        st.plotly_chart(
            fig_ie,
            use_container_width=True,
            config={"displaylogo": False, "displayModeBar": True, "modeBarButtonsToAdd": ["toImage"]},
        )

        st.caption("Egresos se muestran en valor absoluto para facilitar comparación visual.")

    st.markdown("---")
    st.markdown(
        section_heading("⚠️", "Riesgos de cobro y concentración de montos", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 8px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Monitoreo de riesgos y concentración de montos · Vista analítica de cobranza
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        section_heading("📈", "Filtro por centro de costo / situación", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )

    c_top1, c_top2, c_top3, c_top4 = st.columns([2, 1, 1, 1])
    with c_top1:
        dim = st.selectbox("Dimensión", ["Obs", "CC1", "Sit", "CC"], index=1, key="dim_pro")
    with c_top2:
        order_by = st.radio(
            "Ordenar por",
            ["Total CLP", "N° Transacciones"],
            horizontal=True,
            index=0,
            key="order_by_pro",
        )
    with c_top3:
        chart_type = st.selectbox(
            "Visualización",
            ["Barras", "Treemap"],
            index=0,
            key="chart_type_pro",
        )
    with c_top4:
        top_n = st.slider(
            "Top N",
            min_value=5,
            max_value=30,
            value=16,
            step=1,
            key="topn_pro",
        )

    topN_raw = (
        df_f.groupby(dim)["Monto"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Total CLP", "count": "N° Transacciones"})
        .reset_index()
    )

    sort_col = "Total CLP" if order_by == "Total CLP" else "N° Transacciones"
    topN = topN_raw.sort_values(sort_col, ascending=False).head(top_n).copy()

    def color_by_sign_or_cc(series_values, series_dim=None):
        if series_dim == "CC":
            return series_values.map(
                {"INGRESO": "#10B981", "EGRESO": "#EF4444"}
            ).fillna("#2563EB")
        return series_values.apply(lambda v: "#10B981" if v >= 0 else "#EF4444")

    if chart_type == "Barras":
        top_keys = topN[dim].tolist()
        df_dim = df_f[df_f[dim].isin(top_keys)].copy()
        df_dim["CC"] = df_dim["CC"].astype(str).str.strip().str.upper()

        agg_cc = (
            df_dim.groupby([dim, "CC"], as_index=False)["Monto"]
            .sum()
        )

        ingresos = agg_cc[agg_cc["CC"] == "INGRESO"].rename(columns={"Monto": "Ingresos"})
        egresos = agg_cc[agg_cc["CC"] == "EGRESO"].rename(columns={"Monto": "Egresos"})

        base_dim = pd.DataFrame({dim: top_keys})
        base_dim = base_dim.merge(ingresos[[dim, "Ingresos"]], on=dim, how="left")
        base_dim = base_dim.merge(egresos[[dim, "Egresos"]], on=dim, how="left")
        base_dim = base_dim.fillna(0)

        base_dim["Egresos_abs"] = base_dim["Egresos"].abs()
        base_dim["Neto"] = base_dim["Ingresos"] - base_dim["Egresos_abs"]

        if order_by == "Total CLP":
            base_dim = base_dim.merge(topN[[dim, "Total CLP"]], on=dim, how="left")
            base_dim = base_dim.sort_values("Total CLP", ascending=True)
        else:
            base_dim = base_dim.merge(topN[[dim, "N° Transacciones"]], on=dim, how="left")
            base_dim = base_dim.sort_values("N° Transacciones", ascending=True)

        fig_top = go.Figure()

        fig_top.add_trace(
            go.Scatter(
                x=base_dim[dim],
                y=base_dim["Ingresos"],
                mode="lines+markers",
                name="Ingresos",
                line=dict(color="#10B981", width=3),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Ingresos: $%{y:,.0f}<extra></extra>",
            )
        )
        fig_top.add_trace(
            go.Scatter(
                x=base_dim[dim],
                y=base_dim["Egresos_abs"],
                mode="lines+markers",
                name="Egresos",
                line=dict(color="#EF4444", width=3),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Egresos: $%{y:,.0f}<extra></extra>",
            )
        )
        fig_top.add_trace(
            go.Bar(
                x=base_dim[dim],
                y=base_dim["Neto"],
                name="Neto",
                marker_color="#2563EB",
                opacity=0.25,
                hovertemplate="<b>%{x}</b><br>Neto: $%{y:,.0f}<extra></extra>",
            )
        )

        fig_top.update_layout(
            title=dict(
                text=f"📈 Top {top_n} por '{dim}' · {order_by}",
                x=0.02,
                xanchor="left",
                font=dict(size=18, color="#0F2D52"),
            ),
            xaxis_title=dim,
            yaxis_title="Monto (CLP)",
            template="plotly_white",
            margin=dict(l=20, r=20, t=72, b=20),
            legend=dict(
                orientation="h",
                y=1.04,
                x=0.02,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(15,45,82,0.15)",
                borderwidth=1,
            ),
            hovermode="x unified",
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
        )
        fig_top.update_xaxes(showgrid=False, linecolor="rgba(15,45,82,0.25)")
        fig_top.update_yaxes(
            showgrid=True,
            gridcolor="rgba(15,45,82,0.10)",
            zeroline=False,
            tickformat=",.0f",
            linecolor="rgba(15,45,82,0.20)",
        )

        st.plotly_chart(
            fig_top,
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["toImage"],
            },
        )
    else:
        treemap_df = topN.sort_values("Total CLP", ascending=False).copy()
        if dim == "CC":
            color_col = dim
            color_scale = None
        else:
            color_col = "Total CLP"
            color_scale = ["#EF4444", "#F59E0B", "#10B981"]

        fig_tree = px.treemap(
            treemap_df,
            path=[dim],
            values=sort_col if sort_col in treemap_df.columns else "Total CLP",
            color=color_col,
            color_continuous_scale=color_scale,
            title=f"🧭 Distribución Top {top_n} por '{dim}' · {order_by}",
        )
        if dim == "CC":
            fig_tree.update_traces(
                marker_colors=color_by_sign_or_cc(
                    treemap_df[dim], series_dim="CC"
                )
            )
        fig_tree.update_traces(
            hovertemplate="<b>%{label}</b><br>Valor: %{value:,.0f}<extra></extra>",
            textinfo="label+value",
            textfont=dict(size=12),
        )
        fig_tree.update_layout(
            margin=dict(l=0, r=0, t=72, b=0),
            template="plotly_white",
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
            title=dict(
                x=0.01,
                xanchor="left",
                font=dict(size=18, color="#0F2D52"),
            ),
        )
        st.plotly_chart(
            fig_tree,
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["toImage"],
            },
        )

    st.caption("Nota: los montos son la suma de 'Monto' (ingresos positivos, egresos negativos) por categoría.")

    st.markdown("---")
    st.markdown(
        section_heading("⚠️", "Detalle filtrable de movimientos", weight_class="section-heading-title-soft"),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f2d52 0%, #1f4e78 100%);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 8px 0 10px 0;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;">
            Monitoreo detallado por filtros · CC1, OBS y Responsable
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Filtra por CC1, OBS y Responsable para revisar el detalle al final de esta pestaña.")

    df_det = df_f.copy().dropna(subset=["Monto"]).copy()
    for c in ["CC1", "Obs", "Responsable"]:
        df_det[c] = df_det[c].astype(str).str.strip()

    d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
    with d1:
        cc1_opts = ["Todos"] + sorted([v for v in df_det["CC1"].dropna().unique().tolist() if v != ""])
        sel_cc1_det = st.selectbox("CC1", cc1_opts, index=0, key="det_cc1")

    df_det_f = df_det.copy()
    if sel_cc1_det != "Todos":
        df_det_f = df_det_f[df_det_f["CC1"] == sel_cc1_det]

    with d2:
        obs_opts = ["Todos"] + sorted([v for v in df_det_f["Obs"].dropna().unique().tolist() if v != ""])
        sel_obs_det = st.selectbox("OBS", obs_opts, index=0, key="det_obs")
    if sel_obs_det != "Todos":
        df_det_f = df_det_f[df_det_f["Obs"] == sel_obs_det]

    with d3:
        resp_opts = ["Todos"] + sorted([v for v in df_det_f["Responsable"].dropna().unique().tolist() if v != ""])
        sel_resp_det = st.selectbox("Responsable", resp_opts, index=0, key="det_resp")
    if sel_resp_det != "Todos":
        df_det_f = df_det_f[df_det_f["Responsable"] == sel_resp_det]

    with d4:
        if "Año" in df_det_f.columns:
            anio_vals = pd.to_numeric(df_det_f["Año"], errors="coerce").dropna().astype(int).unique().tolist()
            anio_opts = ["Todos"] + [str(y) for y in sorted(anio_vals)]
        else:
            anio_opts = ["Todos"]
        sel_anio_det = st.selectbox("Año", anio_opts, index=0, key="det_anio")
    if sel_anio_det != "Todos" and "Año" in df_det_f.columns:
        df_det_f = df_det_f[pd.to_numeric(df_det_f["Año"], errors="coerce").astype("Int64") == int(sel_anio_det)]

    # KPIs de detalle (debajo de selectores)
    sit_det = df_det_f["Sit"].astype(str).str.strip().str.upper() if "Sit" in df_det_f.columns else pd.Series([], dtype=str)
    monto_total_det = float(df_det_f["Monto"].sum()) if not df_det_f.empty else 0.0
    monto_por_pagar_det = float(df_det_f.loc[sit_det == "NO PAGADO", "Monto"].abs().sum()) if not df_det_f.empty else 0.0
    monto_pagado_det = float(df_det_f.loc[sit_det == "PAGADO", "Monto"].abs().sum()) if not df_det_f.empty else 0.0
    monto_abonos_det = float(
        df_det_f.loc[df_det_f["Obs"].astype(str).str.contains("abono", case=False, na=False), "Monto"].abs().sum()
    ) if not df_det_f.empty else 0.0

    k_det1, k_det2, k_det3, k_det4 = st.columns(4)
    with k_det1:
        st.markdown(
            card_finanza(
                "MONTO TOTAL (FILTRO)",
                fmt_clp_largo(monto_total_det),
                "#1D4ED8",
            ),
            unsafe_allow_html=True,
        )
    with k_det2:
        st.markdown(
            card_finanza(
                "MONTO POR PAGAR",
                fmt_clp_largo(monto_por_pagar_det),
                "#EF4444",
            ),
            unsafe_allow_html=True,
        )
    with k_det3:
        st.markdown(
            card_finanza(
                "MONTO PAGADO",
                fmt_clp_largo(monto_pagado_det),
                "#10B981",
            ),
            unsafe_allow_html=True,
        )
    with k_det4:
        st.markdown(
            card_finanza(
                "ABONOS",
                fmt_clp_largo(monto_abonos_det),
                "#0E7490",
            ),
            unsafe_allow_html=True,
        )

    if "Fecha" in df_det_f.columns:
        df_det_f["Fecha"] = pd.to_datetime(df_det_f["Fecha"], errors="coerce")
        df_det_f = df_det_f.sort_values(["Fecha", "Año", "Mes"], ascending=[False, False, False], na_position="last")
    else:
        df_det_f = df_det_f.sort_values(["Año", "Mes"], ascending=[False, False], na_position="last")

    cols_det = [c for c in ["Fecha", "Año", "Mes", "Esp", "Responsable", "CC", "CC1", "Obs", "Sit", "Monto"] if c in df_det_f.columns]
    df_det_view = df_det_f[cols_det].copy()

    st.caption(f"Registros encontrados: {len(df_det_view):,}".replace(",", "."))
    if df_det_view.empty:
        st.info("No hay movimientos para los filtros seleccionados.")
    else:
        df_det_show = df_det_view.copy()
        if "Fecha" in df_det_show.columns:
            df_det_show["Fecha"] = pd.to_datetime(df_det_show["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.caption("Vista acotada a 20 filas visibles. Usa el scroll interno de la tabla para recorrer todo.")
        visible_rows = 20
        row_height_px = 35
        header_px = 38
        table_height = header_px + (visible_rows * row_height_px)
        left_cols_det = [c for c in ["Responsable", "CC1", "Obs"] if c in df_det_show.columns]
        sit_cols_det = [c for c in ["Sit"] if c in df_det_show.columns]
        cc_cols_det = [c for c in ["CC"] if c in df_det_show.columns]
        monto_cols_det = [c for c in ["Monto"] if c in df_det_show.columns]

        def _highlight_first_row(row):
            if row.name == 0:
                return ["background-color:#E8F1FF; font-weight:700;"] * len(row)
            return [""] * len(row)

        def _style_sit(v):
            s = str(v).strip().upper()
            if s == "NO PAGADO":
                return "background-color:#FEE4E2; color:#B42318; font-weight:700;"
            if s == "PAGADO":
                return "background-color:#ECFDF3; color:#027A48; font-weight:700;"
            return ""

        def _style_cc(v):
            s = str(v).strip().upper()
            if s == "EGRESO":
                return "color:#B42318; font-weight:700;"
            if s == "INGRESO":
                return "color:#027A48; font-weight:700;"
            return ""

        def _style_monto(v):
            try:
                n = float(v)
                if n < 0:
                    return "background-color:#FEE4E2; color:#B42318; font-weight:800;"
                if n > 0:
                    return "background-color:#ECFDF3; color:#027A48; font-weight:800;"
                return "background-color:#F2F4F7; color:#344054; font-weight:700;"
            except Exception:
                pass
            return ""

        styler_det = (
            df_det_show.style
            .format({"Monto": "${:,.0f}"})
            .set_table_styles([
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#163A5F"),
                        ("color", "white"),
                        ("font-weight", "700"),
                        ("font-size", "13px"),
                        ("border-bottom", "1px solid #0F2740"),
                        ("text-align", "center"),
                        ("padding", "8px"),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        ("font-size", "12px"),
                        ("padding", "7px 8px"),
                        ("border-bottom", "1px solid #E5E7EB"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(even)",
                    "props": [("background-color", "#F8FAFC")],
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [("background-color", "#EEF2F7")],
                },
            ])
            .set_properties(subset=left_cols_det, **{"text-align": "left"})
            .set_properties(subset=monto_cols_det, **{"font-weight": "800"})
            .apply(_highlight_first_row, axis=1)
        )
        if sit_cols_det:
            styler_det = styler_det.applymap(_style_sit, subset=sit_cols_det)
        if cc_cols_det:
            styler_det = styler_det.applymap(_style_cc, subset=cc_cols_det)
        if monto_cols_det:
            styler_det = styler_det.applymap(_style_monto, subset=monto_cols_det)
        st.caption("Colores guía: fila principal azul suave · PAGADO verde · NO PAGADO rojo · EGRESO rojo / INGRESO verde.")
        st.dataframe(
            styler_det,
            use_container_width=True,
            height=table_height,
        )

# =========================================================
# ⚡ TAB 6: ELECTRICIDAD (Excel por pestaña)
# =========================================================
if active_section == "⚡ Electricidad":
    title_col, btn_col = st.columns([6, 1])
    with title_col:
        st.markdown(
            section_heading("⚡", "Electricidad — Liquidación por Bodega", weight_class="section-heading-title-soft"),
            unsafe_allow_html=True,
        )
        st.caption("Vista idéntica al Excel: inputs generales, boleta CGE, inputs por bodega y liquidación.")
    with btn_col:
        st.markdown("")
        st.markdown("")
        # Placeholder for PDF button (se setea más abajo cuando tengamos los datos)
        pdf_btn_placeholder = st.empty()

    try:
        parsed_sheets = load_electricidad_parsed(ELECTRICIDAD_XLSX)
    except ImportError:
        parsed_sheets = {}
        st.error(
            "Falta la dependencia `openpyxl` para leer archivos .xlsx. "
            "Instálala en tu entorno con: `pip install openpyxl`"
        )
    except Exception as e:
        parsed_sheets = {}
        st.error(f"No se pudo cargar el Excel de electricidad: {e}")
    if not parsed_sheets:
        st.warning(
            "No se encontraron hojas válidas de electricidad para mostrar. "
            "La fuente puede venir vacía, con una estructura distinta o haber fallado durante el parseo."
        )
        st.stop()

    # Preferir hojas tipo MES-AÑO (ej. FEB-2026)
    month_sheets = [s for s in parsed_sheets.keys() if "-" in s]
    sel_months = st.multiselect(
        "Meses",
        month_sheets or list(parsed_sheets.keys()),
        default=[month_sheets[0]] if month_sheets else list(parsed_sheets.keys())[:1],
        key="elec_months",
    )
    if not sel_months:
        st.info("Selecciona al menos un mes.")
        st.stop()

    # Selector de bodega
    first_parsed = parsed_sheets[sel_months[0]]
    bodega_col = first_parsed["inputs_bodega"].columns[0]
    bodegas = (
        first_parsed["inputs_bodega"][bodega_col]
        .dropna()
        .astype(str)
        .tolist()
    )
    bodegas = [b for b in bodegas if b and b.upper() != "TOTAL"]
    sel_bodega = st.selectbox("Bodega", ["Todas"] + bodegas, index=0, key="elec_bodega")

    # Parse selected months
    parsed_by_month = {m: parsed_sheets[m] for m in sel_months}

    # Encabezado estilo Excel
    st.markdown(
        """
        <div style="background:#1f4e78;color:white;padding:8px 12px;border-radius:6px;font-weight:700;">
        Liquidación Eléctrica por Bodega
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("**INPUTS GENERALES**")
        _render_table(first_parsed["inputs_generales"], header_bg="#1f4e78", header_fg="white", row_alt="#fbf3d6")
    with col_right:
        st.markdown("**BOLETA CGE (INPUTS DE FACTURACIÓN)**")
        # Promedio por concepto según meses seleccionados
        boletas = []
        for m, p in parsed_by_month.items():
            df_b = p["boleta_cge"].copy()
            df_b["Monto $"] = pd.to_numeric(df_b["Monto $"], errors="coerce")
            boletas.append(df_b)
        if boletas:
            boleta_all = pd.concat(boletas, ignore_index=True)
            boleta_avg = (
                boleta_all
                .groupby("Concepto", as_index=False)["Monto $"]
                .mean()
            )
            # mantener orden original de la primera hoja
            orden = first_parsed["boleta_cge"]["Concepto"].tolist()
            boleta_avg["__ord"] = boleta_avg["Concepto"].apply(
                lambda x: orden.index(x) if x in orden else 999
            )
            boleta_avg = boleta_avg.sort_values("__ord").drop(columns="__ord")
            _render_table(
                boleta_avg,
                header_bg="#1f4e78",
                header_fg="white",
                row_alt="#fbf3d6",
            )
        else:
            boleta_avg = first_parsed["boleta_cge"].copy()
            _render_table(boleta_avg, header_bg="#1f4e78", header_fg="white", row_alt="#fbf3d6")

    st.markdown("")
    st.markdown("**INPUTS POR BODEGA (REMARCADOR + HORARIO EFECTIVO)**")
    inputs_bodega = []
    for m, p in parsed_by_month.items():
        df_in = p["inputs_bodega"].copy()
        if sel_bodega != "Todas":
            df_in = df_in[df_in[bodega_col].astype(str) == sel_bodega]
        df_in.insert(0, "Mes", m)
        inputs_bodega.append(df_in)
    inputs_bodega = pd.concat(inputs_bodega, ignore_index=True) if inputs_bodega else pd.DataFrame()
    _render_table(
        inputs_bodega,
        header_bg="#1f4e78",
        header_fg="white",
        row_alt="#fff7e6",
        int_cols={"kWh post-18 (calc)", "kWh día (calc)"},
    )

    st.markdown("")
    st.markdown("**LIQUIDACIÓN POR BODEGA (ASIGNACIÓN DE COSTOS DE BOLETA + CRITERIO HORARIO)**")
    liquidacion = []
    for m, p in parsed_by_month.items():
        df_liq = p["liquidacion"].copy()
        if sel_bodega != "Todas":
            df_liq = df_liq[df_liq[df_liq.columns[0]].astype(str) == sel_bodega]
        df_liq.insert(0, "Mes", m)
        liquidacion.append(df_liq)
    liquidacion = pd.concat(liquidacion, ignore_index=True) if liquidacion else pd.DataFrame()
    _render_table(
        liquidacion,
        header_bg="#1f4e78",
        header_fg="white",
        row_alt="#eef8ee",
        int_cols={"kWh post-18"},
        cur_cols_extra={"IVA"},
    )

    # Gráfico profesional de Liquidación por Bodega
    st.markdown("### 📈 Liquidación por Bodega — Distribución de costos")
    liq_chart = liquidacion.copy()
    if not liq_chart.empty:
        liq_chart = liq_chart[liq_chart[liq_chart.columns[1]].astype(str).str.upper() != "TOTAL"]

    if liq_chart.empty:
        st.info("No hay datos suficientes para el gráfico.")
    else:
        # Columnas esperadas
        col_mes = "Mes"
        col_bodega = liq_chart.columns[1]
        cols_costos = ["$ Energía", "$ Punta", "$ Reactiva", "$ Cargos Fijos", "$ Interés"]
        col_total = "TOTAL c/IVA $"

        # Normalizar numéricos
        for c in cols_costos + [col_total]:
            if c in liq_chart.columns:
                liq_chart[c] = pd.to_numeric(liq_chart[c], errors="coerce")
            else:
                liq_chart[c] = 0

        import plotly.graph_objects as go

        palette = {
            "$ Energía": "#16a34a",    # verde mate
            "$ Punta": "#2563eb",      # azul mate
            "$ Reactiva": "#f59e0b",   # amarillo mate
            "$ Cargos Fijos": "#ef4444",# rojo mate
            "$ Interés": "#9ca3af",    # gris claro mate
        }

        def build_liq_fig(df_plot: pd.DataFrame, x_axis, x_title, height=520, show_legend=True):
            fig = go.Figure()
            for c in cols_costos:
                fig.add_trace(
                    go.Bar(
                        x=x_axis,
                        y=df_plot[c],
                        name=c,
                        marker=dict(color=palette.get(c, "#94a3b8")),
                        text=df_plot[c],
                        texttemplate="%{text:,.0f}",
                        textposition="inside",
                        hovertemplate=f"<b>%{{x}}</b><br>{c}: $%{{y:,.0f}}<extra></extra>",
                    )
                )
            if col_total in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df_plot[col_total],
                        name="TOTAL c/IVA $",
                        mode="lines+markers",
                        line=dict(color="#111827", width=3),
                        marker=dict(size=7, color="#111827"),
                        hovertemplate="<b>%{x}</b><br>Total c/IVA: $%{y:,.0f}<extra></extra>",
                        yaxis="y2",
                    )
                )
            fig.update_layout(
                template="plotly_white",
                barmode="stack",
                height=height,
                margin=dict(l=20, r=40, t=40, b=20),
                legend=dict(
                    orientation="h",
                    y=1.2,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                    traceorder="normal",
                    font=dict(size=11),
                    entrywidth=120,
                    entrywidthmode="pixels",
                ) if show_legend else None,
                xaxis=dict(title=x_title, showgrid=False),
                yaxis=dict(title="Costo (CLP)", tickformat=",.0f", gridcolor="rgba(148,163,184,0.25)"),
                yaxis2=dict(
                    title="Total c/IVA (CLP)",
                    overlaying="y",
                    side="right",
                    tickformat=",.0f",
                    showgrid=False,
                ),
                hovermode="x unified",
            )
            return fig

        def build_liq_single_fig(df_plot: pd.DataFrame, height=420):
            agg_vals = df_plot[cols_costos].sum(numeric_only=True)
            vals = [float(agg_vals.get(c, 0) or 0) for c in cols_costos]
            labels = cols_costos
            total_val = float(pd.to_numeric(df_plot[col_total], errors="coerce").sum())

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=vals,
                        hole=0.58,
                        sort=False,
                        marker=dict(colors=[palette.get(c, "#94a3b8") for c in labels]),
                        texttemplate="%{percent}",
                        textposition="inside",
                        hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>",
                    )
                ]
            )

            bodega_label = str(df_plot[col_bodega].iloc[0]) if not df_plot.empty else ""
            mes_label = str(df_plot[col_mes].iloc[0]) if not df_plot.empty else ""
            center_text = f"<b>{bodega_label}</b><br>{mes_label}<br><span style='font-size:14px'>Total c/IVA</span><br><b>${total_val:,.0f}</b>"

            fig.update_layout(
                template="plotly_white",
                height=height,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(
                    orientation="h",
                    y=1.08,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=11),
                ),
                annotations=[
                    dict(
                        text=center_text,
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        align="center",
                        font=dict(size=15, color="#111827"),
                    )
                ],
            )
            return fig

        single_period = len(sel_months) == 1
        charts_for_pdf = []
        single_month_single_bodega = len(sel_months) == 1 and sel_bodega != "Todas"
        if sel_bodega == "Todas" and len(sel_months) > 1:
            bodegas_plot = liq_chart[col_bodega].dropna().astype(str).unique().tolist()
            cols = st.columns(2)
            for i, b in enumerate(bodegas_plot):
                df_b = liq_chart[liq_chart[col_bodega].astype(str) == b]
                fig = build_liq_fig(df_b, df_b[col_mes], "Mes", height=360, show_legend=True)
                with cols[i % 2]:
                    st.markdown(f"**{b}**")
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                charts_for_pdf.append({
                    "df": df_b,
                    "x_col": col_mes,
                    "title": f"{b}",
                    "cols_costos": cols_costos,
                    "col_total": col_total,
                    "palette": palette,
                    "chart_type": "stacked",
                })
        elif single_month_single_bodega:
            fig = build_liq_single_fig(liq_chart, height=430)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            charts_for_pdf.append({
                "df": liq_chart,
                "x_col": col_bodega if single_period else col_mes,
                "title": "",
                "cols_costos": cols_costos,
                "col_total": col_total,
                "palette": palette,
                "chart_type": "donut",
            })
        else:
            x_axis = liq_chart[col_bodega] if single_period else liq_chart[col_mes]
            x_title = "Bodega" if single_period else "Mes"
            fig = build_liq_fig(liq_chart, x_axis, x_title, height=520, show_legend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            charts_for_pdf.append({
                "df": liq_chart,
                "x_col": col_bodega if single_period else col_mes,
                "title": "",
                "cols_costos": cols_costos,
                "col_total": col_total,
                "palette": palette,
                "chart_type": "stacked",
            })

        # PDF download button (ubicado en el header)
        try:
            pdf_bytes = build_electricidad_pdf(
                title="Liquidación Eléctrica por Bodega",
                sel_months=sel_months,
                sel_bodega=sel_bodega,
                inputs_generales=first_parsed["inputs_generales"],
                boleta_avg=boleta_avg,
                inputs_bodega=inputs_bodega,
                liquidacion=liquidacion,
                charts=charts_for_pdf,
            )
            pdf_btn_placeholder.download_button(
                "⬇️ Descargar PDF de Liquidación Eléctrica",
                data=pdf_bytes,
                file_name="liquidacion_electrica_bodega.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            pdf_btn_placeholder.error(f"No se pudo generar el PDF: {e}")
