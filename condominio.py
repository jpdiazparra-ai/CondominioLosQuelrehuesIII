"""
Condominio Los Queltehues III - Dashboard (General + Ingresos V2.3 + Costos + Obligaciones)
"""

from __future__ import annotations

import base64
import html
import io
import re
from typing import Optional

import pandas as pd


CACHE_VERSION = 2

INGRESOS_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/"
    "pub?gid=1653640714&single=true&output=csv"
)

COSTOS_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/"
    "pub?gid=341023122&single=true&output=csv"
)

OBLIGACIONES_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/"
    "pub?gid=2141405996&single=true&output=csv"
)

PROPIETARIOS_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/"
    "pub?gid=782319858&single=true&output=csv"
)

TD23_CSV_URL = OBLIGACIONES_CSV_URL

MANTENCION_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/"
    "pub?gid=1564429404&single=true&output=csv"
)

POR_PAGAR_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vREdYwR32RK_ecff9UJ-DdGNjvfdnoO55jpToO-KLG62izQTqFovnWUTM-ttfmR9DNt6N1lSNKMzkjZ/"
    "pub?gid=1804083007&single=true&output=csv"
)


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


def load_data(url: str, expected_cols: Optional[set[str]] = None) -> pd.DataFrame:
    df_raw = pd.read_csv(url, header=None, dtype=str)

    expected = expected_cols or {"fecha", "parcela", "abono"}
    header_row = None
    header_norm = None

    for i, row in df_raw.head(50).iterrows():
        norm = [_normalize_colname(c) for c in row.tolist()]
        if expected.issubset(set(norm)):
            header_row = i
            header_norm = norm
            break

    if header_row is None:
        header_row = 0
        header_norm = [_normalize_colname(c) for c in df_raw.iloc[0].tolist()]

    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = header_norm

    # Evita columnas duplicadas
    seen = {}
    new_cols = []
    for c in df.columns:
        count = seen.get(c, 0)
        if count == 0:
            new_cols.append(c)
        else:
            new_cols.append(f"{c}_{count}")
        seen[c] = count + 1
    df.columns = new_cols

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, [c for c in df.columns if not c.startswith("nan")]]
    return df


def _parse_monto_series(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    neg_mask = s.str.contains(r"^\(.*\)$", regex=True)
    s = s.str.replace(r"[^\d\-]", "", regex=True).replace("", pd.NA)
    out = pd.to_numeric(s, errors="coerce")
    out.loc[neg_mask] = -out.loc[neg_mask].abs()
    return out


def _df_to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise RuntimeError("Falta reportlab. Instala con: pip install reportlab") from e

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Heading2"]), Spacer(1, 12)]

    data = [list(df.columns)] + df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F2A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F4F7FA")]),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _fig_to_base64_png(fig, width: int = 1100, height: int = 650) -> str:
    try:
        import plotly.io as pio
    except Exception as e:
        raise RuntimeError("Falta Plotly. Instala con: pip install plotly") from e

    try:
        img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except Exception as e:
        msg = str(e)
        if "kaleido" in msg.lower():
            raise RuntimeError("Falta Kaleido. Instala con: pip install kaleido") from e
        raise
    return base64.b64encode(img_bytes).decode("ascii")


def _build_obligaciones_report_pdf_bytes(
    kpi_data: dict,
    fig_acum,
    tabla_show: pd.DataFrame,
    fig_gc,
    fig_m,
    fig_p,
    fig_cost_cat,
    fig_cost_prov,
) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    except Exception as e:
        raise RuntimeError("Falta reportlab. Instala con: pip install reportlab") from e

    def _fmt_money(v):
        try:
            return f"${float(v):,.0f}"
        except Exception:
            return v
    def _to_num(v):
        try:
            return float(v)
        except Exception:
            s = re.sub(r"[^\d\.\-]", "", str(v))
            try:
                return float(s) if s else 0.0
            except Exception:
                return 0.0

    table_main = tabla_show.copy()
    table_raw = tabla_show.copy()
    for col in table_main.columns:
        if col not in ("Parcela", "Propietario"):
            table_main[col] = table_main[col].apply(_fmt_money)

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    story = [Paragraph("Reporte de Obligaciones", styles["Heading2"]), Spacer(1, 10)]

    story.append(Paragraph("Ingresos, Costos y Neto (mensual)", styles["Heading3"]))
    kpi_rows = [
        ["Total ingresos", _fmt_money(kpi_data.get("total_ing", 0))],
        ["Total costos", _fmt_money(kpi_data.get("total_cost", 0))],
        ["Neto acumulado - banco", _fmt_money(kpi_data.get("total_neto", 0))],
        ["Pendiente de pago total", _fmt_money(kpi_data.get("pendiente_total", 0))],
        ["% no pago total", f'{kpi_data.get("pct_no_pago", 0):.1f}%'],
        ["Mejor año", str(kpi_data.get("best_year", ""))],
    ]
    t_kpi = Table(kpi_rows, colWidths=[220, 160])
    t_kpi.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0B1F2A")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ]
        )
    )
    story.append(t_kpi)
    story.append(Spacer(1, 8))

    if fig_acum is not None:
        img_acum = _fig_to_base64_png(fig_acum, width=1000, height=520)
        story.append(Image(io.BytesIO(base64.b64decode(img_acum)), width=600, height=300))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("Obligación acumulada vs Pagos", styles["Heading3"]))
    data_main = [list(table_main.columns)] + table_main.values.tolist()
    t_main = Table(data_main, repeatRows=1)
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F2A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F4F7FA")]),
    ]
    pending_cols = [c for c in table_raw.columns if c.startswith("Pendiente")]
    col_idx = {c: i for i, c in enumerate(table_raw.columns)}
    total_col_idx = col_idx.get("Total por pagar")
    for r in range(len(table_raw)):
        total_val = _to_num(table_raw.iloc[r]["Total por pagar"]) if "Total por pagar" in table_raw.columns else 0.0
        if total_val > 0:
            table_style.append(("BACKGROUND", (0, r + 1), (-1, r + 1), colors.HexColor("#F4DCDC")))
        if total_col_idx is not None and total_val > 0:
            table_style.append(("BACKGROUND", (total_col_idx, r + 1), (total_col_idx, r + 1), colors.HexColor("#5A2A2A")))
            table_style.append(("TEXTCOLOR", (total_col_idx, r + 1), (total_col_idx, r + 1), colors.white))
            table_style.append(("FONTNAME", (total_col_idx, r + 1), (total_col_idx, r + 1), "Helvetica-Bold"))
    t_main.setStyle(TableStyle(table_style))
    story.append(t_main)
    story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("Distribución de pendientes", styles["Heading3"]))
    pie_imgs = []
    for fig in (fig_gc, fig_m, fig_p):
        if fig is not None:
            img_b64 = _fig_to_base64_png(fig, width=600, height=380)
            pie_imgs.append(Image(io.BytesIO(base64.b64decode(img_b64)), width=240, height=160))
        else:
            pie_imgs.append(Spacer(1, 160))
    story.append(Table([pie_imgs], colWidths=[260, 260, 260]))
    story.append(Spacer(1, 8))

    if fig_cost_cat is not None:
        img_cat = _fig_to_base64_png(fig_cost_cat, width=1000, height=520)
        story.append(Paragraph("Costo por categoría", styles["Heading3"]))
        story.append(Image(io.BytesIO(base64.b64decode(img_cat)), width=600, height=300))
        story.append(Spacer(1, 8))

    if fig_cost_prov is not None:
        img_prov = _fig_to_base64_png(fig_cost_prov, width=900, height=520)
        story.append(Paragraph("Costos por proveedor (top 12)", styles["Heading3"]))
        story.append(Image(io.BytesIO(base64.b64decode(img_prov)), width=520, height=300))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
    def _fmt_money(v):
        try:
            return f"${float(v):,.0f}"
        except Exception:
            return v

    table_obl = oblig_show.copy()
    if "GC total por año" in table_obl.columns:
        table_obl["GC total por año"] = table_obl["GC total por año"].apply(_fmt_money)

    table_main = tabla_show.copy()
    for col in table_main.columns:
        if col not in ("Parcela", "Propietario"):
            table_main[col] = table_main[col].apply(_fmt_money)

    def _table_html(df):
        return df.to_html(index=False, classes="tbl", escape=False)

    img_obl = _fig_to_base64_png(fig_obl_pie) if fig_obl_pie is not None else ""
    img_gc = _fig_to_base64_png(fig_gc) if fig_gc is not None else ""
    img_m = _fig_to_base64_png(fig_m) if fig_m is not None else ""
    img_p = _fig_to_base64_png(fig_p) if fig_p is not None else ""

    html = f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Reporte Obligaciones</title>
<style>
  body {{font-family: Arial, sans-serif; color:#0f172a; margin:24px;}}
  h1 {{margin:0 0 12px 0; font-size:22px;}}
  h2 {{margin:20px 0 10px 0; font-size:16px; color:#0B1F2A;}}
  .row {{display:flex; gap:16px; align-items:flex-start;}}
  .col {{flex:1;}}
  .card {{border:1px solid #E2E8F0; border-radius:12px; padding:12px; background:#fff;}}
  .tbl {{width:100%; border-collapse:collapse; font-size:12px;}}
  .tbl th {{background:#0B1F2A; color:#F8FAFC; font-weight:700; padding:6px 8px;}}
  .tbl td {{border:1px solid #E2E8F0; padding:6px 8px; text-align:right;}}
  .tbl td:first-child, .tbl th:first-child {{text-align:center;}}
  .tbl td:nth-child(2), .tbl th:nth-child(2) {{text-align:left;}}
  .note {{font-size:11px; color:#64748B;}}
  img {{max-width:100%; height:auto;}}
  .pie-row {{display:flex; gap:12px;}}
  .pie-row .card {{flex:1;}}
</style>
</head>
<body>
  <h1>Reporte de Obligaciones</h1>

  <h2>Obligación por año (GC)</h2>
  <div class="row">
    <div class="col card">{_table_html(table_obl)}</div>
    <div class="col card">{f'<img src="data:image/png;base64,{img_obl}" alt="Distribución GC por año"/>' if img_obl else ''}</div>
  </div>

  <h2>Obligación acumulada vs Pagos</h2>
  <div class="card">{_table_html(table_main)}</div>

  <h2>Distribución de pendientes</h2>
  <div class="pie-row">
    <div class="card">{f'<img src="data:image/png;base64,{img_gc}" alt="Pendiente GC"/>' if img_gc else ''}</div>
    <div class="card">{f'<img src="data:image/png;base64,{img_m}" alt="Pendiente mantención"/>' if img_m else ''}</div>
    <div class="card">{f'<img src="data:image/png;base64,{img_p}" alt="Pendiente proyecto"/>' if img_p else ''}</div>
  </div>

  <p class="note">Fuente: Google Sheets (CSV publicado). Generado por dashboard Condominio.</p>
</body>
</html>
"""
    return html


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


def build_obligaciones_vs_pagos(
    df_obl: pd.DataFrame,
    df_ing: pd.DataFrame,
    concepto_col: Optional[str] = None,
    include_keywords: Optional[list[str]] = None,
    exclude_keywords: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_o = list(df_obl.columns)
    col_anio = _pick_col(cols_o, ["ano", "anio", "año"])
    col_gc = _pick_col(cols_o, ["gc", "gasto_comun", "gastos_comunes", "total"])

    if not (col_anio and col_gc):
        empty = pd.DataFrame(columns=["anio", "gc_total"])
        return empty, pd.DataFrame(columns=["parcela", "pagado", "gc_total", "pendiente"])

    base_o = df_obl.copy()
    base_o["anio"] = pd.to_numeric(base_o[col_anio].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
    base_o["gc_total"] = _parse_monto_series(base_o[col_gc])
    base_o = base_o.dropna(subset=["anio", "gc_total"])

    oblig_anual = (
        base_o.groupby("anio", as_index=False)["gc_total"]
        .sum()
        .sort_values("anio")
    )
    gc_total_acum = float(oblig_anual["gc_total"].sum()) if not oblig_anual.empty else 0.0

    cols_i = list(df_ing.columns)
    col_fecha = _pick_col(cols_i, ["fecha"])
    col_parc_i = _pick_col(cols_i, ["parcela"])
    col_abono = next((c for c in cols_i if c.startswith("abono")), None)

    if not (col_fecha and col_parc_i and col_abono):
        empty = pd.DataFrame(columns=["anio", "gc_total"])
        return empty, pd.DataFrame(columns=["parcela", "pagado", "gc_total", "pendiente"])

    base_i = df_ing.copy()
    base_i["fecha_norm"] = pd.to_datetime(base_i[col_fecha], dayfirst=True, errors="coerce")
    base_i["parcela"] = pd.to_numeric(base_i[col_parc_i].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
    base_i["pagado"] = _parse_monto_series(base_i[col_abono])
    base_i = base_i.dropna(subset=["parcela", "pagado"])

    if concepto_col and concepto_col in base_i.columns:
        texto = base_i[concepto_col].astype(str).str.lower()
        if include_keywords:
            inc_mask = False
            for kw in include_keywords:
                inc_mask = inc_mask | texto.str.contains(kw.lower(), regex=False)
            base_i = base_i[inc_mask]
        if exclude_keywords:
            for kw in exclude_keywords:
                base_i = base_i[~texto.str.contains(kw.lower(), regex=False)]

    pagos = base_i.groupby("parcela", as_index=False)["pagado"].sum()

    parcelas = pd.DataFrame({"parcela": list(range(17, 37))})
    out = parcelas.merge(pagos, on="parcela", how="left").fillna({"pagado": 0})
    out["gc_total"] = gc_total_acum
    out["pendiente"] = out["gc_total"] - out["pagado"]
    return oblig_anual, out


def load_td23_table(url: str) -> pd.DataFrame:
    df_raw = pd.read_csv(url, header=None, dtype=str)
    header_row = None
    header_cols = None
    header_idx = None
    for i, row in df_raw.head(50).iterrows():
        norm = [_normalize_colname(c) for c in row.tolist()]
        if "cc" in norm and "monto" in norm and "total" in norm:
            header_row = i
            header_cols = norm
            header_idx = [idx for idx, val in enumerate(norm) if val in ("cc", "monto", "obs", "total")]
            break
    if header_row is None or header_idx is None:
        return pd.DataFrame(columns=["cc", "monto", "obs", "total"])

    df = df_raw.iloc[header_row + 1 :].copy()
    df = df.iloc[:, header_idx]
    df.columns = [header_cols[idx] for idx in header_idx]
    df = df.dropna(how="all")
    df = df.rename(columns={"cc": "cc", "monto": "monto", "obs": "obs", "total": "total"})
    df = df[df["cc"].notna()]
    return df


def load_mantencion_table(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [_normalize_colname(c) for c in df.columns]
    col_parc = _pick_col(list(df.columns), ["parcela", "n_parcela", "numero_parcela", "lote", "unidad", "sitio"])
    col_val = _pick_col(list(df.columns), ["monto", "valor", "mantencion", "mantenimiento", "total"])
    if not col_parc or not col_val:
        return pd.DataFrame(columns=["parcela", "mantencion"])
    out = pd.DataFrame()
    out["parcela"] = pd.to_numeric(df[col_parc].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
    out["mantencion"] = _parse_monto_series(df[col_val])
    out = out.dropna(subset=["parcela"])
    return out


def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="Condominio Los Quelrehues III - Dashboard",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
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
        spark = _sparkline_svg(color, spark_points) if spark_points else ""
        return f"""
        <div class="dash-kpi-card dash-{tone}">
          <div class="dash-icon">{icon}</div>
          <div class="dash-kpi-title">{html.escape(title)}</div>
          <div class="dash-kpi-value">{html.escape(value)}</div>
          <div class="dash-kpi-sub">{html.escape(subtitle)}</div>
          {spark}
        </div>
        """

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
          padding-top: 0.65rem;
          padding-left: 2.75rem;
          padding-right: 2.75rem;
          max-width: 1680px;
        }
        [data-testid="stSidebar"] {
          background: rgba(255, 255, 255, 0.93);
          border-right: 1px solid #e6edf5;
          box-shadow: 10px 0 28px rgba(15, 23, 42, 0.04);
          min-width: 138px !important;
          max-width: 138px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
          padding: 1.7rem 1rem 1rem 1rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stButton,
        [data-testid="stSidebar"] hr {
          display: none;
        }
        .side-logo {
          width: 58px;
          height: 58px;
          margin: 0 auto 30px auto;
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
          gap: 13px;
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
          margin: 0;
        }
        .side-about {
          position: fixed;
          bottom: 2rem;
          left: 0;
          width: 138px;
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
          margin-bottom: 22px;
        }
        .app-title {
          margin: 0;
          color: var(--dash-ink);
          font-size: clamp(2.2rem, 4vw, 3.25rem);
          line-height: 1;
          letter-spacing: 0;
          font-weight: 900;
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
          min-height: 60px;
          border-radius: 8px;
          border: 1px solid #d8e0eb;
          background: #ffffff;
          color: #101828;
          box-shadow: 0 8px 18px rgba(15,23,42,0.12);
          font-weight: 800;
          padding: 0 1.55rem;
        }
        div.stDownloadButton > button:hover {
          border-color: #c5cfdd;
          background: #fbfdff;
          color: #0f172a;
        }
        .dash-hero {
          min-height: 386px;
          display: grid;
          grid-template-columns: minmax(0, 1fr) 360px;
          gap: 32px;
          align-items: center;
          margin: 38px 0 36px 0;
          padding: 38px 42px;
          border-radius: 17px;
          position: relative;
          overflow: hidden;
          color: #ffffff;
          background:
            radial-gradient(circle at 89% 18%, rgba(122, 231, 230, 0.16), transparent 18%),
            radial-gradient(circle at 84% 76%, rgba(122, 231, 230, 0.14), transparent 28%),
            linear-gradient(130deg, #002b41 0%, #003a50 45%, #00465a 100%);
          box-shadow: 0 18px 28px rgba(15, 23, 42, 0.22);
        }
        .dash-hero::before {
          content: "";
          position: absolute;
          inset: auto -6% -18% 16%;
          height: 150px;
          background: rgba(22, 92, 116, 0.25);
          border-radius: 55% 45% 0 0;
        }
        .dash-hero-eyebrow,
        .dash-hero-label {
          position: relative;
          z-index: 1;
          font-size: 0.96rem;
          letter-spacing: 0.1em;
          font-weight: 900;
          text-transform: uppercase;
        }
        .dash-hero-value {
          position: relative;
          z-index: 1;
          margin-top: 22px;
          font-size: clamp(3.1rem, 5.2vw, 4.85rem);
          line-height: 0.98;
          font-weight: 900;
          text-shadow: 0 3px 0 rgba(255, 255, 255, 0.12), 0 7px 18px rgba(0, 0, 0, 0.25);
        }
        .dash-hero-note,
        .dash-hero-sub {
          position: relative;
          z-index: 1;
          margin-top: 18px;
          color: #edf7fb;
          font-size: 1.02rem;
          font-weight: 700;
        }
        .dash-hero-rule {
          position: relative;
          z-index: 1;
          width: 68%;
          height: 1px;
          background: rgba(223, 244, 248, 0.25);
          margin: 32px 0 26px 0;
        }
        .dash-hero-secondary-value {
          position: relative;
          z-index: 1;
          margin-top: 14px;
          font-size: 2.45rem;
          font-weight: 900;
          line-height: 1;
        }
        .bank-art {
          position: relative;
          z-index: 1;
          height: 260px;
          display: grid;
          place-items: end center;
          opacity: 0.88;
        }
        .bank-art .bank {
          position: relative;
          width: 250px;
          height: 170px;
        }
        .bank-roof {
          position: absolute;
          left: 18px;
          top: 4px;
          width: 214px;
          height: 54px;
          background: #86dce1;
          clip-path: polygon(50% 0%, 100% 72%, 95% 100%, 5% 100%, 0% 72%);
          box-shadow: inset 0 -8px 0 rgba(14, 84, 98, 0.18);
        }
        .bank-row {
          position: absolute;
          left: 28px;
          top: 68px;
          width: 194px;
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 20px;
        }
        .bank-col {
          height: 92px;
          border-radius: 8px 8px 2px 2px;
          background: linear-gradient(180deg, #9be8e9, #5dc5cb);
          box-shadow: inset 7px 0 0 rgba(255,255,255,0.18);
        }
        .bank-base {
          position: absolute;
          left: 8px;
          bottom: 0;
          width: 234px;
          height: 21px;
          border-radius: 3px;
          background: #a8edf0;
          box-shadow: 0 -13px 0 #73d4d8;
        }
        .bank-coin {
          position: absolute;
          left: 84px;
          top: 116px;
          width: 74px;
          height: 74px;
          border-radius: 50%;
          display: grid;
          place-items: center;
          background: #ecffff;
          color: #0c6671;
          font-size: 42px;
          font-weight: 900;
          box-shadow: 0 0 0 9px rgba(129, 218, 224, 0.74);
        }
        .dash-kpi-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 28px;
          margin: 0 0 34px 0;
        }
        .dash-kpi-card {
          min-height: 238px;
          background: #ffffff;
          border: 1px solid #e2eaf4;
          border-top: 4px solid var(--tone);
          border-radius: 14px;
          padding: 20px 20px 13px 20px;
          box-shadow: 0 11px 22px rgba(15,23,42,0.10);
          overflow: hidden;
        }
        .dash-green { --tone: #00a86b; }
        .dash-red { --tone: #ff2d2d; }
        .dash-violet { --tone: #8a2be2; }
        .dash-teal { --tone: #008b78; }
        .dash-blue { --tone: #0f6bff; }
        .dash-icon {
          width: 52px;
          height: 52px;
          border-radius: 50%;
          display: grid;
          place-items: center;
          color: var(--tone);
          background: color-mix(in srgb, var(--tone) 15%, white);
          font-size: 25px;
          font-weight: 900;
          margin-bottom: 14px;
        }
        .dash-kpi-title {
          color: #3d485c;
          font-size: 0.78rem;
          font-weight: 900;
          text-transform: uppercase;
          line-height: 1.25;
        }
        .dash-kpi-value {
          margin-top: 8px;
          color: #071326;
          font-size: clamp(1.55rem, 2vw, 2rem);
          line-height: 1.05;
          font-weight: 900;
        }
        .dash-kpi-sub {
          margin-top: 12px;
          color: var(--tone);
          font-size: 0.86rem;
          font-weight: 700;
          min-height: 21px;
        }
        .dash-sparkline {
          width: 100%;
          height: 38px;
          margin-top: 20px;
        }
        @media (max-width: 1300px) {
          .dash-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .dash-hero { grid-template-columns: 1fr 260px; }
        }
        @media (max-width: 780px) {
          .block-container { padding-left: 1.1rem; padding-right: 1.1rem; }
          [data-testid="stSidebar"] { min-width: 112px !important; max-width: 112px !important; }
          .top-shell { display: block; }
          .top-actions { justify-content: flex-start; margin-top: 20px; }
          .dash-hero { grid-template-columns: 1fr; padding: 28px 24px; }
          .bank-art { display: none; }
          .dash-kpi-grid { grid-template-columns: 1fr; gap: 16px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="side-logo">🏢</div>
            <div class="side-about"><div style="font-size:24px;">ⓘ</div><div>Acerca de</div></div>
            """,
            unsafe_allow_html=True,
        )
        section_options = ["General", "Ingresos V2.3", "Costos", "Obligaciones"]
        section_labels = {
            "General": "⌂\nGeneral",
            "Ingresos V2.3": "↗\nIngresos V2.3",
            "Costos": "$\nCostos",
            "Obligaciones": "▤\nObligaciones",
        }
        selected_section = st.radio(
            "Navegación",
            section_options,
            index=0,
            format_func=lambda option: section_labels[option],
            label_visibility="collapsed",
        )

    @st.cache_data(show_spinner=False)
    def _load(url_value: str, cache_version: int, expected_cols: Optional[set[str]] = None) -> pd.DataFrame:
        return load_data(url_value, expected_cols)

    @st.cache_data(show_spinner=False)
    def _make_obligaciones_report() -> bytes:
        df_obl = _load(OBLIGACIONES_CSV_URL, CACHE_VERSION, {"ano", "anio", "año", "parcela", "gc"})
        df_ing_o = _load(INGRESOS_CSV_URL, CACHE_VERSION, {"fecha", "parcela", "abono"})
        df_prop = _load(PROPIETARIOS_CSV_URL, CACHE_VERSION, {"parcela", "propietario"})
        df_td = load_td23_table(TD23_CSV_URL)
        df_mant = load_mantencion_table(MANTENCION_CSV_URL)

        cols_ing_o = list(df_ing_o.columns)
        cand_concepto = ["detalle", "concepto", "glosa", "descripcion", "tipo", "categoria", "cc", "ccc", "medio"]
        concepto_col_val = next((c for c in cand_concepto if c in cols_ing_o), None)
        include_list = ["gasto", "gc"]
        exclude_list = ["proyecto"]

        oblig_anual, tabla = build_obligaciones_vs_pagos(
            df_obl,
            df_ing_o,
            concepto_col=concepto_col_val,
            include_keywords=include_list,
            exclude_keywords=exclude_list,
        )
        if tabla.empty:
            raise RuntimeError("No se pudieron construir obligaciones vs pagos. Revisa columnas de año/parcela/gc.")

        tabla_full = tabla.copy()
        tabla_full["pendiente_pos"] = tabla_full["pendiente"].clip(lower=0)
        tabla_full["saldo_favor"] = (-tabla_full["pendiente"]).clip(lower=0)

        if not df_mant.empty:
            df_mant = df_mant.groupby("parcela", as_index=False)["mantencion"].sum()
            tabla_full = tabla_full.merge(df_mant, on="parcela", how="left").fillna({"mantencion": 0})

            cols_ing = list(df_ing_o.columns)
            col_cc_ing = _pick_col(cols_ing, ["cc", "categoria", "rubro", "ccc"])
            col_abono_ing = _pick_col(cols_ing, ["abono"])
            col_parc_ing = _pick_col(cols_ing, ["parcela"])
            if col_cc_ing and col_abono_ing and col_parc_ing:
                ing_m = df_ing_o.copy()
                ing_m["parcela"] = pd.to_numeric(
                    ing_m[col_parc_ing].astype(str).str.replace(r"[^\d]", "", regex=True),
                    errors="coerce",
                )
                ing_m["monto_norm"] = _parse_monto_series(ing_m[col_abono_ing])
                ing_m = ing_m.dropna(subset=["parcela", "monto_norm"])
                cc_text = (
                    ing_m[col_cc_ing]
                    .astype(str)
                    .str.lower()
                    .str.replace("á", "a")
                    .str.replace("é", "e")
                    .str.replace("í", "i")
                    .str.replace("ó", "o")
                    .str.replace("ú", "u")
                    .str.replace("ñ", "n")
                )
                mask_mant = cc_text.str.contains("mantencion", regex=False) | cc_text.str.contains("mantenimiento", regex=False)
                pagos_mant = (
                    ing_m[mask_mant]
                    .groupby("parcela", as_index=False)["monto_norm"]
                    .sum()
                    .rename(columns={"monto_norm": "pagado_mant"})
                )
                tabla_full = tabla_full.merge(pagos_mant, on="parcela", how="left").fillna({"pagado_mant": 0})
                tabla_full["mantencion"] = (tabla_full["mantencion"] - tabla_full["pagado_mant"]).clip(lower=0)
                tabla_full = tabla_full.drop(columns=["pagado_mant"])

            tabla_full = tabla_full.rename(columns={"mantencion": "Mantención"})

        if not df_td.empty:
            cols_ing = list(df_ing_o.columns)
            col_cc_ing = _pick_col(cols_ing, ["cc", "categoria", "rubro", "ccc"])
            if col_cc_ing:
                df_ing_cc = df_ing_o.copy()
                df_ing_cc["parcela"] = pd.to_numeric(
                    df_ing_cc[_pick_col(cols_ing, ["parcela"])].astype(str).str.replace(r"[^\d]", "", regex=True),
                    errors="coerce",
                )
                df_ing_cc["monto_norm"] = _parse_monto_series(df_ing_cc[_pick_col(cols_ing, ["abono"])])
                df_ing_cc = df_ing_cc.dropna(subset=["parcela", "monto_norm"])
                df_ing_cc["cc_norm"] = df_ing_cc[col_cc_ing].astype(str).str.lower()

                df_td = df_td.copy()
                df_td["cc_norm"] = df_td["cc"].astype(str).str.lower()
                df_td["monto_norm"] = _parse_monto_series(df_td["monto"])

                for _, row in df_td.iterrows():
                    cc_name = str(row["cc"]).strip()
                    if not cc_name:
                        continue
                    monto_cc = float(row["monto_norm"]) if pd.notna(row["monto_norm"]) else 0.0
                    if monto_cc == 0:
                        continue
                    mask_cc = df_ing_cc["cc_norm"].str.contains(cc_name.lower(), regex=False)
                    pagos_cc = (
                        df_ing_cc[mask_cc]
                        .groupby("parcela", as_index=False)["monto_norm"]
                        .sum()
                        .rename(columns={"monto_norm": "pagado_cc"})
                    )
                    col_name = f"Pendiente {cc_name}"
                    tabla_full = tabla_full.merge(pagos_cc, on="parcela", how="left").fillna({"pagado_cc": 0})
                    tabla_full[col_name] = (monto_cc - tabla_full["pagado_cc"]).clip(lower=0)
                    tabla_full = tabla_full.drop(columns=["pagado_cc"])

        gc_total_parcela = float(tabla_full["gc_total"].max()) if not tabla_full.empty else 0.0
        total_pendiente = float(tabla_full["pendiente_pos"].sum()) if not tabla_full.empty else 0.0
        pendiente_mant = float(tabla_full["Mantención"].sum()) if "Mantención" in tabla_full.columns else 0.0
        cc_cols = [c for c in tabla_full.columns if c.startswith("Pendiente ")]
        pendiente_proy = float(tabla_full[cc_cols].sum().sum()) if cc_cols else 0.0

        tabla_show = tabla_full.copy()
        tabla_show = tabla_show.rename(
            columns={
                "parcela": "Parcela",
                "pagado": "Pagado",
                "gc_total": "GC total",
                "pendiente": "Diferencia",
                "pendiente_pos": "Pendiente",
                "saldo_favor": "GC por anticipado",
                "Mantención": "Pendiente mantención",
            }
        )
        cols_prop = list(df_prop.columns)
        col_parc_p = _pick_col(cols_prop, ["n_parcela", "numero_parcela", "parcela", "lote", "unidad", "sitio"])
        col_name = _pick_col(cols_prop, ["nombre", "propietario", "dueno", "dueño"])
        if col_parc_p and col_name:
            prop_map = df_prop.copy()
            prop_map["Parcela"] = pd.to_numeric(
                prop_map[col_parc_p].astype(str).str.replace(r"[^\d]", "", regex=True),
                errors="coerce",
            )
            prop_map = prop_map.dropna(subset=["Parcela"])
            prop_map = prop_map[["Parcela", col_name]].rename(columns={col_name: "Propietario"})
            tabla_show = tabla_show.merge(prop_map, on="Parcela", how="left")
        else:
            tabla_show["Propietario"] = ""
        if "pendiente" in tabla_show.columns:
            tabla_show = tabla_show.drop(columns=["pendiente"])
        if "Diferencia" in tabla_show.columns:
            tabla_show = tabla_show.drop(columns=["Diferencia"])
        extra_cc_cols = [c for c in tabla_show.columns if c.startswith("Pendiente ")]
        total_cols = ["Pendiente", "Pendiente mantención"] + extra_cc_cols
        tabla_show["Total por pagar"] = tabla_show[total_cols].fillna(0).sum(axis=1)
        tabla_show = tabla_show.rename(columns={"Pendiente": "Pendiente GC"})
        cols_front = ["Parcela", "Propietario"]
        cols_rest = [c for c in tabla_show.columns if c not in cols_front]
        tabla_show = tabla_show[cols_front + cols_rest]

        pie_gc = tabla_show[tabla_show["Pendiente GC"] > 0][["Parcela", "Pendiente GC"]].copy()
        if "Pendiente mantención" in tabla_show.columns:
            pie_mant = tabla_show[tabla_show["Pendiente mantención"] > 0][["Parcela", "Pendiente mantención"]].copy()
        else:
            pie_mant = pd.DataFrame(columns=["Parcela", "Pendiente mantención"])
        proj_cols = [c for c in tabla_show.columns if c.startswith("Pendiente ") and c not in ("Pendiente mantención", "Pendiente GC")]
        if proj_cols:
            pie_proj = tabla_show[["Parcela"] + proj_cols].copy()
            pie_proj["Pendiente proyecto"] = pie_proj[proj_cols].sum(axis=1)
            pie_proj = pie_proj[pie_proj["Pendiente proyecto"] > 0][["Parcela", "Pendiente proyecto"]]
        else:
            pie_proj = pd.DataFrame(columns=["Parcela", "Pendiente proyecto"])

        fig_gc = None
        fig_m = None
        fig_p = None
        fig_acum = None
        fig_cost_cat = None
        fig_cost_prov = None

        df_cost_r = _load(COSTOS_CSV_URL, CACHE_VERSION, {"monto", "proveedor", "cc"})
        s_ing = build_series_mensual_ingresos(df_ing_o)
        s_cost = build_series_mensual_costos(df_cost_r)
        df_m = (
            s_ing.merge(s_cost, on="periodo", how="outer")
            .fillna(0)
            .sort_values("periodo")
        )
        df_m["neto"] = df_m["ingresos"] - df_m["costos"].abs()
        df_m["anio"] = pd.to_datetime(df_m["periodo"] + "-01", errors="coerce").dt.year
        df_y = (
            df_m.groupby("anio", as_index=False)[["ingresos", "costos", "neto"]]
            .sum()
            .fillna(0)
        )
        df_y["costos"] = df_y["costos"].abs()

        total_ing = float(df_y["ingresos"].sum()) if not df_y.empty else 0.0
        total_cost = float(df_y["costos"].sum()) if not df_y.empty else 0.0
        total_neto = float(df_y["neto"].sum()) if not df_y.empty else 0.0
        best_year = int(df_y.sort_values("neto", ascending=False)["anio"].iloc[0]) if not df_y.empty else 0
        total_gc = float(gc_total_parcela * 20) if gc_total_parcela else 0.0
        pendiente_total_all = float(total_pendiente + pendiente_mant + pendiente_proy)
        pct_no_pago = (pendiente_total_all / total_gc) * 100 if total_gc > 0 else 0.0

        kpi_data = {
            "total_ing": total_ing,
            "total_cost": total_cost,
            "total_neto": total_neto,
            "pendiente_total": pendiente_total_all,
            "pct_no_pago": pct_no_pago,
            "best_year": best_year,
        }

        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except Exception:
            pass
        else:
            if not pie_gc.empty:
                fig_gc = px.pie(
                    pie_gc,
                    names="Parcela",
                    values="Pendiente GC",
                    title="Distribución pendiente GC por parcela",
                    hole=0.35,
                    color_discrete_sequence=["#0B1F2A", "#1F4F5B", "#2C5B4A", "#3A6B5A", "#8DA2C8", "#A4463F"],
                )
                fig_gc.update_traces(textinfo="percent+label")
                fig_gc.update_layout(height=380, margin=dict(l=5, r=5, t=40, b=10), legend_title_text="Parcela")
            if not pie_mant.empty:
                fig_m = px.pie(
                    pie_mant,
                    names="Parcela",
                    values="Pendiente mantención",
                    title="Distribución pendiente mantención",
                    hole=0.35,
                    color_discrete_sequence=["#2C5B4A", "#3A6B5A", "#8DA2C8", "#0B1F2A", "#1F4F5B", "#A4463F"],
                )
                fig_m.update_traces(textinfo="percent+label")
                fig_m.update_layout(height=380, margin=dict(l=5, r=5, t=40, b=10), legend_title_text="Parcela")
            if not pie_proj.empty:
                fig_p = px.pie(
                    pie_proj,
                    names="Parcela",
                    values="Pendiente proyecto",
                    title="Distribución pendiente proyecto",
                    hole=0.35,
                    color_discrete_sequence=["#A4463F", "#8DA2C8", "#3A6B5A", "#2C5B4A", "#1F4F5B", "#0B1F2A"],
                )
                fig_p.update_traces(textinfo="percent+label")
                fig_p.update_layout(height=380, margin=dict(l=5, r=5, t=40, b=10), legend_title_text="Parcela")

            df_long = df_y.melt(
                id_vars=["anio"],
                value_vars=["ingresos", "costos", "neto"],
                var_name="tipo",
                value_name="monto",
            )
            fig_acum = px.bar(
                df_long,
                x="anio",
                y="monto",
                color="tipo",
                barmode="group",
                title="Ingresos, Costos y Neto — Acumulado por año",
                labels={"anio": "Año", "monto": "Monto (CLP)", "tipo": ""},
                color_discrete_map={"ingresos": "#2C5B4A", "costos": "#A4463F", "neto": "#8DA2C8"},
            )
            fig_acum.update_layout(hovermode="x unified", height=520)
            fig_acum.update_traces(
                texttemplate="%{y:,.0f}",
                textposition="inside",
                textfont=dict(color="white", size=11),
                cliponaxis=False,
            )

            cols_cost = list(df_cost_r.columns)
            col_monto_c = _pick_col(cols_cost, ["monto", "total", "importe", "valor"])
            col_cc = _pick_col(cols_cost, ["cc", "categoria", "rubro"])
            col_prov = _pick_col(cols_cost, ["proveedor"])
            tmp_cost = None
            if col_monto_c:
                tmp_cost = df_cost_r.copy()
                tmp_cost["monto_norm"] = _parse_monto_series(tmp_cost[col_monto_c])

            if tmp_cost is not None and col_cc:
                cat = (
                    tmp_cost.groupby(col_cc, as_index=False)["monto_norm"]
                    .sum()
                    .sort_values("monto_norm", ascending=False)
                )
                cat["cum_pct"] = cat["monto_norm"].cumsum() / cat["monto_norm"].sum() * 100
                fig_cost_cat = go.Figure()
                fig_cost_cat.add_trace(go.Bar(
                    x=cat[col_cc].head(12),
                    y=cat["monto_norm"].head(12),
                    name="Costo",
                    marker_color="#2C5B4A",
                    text=[f"{v:,.0f}" for v in cat["monto_norm"].head(12)],
                    textposition="inside",
                    textfont=dict(color="white", size=11),
                ))
                fig_cost_cat.add_trace(go.Scatter(
                    x=cat[col_cc].head(12),
                    y=cat["cum_pct"].head(12),
                    name="% acumulado",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#0B1F2A", width=2),
                ))
                fig_cost_cat.update_layout(
                    title="Costo por categoría",
                    yaxis=dict(title="Costo (CLP)"),
                    yaxis2=dict(title="% acumulado", overlaying="y", side="right"),
                    hovermode="x unified",
                    height=420,
                )

            if tmp_cost is not None and col_prov:
                prov = (
                    tmp_cost.groupby(col_prov, as_index=False)["monto_norm"]
                    .sum()
                    .assign(monto_abs=lambda d: d["monto_norm"].abs())
                    .query("monto_abs > 0")
                    .sort_values("monto_abs", ascending=False)
                    .head(12)
                )
                fig_cost_prov = px.pie(
                    prov,
                    names=col_prov,
                    values="monto_abs",
                    title="Costos por proveedor (top 12)",
                    hole=0.35,
                    color_discrete_sequence=["#0B1F2A", "#153A52", "#1F4F5B", "#1E3D36", "#2C5B4A", "#3A6B5A"],
                )
                fig_cost_prov.update_traces(textinfo="percent+label")
                fig_cost_prov.update_layout(height=420, legend_title_text="Proveedor")

        report_pdf = _build_obligaciones_report_pdf_bytes(
            kpi_data,
            fig_acum,
            tabla_show,
            fig_gc,
            fig_m,
            fig_p,
            fig_cost_cat,
            fig_cost_prov,
        )
        return report_pdf

    st.markdown(
        """
        <div class="top-shell">
          <div>
            <h1 class="app-title">Condominio Los Queltehues III</h1>
          </div>
          <div class="top-actions" aria-hidden="true">
            <span>☼</span><span>☆</span><span>♢</span><span>◉</span><span>⋮</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_title, col_btn = st.columns([0.78, 0.22])
    with col_title:
        st.empty()
    with col_btn:
        try:
            report_pdf = _make_obligaciones_report()
            st.download_button(
                "⇩  Descargar reporte (PDF)",
                data=report_pdf,
                file_name="reporte_obligaciones.pdf",
                mime="application/pdf",
            )
        except RuntimeError as e:
            st.info(str(e))
        except Exception as e:
            st.error(f"No se pudo generar el reporte. Detalle: {e}")

    if selected_section == "General":
        with st.spinner("Cargando datos generales..."):
            df_ing_g = _load(INGRESOS_CSV_URL, CACHE_VERSION, {"fecha", "parcela", "abono"})
            df_cost_g = _load(COSTOS_CSV_URL, CACHE_VERSION, {"monto", "proveedor", "cc"})

        serie_ing = build_series_mensual_ingresos(df_ing_g)
        serie_cost = build_series_mensual_costos(df_cost_g)
        if serie_ing.empty or serie_cost.empty:
            st.warning("No se pudieron construir series mensuales de ingresos o costos.")

        df_m = pd.merge(serie_ing, serie_cost, on="periodo", how="outer").fillna(0)
        df_m = df_m.sort_values("periodo")
        df_m["neto"] = df_m["ingresos"] - df_m["costos"].abs()
        df_m["anio"] = pd.to_datetime(df_m["periodo"] + "-01", errors="coerce").dt.year

        df_y = (
            df_m.groupby("anio", as_index=False)[["ingresos", "costos", "neto"]]
            .sum()
            .fillna(0)
        )
        df_y["costos"] = df_y["costos"].abs()

        total_ing = float(df_y["ingresos"].sum()) if not df_y.empty else 0.0
        total_cost = float(df_y["costos"].sum()) if not df_y.empty else 0.0
        total_neto = float(df_y["neto"].sum()) if not df_y.empty else 0.0
        best_year = int(df_y.sort_values("neto", ascending=False)["anio"].iloc[0]) if not df_y.empty else 0
        pendiente_proveedores = 0.0

        try:
            df_por_pagar = _load(
                POR_PAGAR_CSV_URL,
                CACHE_VERSION,
                {"motivo", "presupuesto", "abono", "pendiente_a_proveedor"},
            )
            col_pend_prov = _pick_col(list(df_por_pagar.columns), ["pendiente_a_proveedor", "pendiente_proveedores"])
            if col_pend_prov:
                pendiente_proveedores = float(_parse_monto_series(df_por_pagar[col_pend_prov]).sum())
        except Exception:
            pendiente_proveedores = 0.0
        banco_futuro = total_neto - pendiente_proveedores

        # KPI de pendientes (desde Obligaciones)
        try:
            df_obl_g = _load(OBLIGACIONES_CSV_URL, CACHE_VERSION, {"ano", "anio", "año", "parcela", "gc"})
            df_ing_g2 = _load(INGRESOS_CSV_URL, CACHE_VERSION, {"fecha", "parcela", "abono"})
            cols_ing_o = list(df_ing_g2.columns)
            cand_concepto = ["detalle", "concepto", "glosa", "descripcion", "tipo", "categoria", "cc", "ccc", "medio"]
            concepto_col_val = next((c for c in cand_concepto if c in cols_ing_o), None)
            oblig_anual_g, tabla_g = build_obligaciones_vs_pagos(
                df_obl_g,
                df_ing_g2,
                concepto_col=concepto_col_val,
                include_keywords=["gasto", "gc"],
                exclude_keywords=["proyecto"],
            )
            if not tabla_g.empty:
                gc_total_parcela = float(tabla_g["gc_total"].max())
                total_gc = gc_total_parcela * 20
                total_pagado_gc = float(tabla_g["pagado"].sum())
                pendiente_gc = float(tabla_g["pendiente"].clip(lower=0).sum())
                pct_no_pago = (pendiente_gc / total_gc) * 100 if total_gc > 0 else 0.0
                pendiente_mant = 0.0
                pendiente_proy = 0.0

                col_cc_ing = _pick_col(cols_ing_o, ["cc", "categoria", "rubro", "ccc"])
                col_abono_ing = _pick_col(cols_ing_o, ["abono"])
                col_parc_ing = _pick_col(cols_ing_o, ["parcela"])
                if col_parc_ing and col_abono_ing:
                    ing_all = df_ing_g2.copy()
                    ing_all["parcela"] = pd.to_numeric(
                        ing_all[col_parc_ing].astype(str).str.replace(r"[^\d]", "", regex=True),
                        errors="coerce",
                    )
                    ing_all["monto_norm"] = _parse_monto_series(ing_all[col_abono_ing])
                    ing_all = ing_all.dropna(subset=["parcela", "monto_norm"])

                    # Pendiente mantención
                    df_mant_g = load_mantencion_table(MANTENCION_CSV_URL)
                    if not df_mant_g.empty and col_cc_ing:
                        mant = df_mant_g.groupby("parcela", as_index=False)["mantencion"].sum()
                        cc_text = (
                            ing_all[col_cc_ing]
                            .astype(str)
                            .str.lower()
                            .str.replace("á", "a")
                            .str.replace("é", "e")
                            .str.replace("í", "i")
                            .str.replace("ó", "o")
                            .str.replace("ú", "u")
                            .str.replace("ñ", "n")
                        )
                        mask_mant = cc_text.str.contains("mantencion", regex=False) | cc_text.str.contains("mantenimiento", regex=False)
                        pagos_m = (
                            ing_all[mask_mant]
                            .groupby("parcela", as_index=False)["monto_norm"]
                            .sum()
                            .rename(columns={"monto_norm": "pagado_mant"})
                        )
                        mant = mant.merge(pagos_m, on="parcela", how="left").fillna({"pagado_mant": 0})
                        pendiente_mant = float((mant["mantencion"] - mant["pagado_mant"]).clip(lower=0).sum())

                    # Pendiente proyecto por CC desde TD 2.3
                    df_td_g = load_td23_table(TD23_CSV_URL)
                    if not df_td_g.empty and col_cc_ing:
                        ing_cc = ing_all.copy()
                        ing_cc["cc_norm"] = ing_cc[col_cc_ing].astype(str).str.lower()
                        parcelas = sorted(tabla_g["parcela"].dropna().unique().tolist())
                        for _, row in df_td_g.iterrows():
                            cc_name = str(row.get("cc", "")).strip()
                            if not cc_name:
                                continue
                            monto_cc = float(_parse_monto_series(pd.Series([row.get("monto")])).iloc[0] or 0)
                            if monto_cc == 0:
                                continue
                            mask_cc = ing_cc["cc_norm"].str.contains(cc_name.lower(), regex=False)
                            pagos_cc = (
                                ing_cc[mask_cc]
                                .groupby("parcela", as_index=False)["monto_norm"]
                                .sum()
                                .set_index("parcela")
                            )
                            pagos_series = pagos_cc["monto_norm"].reindex(parcelas, fill_value=0)
                            pendiente_proy += float((monto_cc - pagos_series).clip(lower=0).sum())

                pendiente_total = pendiente_gc + pendiente_mant + pendiente_proy
            else:
                pendiente_gc = 0.0
                pendiente_total = 0.0
                pendiente_mant = 0.0
                pendiente_proy = 0.0
                pct_no_pago = 0.0
        except Exception:
            pendiente_gc = 0.0
            pendiente_total = 0.0
            pendiente_mant = 0.0
            pendiente_proy = 0.0
            pct_no_pago = 0.0

        kpi_tiles = "".join(
            [
                _kpi_tile(
                    "↗",
                    "Total ingresos",
                    f"${total_ing:,.0f}",
                    "Suma histórica",
                    "green",
                    "0,28 20,18 38,21 56,10 76,22 96,4 116,15 136,10 156,21",
                ),
                _kpi_tile(
                    "⊥",
                    "Total costos",
                    f"${total_cost:,.0f}",
                    "Suma histórica",
                    "red",
                    "0,20 20,19 38,21 56,8 76,20 96,9 116,18 136,13 156,23",
                ),
                _kpi_tile(
                    "◔",
                    "Pendiente de pago total",
                    f"${pendiente_total:,.0f}",
                    "GC + mantención + proyecto",
                    "violet",
                    "0,25 20,13 38,17 56,11 76,18 96,2 116,16 136,9 156,21",
                ),
                _kpi_tile(
                    "●",
                    "Pendiente a proveedores",
                    f"${pendiente_proveedores:,.0f}",
                    "Pestaña Por pagar",
                    "teal",
                    "0,20 156,20",
                ),
                _kpi_tile(
                    "%",
                    "% no pago total",
                    f"{pct_no_pago:,.1f}%",
                    "Pendiente / GC total",
                    "red",
                    "0,25 20,18 38,21 56,14 76,22 96,7 116,20 136,13 156,23",
                ),
                _kpi_tile(
                    "▣",
                    "Mejor año",
                    f"{best_year}",
                    "Mayor neto",
                    "blue",
                    "0,27 20,15 38,19 56,18 76,11 96,18 116,5 136,17 156,10",
                ),
            ]
        )
        st.markdown(
            f"""
            <div class="dash-hero">
              <div>
                <div class="dash-hero-eyebrow">NETO ACUMULADO - BANCO</div>
                <div class="dash-hero-value">${total_neto:,.0f}</div>
                <div class="dash-hero-note">↗&nbsp;&nbsp;Resultado histórico acumulado: ingresos menos costos</div>
                <div class="dash-hero-rule"></div>
                <div class="dash-hero-label">BANCO FUTURO SIN CUENTAS POR PAGAR</div>
                <div class="dash-hero-secondary-value">${banco_futuro:,.0f}</div>
                <div class="dash-hero-sub">⚖&nbsp;&nbsp;Neto acumulado menos pendiente a proveedores</div>
              </div>
              <div class="bank-art" aria-hidden="true">
                <div class="bank">
                  <div class="bank-roof"></div>
                  <div class="bank-row">
                    <div class="bank-col"></div>
                    <div class="bank-col"></div>
                    <div class="bank-col"></div>
                    <div class="bank-col"></div>
                  </div>
                  <div class="bank-coin">$</div>
                  <div class="bank-base"></div>
                </div>
              </div>
            </div>
            <div class="dash-kpi-grid">
              {kpi_tiles}
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            import plotly.express as px
        except Exception:
            st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
        else:
            df_long = df_y.melt(
                id_vars=["anio"],
                value_vars=["ingresos", "costos", "neto"],
                var_name="tipo",
                value_name="monto",
            )
            fig_g = px.bar(
                df_long,
                x="anio",
                y="monto",
                color="tipo",
                barmode="group",
                text="monto",
                title="Ingresos, Costos y Neto — Totales por año",
                labels={"anio": "Año", "monto": "Monto (CLP)", "tipo": "Tipo"},
                color_discrete_map={
                    "ingresos": "#1F6F5B",
                    "costos": "#A4463F",
                    "neto": "#8DA2C8",
                },
            )
            fig_g.update_traces(texttemplate="$%{text:,.0f}", textposition="inside", textfont_color="white")
            fig_g.update_layout(
                hovermode="x unified",
                height=520,
                plot_bgcolor="#ffffff",
                paper_bgcolor="white",
                font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#273043"),
                xaxis=dict(title="Año", showgrid=False),
                yaxis=dict(title="Monto (CLP)", gridcolor="#e9edf3"),
                legend=dict(orientation="h", y=1.12, x=0.01),
                margin=dict(l=40, r=20, t=70, b=40),
            )

            df_y_cum = df_y.copy()
            df_y_cum[["ingresos", "costos", "neto"]] = df_y_cum[["ingresos", "costos", "neto"]].cumsum()
            df_long_cum = df_y_cum.melt(
                id_vars=["anio"],
                value_vars=["ingresos", "costos", "neto"],
                var_name="tipo",
                value_name="monto",
            )
            fig_cum = px.bar(
                df_long_cum,
                x="anio",
                y="monto",
                color="tipo",
                barmode="group",
                text="monto",
                title="Ingresos, Costos y Neto — Acumulado por año",
                labels={"anio": "Año", "monto": "Monto (CLP)", "tipo": "Tipo"},
                color_discrete_map={
                    "ingresos": "#1F6F5B",
                    "costos": "#A4463F",
                    "neto": "#8DA2C8",
                },
            )
            fig_cum.update_traces(texttemplate="$%{text:,.0f}", textposition="inside", textfont_color="white")
            fig_cum.update_layout(
                hovermode="x unified",
                height=520,
                plot_bgcolor="#ffffff",
                paper_bgcolor="white",
                font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#273043"),
                xaxis=dict(title="Año", showgrid=False),
                yaxis=dict(title="Monto (CLP)", gridcolor="#e9edf3"),
                legend=dict(orientation="h", y=1.12, x=0.01),
                margin=dict(l=40, r=20, t=70, b=40),
            )
            st.plotly_chart(fig_cum, use_container_width=True)
            st.plotly_chart(fig_g, use_container_width=True)

        st.subheader("Resumen por año")
        df_y_show = df_y.copy()
        df_y_show = df_y_show.rename(columns={"ingresos": "Ingresos", "costos": "Costos", "neto": "Neto", "anio": "Año"})
        for col in ["Ingresos", "Costos", "Neto"]:
            df_y_show[col] = df_y_show[col].map(lambda x: f"${x:,.0f}")
        left_tbl, right_pie = st.columns([1.2, 1])
        with left_tbl:
            st.dataframe(df_y_show, use_container_width=True, height=360, hide_index=True)
        with right_pie:
            try:
                import plotly.express as px
            except Exception:
                st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
            else:
                if not df_y.empty:
                    fig_pie_y = px.pie(
                        df_y,
                        names="anio",
                        values="ingresos",
                        title="Distribución de ingresos por año",
                        hole=0.35,
                        color_discrete_sequence=["#0B1F2A", "#1F4F5B", "#2C5B4A", "#3A6B5A", "#8DA2C8", "#A4463F"],
                    )
                    fig_pie_y.update_traces(textinfo="percent+label")
                    fig_pie_y.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Año")
                    st.plotly_chart(fig_pie_y, use_container_width=True)
                else:
                    st.info("Sin datos para el gráfico.")

    if selected_section == "Ingresos V2.3":
        st.subheader("Ingresos — Análisis técnico")
        with st.spinner("Cargando ingresos..."):
            df_ing = _load(INGRESOS_CSV_URL, CACHE_VERSION, {"fecha", "parcela", "abono"})

        cols = list(df_ing.columns)
        col_parcela = "parcela" if "parcela" in cols else None
        col_monto = next((c for c in cols if c.startswith("abono")), None)
        col_fecha = _pick_col(cols, ["fecha"])
        col_concepto = next((c for c in ["detalle", "concepto", "glosa", "descripcion", "tipo", "categoria", "cc", "ccc", "medio"] if c in cols), None)

        if not col_parcela or not col_monto:
            st.error("No pude identificar columnas de parcela y abono. Selecciónalas manualmente.")
            col_parcela = st.selectbox("Columna parcela", cols, key="sel_parcela")
            col_monto = st.selectbox("Columna monto/abono", cols, key="sel_monto")

        base = df_ing.copy()
        base["monto_norm"] = _parse_monto_series(base[col_monto])
        base["parcela_norm"] = pd.to_numeric(
            base[col_parcela].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce",
        )
        if col_fecha:
            base["fecha_norm"] = pd.to_datetime(base[col_fecha], dayfirst=True, errors="coerce")
            base["anio"] = base["fecha_norm"].dt.year
            base["mes"] = base["fecha_norm"].dt.month
            base["periodo"] = base["fecha_norm"].dt.to_period("M").astype(str)
        base = base.dropna(subset=["monto_norm", "parcela_norm"])

        years = sorted([int(y) for y in base["anio"].dropna().unique().tolist()]) if "anio" in base.columns else []
        sel_years = years
        sel_parcelas = []
        sel_conceptos = sorted(base[col_concepto].dropna().astype(str).unique().tolist()) if col_concepto else []

        filt = base.copy()
        if "anio" in base.columns and sel_years:
            filt = filt[filt["anio"].isin(sel_years)]
        if sel_parcelas:
            filt = filt[filt["parcela_norm"].isin(sel_parcelas)]
        if col_concepto and sel_conceptos:
            filt = filt[filt[col_concepto].astype(str).isin(sel_conceptos)]

        total_ing = float(filt["monto_norm"].sum())
        n_reg = int(len(filt))
        avg_mensual = float(filt.groupby("periodo")["monto_norm"].sum().mean()) if "periodo" in filt.columns and not filt.empty else 0.0
        top_parc = (
            filt.groupby("parcela_norm")["monto_norm"].sum().sort_values(ascending=False).index[0]
            if not filt.empty else "-"
        )
        # Ingresos GC vs Mantención vs Proyectos
        if col_concepto:
            texto_conc = filt[col_concepto].astype(str).str.lower()
            ing_gc = float(filt[texto_conc.str.contains("gasto", regex=False) | texto_conc.str.contains("gc", regex=False)]["monto_norm"].sum())
            ing_mant = float(
                filt[
                    texto_conc.str.contains("mantencion", regex=False)
                    | texto_conc.str.contains("mantenimiento", regex=False)
                ]["monto_norm"].sum()
            )
            ing_proy = float(filt[texto_conc.str.contains("proyecto", regex=False)]["monto_norm"].sum())
        else:
            ing_gc = 0.0
            ing_mant = 0.0
            ing_proy = 0.0

        st.markdown(
            """
            <style>
            .kpi-row {display:flex;gap:16px;overflow-x:auto;padding-bottom:6px;margin:8px 0 18px 0;}
            .kpi-card {min-width:220px;flex:0 0 220px;background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:14px 16px;box-shadow:0 2px 12px rgba(15,23,42,0.06);position:relative;}
            .kpi-card:before {content:"";position:absolute;left:0;top:0;height:100%;width:6px;border-radius:16px 0 0 16px;}
            .kpi-title {font-size:11px;letter-spacing:0.08em;color:#6B7280;font-weight:700;}
            .kpi-value {font-size:22px;font-weight:800;margin-top:6px;}
            .kpi-sub {font-size:11px;color:#94A3B8;margin-top:6px;}
            .kpi-green:before {background:#22C55E;}
            .kpi-navy:before {background:#0B1F2A;}
            .kpi-teal:before {background:#2C5B4A;}
            .kpi-red:before {background:#EF4444;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi-card kpi-green">
                <div class="kpi-title">TOTAL INGRESOS</div>
                <div class="kpi-value">${total_ing:,.0f}</div>
                <div class="kpi-sub">Suma filtrada</div>
              </div>
              <div class="kpi-card kpi-teal">
                <div class="kpi-title">INGRESOS GC</div>
                <div class="kpi-value">${ing_gc:,.0f}</div>
                <div class="kpi-sub">Solo gasto común</div>
              </div>
              <div class="kpi-card kpi-teal">
                <div class="kpi-title">INGRESOS MANTENCIÓN</div>
                <div class="kpi-value">${ing_mant:,.0f}</div>
                <div class="kpi-sub">Solo mantención</div>
              </div>
              <div class="kpi-card kpi-red">
                <div class="kpi-title">INGRESOS PROYECTO</div>
                <div class="kpi-value">${ing_proy:,.0f}</div>
                <div class="kpi-sub">Solo proyectos</div>
              </div>
              <div class="kpi-card kpi-navy">
                <div class="kpi-title">PROMEDIO MENSUAL</div>
                <div class="kpi-value">${avg_mensual:,.0f}</div>
                <div class="kpi-sub">Ingreso medio por mes</div>
              </div>
              <div class="kpi-card kpi-teal">
                <div class="kpi-title">PARCELA TOP</div>
                <div class="kpi-value">{top_parc}</div>
                <div class="kpi-sub">Mayor aporte</div>
              </div>
              <div class="kpi-card kpi-red">
                <div class="kpi-title">REGISTROS</div>
                <div class="kpi-value">{n_reg}</div>
                <div class="kpi-sub">Transacciones filtradas</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pagos = (
            filt.groupby("parcela_norm", as_index=False)["monto_norm"]
            .sum()
            .rename(columns={"parcela_norm": "parcela", "monto_norm": "pagos_total"})
            .sort_values("parcela")
        )
        rango = list(range(17, 37))
        pagos = (
            pd.DataFrame({"parcela": rango})
            .merge(pagos, on="parcela", how="left")
            .fillna({"pagos_total": 0})
        )

        try:
            import plotly.express as px
        except Exception:
            st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
        else:
            muted_palette = [
                "#0B1F2A",
                "#153A52",
                "#1F4F5B",
                "#1E3D36",
                "#2C5B4A",
                "#3A6B5A",
                "#1D2B3A",
                "#2A3F4D",
            ]
            fig = px.bar(
                pagos,
                x="parcela",
                y="pagos_total",
                text="pagos_total",
                title="Pagos por parcela (17–36)",
                labels={"parcela": "Parcela", "pagos_total": "Monto abonado (CLP)"},
                color_discrete_sequence=muted_palette,
            )
            fig.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="inside",
                textfont_color="white",
                hovertemplate="Parcela %{x}<br>Monto CLP %{y:,.0f}<extra></extra>",
            )
            fig.update_layout(hovermode="x unified", height=520)
            fig.update_xaxes(type="category", tickmode="linear", dtick=1)
            st.plotly_chart(fig, use_container_width=True)

            if "periodo" in filt.columns:
                per = (
                    filt.groupby("periodo", as_index=False)["monto_norm"]
                    .sum()
                    .sort_values("periodo")
                )
                fig_p = px.bar(
                    per,
                    x="periodo",
                    y="monto_norm",
                    title="Ingresos por periodo",
                    labels={"periodo": "Periodo", "monto_norm": "Ingreso (CLP)"},
                    color_discrete_sequence=muted_palette,
                )
                fig_p.update_layout(hovermode="x unified", height=420)
                st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Detalle de ingresos (filtrado)")
        st.dataframe(filt, use_container_width=True)

    if selected_section == "Costos":
        st.subheader("Costos — Análisis técnico")
        with st.spinner("Cargando costos..."):
            df_cost = _load(COSTOS_CSV_URL, CACHE_VERSION, {"monto", "proveedor", "cc"})

        cols_cost = list(df_cost.columns)
        col_monto = _pick_col(cols_cost, ["monto", "total", "importe", "valor"])
        col_fecha = _pick_col(cols_cost, ["d", "fecha"])
        col_cc = _pick_col(cols_cost, ["cc", "categoria", "rubro"])
        col_ccc = _pick_col(cols_cost, ["ccc", "subcategoria"])
        col_prov = _pick_col(cols_cost, ["proveedor"])

        if not col_monto:
            st.error("No se encontró columna de monto en Costos.")
            return

        base = df_cost.copy()
        base["monto_norm"] = _parse_monto_series(base[col_monto])
        if col_fecha:
            base["fecha_norm"] = pd.to_datetime(base[col_fecha], dayfirst=True, errors="coerce")
            base["anio"] = base["fecha_norm"].dt.year
            base["mes"] = base["fecha_norm"].dt.month
            base["periodo"] = base["fecha_norm"].dt.to_period("M").astype(str)
        base = base.dropna(subset=["monto_norm"])

        years = sorted([int(y) for y in base["anio"].dropna().unique().tolist()]) if "anio" in base.columns else []
        sel_years = years
        sel_cats = sorted(base[col_cc].dropna().astype(str).unique().tolist()) if col_cc else []
        sel_prov = sorted(base[col_prov].dropna().astype(str).unique().tolist()) if col_prov else []

        filt = base.copy()
        if "anio" in base.columns and sel_years:
            filt = filt[filt["anio"].isin(sel_years)]
        if col_cc and sel_cats:
            filt = filt[filt[col_cc].astype(str).isin(sel_cats)]
        if col_prov and sel_prov:
            filt = filt[filt[col_prov].astype(str).isin(sel_prov)]

        total_cost = float(filt["monto_norm"].sum())
        n_reg = int(len(filt))
        avg_mensual = float(filt.groupby("periodo")["monto_norm"].sum().mean()) if "periodo" in filt.columns and not filt.empty else 0.0
        top_prov = (
            filt.groupby(col_prov)["monto_norm"].sum().sort_values(ascending=True).index[0]
            if col_prov and not filt.empty else "-"
        )
        top_cat = (
            filt.groupby(col_cc)["monto_norm"].sum().sort_values(ascending=True).index[0]
            if col_cc and not filt.empty else "-"
        )

        st.markdown(
            """
            <style>
            .kpi-grid-3 {display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:8px 0 18px 0;}
            .kpi-card {background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:18px 20px;box-shadow:0 2px 12px rgba(15,23,42,0.06);position:relative;}
            .kpi-card:before {content:"";position:absolute;left:0;top:0;height:100%;width:6px;border-radius:16px 0 0 16px;}
            .kpi-title {font-size:12px;letter-spacing:0.08em;color:#6B7280;font-weight:700;}
            .kpi-value {font-size:24px;font-weight:800;margin-top:6px;}
            .kpi-sub {font-size:12px;color:#94A3B8;margin-top:6px;}
            .kpi-navy:before {background:#0B1F2A;}
            .kpi-red:before {background:#EF4444;}
            .kpi-green:before {background:#22C55E;}
            .kpi-teal:before {background:#2C5B4A;}
            @media (max-width: 1100px){.kpi-grid-3{grid-template-columns:repeat(2,1fr);}}
            @media (max-width: 700px){.kpi-grid-3{grid-template-columns:1fr;}}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-grid-3">
              <div class="kpi-card kpi-red">
                <div class="kpi-title">TOTAL COSTOS</div>
                <div class="kpi-value">${total_cost:,.0f}</div>
                <div class="kpi-sub">Suma filtrada</div>
              </div>
              <div class="kpi-card kpi-navy">
                <div class="kpi-title">PROMEDIO MENSUAL</div>
                <div class="kpi-value">${avg_mensual:,.0f}</div>
                <div class="kpi-sub">Costo medio por mes</div>
              </div>
              <div class="kpi-card kpi-teal">
                <div class="kpi-title">TOP CATEGORÍA</div>
                <div class="kpi-value">{top_cat}</div>
                <div class="kpi-sub">Mayor gasto</div>
              </div>
              <div class="kpi-card kpi-green">
                <div class="kpi-title">TOP PROVEEDOR</div>
                <div class="kpi-value">{top_prov}</div>
                <div class="kpi-sub">Mayor gasto</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except Exception:
            st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
        else:
            muted_palette = [
                "#0B1F2A",
                "#153A52",
                "#1F4F5B",
                "#1E3D36",
                "#2C5B4A",
                "#3A6B5A",
                "#1D2B3A",
                "#2A3F4D",
            ]

            if "periodo" in filt.columns:
                per = (
                    filt.groupby("periodo", as_index=False)["monto_norm"]
                    .sum()
                    .sort_values("periodo")
                )
                fig_p = px.bar(
                    per,
                    x="periodo",
                    y="monto_norm",
                    title="Costos por periodo",
                    labels={"periodo": "Periodo", "monto_norm": "Costo (CLP)"},
                    color_discrete_sequence=["#0B1F2A"],
                    text="monto_norm",
                )
                fig_p.update_traces(
                    texttemplate="$%{text:,.0f}",
                    textposition="inside",
                    textfont_color="white",
                    hovertemplate="Periodo %{x}<br>Costo CLP %{y:,.0f}<extra></extra>",
                )
                fig_p.update_layout(
                    hovermode="x unified",
                    height=420,
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="#e9edf3"),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            if col_cc:
                cat = (
                    filt.groupby(col_cc, as_index=False)["monto_norm"]
                    .sum()
                    .sort_values("monto_norm", ascending=False)
                )
                cat["cum_pct"] = cat["monto_norm"].cumsum() / cat["monto_norm"].sum() * 100
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(
                    x=cat[col_cc].head(12),
                    y=cat["monto_norm"].head(12),
                    name="Costo",
                    marker_color="#2C5B4A",
                    text=cat["monto_norm"].head(12),
                    textposition="inside",
                    texttemplate="$%{text:,.0f}",
                    textfont=dict(color="white"),
                ))
                fig_c.add_trace(go.Scatter(
                    x=cat[col_cc].head(12),
                    y=cat["cum_pct"].head(12),
                    name="% acumulado",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#0B1F2A", width=2),
                ))
                fig_c.update_layout(
                    title="Costo por categoría",
                    yaxis=dict(title="Costo (CLP)"),
                    yaxis2=dict(title="% acumulado", overlaying="y", side="right"),
                    hovermode="x unified",
                    height=420,
                )
                st.plotly_chart(fig_c, use_container_width=True)

            if col_prov:
                prov = (
                    filt.groupby(col_prov, as_index=False)["monto_norm"]
                    .sum()
                    .assign(monto_abs=lambda d: d["monto_norm"].abs())
                    .query("monto_abs > 0")
                    .sort_values("monto_abs", ascending=False)
                    .head(12)
                )
                fig_v = px.pie(
                    prov,
                    names=col_prov,
                    values="monto_abs",
                    title="Costos por proveedor (top 12)",
                    hole=0.35,
                    color_discrete_sequence=muted_palette,
                )
                fig_v.update_traces(textinfo="percent+label")
                fig_v.update_layout(height=462, legend_title_text="Proveedor")
                st.plotly_chart(fig_v, use_container_width=True)

        st.subheader("Detalle de costos (filtrado)")
        st.dataframe(filt, use_container_width=True)

    if selected_section == "Obligaciones":
        st.subheader("Obligaciones vs Pagos (por año y parcela)")
        with st.spinner("Cargando obligaciones y pagos..."):
            df_obl = _load(OBLIGACIONES_CSV_URL, CACHE_VERSION, {"ano", "anio", "año", "parcela", "gc"})
            df_ing_o = _load(INGRESOS_CSV_URL, CACHE_VERSION, {"fecha", "parcela", "abono"})
            df_prop = _load(PROPIETARIOS_CSV_URL, CACHE_VERSION, {"parcela", "propietario"})
            df_td = load_td23_table(TD23_CSV_URL)
            df_mant = load_mantencion_table(MANTENCION_CSV_URL)
            df_por_pagar_o = _load(
                POR_PAGAR_CSV_URL,
                CACHE_VERSION,
                {"motivo", "presupuesto", "abono", "pendiente_a_proveedor"},
            )

        cols_ing_o = list(df_ing_o.columns)
        cand_concepto = ["detalle", "concepto", "glosa", "descripcion", "tipo", "categoria", "cc", "ccc", "medio"]
        concepto_col_val = next((c for c in cand_concepto if c in cols_ing_o), None)
        include_list = ["gasto", "gc"]
        exclude_list = ["proyecto"]

        oblig_anual, tabla = build_obligaciones_vs_pagos(
            df_obl,
            df_ing_o,
            concepto_col=concepto_col_val,
            include_keywords=include_list,
            exclude_keywords=exclude_list,
        )
        if tabla.empty:
            st.warning("No se pudieron construir obligaciones vs pagos. Revisa columnas de año/parcela/gc.")
        else:
            fig_obl_pie = None
            fig_gc = None
            fig_m = None
            fig_p = None
            oblig_show = pd.DataFrame()
            if not oblig_anual.empty:
                st.subheader("Obligación por año (GC)")
                c_left, c_right = st.columns([1.2, 1])
                with c_left:
                    oblig_show = oblig_anual.copy()
                    oblig_show = oblig_show.rename(columns={"anio": "Año", "gc_total": "GC total por año"})
                    oblig_show["GC total por año"] = oblig_show["GC total por año"].map(lambda x: f"${x:,.0f}")
                    st.dataframe(oblig_show, use_container_width=True, height=260, hide_index=True)
                with c_right:
                    try:
                        import plotly.express as px
                    except Exception:
                        st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
                    else:
                        fig_obl_pie = px.pie(
                            oblig_anual,
                            names="anio",
                            values="gc_total",
                            title="Distribución GC por año",
                            hole=0.35,
                            color_discrete_sequence=["#0B1F2A", "#1F4F5B", "#2C5B4A", "#3A6B5A", "#8DA2C8", "#A4463F"],
                        )
                        fig_obl_pie.update_traces(textinfo="percent+label")
                        fig_obl_pie.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_obl_pie, use_container_width=True)

            tabla_full = tabla.copy()
            tabla_full["pendiente_pos"] = tabla_full["pendiente"].clip(lower=0)
            tabla_full["saldo_favor"] = (-tabla_full["pendiente"]).clip(lower=0)

            # Mantención por parcela
            if not df_mant.empty:
                df_mant = df_mant.groupby("parcela", as_index=False)["mantencion"].sum()
                tabla_full = tabla_full.merge(df_mant, on="parcela", how="left").fillna({"mantencion": 0})

                # Resta pagos de mantención desde ingresos (cc = mantención/mantenimiento)
                cols_ing = list(df_ing_o.columns)
                col_cc_ing = _pick_col(cols_ing, ["cc", "categoria", "rubro", "ccc"])
                col_abono_ing = _pick_col(cols_ing, ["abono"])
                col_parc_ing = _pick_col(cols_ing, ["parcela"])
                if col_cc_ing and col_abono_ing and col_parc_ing:
                    ing_m = df_ing_o.copy()
                    ing_m["parcela"] = pd.to_numeric(
                        ing_m[col_parc_ing].astype(str).str.replace(r"[^\d]", "", regex=True),
                        errors="coerce",
                    )
                    ing_m["monto_norm"] = _parse_monto_series(ing_m[col_abono_ing])
                    ing_m = ing_m.dropna(subset=["parcela", "monto_norm"])
                    cc_text = (
                        ing_m[col_cc_ing]
                        .astype(str)
                        .str.lower()
                        .str.replace("á", "a")
                        .str.replace("é", "e")
                        .str.replace("í", "i")
                        .str.replace("ó", "o")
                        .str.replace("ú", "u")
                        .str.replace("ñ", "n")
                    )
                    mask_mant = cc_text.str.contains("mantencion", regex=False) | cc_text.str.contains("mantenimiento", regex=False)
                    pagos_mant = (
                        ing_m[mask_mant]
                        .groupby("parcela", as_index=False)["monto_norm"]
                        .sum()
                        .rename(columns={"monto_norm": "pagado_mant"})
                    )
                    tabla_full = tabla_full.merge(pagos_mant, on="parcela", how="left").fillna({"pagado_mant": 0})
                    tabla_full["mantencion"] = (tabla_full["mantencion"] - tabla_full["pagado_mant"]).clip(lower=0)
                    tabla_full = tabla_full.drop(columns=["pagado_mant"])

                tabla_full = tabla_full.rename(columns={"mantencion": "Mantención"})

            # Cruce por CC desde TD 2.3
            if not df_td.empty:
                cols_ing = list(df_ing_o.columns)
                col_cc_ing = _pick_col(cols_ing, ["cc", "categoria", "rubro", "ccc"])
                if col_cc_ing:
                    df_ing_cc = df_ing_o.copy()
                    df_ing_cc["parcela"] = pd.to_numeric(
                        df_ing_cc[_pick_col(cols_ing, ["parcela"])].astype(str).str.replace(r"[^\d]", "", regex=True),
                        errors="coerce",
                    )
                    df_ing_cc["monto_norm"] = _parse_monto_series(df_ing_cc[_pick_col(cols_ing, ["abono"])])
                    df_ing_cc = df_ing_cc.dropna(subset=["parcela", "monto_norm"])
                    df_ing_cc["cc_norm"] = df_ing_cc[col_cc_ing].astype(str).str.lower()

                    df_td = df_td.copy()
                    df_td["cc_norm"] = df_td["cc"].astype(str).str.lower()
                    df_td["monto_norm"] = _parse_monto_series(df_td["monto"])

                    for _, row in df_td.iterrows():
                        cc_name = str(row["cc"]).strip()
                        if not cc_name:
                            continue
                        monto_cc = float(row["monto_norm"]) if pd.notna(row["monto_norm"]) else 0.0
                        if monto_cc == 0:
                            continue
                        mask_cc = df_ing_cc["cc_norm"].str.contains(cc_name.lower(), regex=False)
                        pagos_cc = (
                            df_ing_cc[mask_cc]
                            .groupby("parcela", as_index=False)["monto_norm"]
                            .sum()
                            .rename(columns={"monto_norm": "pagado_cc"})
                        )
                        col_name = f"Pendiente {cc_name}"
                        tabla_full = tabla_full.merge(pagos_cc, on="parcela", how="left").fillna({"pagado_cc": 0})
                        tabla_full[col_name] = (monto_cc - tabla_full["pagado_cc"]).clip(lower=0)
                        tabla_full = tabla_full.drop(columns=["pagado_cc"])

            gc_total_parcela = float(tabla_full["gc_total"].max()) if not tabla_full.empty else 0.0
            total_pendiente = float(tabla_full["pendiente_pos"].sum()) if not tabla_full.empty else 0.0
            total_favor = float(tabla_full["saldo_favor"].sum()) if not tabla_full.empty else 0.0
            pendiente_mant = float(tabla_full["Mantención"].sum()) if "Mantención" in tabla_full.columns else 0.0
            col_pend_prov_o = _pick_col(list(df_por_pagar_o.columns), ["pendiente_a_proveedor", "pendiente_proveedores"])
            pendiente_proveedores = (
                float(_parse_monto_series(df_por_pagar_o[col_pend_prov_o]).sum())
                if col_pend_prov_o else 0.0
            )
            # Suma pendientes por CC (ej. Proyecto)
            cc_cols = [c for c in tabla_full.columns if c.startswith("Pendiente ")]
            pendiente_proy = float(tabla_full[cc_cols].sum().sum()) if cc_cols else 0.0

            st.markdown(
                """
                <style>
                .kpi-grid {display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:8px 0 18px 0;}
                .kpi-card {background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:18px 20px;box-shadow:0 2px 12px rgba(15,23,42,0.06);position:relative;}
                .kpi-card:before {content:"";position:absolute;left:0;top:0;height:100%;width:6px;border-radius:16px 0 0 16px;}
                .kpi-title {font-size:11px;letter-spacing:0.08em;color:#6B7280;font-weight:700;}
                .kpi-value {font-size:22px;font-weight:800;margin-top:6px;}
                .kpi-sub {font-size:12px;color:#94A3B8;margin-top:6px;}
                .kpi-green:before {background:#22C55E;}
                .kpi-red:before {background:#EF4444;}
                .kpi-navy:before {background:#0B1F2A;}
                .kpi-teal:before {background:#2C5B4A;}
                @media (max-width: 1100px){.kpi-grid{grid-template-columns:repeat(2,1fr);}}
                @media (max-width: 700px){.kpi-grid{grid-template-columns:1fr;}}
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="kpi-grid">
                  <div class="kpi-card kpi-navy">
                    <div class="kpi-title">OBLIGACIÓN POR PARCELA</div>
                    <div class="kpi-value">${gc_total_parcela:,.0f}</div>
                    <div class="kpi-sub">Total GC acumulado</div>
                  </div>
                  <div class="kpi-card kpi-teal">
                    <div class="kpi-title">PENDIENTE A PROVEEDORES</div>
                    <div class="kpi-value">${pendiente_proveedores:,.0f}</div>
                    <div class="kpi-sub">Pestaña Por pagar</div>
                  </div>
                  <div class="kpi-card kpi-red">
                    <div class="kpi-title">PENDIENTE GC</div>
                    <div class="kpi-value">${total_pendiente:,.0f}</div>
                    <div class="kpi-sub">Solo montos positivos</div>
                  </div>
                  <div class="kpi-card kpi-red">
                    <div class="kpi-title">PENDIENTE MANTENCIÓN</div>
                    <div class="kpi-value">${pendiente_mant:,.0f}</div>
                    <div class="kpi-sub">Mantención por parcela</div>
                  </div>
                  <div class="kpi-card kpi-red">
                    <div class="kpi-title">PENDIENTE PROYECTO</div>
                    <div class="kpi-value">${pendiente_proy:,.0f}</div>
                    <div class="kpi-sub">Suma CC proyectos</div>
                  </div>
                  <div class="kpi-card kpi-teal">
                    <div class="kpi-title">GC POR ANTICIPADO</div>
                    <div class="kpi-value">${total_favor:,.0f}</div>
                    <div class="kpi-sub">Pagos sobre obligación</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Obligación acumulada vs Pagos")
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except Exception:
                st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
            else:
                viz_df = tabla_full.copy()
                viz_df["cumplimiento_pct"] = (
                    (viz_df["pagado"] / viz_df["gc_total"].replace(0, pd.NA)) * 100
                ).fillna(0).clip(lower=0)
                viz_df["pendiente_plot"] = viz_df["pendiente_pos"]
                viz_df["favor_plot"] = -viz_df["saldo_favor"]
                parcelas_orden = list(range(17, 37))
                viz_df["parcela"] = pd.to_numeric(viz_df["parcela"], errors="coerce")
                viz_df = (
                    pd.DataFrame({"parcela": parcelas_orden})
                    .merge(viz_df, on="parcela", how="left")
                    .fillna(
                        {
                            "gc_total": 0,
                            "pagado": 0,
                            "cumplimiento_pct": 0,
                            "pendiente_plot": 0,
                            "favor_plot": 0,
                            "saldo_favor": 0,
                        }
                    )
                )
                viz_df["Parcela"] = viz_df["parcela"].astype(int).astype(str)
                fig_obl_comp = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=False,
                    vertical_spacing=0.12,
                    row_heights=[0.68, 0.32],
                    subplot_titles=(
                        "Cobertura de pagos por parcela (17 al 36)",
                        "Brecha por parcela: pendiente o saldo a favor",
                    ),
                )
                fig_obl_comp.add_trace(
                    go.Bar(
                        y=viz_df["Parcela"],
                        x=viz_df["gc_total"],
                        name="Obligación",
                        orientation="h",
                        marker=dict(color="#E2E8F0", line=dict(color="#CBD5E1", width=1)),
                        hovertemplate="Parcela %{y}<br>Obligación CLP %{x:,.0f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig_obl_comp.add_trace(
                    go.Bar(
                        y=viz_df["Parcela"],
                        x=viz_df["pagado"],
                        name="Pagado",
                        orientation="h",
                        marker=dict(color="#15803D"),
                        text=viz_df["cumplimiento_pct"].map(lambda v: f"{v:.0f}%"),
                        textposition="inside",
                        textfont=dict(color="white", size=11),
                        hovertemplate="Parcela %{y}<br>Pagado CLP %{x:,.0f}<br>Cumplimiento %{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig_obl_comp.add_trace(
                    go.Bar(
                        y=viz_df["Parcela"],
                        x=viz_df["pendiente_plot"],
                        name="Pendiente GC",
                        orientation="h",
                        marker=dict(color="#DC2626"),
                        hovertemplate="Parcela %{y}<br>Pendiente CLP %{x:,.0f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                fig_obl_comp.add_trace(
                    go.Bar(
                        y=viz_df["Parcela"],
                        x=viz_df["favor_plot"],
                        name="Saldo a favor",
                        orientation="h",
                        marker=dict(color="#0EA5A4"),
                        customdata=viz_df["saldo_favor"],
                        hovertemplate="Parcela %{y}<br>Saldo a favor CLP %{customdata:,.0f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                fig_obl_comp.add_vline(
                    x=gc_total_parcela,
                    line_dash="dot",
                    line_color="#475569",
                    line_width=2,
                    row=1,
                    col=1,
                )
                fig_obl_comp.update_layout(
                    barmode="overlay",
                    height=920,
                    margin=dict(l=20, r=20, t=80, b=30),
                    paper_bgcolor="white",
                    plot_bgcolor="#F8FAFC",
                    hovermode="y unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
                    bargap=0.16,
                )
                fig_obl_comp.update_xaxes(
                    title_text="Monto (CLP)",
                    gridcolor="#CBD5E1",
                    zeroline=False,
                    tickformat=",.0f",
                    row=1,
                    col=1,
                )
                fig_obl_comp.update_xaxes(
                    title_text="Pendiente (+) / Saldo a favor (-)",
                    gridcolor="#CBD5E1",
                    zeroline=True,
                    zerolinecolor="#64748B",
                    tickformat=",.0f",
                    row=2,
                    col=1,
                )
                fig_obl_comp.update_yaxes(
                    title_text="Parcela",
                    categoryorder="array",
                    categoryarray=viz_df["Parcela"].tolist()[::-1],
                    tickmode="array",
                    tickvals=viz_df["Parcela"].tolist(),
                    ticktext=viz_df["Parcela"].tolist(),
                    tickfont=dict(size=12),
                    row=1,
                    col=1,
                )
                fig_obl_comp.update_yaxes(
                    title_text="Parcela",
                    categoryorder="array",
                    categoryarray=viz_df["Parcela"].tolist()[::-1],
                    tickmode="array",
                    tickvals=viz_df["Parcela"].tolist(),
                    ticktext=viz_df["Parcela"].tolist(),
                    tickfont=dict(size=12),
                    row=2,
                    col=1,
                )
                fig_obl_comp.add_annotation(
                    x=1,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    text=f"Obligación referencial por parcela: ${gc_total_parcela:,.0f}",
                    showarrow=False,
                    font=dict(size=12, color="#64748B"),
                    xanchor="right",
                )
                st.plotly_chart(fig_obl_comp, use_container_width=True)

            tabla_show = tabla_full.copy()
            tabla_show = tabla_show.rename(
                columns={
                    "parcela": "Parcela",
                    "pagado": "Pagado",
                    "gc_total": "GC total",
                    "pendiente": "Diferencia",
                    "pendiente_pos": "Pendiente",
                    "saldo_favor": "GC por anticipado",
                    "Mantención": "Pendiente mantención",
                }
            )
            # Agrega Propietario junto a Parcela
            cols_prop = list(df_prop.columns)
            col_parc_p = _pick_col(cols_prop, ["n_parcela", "numero_parcela", "parcela", "lote", "unidad", "sitio"])
            col_name = _pick_col(cols_prop, ["nombre", "propietario", "dueno", "dueño"])
            if col_parc_p and col_name:
                prop_map = df_prop.copy()
                prop_map["Parcela"] = pd.to_numeric(
                    prop_map[col_parc_p].astype(str).str.replace(r"[^\d]", "", regex=True),
                    errors="coerce",
                )
                prop_map = prop_map.dropna(subset=["Parcela"])
                prop_map = prop_map[["Parcela", col_name]].rename(columns={col_name: "Propietario"})
                tabla_show = tabla_show.merge(prop_map, on="Parcela", how="left")
            else:
                tabla_show["Propietario"] = ""
            if "pendiente" in tabla_show.columns:
                tabla_show = tabla_show.drop(columns=["pendiente"])
            if "Diferencia" in tabla_show.columns:
                tabla_show = tabla_show.drop(columns=["Diferencia"])
            # Formato para columnas adicionales de CC
            extra_cc_cols = [c for c in tabla_show.columns if c.startswith("Pendiente ")]
            # Total por pagar = Pendiente + Pendiente mantención + CCs
            total_cols = ["Pendiente", "Pendiente mantención"] + extra_cc_cols
            tabla_show["Total por pagar"] = tabla_show[total_cols].fillna(0).sum(axis=1)
            # Renombre pendiente
            tabla_show = tabla_show.rename(columns={"Pendiente": "Pendiente GC"})
            # Orden columnas: Parcela, Propietario, ...
            cols_front = ["Parcela", "Propietario"]
            cols_rest = [c for c in tabla_show.columns if c not in cols_front]
            tabla_show = tabla_show[cols_front + cols_rest]

            def _style_obl(s: pd.Series):
                if s.name == "Total por pagar":
                    return ["background-color: #5A2A2A; color: #FFFFFF; font-weight:700;" if v > 0 else "" for v in s]
                return [""] * len(s)

            def _style_row_total(row: pd.Series):
                if row.get("Total por pagar", 0) > 0:
                    return ["background-color: #F4DCDC;" for _ in row]
                return ["" for _ in row]

            def _style_total_col(s: pd.Series):
                if s.name == "Total por pagar":
                    return ["background-color: #5A2A2A; color: #FFFFFF; font-weight:700;" if v > 0 else "" for v in s]
                return [""] * len(s)

            styler = (
                tabla_show.style
                .format({col: "${:,.0f}" for col in ["Pagado", "GC total", "Pendiente GC", "GC por anticipado", "Pendiente mantención", "Total por pagar"] + extra_cc_cols})
                .apply(_style_obl)
                .apply(_style_row_total, axis=1)
                .apply(_style_total_col, axis=0)
                .set_properties(subset=["Parcela"], **{"text-align": "center"})
                .set_properties(subset=["Propietario"], **{"text-align": "left"})
                .set_properties(subset=[c for c in tabla_show.columns if c not in ("Parcela", "Propietario")], **{"text-align": "right"})
                .set_table_styles(
                    [
                        {"selector": "th", "props": "background:#0B1F2A;color:#F8FAFC;font-weight:700;font-size:12px;padding:6px 8px;"},
                        {"selector": "td", "props": "border-color:#E2E8F0;font-size:12px;padding:6px 8px;"},
                        {"selector": "tr:nth-child(even) td", "props": "background:#EAF0F5;"},
                        {"selector": "tr:hover td", "props": "background:#DCE7F0;"},
                        {"selector": "table", "props": "border-radius:10px;overflow:hidden;"},
                    ]
                )
            )
            st.dataframe(styler, use_container_width=True, height=800, hide_index=True)

            pie_gc = tabla_show[tabla_show["Pendiente GC"] > 0][["Parcela", "Pendiente GC"]].copy()
            pie_mant = tabla_show[tabla_show["Pendiente mantención"] > 0][["Parcela", "Pendiente mantención"]].copy()
            # Para proyecto, suma de columnas "Pendiente X"
            proj_cols = [c for c in tabla_show.columns if c.startswith("Pendiente ") and c not in ("Pendiente mantención", "Pendiente GC")]
            if proj_cols:
                pie_proj = tabla_show[["Parcela"] + proj_cols].copy()
                pie_proj["Pendiente proyecto"] = pie_proj[proj_cols].sum(axis=1)
                pie_proj = pie_proj[pie_proj["Pendiente proyecto"] > 0][["Parcela", "Pendiente proyecto"]]
            else:
                pie_proj = pd.DataFrame(columns=["Parcela", "Pendiente proyecto"])

            try:
                import plotly.express as px
            except Exception:
                st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    if not pie_gc.empty:
                        fig_gc = px.pie(
                            pie_gc,
                            names="Parcela",
                            values="Pendiente GC",
                            title="Distribución pendiente GC por parcela",
                            hole=0.35,
                            color_discrete_sequence=["#0B1F2A", "#1F4F5B", "#2C5B4A", "#3A6B5A", "#8DA2C8", "#A4463F"],
                        )
                        fig_gc.update_traces(
                            textinfo="label+percent",
                            textposition="auto",
                            texttemplate="%{label}<br>%{percent}",
                            automargin=True,
                            hovertemplate="Parcela %{label}<br>Participación %{percent}<extra></extra>",
                        )
                        fig_gc.update_layout(
                            height=520,
                            margin=dict(l=20, r=20, t=40, b=80),
                            legend_title_text="Parcela",
                        )
                        st.plotly_chart(fig_gc, use_container_width=True)
                    else:
                        st.info("Sin pendiente GC.")
                with c2:
                    if not pie_mant.empty:
                        fig_m = px.pie(
                            pie_mant,
                            names="Parcela",
                            values="Pendiente mantención",
                            title="Distribución pendiente mantención",
                            hole=0.35,
                            color_discrete_sequence=["#2C5B4A", "#3A6B5A", "#8DA2C8", "#0B1F2A", "#1F4F5B", "#A4463F"],
                        )
                        fig_m.update_traces(
                            textinfo="label+percent",
                            textposition="auto",
                            texttemplate="%{label}<br>%{percent}",
                            automargin=True,
                            hovertemplate="Parcela %{label}<br>Participación %{percent}<extra></extra>",
                        )
                        fig_m.update_layout(
                            height=520,
                            margin=dict(l=20, r=20, t=40, b=80),
                            legend_title_text="Parcela",
                        )
                        st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.info("Sin pendiente mantención.")
                with c3:
                    if not pie_proj.empty:
                        fig_p = px.pie(
                            pie_proj,
                            names="Parcela",
                            values="Pendiente proyecto",
                            title="Distribución pendiente proyecto",
                            hole=0.35,
                            color_discrete_sequence=["#A4463F", "#8DA2C8", "#3A6B5A", "#2C5B4A", "#1F4F5B", "#0B1F2A"],
                        )
                        fig_p.update_traces(
                            textinfo="label+percent",
                            textposition="auto",
                            texttemplate="%{label}<br>%{percent}",
                            automargin=True,
                            hovertemplate="Parcela %{label}<br>Participación %{percent}<extra></extra>",
                        )
                        fig_p.update_layout(
                            height=520,
                            margin=dict(l=20, r=20, t=40, b=80),
                            legend_title_text="Parcela",
                        )
                        st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        st.info("Sin pendiente proyecto.")

            st.subheader("Detalle de GC pendientes por parcela")
            tabla_prop = tabla_show[["Parcela", "Propietario", "Total por pagar"]].copy()
            tabla_prop = tabla_prop.rename(columns={"Total por pagar": "Pendiente"})

            total_pend_global = float(tabla_prop["Pendiente"].sum()) if not tabla_prop.empty else 0.0
            if total_pend_global > 0:
                tabla_prop["pct_pendiente"] = (tabla_prop["Pendiente"] / total_pend_global) * 100
            else:
                tabla_prop["pct_pendiente"] = 0.0
            # Último depósito por parcela
            cols_ing_det = list(df_ing_o.columns)
            col_fecha_det = _pick_col(cols_ing_det, ["fecha"])
            col_parc_det = _pick_col(cols_ing_det, ["parcela"])
            if col_fecha_det and col_parc_det:
                ult = df_ing_o.copy()
                ult["Parcela"] = pd.to_numeric(
                    ult[col_parc_det].astype(str).str.replace(r"[^\d]", "", regex=True),
                    errors="coerce",
                )
                ult["Fecha"] = pd.to_datetime(ult[col_fecha_det], dayfirst=True, errors="coerce")
                ult = ult.dropna(subset=["Parcela", "Fecha"])
                ult = ult.groupby("Parcela", as_index=False)["Fecha"].max()
                tabla_prop = tabla_prop.merge(ult, on="Parcela", how="left")
            else:
                tabla_prop["Fecha"] = pd.NaT
            tabla_prop = tabla_prop.rename(
                columns={
                    "pct_pendiente": "% Pendiente",
                }
            )
            tabla_prop = tabla_prop[["Parcela", "Propietario", "Pendiente", "% Pendiente", "Fecha"]]
            tabla_prop = tabla_prop.rename(columns={"Fecha": "Último pago"})

            left_col, right_col = st.columns([1.2, 1])
            with left_col:
                styler_prop = (
                    tabla_prop.style
                    .format({"Pendiente": "${:,.0f}", "% Pendiente": "{:.1f}%", "Último pago": "{:%d-%m-%Y}"})
                    .apply(
                        lambda s: ["background-color: #4A1B1B; color:#F8FAFC;" if v > 0 else "" for v in s]
                        if s.name in ("Pendiente", "% Pendiente")
                        else [""] * len(s)
                    )
                    .set_table_styles(
                        [
                            {"selector": "th", "props": "background:#0B1F2A;color:#F8FAFC;font-weight:600;"},
                            {"selector": "th", "props": "padding:5px 8px;"},
                            {"selector": "td", "props": "padding:5px 8px;"},
                            {"selector": "tr:nth-child(even) td", "props": "background:#F4F7FA;"},
                        ]
                    )
                )
                st.dataframe(styler_prop, use_container_width=True, height=800, hide_index=True)

            with right_col:
                pie_df = tabla_prop[tabla_prop["% Pendiente"] > 0].copy()
                if not pie_df.empty:
                    try:
                        import plotly.express as px
                    except Exception:
                        st.error("Falta Plotly para el gráfico avanzado. Instala con: pip install plotly")
                    else:
                        pie_df["Etiqueta"] = pie_df["Parcela"].astype(int).astype(str)
                        fig_pie = px.pie(
                            pie_df,
                            names="Etiqueta",
                            values="% Pendiente",
                            title="Distribución % deuda por parcela",
                            hole=0.35,
                            color_discrete_sequence=["#0B1F2A", "#1F4F5B", "#2C5B4A", "#3A6B5A", "#8DA2C8", "#A4463F"],
                        )
                        fig_pie.update_layout(legend_title_text="Parcela")
                        fig_pie.update_traces(
                            textinfo="label+percent",
                            textposition="auto",
                            texttemplate="%{label}<br>%{percent}",
                            automargin=True,
                            hovertemplate="Parcela %{label}<br>Participación %{percent}<extra></extra>",
                        )
                        fig_pie.update_layout(
                            height=520,
                            margin=dict(l=20, r=20, t=40, b=80),
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No hay pendientes positivos para graficar.")

            st.subheader("Detalle de abonos por parcela")
            cols_ing_det = list(df_ing_o.columns)
            col_fecha_det = _pick_col(cols_ing_det, ["fecha"])
            col_parc_det = _pick_col(cols_ing_det, ["parcela"])
            col_abono_det = next((c for c in cols_ing_det if c.startswith("abono")), None)
            col_concepto_det = next((c for c in ["detalle", "concepto", "glosa", "descripcion", "tipo", "categoria", "cc", "ccc"] if c in cols_ing_det), None)

            if col_fecha_det and col_parc_det and col_abono_det:
                det = df_ing_o.copy()
                det["Parcela"] = pd.to_numeric(
                    det[col_parc_det].astype(str).str.replace(r"[^\d]", "", regex=True),
                    errors="coerce",
                )
                det["Fecha"] = pd.to_datetime(det[col_fecha_det], dayfirst=True, errors="coerce").dt.date
                det["Abono"] = _parse_monto_series(det[col_abono_det])
                cols_keep = ["Fecha", "Parcela", "Abono"]
                if col_concepto_det:
                    det["Concepto"] = det[col_concepto_det].astype(str)
                    cols_keep.append("Concepto")
                det = det[cols_keep].dropna(subset=["Parcela", "Abono"])
                det = det.sort_values(["Parcela", "Fecha"])

                parcelas_det = sorted(det["Parcela"].dropna().unique().tolist())
                sel_parcela = st.selectbox("Filtrar parcela", options=["(todas)"] + parcelas_det)
                if sel_parcela != "(todas)":
                    det = det[det["Parcela"] == sel_parcela]

                det_show = det.copy()
                det_show["Abono"] = det_show["Abono"].map(lambda x: f"${x:,.0f}")
                st.dataframe(det_show, use_container_width=True, height=520, hide_index=True)

                try:
                    pdf_bytes = _df_to_pdf_bytes(det_show, "Detalle de abonos por parcela")
                    st.download_button(
                        "Descargar PDF",
                        data=pdf_bytes,
                        file_name="detalle_abonos_parcela.pdf",
                        mime="application/pdf",
                    )
                except RuntimeError as e:
                    st.error(str(e))
            else:
                st.warning("No se pudo construir el detalle de abonos. Revisa columnas de Fecha/Parcela/Abono.")


if __name__ == "__main__":
    run_streamlit()
