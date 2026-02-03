"""
Condominio - Dashboard (General + Ingresos V2.3 + Costos + Obligaciones)
"""

from __future__ import annotations

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


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Condominio - Dashboard", layout="wide")
    st.title("Condominio - Dashboard")
    st.caption("Fuente: Google Sheets (CSV publicado)")

    @st.cache_data(show_spinner=False)
    def _load(url_value: str, cache_version: int, expected_cols: Optional[set[str]] = None) -> pd.DataFrame:
        return load_data(url_value, expected_cols)

    with st.sidebar:
        st.header("Actualización")
        if st.button("Actualizar información"):
            st.cache_data.clear()

    tab_general, tab_ing, tab_cost, tab_obl = st.tabs(["General", "Ingresos V2.3", "Costos", "Obligaciones"])

    with tab_general:
        st.subheader("Ingresos, Costos y Neto (mensual)")

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

        st.markdown(
            """
            <style>
            .kpi-grid-3 {display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:8px 0 18px 0;}
            .kpi-card {background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:18px 20px;box-shadow:0 2px 12px rgba(15,23,42,0.06);position:relative;}
            .kpi-card:before {content:"";position:absolute;left:0;top:0;height:100%;width:6px;border-radius:16px 0 0 16px;}
            .kpi-title {font-size:12px;letter-spacing:0.08em;color:#6B7280;font-weight:700;}
            .kpi-value {font-size:24px;font-weight:800;margin-top:6px;}
            .kpi-sub {font-size:12px;color:#94A3B8;margin-top:6px;}
            .kpi-green:before {background:#22C55E;}
            .kpi-red:before {background:#EF4444;}
            .kpi-navy:before {background:#0B1F2A;}
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
              <div class="kpi-card kpi-green">
                <div class="kpi-title">TOTAL INGRESOS</div>
                <div class="kpi-value">${total_ing:,.0f}</div>
                <div class="kpi-sub">Suma histórica</div>
              </div>
              <div class="kpi-card kpi-red">
                <div class="kpi-title">TOTAL COSTOS</div>
                <div class="kpi-value">${total_cost:,.0f}</div>
                <div class="kpi-sub">Suma histórica</div>
              </div>
              <div class="kpi-card kpi-teal">
                <div class="kpi-title">NETO ACUMULADO</div>
                <div class="kpi-value">${total_neto:,.0f}</div>
                <div class="kpi-sub">Ingresos - Costos</div>
              </div>
              <div class="kpi-card kpi-navy">
                <div class="kpi-title">MEJOR AÑO</div>
                <div class="kpi-value">{best_year}</div>
                <div class="kpi-sub">Mayor neto</div>
              </div>
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
        st.dataframe(df_y_show, use_container_width=True, height=360, hide_index=True)

    with tab_ing:
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

        st.markdown(
            """
            <style>
            .kpi-grid-3 {display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:8px 0 18px 0;}
            .kpi-card {background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:18px 20px;box-shadow:0 2px 12px rgba(15,23,42,0.06);position:relative;}
            .kpi-card:before {content:"";position:absolute;left:0;top:0;height:100%;width:6px;border-radius:16px 0 0 16px;}
            .kpi-title {font-size:12px;letter-spacing:0.08em;color:#6B7280;font-weight:700;}
            .kpi-value {font-size:24px;font-weight:800;margin-top:6px;}
            .kpi-sub {font-size:12px;color:#94A3B8;margin-top:6px;}
            .kpi-green:before {background:#22C55E;}
            .kpi-navy:before {background:#0B1F2A;}
            .kpi-teal:before {background:#2C5B4A;}
            .kpi-red:before {background:#EF4444;}
            @media (max-width: 1100px){.kpi-grid-3{grid-template-columns:repeat(2,1fr);}}
            @media (max-width: 700px){.kpi-grid-3{grid-template-columns:1fr;}}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-grid-3">
              <div class="kpi-card kpi-green">
                <div class="kpi-title">TOTAL INGRESOS</div>
                <div class="kpi-value">${total_ing:,.0f}</div>
                <div class="kpi-sub">Suma filtrada</div>
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

    with tab_cost:
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
                    title="Pareto por categoría (Top 12)",
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
                    .sort_values("monto_norm", ascending=False)
                    .head(12)
                )
                fig_v = px.bar(
                    prov,
                    x=col_prov,
                    y="monto_norm",
                    title="Costos por proveedor (top 12)",
                    labels={col_prov: "Proveedor", "monto_norm": "Costo (CLP)"},
                    color_discrete_sequence=muted_palette,
                    text="monto_norm",
                )
                fig_v.update_traces(texttemplate="$%{text:,.0f}", textposition="inside", textfont_color="white")
                fig_v.update_layout(hovermode="x unified", height=420)
                st.plotly_chart(fig_v, use_container_width=True)

        st.subheader("Detalle de costos (filtrado)")
        st.dataframe(filt, use_container_width=True)

    with tab_obl:
        st.subheader("Obligaciones vs Pagos (por año y parcela)")
        with st.spinner("Cargando obligaciones y pagos..."):
            df_obl = _load(OBLIGACIONES_CSV_URL, CACHE_VERSION, {"ano", "anio", "año", "parcela", "gc"})
            df_ing_o = _load(INGRESOS_CSV_URL, CACHE_VERSION, {"fecha", "parcela", "abono"})
            df_prop = _load(PROPIETARIOS_CSV_URL, CACHE_VERSION, {"parcela", "propietario"})

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
                        fig_pie = px.pie(
                            oblig_anual,
                            names="anio",
                            values="gc_total",
                            title="Distribución GC por año",
                            hole=0.35,
                            color_discrete_sequence=["#0B1F2A", "#1F4F5B", "#2C5B4A", "#3A6B5A", "#8DA2C8", "#A4463F"],
                        )
                        fig_pie.update_traces(textinfo="percent+label")
                        fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_pie, use_container_width=True)

            tabla_full = tabla.copy()
            tabla_full["pendiente_pos"] = tabla_full["pendiente"].clip(lower=0)
            tabla_full["saldo_favor"] = (-tabla_full["pendiente"]).clip(lower=0)

            gc_total_parcela = float(tabla_full["gc_total"].max()) if not tabla_full.empty else 0.0
            total_pagado = float(tabla_full["pagado"].sum()) if not tabla_full.empty else 0.0
            total_pendiente = float(tabla_full["pendiente_pos"].sum()) if not tabla_full.empty else 0.0
            total_favor = float(tabla_full["saldo_favor"].sum()) if not tabla_full.empty else 0.0

            st.markdown(
                """
                <style>
                .kpi-grid {display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:8px 0 18px 0;}
                .kpi-card {background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:18px 20px;box-shadow:0 2px 12px rgba(15,23,42,0.06);position:relative;}
                .kpi-card:before {content:"";position:absolute;left:0;top:0;height:100%;width:6px;border-radius:16px 0 0 16px;}
                .kpi-title {font-size:12px;letter-spacing:0.08em;color:#6B7280;font-weight:700;}
                .kpi-value {font-size:28px;font-weight:800;margin-top:6px;}
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
                  <div class="kpi-card kpi-green">
                    <div class="kpi-title">TOTAL PAGADO</div>
                    <div class="kpi-value">${total_pagado:,.0f}</div>
                    <div class="kpi-sub">Ingresos reconocidos como GC</div>
                  </div>
                  <div class="kpi-card kpi-red">
                    <div class="kpi-title">PENDIENTE TOTAL</div>
                    <div class="kpi-value">${total_pendiente:,.0f}</div>
                    <div class="kpi-sub">Solo montos positivos</div>
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

            st.subheader("Obligación acumulada vs Pagos (por parcela)")
            tabla_show = tabla_full.copy()
            tabla_show = tabla_show.rename(
                columns={
                    "parcela": "Parcela",
                    "pagado": "Pagado",
                    "gc_total": "GC total",
                    "pendiente": "Diferencia",
                    "pendiente_pos": "Pendiente",
                    "saldo_favor": "Saldo a favor",
                }
            )

            def _style_obl(s: pd.Series):
                if s.name in ("Pendiente", "Diferencia"):
                    return ["background-color: #4A1B1B; color: #F8FAFC;" if v > 0 else "" for v in s]
                if s.name == "Saldo a favor":
                    return ["background-color: #0F3D2E; color: #F8FAFC;" if v > 0 else "" for v in s]
                return [""] * len(s)

            styler = (
                tabla_show.style
                .format({col: "${:,.0f}" for col in ["Pagado", "GC total", "Diferencia", "Pendiente", "Saldo a favor"]})
                .apply(_style_obl)
                .set_table_styles(
                    [
                        {"selector": "th", "props": "background:#0B1F2A;color:#F8FAFC;font-weight:600;"},
                        {"selector": "td", "props": "border-color:#E2E8F0;"},
                        {"selector": "tr:nth-child(even) td", "props": "background:#F4F7FA;"},
                    ]
                )
            )
            st.dataframe(styler, use_container_width=True, height=520, hide_index=True)

            st.subheader("Tabla para propietarios (solo pendientes positivos)")
            tabla_prop = tabla_full[["parcela", "pendiente_pos", "gc_total"]].copy()
            cols_prop = list(df_prop.columns)
            col_parc_p = _pick_col(cols_prop, ["n_parcela", "numero_parcela", "parcela", "lote", "unidad", "sitio"])
            col_name = _pick_col(cols_prop, ["nombre", "propietario", "dueno", "dueño"])
            if col_parc_p and col_name:
                prop_map = df_prop.copy()
                prop_map["parcela"] = pd.to_numeric(
                    prop_map[col_parc_p].astype(str).str.replace(r"[^\d]", "", regex=True),
                    errors="coerce",
                )
                prop_map = prop_map.dropna(subset=["parcela"])
                prop_map = prop_map[["parcela", col_name]].rename(columns={col_name: "Propietario"})
                tabla_prop = tabla_prop.merge(prop_map, on="parcela", how="left")
            else:
                tabla_prop["Propietario"] = ""

            total_pend_global = float(tabla_prop["pendiente_pos"].sum()) if not tabla_prop.empty else 0.0
            if total_pend_global > 0:
                tabla_prop["pct_pendiente"] = (tabla_prop["pendiente_pos"] / total_pend_global) * 100
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
                tabla_prop = tabla_prop.merge(ult, left_on="parcela", right_on="Parcela", how="left")
                tabla_prop = tabla_prop.drop(columns=["Parcela"])
            else:
                tabla_prop["Fecha"] = pd.NaT
            tabla_prop = tabla_prop.rename(
                columns={
                    "parcela": "Parcela",
                    "pendiente_pos": "Pendiente",
                    "pct_pendiente": "% Pendiente",
                }
            )
            tabla_prop = tabla_prop[["Parcela", "Propietario", "Pendiente", "% Pendiente", "Fecha"]]
            tabla_prop = tabla_prop.rename(columns={"Fecha": "Último pago"})
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
