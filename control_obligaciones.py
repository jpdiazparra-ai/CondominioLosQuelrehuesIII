"""Control de cobranza conciliado con las fuentes publicadas del condominio."""
from __future__ import annotations

import unicodedata
from html import escape

import pandas as pd


def _text(value):
    return " ".join(unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower().split())


def build_control(obligaciones, ingresos, propietarios, mantencion, proyectos, parse_money):
    """Una fila por parcela/concepto; los créditos no compensan otros conceptos."""
    alerts = []
    owners = propietarios.copy()
    parcel_col = next((c for c in ("n_parcela", "parcela") if c in owners), None)
    name_col = next((c for c in ("nombre", "propietario") if c in owners), None)
    if not parcel_col or not name_col:
        raise ValueError("La fuente de propietarios debe contener parcela y nombre.")
    owners["Parcela"] = pd.to_numeric(owners[parcel_col], errors="coerce")
    owners = owners[owners["Parcela"].isin(range(17, 37))]
    if owners["Parcela"].duplicated().any():
        alerts.append("Hay parcelas duplicadas en propietarios; se muestra el primer nombre publicado.")
    owner_map = owners.drop_duplicates("Parcela").set_index("Parcela")[name_col].to_dict()
    if len(owner_map) < 20:
        alerts.append(f"Faltan nombres de propietarios para {20-len(owner_map)} parcelas del universo 17–36.")

    annual = obligaciones.copy()
    year_col = next((c for c in ("ano", "anio") if c in annual), None)
    if not year_col or "gc" not in annual:
        raise ValueError("La fuente de obligaciones debe contener Año y GC.")
    annual["Año"] = pd.to_numeric(annual[year_col], errors="coerce")
    annual["Cargo por parcela"] = parse_money(annual["gc"])
    bad_annual = annual["Año"].notna() & annual["Cargo por parcela"].isna()
    if bad_annual.any():
        alerts.append(f"{int(bad_annual.sum())} años tienen GC sin monto válido; no se incluyeron en los cargos.")
    annual = annual.dropna(subset=["Año", "Cargo por parcela"])
    annual["Año"] = annual["Año"].astype(int)
    if annual["Año"].duplicated().any():
        raise ValueError("Hay años repetidos en obligaciones. Corrige la fuente para evitar duplicar cargos.")
    gc = float(annual["Cargo por parcela"].sum())
    charges = {(p, "Gastos comunes"): gc for p in range(17, 37)}
    project_map = {}
    for _, row in proyectos.iterrows():
        name = str(row["cc"]).strip()
        amount = parse_money(pd.Series([row["monto"]])).iloc[0]
        if pd.isna(amount):
            alerts.append(f"Proyecto {name}: monto no válido; requiere revisión.")
            continue
        normalized = _text(name)
        if normalized in {"gasto comun", "gastos comunes", "gc", "mantencion", "mantenimiento"}:
            alerts.append(f"El concepto {name} aparece también en proyectos; no se duplicó el cargo.")
            continue
        project_map[normalized] = name
        for p in range(17, 37):
            charges[p, name] = charges.get((p, name), 0) + float(amount)
    maintenance = mantencion.copy()
    if not maintenance.empty:
        maintenance["Parcela"] = pd.to_numeric(maintenance["parcela"], errors="coerce")
        maintenance["Cargo"] = parse_money(maintenance["monto"])
        missing = maintenance["Parcela"].notna() & maintenance["Cargo"].isna()
        if missing.any():
            alerts.append(f"{int(missing.sum())} trabajos de mantención tienen parcela pero no monto: no se consideran deuda hasta valorizarlos.")
        for p, amount in maintenance.dropna(subset=["Cargo"]).groupby("Parcela")["Cargo"].sum().items():
            if p not in range(17, 37):
                alerts.append(f"Mantención de parcela {p:g} fuera del universo 17–36; no incluida.")
            else:
                charges[int(p), "Mantención"] = float(amount)

    payments = ingresos.copy()
    payments["Parcela"] = pd.to_numeric(payments["parcela"], errors="coerce")
    payments["Abono"] = parse_money(payments["abono"])
    payments["Fecha"] = pd.to_datetime(payments["fecha"], dayfirst=True, errors="coerce")
    category = payments["cc"].map(_text)
    mapping = {"gasto comun": "Gastos comunes", "gastos comunes": "Gastos comunes", "gc": "Gastos comunes", "mantencion": "Mantención", "mantenimiento": "Mantención", **project_map}
    payments["Concepto"] = category.map(mapping)
    populated = payments[["parcela", "abono", "cc", "fecha"]].notna().any(axis=1)
    invalid = populated & (payments["Abono"].isna() | ~payments["Parcela"].isin(range(17, 37)) | payments["Concepto"].isna())
    review = payments.loc[invalid, ["fecha", "parcela", "cc", "abono"]].copy()
    if invalid.any():
        alerts.append(f"{int(invalid.sum())} abonos no se pudieron asignar por parcela, concepto o monto; revisa el detalle de conciliación.")
    valid = payments.loc[populated & ~invalid].copy()
    if valid["Fecha"].isna().any():
        alerts.append(f"{int(valid['Fecha'].isna().sum())} abonos asignados no tienen fecha válida; se incluyen en el saldo pero no en la evolución mensual.")
    totals = valid.groupby(["Parcela", "Concepto"])["Abono"].sum().to_dict()
    last = valid.groupby(["Parcela", "Concepto"])["Fecha"].max().to_dict()
    rows = []
    for p, concept in sorted(set(charges) | set(totals)):
        charge, paid = charges.get((p, concept), 0), totals.get((p, concept), 0)
        balance = charge - paid
        rows.append({"Parcela": int(p), "Propietario": str(owner_map.get(p, "Sin propietario")), "Concepto": concept, "Cargo": charge, "Abonado": paid, "Pendiente": max(balance, 0), "A favor": max(-balance, 0), "Estado": "Pendiente" if balance > 0 else "A favor" if balance < 0 else "Al día", "Último abono": last.get((p, concept), pd.NaT)})
    return pd.DataFrame(rows), valid, annual, alerts, review


def render_control(st, load, load_projects, urls, version, parse_money, prepared=None):
    from control_finance import analyze_portfolio, portfolio_table, financial_figures, cash_flow, cash_window, cash_figure, dashboard_headlines, money, percent
    from control_dashboard_pdf import build_dashboard_pdf

    st.markdown(FINANCE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="fin-heading"><div class="fin-heading-icon">▤</div><div><div class="fin-eyebrow">LOS QUELTEHUES III / CONTROL FINANCIERO</div><h1>Control de obligaciones</h1><p>Cartera de cobros, aplicación de abonos y exposición por parcela.</p></div><span class="fin-heading-tag">CLP · SALDOS CONCILIADOS</span></div>',unsafe_allow_html=True)
    if prepared is not None:
        ledger, payments, annual, alerts, review = prepared.ledger, prepared.payments, prepared.annual, prepared.alerts, prepared.review
    else:
        try:
            sources={key:load(urls[key],version,cols) for key,cols in [("obligaciones",{"ano","gc"}),("ingresos",{"fecha","parcela","abono"}),("propietarios",{"n_parcela","nombre"}),("mantencion",{"parcela","monto"})]}
            ledger,payments,annual,alerts,review=build_control(**sources,proyectos=load_projects(urls.get("proyectos",urls["obligaciones"]),version),parse_money=parse_money)
        except Exception as exc:
            st.error(f"No fue posible preparar el control financiero: {exc}")
            return
    def section(number,title,description):
        st.markdown(f'<div class="fin-section"><span>{number}</span><div><h2>{escape(title)}</h2><p>{escape(description)}</p></div></div>',unsafe_allow_html=True)
    def chart_header(title,subtitle):
        st.markdown(f'<div class="fin-chart-heading">{escape(title)}</div><div class="fin-chart-sub">{escape(subtitle)}</div>',unsafe_allow_html=True)
    config={"displayModeBar":True,"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"],"toImageButtonOptions":{"format":"png","scale":3}}
    latest=payments.Fecha.max()
    date_text=latest.strftime('%d/%m/%Y') if pd.notna(latest) else 'Sin fecha'
    st.caption(f'Base de cobranza: parcelas 17–36 · Último abono del libro Ingresos: {date_text}. Los cargos corresponden al total publicado.')
    with st.container(border=True):
        a,b,c=st.columns([2,1.35,1.35])
        query=a.text_input('Buscar parcela o propietario',placeholder='Parcela, nombre o apellido',key='control_search')
        concept=b.selectbox('Concepto',['Todos']+sorted(ledger.Concepto.unique()),key='control_concept')
        state=c.selectbox('Estado de la obligación',['Todos','Pendiente','Al día','A favor'],key='control_status')
    filtered=ledger.copy()
    if query.strip():
        q=_text(query)
        filtered=filtered[filtered.Propietario.map(_text).str.contains(q,regex=False)|filtered.Parcela.astype(str).eq(q)]
    if concept!='Todos':filtered=filtered[filtered.Concepto.eq(concept)]
    if state!='Todos':filtered=filtered[filtered.Estado.eq(state)]
    m=analyze_portfolio(filtered,payments)
    headlines=dashboard_headlines(prepared,m)
    global_money=lambda value:money(value) if value is not None else '—'
    a,b=st.columns([1.2,1.8],gap='large')
    try:
        selected_order=st.session_state.get('finance_order','Mayor pendiente')
        selected_cash=st.session_state.get('finance_cash_window','Últimos 12 meses de caja')
        selected_trend=st.session_state.get('finance_window','Últimos 12 meses registrados')
        cash_length=12 if selected_cash.startswith('Últimos 12') else 24 if selected_cash.startswith('Últimos 24') else 0
        trend_length=12 if selected_trend.startswith('Últimos 12') else 24 if selected_trend.startswith('Últimos 24') else 0
        pdf=build_dashboard_pdf(filtered,payments,prepared,{'Búsqueda':query.strip() or 'Todas las parcelas','Concepto':concept,'Estado':state},order=selected_order,cash_length=cash_length,trend_length=trend_length) if len(filtered) else b''
        a.download_button('Descargar informe para compartir · PDF',pdf,'informe_obligaciones_condominio.pdf','application/pdf',key='control_pdf_download',type='primary',icon=':material/picture_as_pdf:',width='stretch',disabled=filtered.empty)
        b.caption('Dashboard hasta conciliación y abonos · KPI, gráficos, panorama completo y flujo de caja global. Respeta filtros, orden y ventanas temporales.')
    except (ImportError,ValueError,RuntimeError) as exc:st.error(f'No fue posible preparar el informe: {exc}')
    coverage=percent(m['coverage']) if m['base'] else '—'
    hero_items=[
        ('Caja al día de hoy',global_money(headlines['cash_today']),f'Global · movimientos Pagado al {headlines["as_of"]:%d/%m/%Y}'),
        ('Parcelas con pendientes',f'{m["debtors"]} / {m["count"]}',f'{percent(m["debtor_share"])} de la selección'),
        ('Cuentas por cobrar',money(m['due']),'Cartera seleccionada · créditos separados'),
        ('Caja tras pagar proveedores',global_money(headlines['cash_after_payables']),'Caja global menos Costos No pagado'),
    ]
    st.markdown('<div class="fin-hero">'+''.join(f'<div class="fin-hero-item"><div class="fin-hero-label">{escape(label)}</div><div class="fin-hero-value">{escape(value)}</div><p>{escape(note)}</p></div>' for label,value,note in hero_items)+'</div>',unsafe_allow_html=True)
    if headlines['undated_count']:st.caption(f'La caja incluye {headlines["undated_count"]} movimientos reconocidos sin fecha, identificados en las observaciones de origen.')
    if headlines['future_count']:st.caption(f'Caja de hoy: se excluyen {headlines["future_count"]} movimientos con fecha posterior al {headlines["as_of"]:%d/%m/%Y}.')
    kpis=[
        ('Cargos netos publicados',money(m['charges']),'Incluyen ajustes netos de mantención','#7457ce','CN'),
        ('Abonos del libro de caja',money(m['paid']),'Asignados a su parcela y concepto','#16957f','AB'),
        ('Abonos aplicados a cargos',money(m['applied']),'Se limita cada abono al cargo positivo','#3279d5','AP'),
        ('Créditos a favor',money(m['credit']),'Excedentes separados de la cobranza','#5285bd','CR'),
        ('Cobertura ponderada',coverage,'Aplicado / cargos positivos de la selección','#16957f','%'),
        ('Cargos por cobrar',percent(m['uncovered']) if m['base'] else '—','Pendiente / cargos positivos de la selección','#d9803f','EX'),
        ('Concentración: 3 mayores',percent(m['top3']),'Participación en el pendiente seleccionado','#7457ce','C3'),
        ('Cuentas por pagar a proveedores',global_money(headlines['payables']),'Global · Costos con Estado No pagado','#8b5ba9','CP'),
    ]
    cards=''.join(f'<div class="fin-kpi" style="--tone:{tone}"><div class="fin-kpi-top"><span>{escape(title)}</span><i>{icon}</i></div><strong>{escape(value)}</strong><p>{escape(note)}</p></div>' for title,value,note,tone,icon in kpis)
    st.markdown('<div class="fin-kpi-grid">'+cards+'</div>',unsafe_allow_html=True)
    st.caption('Caja y proveedores tienen alcance global. Parcelas, cuentas por cobrar y demás indicadores responden a los filtros de cartera. La caja tras pagar proveedores no incorpora cobros futuros; los créditos permanecen separados.')
    if filtered.empty:
        st.info('No hay obligaciones para los filtros seleccionados.')
        return
    figures=financial_figures(m)
    section('01','Exposición y concentración de cartera','Dónde se concentra el saldo pendiente y qué proporción del cargo está cubierta.')
    a,b=st.columns([1.35,1],gap='large')
    with a,st.container(border=True):
        chart_header('Concentración del pendiente por parcela','Barras: saldo por cobrar · Línea: participación acumulada')
        if m['debtors']:
            st.plotly_chart(figures['pareto'],width='stretch',config=config,key='finance_pareto')
            top_count=min(3,m['debtors'])
            st.markdown(f'<div class="fin-chart-insight"><b>{top_count} parcelas concentran {percent(m["top3"])} de la deuda.</b> Las {min(5,m["debtors"])} mayores representan {percent(m["top5"])}.</div>',unsafe_allow_html=True)
        else:st.success('No hay saldos pendientes en esta selección.')
    with b,st.container(border=True):
        chart_header('Cobertura financiera por concepto','Abonos aplicables / cargos positivos · escala común de 0 a 100%')
        if m['base']:st.plotly_chart(figures['coverage'],width='stretch',config=config,key='finance_coverage')
        else:st.info('La selección no tiene cargos positivos para calcular cobertura.')
        concepts=m['concepts']
        rows=''.join(f'<tr><td><i style="background:{color}"></i>{escape(r.Concepto)}</td><td>{money(r.Pendiente)}</td><td>{percent(r.Cobertura) if r["Base de cobertura"] else "—"}</td></tr>' for (_,r),color in zip(concepts.iterrows(),['#7457ce','#df9635','#3279d5']*max(1,len(concepts))))
        st.markdown('<table class="fin-concept-table"><thead><tr><th>Concepto</th><th>Por cobrar</th><th>Cobertura</th></tr></thead><tbody>'+rows+'</tbody></table>',unsafe_allow_html=True)
    section('02','Panorama financiero por parcela','Cuenta consolidada, detalle del pendiente por concepto e indicadores de seguimiento.')
    a,b=st.columns([1,2])
    order=a.selectbox('Orden de la cartera',['Mayor pendiente','Número de parcela','Menor cobertura','Mayor saldo a favor'],key='finance_order')
    positions=m['positions']
    if order=='Número de parcela':positions=positions.sort_values('Parcela')
    elif order=='Menor cobertura':positions=positions.sort_values(['Cobertura','Pendiente'],ascending=[True,False])
    elif order=='Mayor saldo a favor':positions=positions.sort_values(['A favor','Parcela'],ascending=[False,True])
    else:positions=positions.sort_values(['Pendiente','Parcela'],ascending=[False,True])
    b.caption(f'{len(positions)} parcelas · {m["debtors"]} con pendientes · {m["current"]} sin pendientes en los conceptos seleccionados. La tabla permite desplazamiento horizontal.')
    st.markdown(portfolio_table(positions,m),unsafe_allow_html=True)
    export_cols=['Parcela','Propietario','Cargo','Abonado','Gastos comunes','Proyecto','Mantención','Pendiente','A favor','Cobertura','Participación','Último ingreso','Situación']
    export_positions=positions[export_cols].copy()
    if prepared is not None:
        export_positions['Corte de datos']=prepared.revision
        export_positions['Consulta']=prepared.captured_at
    st.download_button('Descargar panorama financiero · CSV',export_positions.to_csv(index=False).encode('utf-8-sig'),'panorama_financiero_parcelas.csv','text/csv',key='finance_summary_download')
    section('03','Flujo de caja del condominio','Pagos realizados, compromisos No pagado y efecto sobre la caja acumulada.')
    if prepared is not None:
        flow=cash_flow(prepared)
        with st.container(border=True):
            left,right=st.columns([2,1])
            left.markdown('<div class="fin-chart-heading">Caja operativa · alcance global</div><div class="fin-chart-sub">Alcance global. Caja pagada y escenario separado al atender Costos No pagado; no aplica filtros de cobranza superiores.</div>',unsafe_allow_html=True)
            flow_choice=right.selectbox('Ventana de flujo de caja',['Últimos 12 meses de caja','Últimos 24 meses de caja','Todo el histórico de caja'],key='finance_cash_window')
            length=12 if flow_choice.startswith('Últimos 12') else 24 if flow_choice.startswith('Últimos 24') else 0
            window=cash_window(flow,length)
            cash_cards=[('Caja de apertura',window['opening']),('Ingresos de la ventana',window['income']),('Egresos pagados',window['expenses']),('Proveedores por pagar',window['pending']),('Caja pagada al cierre',window['closing']),('Caja al pagar pendientes',window['adjusted_closing'])]
            st.markdown('<div class="fin-cash-kpis">'+''.join(f'<div><span>{label}</span><strong>{money(value)}</strong></div>' for label,value in cash_cards)+'</div>',unsafe_allow_html=True)
            if len(window['data']):
                st.plotly_chart(cash_figure(window),width='stretch',config=config,key='finance_cashflow')
                st.markdown(f'<div class="fin-chart-insight"><b>{window["data"].Mes.iloc[0]} a {window["data"].Mes.iloc[-1]}:</b> flujo pagado {money(window["net"])} · pendientes registrados en la ventana {money(window["pending_window"])} · reintegros {money(window["refunds"])} · cobertura de egresos {percent(window["coverage"]) if window["coverage"] is not None else "—"}.</div>',unsafe_allow_html=True)
            else:st.info('Sin movimientos de caja fechados para graficar.')
            st.caption('Caja pagada: apertura + ingresos + reintegros − egresos con Estado Pagado. Caja al pagar pendientes: caja pagada − Costos No pagado acumulados. La barra y línea moradas son un escenario según la fecha registrada; No pagado no acredita un pago. No se suma el auxiliar Deudas proveedores.')
            if flow.get('pending_undated',0):st.warning(f'Pendientes sin fecha: {money(flow["pending_undated"])}. Se muestran en el último punto del escenario sin asignarles una fecha de pago real.')
            if flow['undated_count']:st.warning(f'{flow["undated_count"]} movimientos con monto sin fecha están fuera del gráfico. Efecto neto: {money(flow["undated_net"])}.')
    else:st.info('El flujo de caja requiere los libros globales Ingresos y Costos de la publicación activa.')
    section('04','Conciliación y comportamiento de abonos','Relación entre cargos y saldos, y ritmo de ingresos del libro de caja.')
    a,b=st.columns([1,1.25],gap='large')
    with a,st.container(border=True):
        chart_header('Puente de conciliación de la cartera','Cargos netos − abonos + créditos separados = pendiente por cobrar')
        st.plotly_chart(figures['bridge'],width='stretch',config=config,key='finance_bridge')
        st.markdown(f'<div class="fin-chart-insight"><b>Control de saldo:</b> {money(m["charges"])} − {money(m["paid"])} + {money(m["credit"])} = <b>{money(m["due"])}</b></div>',unsafe_allow_html=True)
    with b,st.container(border=True):
        chart_header('Histórico mensual de abonos de caja','Barras: abonos recibidos · Línea: promedio móvil de tres meses')
        window=st.selectbox('Ventana del gráfico',['Últimos 12 meses registrados','Últimos 24 meses registrados','Todo el histórico'],key='finance_window',label_visibility='collapsed')
        length=12 if window.startswith('Últimos 12') else 24 if window.startswith('Últimos 24') else 0
        if len(m['monthly']):
            trend=figures['trend'] if length==12 else financial_figures(m,length)['trend']
            st.plotly_chart(trend,width='stretch',config=config,key='finance_trend')
            series=m['monthly'].tail(length) if length else m['monthly']
            st.markdown(f'<div class="fin-chart-insight"><b>{series.iloc[0].Mes} a {series.iloc[-1].Mes}:</b> {money(series.Abonos.sum())} recibidos. Se incluyen meses sin abonos con valor cero.</div>',unsafe_allow_html=True)
        else:st.info('No hay abonos fechados para la selección.')
        st.caption('La ventana modifica solo este gráfico. Los KPI y la cartera conservan el total seleccionado. El histórico no reconstruye deuda ni vencimientos.')
    section('05','Detalle técnico y estado de cuenta','Revisa cada obligación y sus referencias antes de consultar la cuenta completa de una parcela.')
    with st.expander('Detalle de cargos, abonos y saldos por obligación'):
        st.dataframe(filtered.sort_values(['Pendiente','Parcela'],ascending=[False,True]),hide_index=True,width='stretch',column_config={**{c:st.column_config.NumberColumn(c+' · CLP',format='dollar') for c in ['Cargo','Abonado','Pendiente','A favor']},'Último abono':st.column_config.DateColumn(format='DD/MM/YYYY')})
        st.download_button('Descargar cartera filtrada · CSV',filtered.to_csv(index=False).encode('utf-8-sig'),'control_obligaciones.csv','text/csv',key='control_download')
    parcels=positions.Parcela.tolist()
    parcel=st.selectbox('Seleccionar parcela',parcels,format_func=lambda p:f'Parcela {p} · {ledger.loc[ledger.Parcela.eq(p),"Propietario"].iloc[0]}',key='control_parcel')
    account=ledger[ledger.Parcela.eq(parcel)]
    account_metrics=analyze_portfolio(account,payments)
    st.markdown(f'<div class="fin-account"><div><div class="fin-eyebrow">CUENTA COMPLETA · PARCELA {parcel}</div><h3>{escape(str(account.iloc[0].Propietario))}</h3><p>Todos los conceptos de la parcela, independientemente de los filtros superiores.</p></div><div><span>Por cobrar</span><strong>{money(account_metrics["due"])}</strong></div><div><span>Crédito a favor</span><strong>{money(account_metrics["credit"])}</strong></div><div><span>Cobertura</span><strong>{percent(account_metrics["coverage"]) if account_metrics["base"] else "—"}</strong></div></div>',unsafe_allow_html=True)
    detail_tab,income_tab,adjustments_tab=st.tabs(['Obligaciones y referencias','Abonos del libro de caja','Mantención y ajustes'])
    with detail_tab:
        st.dataframe(account.drop(columns=['Parcela','Propietario']),hide_index=True,width='stretch',column_config={**{c:st.column_config.NumberColumn(c+' · CLP',format='dollar') for c in ['Cargo','Abonado','Pendiente','A favor']},'Último abono':st.column_config.DateColumn(format='DD/MM/YYYY')})
    with income_tab:
        history_cols=[c for c in ['Fecha','Concepto','Abono','medio','id','fila_origen','fila_publicada','Fuente'] if c in payments]
        history=payments[payments.Parcela.eq(parcel)].sort_values('Fecha',ascending=False)[history_cols].rename(columns={'medio':'Medio','id':'ID de origen','fila_origen':'Fila histórica','fila_publicada':'Fila publicada'})
        st.dataframe(history,hide_index=True,width='stretch',column_config={'Fecha':st.column_config.DateColumn(format='DD/MM/YYYY'),'Abono':st.column_config.NumberColumn(format='dollar')})
        st.download_button('Descargar abonos de la parcela',history.to_csv(index=False).encode('utf-8-sig'),f'abonos_parcela_{parcel}.csv','text/csv',key='control_history_download')
    with adjustments_tab:
        if prepared is not None:
            maintenance=prepared.frames['mantencion']
            maintenance=maintenance[maintenance.parcela_num.eq(parcel)].copy()
            if len(maintenance):
                maintenance['Monto firmado']=parse_money(maintenance.monto)
                cols=[c for c in ['id','fecha','obs','tipo_de_movimiento','Monto firmado','fila_origen','fila_publicada'] if c in maintenance]
                st.dataframe(maintenance[cols],hide_index=True,width='stretch',column_config={'Monto firmado':st.column_config.NumberColumn(format='dollar')})
                st.caption('Positivo = cargo; negativo = ajuste/abono del trabajo. Se aplica al cargo neto y no se agrega a ingresos de caja.')
            else:st.info('Esta parcela no tiene movimientos de mantención publicados.')
        else:st.info('El detalle de ajustes requiere la fuente de mantención por trabajo.')
    with st.expander(f'Metodología, conciliación y calidad · {len(alerts)} observaciones'):
        st.markdown('**Cobertura:** suma de abonos aplicables a cargos positivos / suma de cargos positivos. **Concentración:** deuda de las tres mayores parcelas / deuda seleccionada. **Saldo neto contable:** pendiente − créditos; no es el importe de cobranza. **Pendiente medio:** pendiente / parcelas con deuda.')
        st.caption('No se calcula antigüedad de deuda, mora ni DSO porque los datos no definen vencimientos ni imputación histórica por período. Mantención conserva los ajustes por parcela y trabajo. Los créditos no compensan deudas ajenas.')
        for alert in alerts:st.warning(alert)
        if not review.empty:st.dataframe(review,hide_index=True,width='stretch')
        st.dataframe(annual[['Año','Cargo por parcela']],hide_index=True,width='stretch')
        for key,label in [('obligaciones','Cobros GC'),('proyectos','Cámaras'),('ingresos','Ingresos'),('mantencion','Mantención')]:
            if key in urls:st.markdown(f'[{label} · fuente publicada]({urls[key]})')
    st.caption('Los registros se actualizan en la planilla de origen. Usa Actualizar data del URL para consultar un nuevo corte.')

FINANCE_CSS = '''<style>
.fin-heading{display:flex;align-items:center;gap:15px;margin:4px 0 16px}.fin-heading-icon{width:50px;height:50px;border-radius:14px;background:linear-gradient(135deg,#123e56,#13697a);display:grid;place-items:center;color:white;font-size:25px;flex-shrink:0}.fin-heading h1{color:#102e43;font-size:29px!important;font-weight:850;letter-spacing:-.8px;padding:4px 0!important;margin:0!important}.fin-heading p{font-size:13px;color:#75869a;margin:3px 0 0}.fin-eyebrow{font-size:10px;font-weight:800;letter-spacing:1.7px;color:#6c8797}.fin-heading-tag{margin-left:auto;font-size:10px;font-weight:750;color:#557083;background:#edf4f8;border:1px solid #dde7ef;border-radius:6px;padding:8px 11px;white-space:nowrap}
.fin-hero{background:linear-gradient(110deg,#102f45,#154e65 70%,#1b6470);border-radius:15px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:22px;padding:26px 28px;margin:20px 0 17px;box-shadow:0 12px 28px #0e3b5115;align-items:start}.fin-hero-item{min-width:0}.fin-hero-item+.fin-hero-item{border-left:1px solid #ffffff25;padding-left:22px}.fin-hero-label{color:#b7d1df;font-size:10px;line-height:1.5;text-transform:uppercase;font-weight:750;letter-spacing:1px;min-height:30px}.fin-hero-value{font-size:31px;letter-spacing:-1px;color:white;font-weight:850;line-height:1.25;margin:9px 0}.fin-hero p{color:#b9d3df;font-size:10px;line-height:1.6;margin:3px 0}
.fin-cash-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:10px 0 17px;padding:15px 18px;border:1px solid #e2eaf1;border-radius:10px;background:#f6f9fc}.fin-cash-kpis span{display:block;font-size:10px;color:#7990a3}.fin-cash-kpis strong{display:block;font-size:23px;color:#214960;font-weight:800;letter-spacing:-.5px;margin-top:8px}.fin-kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:15px 0 10px}.fin-kpi{padding:16px 18px 15px;background:#fff;border:1px solid #e0e8f0;border-radius:11px;box-shadow:0 4px 15px #15364e05;border-top:3px solid var(--tone)}.fin-kpi-top{display:flex;align-items:center;justify-content:space-between;gap:6px}.fin-kpi-top span{font-size:11px;font-weight:700;color:#657b90;line-height:1.5}.fin-kpi-top i{width:28px;height:25px;border:1px solid #e8edf2;border-radius:6px;display:grid;place-items:center;color:var(--tone);font-style:normal;font-size:9px;font-weight:850;flex-shrink:0}.fin-kpi strong{display:block;font-size:25px;color:#153149;letter-spacing:-.7px;line-height:1.3;margin:8px 0}.fin-kpi p{font-size:10px;color:#8190a1;line-height:1.5;margin:0}
.fin-section{display:flex;gap:12px;align-items:center;margin:30px 0 16px}.fin-section>span{display:grid;place-items:center;width:29px;height:29px;border:1px solid #dfe8f0;color:#678295;border-radius:7px;font-size:10px;font-weight:800;flex-shrink:0}.fin-section h2{font-size:18px!important;letter-spacing:-.4px;font-weight:800;color:#18354b;padding:0!important;margin:0!important;line-height:1.4}.fin-section p{font-size:11px;line-height:1.6;color:#8190a2;margin:3px 0 0}.fin-chart-heading{font-size:14px;font-weight:800;color:#1a374d;margin:3px 0 5px}.fin-chart-sub{font-size:10px;color:#8796a7;line-height:1.7;margin-bottom:12px}.fin-chart-insight{font-size:11px;line-height:1.7;background:#f4f8fb;border-radius:8px;border:1px solid #e8eef3;color:#71879a;padding:10px 13px;margin:5px 0}.fin-chart-insight b{font-weight:750;color:#2d5268}
.fin-concept-table{width:100%;border-collapse:collapse;margin-top:5px;font-size:11px}.fin-concept-table th{font-weight:700;color:#8797a8;text-align:right;font-size:9px;text-transform:uppercase;letter-spacing:.5px;padding:9px 6px;border-bottom:1px solid #e5edf3}.fin-concept-table td{padding:10px 6px;color:#35536b;text-align:right;border-bottom:1px solid #edf2f6;font-variant-numeric:tabular-nums}.fin-concept-table th:first-child,.fin-concept-table td:first-child{text-align:left}.fin-concept-table i{display:inline-block;width:6px;height:6px;border-radius:100%;margin-right:7px}
.fin-table-shell{border:1px solid #dce6ee;border-radius:12px;overflow:hidden;background:#fff;box-shadow:0 7px 24px #14324c06;margin:7px 0 13px}.fin-table-scroll{overflow:auto;max-height:710px;scrollbar-color:#c5d7e3 #f3f7fa;scrollbar-width:thin}.fin-table{border-collapse:separate;border-spacing:0;width:100%;min-width:1470px;font-size:11px;line-height:1.4;font-variant-numeric:tabular-nums}.fin-table th{white-space:nowrap;border-bottom:1px solid #dfe8f0;text-align:right;padding:10px 13px}.fin-table thead tr:first-child th{position:sticky;top:0;z-index:3;background:#edf4f8;color:#72899c;font-size:9px;letter-spacing:.8px;text-transform:uppercase;height:38px;box-sizing:border-box;font-weight:800;text-align:center}.fin-table thead tr:nth-child(2) th{position:sticky;top:38px;z-index:3;background:#f7fafc;color:#58748a;font-size:10px;height:38px;box-sizing:border-box;font-weight:750}.fin-table td{padding:12px 13px;border-bottom:1px solid #edf2f6;color:#506d83;background:#fff;white-space:nowrap;vertical-align:middle}.fin-table tr:nth-child(even) td{background:#fbfcfe}.fin-table tbody tr:hover td{background:#f0f7fb}.fin-table .fin-sticky{position:sticky;left:0;z-index:2;min-width:230px;box-shadow:2px 0 3px #193d5510;text-align:left}.fin-table thead .fin-sticky{z-index:5!important;text-align:left!important;vertical-align:middle}.fin-owner{display:flex;align-items:center;gap:10px;max-width:260px}.fin-owner b{color:#2b4a62;font-size:11px;font-weight:750;display:block;max-width:207px;overflow:hidden;text-overflow:ellipsis}.fin-owner small{font-size:9px;color:#8b9dac;display:block;margin-top:3px}.fin-unit{width:31px;height:31px;display:grid;place-items:center;background:#eff4f8;border:1px solid #e0e9f0;border-radius:7px;color:#668397;font-size:11px;font-weight:800;flex-shrink:0}.fin-number{text-align:right}.fin-table td.fin-due{font-weight:800;color:#c95759;background:#fff9f9}.fin-table td.fin-credit{font-weight:750;color:#377db9}.fin-table td.fin-muted{color:#bac6cf}.fin-date{font-size:10px;color:#8497a8!important;text-align:center}.fin-coverage{width:80px;margin-left:auto}.fin-coverage b{font-size:10px;font-weight:750;color:#397767;display:block;text-align:right}.fin-coverage>span{height:4px;display:block;overflow:hidden;background:#edf2f6;border-radius:5px;margin-top:6px}.fin-coverage i{height:100%;display:block;border-radius:5px;background:#4eaa94}.fin-status{display:inline-block;font-size:9px;font-weight:750;padding:5px 8px;border-radius:5px;line-height:1.4}.fin-status.red{background:#fceff0;color:#ba565e}.fin-status.blue{background:#eef4fc;color:#4c7cb0}.fin-status.green{background:#eaf5f0;color:#438574}.fin-table .fin-total td{border-top:2px solid #bad0de;background:#eaf2f7!important;color:#244f69!important;font-weight:800;font-size:10px;padding:15px 13px}.fin-table-footer{display:flex;justify-content:space-between;gap:18px;padding:11px 15px;color:#8a9dac;font-size:9px;background:#f9fbfd;border-top:1px solid #e3ecf3;line-height:1.7}
.fin-account{display:grid;grid-template-columns:2fr repeat(3,1fr);align-items:center;gap:22px;background:linear-gradient(110deg,#f0f7fa,#f8fbfd);border:1px solid #dce8ef;border-radius:12px;padding:22px 25px;margin:16px 0 20px}.fin-account h3{font-size:19px!important;color:#20465d;margin:7px 0!important;padding:0!important}.fin-account p{font-size:10px;color:#8095a6;line-height:1.6;margin:0;max-width:350px}.fin-account>div+div{border-left:1px solid #dce7ee;padding-left:20px}.fin-account>div>span{font-size:10px;color:#7c94a6}.fin-account strong{display:block;font-size:22px;font-weight:800;color:#244b63;margin-top:7px;letter-spacing:-.4px}
@media(max-width:1100px){.fin-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.fin-hero{grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.fin-hero-item:nth-child(3){border-left:0;padding-left:0}.fin-hero-item:nth-child(n+3){border-top:1px solid #ffffff25;padding-top:18px}.fin-account{grid-template-columns:repeat(3,1fr)}.fin-account>div:first-child{grid-column:1/-1}.fin-account>div:nth-child(2){border:0;padding:0}.fin-heading-tag{display:none}}
@media(max-width:650px){.fin-cash-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.fin-cash-kpis strong{font-size:20px}.fin-hero{grid-template-columns:1fr;padding:22px;gap:18px}.fin-hero-item+.fin-hero-item{border-left:0;border-top:1px solid #ffffff25;padding:16px 0 0}.fin-hero-label{min-height:0}.fin-hero-value{font-size:31px}.fin-kpi{padding:13px}.fin-kpi strong{font-size:22px}.fin-heading h1{font-size:25px!important}.fin-heading-icon{width:43px;height:43px}.fin-table .fin-sticky{min-width:190px}.fin-owner b{max-width:140px}.fin-table-footer{display:block}.fin-account{gap:12px;padding:18px}.fin-account strong{font-size:18px}.fin-account>div+div{padding-left:10px}.fin-section p{font-size:10px}}

.fin-table{--fin-owner-width:250px;min-width:1515px;table-layout:fixed}.fin-table th,.fin-table td{box-sizing:border-box}.fin-table th{border-right:1px solid #d1dde7}.fin-table td{padding-top:14px;padding-bottom:14px;border-bottom:1px solid #d6e2eb}.fin-table .fin-sticky{width:var(--fin-owner-width);min-width:var(--fin-owner-width);max-width:var(--fin-owner-width);border-right:1px solid #cbdce8}.fin-table .fin-due-col{position:sticky;left:var(--fin-owner-width);z-index:2;width:135px;min-width:135px;background:#fff1f2!important;color:#b23e49!important;border-right:2px solid #e6b4bb;font-size:15px;font-weight:850;letter-spacing:-.3px;text-align:right}.fin-table thead .fin-due-col{z-index:6!important;background:#b7474d!important;color:white!important;font-size:10px;letter-spacing:.5px;text-align:center!important;top:0}.fin-table td.fin-due-col.is-clear{color:#aab6c1!important;font-size:12px}.fin-table td:nth-child(5),.fin-table td:nth-child(8),.fin-table td:nth-child(11){border-right:1px solid #cbdce8}.fin-table .fin-total .fin-due-col{background:#f7dce1!important;color:#a93240!important;font-size:15px}.fin-owner{max-width:100%;min-width:0}.fin-owner>span:last-child{min-width:0;flex:1}.fin-owner b{font-size:12px;max-width:100%}.fin-unit{color:#254d68;background:#eaf1f6;font-size:12px}.fin-table tbody tr:hover td{background:#edf5fa}.fin-table tbody tr:hover .fin-due-col{background:#fbe0e5!important}.fin-table tbody tr:hover .fin-sticky{background:#edf5fa!important}
@media(max-width:650px){.fin-table{--fin-owner-width:195px}.fin-owner b{max-width:132px}.fin-table .fin-due-col{font-size:14px}}
</style>'''
