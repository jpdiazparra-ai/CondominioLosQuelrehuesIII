"""Presentación compartida, registro de fuentes y controles de conciliación."""
from html import escape
import pandas as pd
from condominio_data import URLS, SHEETS, PUBLICATION_URL, parse_money


def money(value):
    return '$'+f'{0 if abs(value)<.5 else value:,.0f}'.replace(',','.')


def header(st,title,subtitle,icon='▤'):
    st.markdown(f'<div class="data-page-head"><div class="data-icon">{icon}</div><div><div class="data-eyebrow">LOS QUELTEHUES III · ADMINISTRACIÓN</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div></div>',unsafe_allow_html=True)


def kpis(st,items):
    html='<div class="data-kpis">'
    for label,value,sub,tone in items:
        html+=f'<div class="data-card" style="--tone:{tone}"><div class="data-label">{escape(label)}</div><strong>{escape(str(value))}</strong><small>{escape(sub)}</small></div>'
    st.markdown(html+'</div>',unsafe_allow_html=True)


def render_source_bar(st,snapshot):
    st.markdown('''<style>
    .source-band{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:11px 15px;background:#f1f7fa;border:1px solid #dce8ef;border-radius:10px;color:#61758a;font-size:11px;margin:0 0 24px;flex-wrap:wrap}.source-band a{color:#087f91!important;font-weight:750;text-decoration:none}.source-status{color:#008a65;font-weight:800}.source-revision{font-family:monospace;font-size:10px}
    .data-page-head{display:flex;align-items:center;gap:16px;margin:8px 0 22px}.data-icon{width:52px;height:52px;display:grid;place-items:center;border-radius:15px;background:linear-gradient(135deg,#003a4b,#0d8587);color:white;font-size:27px;flex-shrink:0}.data-eyebrow{font-size:10px;letter-spacing:1.7px;font-weight:800;color:#63808c}.data-page-head h1{font-size:28px!important;letter-spacing:-.7px;font-weight:850;margin:0;padding:4px 0!important;color:#10263c}.data-page-head p{color:#718398;font-size:13px;margin:3px 0 0}.data-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:18px 0 25px}.data-card{padding:20px;background:white;border:1px solid #e1e9f0;border-left:3px solid var(--tone);border-radius:12px;box-shadow:0 6px 18px #112e4807;min-width:0}.data-label{color:#697c90;font-size:12px;font-weight:700}.data-card strong{display:block;font-size:25px;color:#10263c;letter-spacing:-.6px;margin:10px 0}.data-card small{font-size:11px;color:#73889a}.data-chart-head{font-size:15px;font-weight:800;color:#142b43;margin:3px 0 12px}.data-note{background:#f3f8fb;border:1px solid #e0eaf1;border-radius:10px;padding:15px 18px;color:#61768c;font-size:12px;line-height:1.65;margin:12px 0 18px}
    @media(max-width:1000px){.data-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.source-band{display:block;line-height:1.9}.data-card{padding:14px}.data-card strong{font-size:21px}}
    </style>''',unsafe_allow_html=True)
    count=int(snapshot.checks.Estado.eq('Coincide').sum())
    tone='#008a65' if count==len(snapshot.checks) else '#b77d22'
    st.markdown(f'<div class="source-band"><span><a href="{escape(PUBLICATION_URL,quote=True)}" target="_blank">↗ Publicación activa · 8 hojas</a></span><span>Corte de consulta: {snapshot.captured_at} · Santiago</span><span class="source-status" style="color:{tone}">{count}/{len(snapshot.checks)} conciliaciones coinciden</span><span class="source-revision">Corte {snapshot.revision}</span></div>',unsafe_allow_html=True)


def render_sources(st,snapshot):
    header(st,'Fuentes y control','Cada cifra vinculada a su hoja, su registro y el mismo corte de datos.','✓')
    matched=int(snapshot.checks.Estado.eq('Coincide').sum())
    kpis(st,[('Hojas conectadas','8','Una única publicación activa','#0f6bff'),('Conciliaciones correctas',f'{matched}/{len(snapshot.checks)}','Comparación con Origen y control','#008a65'),('Observaciones',str(len(snapshot.alerts)),'Importes, fechas y asignación','#d87917'),('Caja calculada',money(snapshot.totals['caja']),'Ingresos + costos firmados','#8a3ffc')])
    st.subheader('Conciliación de importes')
    st.caption('Las cifras publicadas son controles internos del mismo archivo; no constituyen una validación contra cartola bancaria.')
    st.dataframe(snapshot.checks,hide_index=True,width='stretch',column_config={c:st.column_config.NumberColumn(format='dollar') for c in ['Calculado','Publicado','Diferencia']})
    st.subheader('Registros que requieren revisión')
    for alert in snapshot.alerts:st.warning(alert)
    if not snapshot.review.empty:st.dataframe(snapshot.review,hide_index=True,width='stretch')
    st.subheader('Catálogo de fuentes')
    catalog=[]
    for key,(label,gid) in SHEETS.items():
        frame=snapshot.frames.get('tramos' if key=='obligaciones' else key)
        catalog.append({'Hoja':label,'GID':gid,'Registros operativos':len(frame) if frame is not None else None,'Abrir fuente':URLS[key]})
    st.dataframe(pd.DataFrame(catalog),hide_index=True,width='stretch',column_config={'Abrir fuente':st.column_config.LinkColumn('Fuente publicada',display_text='Abrir hoja')})
    st.subheader('Explorar registros con trazabilidad')
    options=['ingresos','costos','obligaciones','proyectos','mantencion','proveedores']
    key=st.selectbox('Hoja de origen',options,format_func=lambda k:SHEETS[k][0],key='trace_sheet')
    frame=snapshot.frames['tramos' if key=='obligaciones' else key].copy()
    query=st.text_input('Buscar ID o texto del registro',key='trace_query',placeholder='Ej.: ING-0958, MAN-008 o parcela')
    if query.strip():
        mask=frame.fillna('').astype(str).apply(lambda col:col.str.contains(query.strip(),case=False,regex=False)).any(axis=1)
        frame=frame[mask]
    st.dataframe(frame,hide_index=True,width='stretch')
    st.download_button('Descargar registros con referencias · CSV',frame.to_csv(index=False).encode('utf-8-sig'),f'trazabilidad_{key}_{snapshot.revision}.csv','text/csv',key='trace_download')
    st.markdown('<div class="data-note"><b>Reglas compartidas en todas las vistas</b><br>Caja pagada = ingresos recibidos + costos con Estado Pagado. Costos No pagado es la fuente operativa de proveedores pendientes; el auxiliar Deudas proveedores se concilia y no se suma. Cámaras se compara con su auxiliar y se cobra una sola vez. Mantención conserva cargos y ajustes con su signo. Proveedores se mantiene fuera de los costos pagados. GC suma los tramos por año sin inventar los meses exactos del cambio de tarifa.<br>Los propietarios se identifican con el último nombre registrado en Ingresos o Proyectos de esta publicación. “Fila publicada” corresponde a la hoja actual; “Fila origen” identifica el archivo histórico migrado.</div>',unsafe_allow_html=True)
    st.caption('El corte identifica el contenido de las ocho hojas. Actualizar data del URL vuelve a consultar todas las fuentes y recalcula las vistas y los reportes.')


def render_transactions(st,snapshot,kind):
    import plotly.express as px
    if kind=='costos':
        render_costs(st,snapshot)
        return
    income=kind=='ingresos'
    header(st,'Ingresos recibidos' if income else 'Costos y reintegros','Movimientos del libro de caja, con filtros reales y referencia al registro de origen.','↗' if income else '$')
    frame=snapshot.frames[kind].copy()
    frame['Monto']=parse_money(frame.abono if income else frame.monto)
    frame['Fecha']=pd.to_datetime(frame.fecha,dayfirst=True,errors='coerce')
    years=sorted(frame.Fecha.dt.year.dropna().astype(int).unique())
    a,b,c=st.columns([1,1.4,2])
    year=a.selectbox('Año',['Todos']+years+(['Sin fecha'] if frame.Fecha.isna().any() else []),key=kind+'_year')
    cat=b.selectbox('Clasificación' if income else 'Centro de costo',['Todas']+sorted(frame.cc.dropna().unique()),key=kind+'_category')
    search=c.text_input('Buscar parcela, propietario o ID' if income else 'Buscar proveedor, trabajo o ID',key=kind+'_search')
    if year=='Sin fecha':frame=frame[frame.Fecha.isna()]
    elif year!='Todos':frame=frame[frame.Fecha.dt.year.eq(year)]
    if cat!='Todas':frame=frame[frame.cc.eq(cat)]
    if search.strip():
        cols=['id','parcela','propietario'] if income else ['id','proveedor','detalle']
        mask=frame[cols].fillna('').astype(str).apply(lambda col:col.str.contains(search.strip(),case=False,regex=False)).any(axis=1)
        frame=frame[mask]
    valid=frame[frame.Monto.notna()]
    total=valid.Monto.sum()
    if income:
        latest=valid.Fecha.max()
        items=[('Ingresos recibidos',money(total),'Montos de la selección','#008a65'),('Movimientos con monto',str(len(valid)),'Incluye montos cero explícitos','#0f6bff'),('Aporte medio',money(total/len(valid) if len(valid) else 0),'Por movimiento con monto','#8a3ffc'),('Último abono',latest.strftime('%d/%m/%Y') if pd.notna(latest) else 'Sin fecha','Según registros seleccionados','#087f91')]
    else:
        outgoing=-valid.loc[valid.Monto.lt(0),'Monto'].sum();refund=valid.loc[valid.Monto.gt(0),'Monto'].sum()
        items=[('Egreso neto',money(-total),'Egresos menos reintegros','#de404c'),('Egresos brutos',money(outgoing),'Pagos con signo negativo','#8a3ffc'),('Reintegros / abonos',money(refund),'Movimientos con signo positivo','#008a65'),('Movimientos con monto',str(len(valid)),'Registros operativos de la selección','#0f6bff')]
    kpis(st,items)
    st.caption(f'Corte {snapshot.revision} · {len(frame)-len(valid)} registros sin monto válido en la selección. Los montos sin fecha se incluyen en los KPI y se excluyen de los gráficos temporales.')
    left,right=st.columns([1.4,1],gap='large')
    with left,st.container(border=True):
        st.markdown('<div class="data-chart-head">Evolución mensual</div>',unsafe_allow_html=True)
        monthly=valid.dropna(subset=['Fecha']).copy();monthly['Período']=monthly.Fecha.dt.to_period('M').astype(str)
        monthly=monthly.groupby('Período',as_index=False).Monto.sum()
        if not income:monthly.Monto=-monthly.Monto
        if monthly.empty:st.info('Sin movimientos fechados para graficar.')
        else:
            fig=px.bar(monthly,x='Período',y='Monto',color_discrete_sequence=['#087f91' if income else '#8a3ffc'],labels={'Monto':'Ingresos · CLP' if income else 'Egreso neto · CLP'})
            fig.update_layout(height=330,margin=dict(l=0,r=10,t=10,b=10),paper_bgcolor='white',plot_bgcolor='white',font=dict(color='#62748a'),xaxis=dict(title=None),yaxis=dict(gridcolor='#edf2f7'))
            st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
    with right,st.container(border=True):
        st.markdown('<div class="data-chart-head">Ingresos por clasificación</div>' if income else '<div class="data-chart-head">Egreso neto por centro</div>',unsafe_allow_html=True)
        grouped=valid.groupby('cc',as_index=False).Monto.sum()
        if not income:grouped.Monto=-grouped.Monto
        if grouped.empty:st.info('Sin montos para graficar.')
        else:
            fig=px.bar(grouped.sort_values('Monto'),x='Monto',y='cc',orientation='h',color_discrete_sequence=['#0f6bff'],labels={'Monto':'CLP','cc':''})
            fig.update_layout(height=330,margin=dict(l=0,r=10,t=10,b=10),paper_bgcolor='white',plot_bgcolor='white',font=dict(color='#62748a'),xaxis=dict(gridcolor='#edf2f7'))
            st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
    st.subheader('Libro de movimientos')
    cols=['id','Fecha','parcela','propietario','cc','Monto','medio','fila_publicada','fila_origen','revision'] if income else ['id','Fecha','detalle','proveedor','cc','ccc','Monto','medio','boleta_factura','tipo_de_flujo','fila_publicada','fila_origen','revision']
    table=frame[[col for col in cols if col in frame]].sort_values('Fecha',ascending=False,na_position='last').rename(columns={'id':'ID','parcela':'Parcela','propietario':'Propietario','cc':'Clasificación' if income else 'Centro','medio':'Medio de pago','fila_publicada':'Fila publicada','fila_origen':'Fila histórica','detalle':'Trabajo','proveedor':'Proveedor','ccc':'Subcentro','revision':'Revisión fuente'})
    st.dataframe(table,hide_index=True,width='stretch',column_config={'Fecha':st.column_config.DateColumn(format='DD/MM/YYYY'),'Monto':st.column_config.NumberColumn('Monto firmado · CLP',format='dollar')})
    st.download_button('Descargar movimientos filtrados · CSV',table.to_csv(index=False).encode('utf-8-sig'),f'{kind}_{snapshot.revision}.csv','text/csv',key=kind+'_csv')
    from control_obligaciones_pdf import build_table_pdf
    pdf_cols = [c for c in ['ID','Fecha','Parcela','Propietario','Trabajo','Proveedor','Clasificación','Centro','Monto','Fila publicada'] if c in table]
    pdf = build_table_pdf(table[pdf_cols], 'Ingresos recibidos' if income else 'Costos y reintegros', snapshot.revision, snapshot.captured_at, URLS[kind], f'Año: {year} · Clasificación: {cat} · Búsqueda: {search or "Todos"}. Montos con signo de origen; registros sin monto conservados para revisión.') if len(table) else b''
    st.download_button('Descargar movimientos filtrados · PDF', pdf, f'{kind}_{snapshot.revision}.pdf','application/pdf',key=kind+'_pdf',disabled=table.empty)
    st.markdown(f'[Consultar hoja de origen · {SHEETS[kind][0]}]({URLS[kind]})')


def render_costs(st,snapshot):
    import plotly.express as px
    from condominio_data import classify_costs
    from control_obligaciones_pdf import build_table_pdf
    header(st,'Costos y compromisos de pago','Control de proveedores: egresos pagados, reintegros realizados y costos No pagado.','$')
    source=classify_costs(snapshot.frames['costos'])
    pending=source[source.estado_pago.eq('No pagado')]
    pending_total=float(source.por_pagar.sum())
    kpis(st,[('Proveedores por pagar',money(pending_total),'Costos · Estado No pagado','#a060b4'),('Compromisos abiertos',str(int(pending.por_pagar.gt(0).sum())),'Registros No pagado con monto a pagar','#d87917'),('Caja pagada acumulada',money(snapshot.totals['caja']),'Solo movimientos realizados','#087f91'),('Caja al pagar pendientes',money(snapshot.totals['caja']-pending_total),'Escenario: caja pagada − No pagado','#8a3ffc')])
    st.caption('Lectura global de la publicación: estos cuatro KPI no cambian con los filtros del libro. El auxiliar Deudas proveedores se muestra en Fuentes y control; no se suma a Costos No pagado.')
    source['Monto']=parse_money(source.monto)
    source['Fecha']=pd.to_datetime(source.fecha,dayfirst=True,errors='coerce')
    years=sorted(source.Fecha.dt.year.dropna().astype(int).unique())
    a,b,c,d=st.columns([1,1.25,1.05,1.7])
    year=a.selectbox('Año',['Todos']+years+(['Sin fecha'] if source.Fecha.isna().any() else []),key='costos_year')
    cat=b.selectbox('Centro de costo',['Todas']+sorted(source.cc.dropna().unique()),key='costos_category')
    state=c.selectbox('Estado de pago',['Todos','Pagado','No pagado']+(['Sin clasificar'] if source.estado_pago.eq('Sin clasificar').any() else []),key='costos_status')
    search=d.text_input('Buscar proveedor, trabajo o ID',key='costos_search')
    frame=source.copy()
    if year=='Sin fecha':frame=frame[frame.Fecha.isna()]
    elif year!='Todos':frame=frame[frame.Fecha.dt.year.eq(year)]
    if cat!='Todas':frame=frame[frame.cc.eq(cat)]
    if state!='Todos':frame=frame[frame.estado_pago.eq(state)]
    if search.strip():
        mask=frame[['id','proveedor','detalle']].fillna('').astype(str).apply(lambda col:col.str.contains(search.strip(),case=False,regex=False)).any(axis=1)
        frame=frame[mask]
    valid=frame[frame.Monto.notna()];paid=valid[valid.estado_pago.eq('Pagado')]
    outgoing=-paid.loc[paid.Monto.lt(0),'Monto'].sum();refund=paid.loc[paid.Monto.gt(0),'Monto'].sum()
    kpis(st,[('Egreso neto pagado',money(outgoing-refund),'Pagado: egresos menos reintegros','#de404c'),('Egresos pagados brutos',money(outgoing),'Estado Pagado · signo negativo','#d87917'),('Reintegros pagados',money(refund),'Estado Pagado · signo positivo','#008a65'),('No pagado · selección',money(valid.por_pagar.sum()),'Compromisos incluidos en los filtros','#a060b4')])
    st.caption(f'Corte {snapshot.revision} · {len(frame)} registros seleccionados · {len(frame)-len(valid)} sin monto válido. No pagado no se descuenta de la caja pagada. Su fecha es una referencia del compromiso, no acredita un pago.')
    a,b=st.columns([1.4,1],gap='large')
    plot=valid.dropna(subset=['Fecha']).copy();plot['Período']=plot.Fecha.dt.to_period('M').astype(str)
    plot['Egreso neto pagado']=-plot.Monto.where(plot.estado_pago.eq('Pagado'),0.)
    plot['Por pagar']=plot.por_pagar
    for panel,key,title,horizontal in [(a,'Período','Egresos pagados y pendientes por mes',False),(b,'cc','Egresos pagados y pendientes por centro',True)]:
        with panel,st.container(border=True):
            st.markdown(f'<div class="data-chart-head">{title}</div>',unsafe_allow_html=True)
            data=plot.groupby(key,as_index=False)[['Egreso neto pagado','Por pagar']].sum().melt(id_vars=key,var_name='Situación',value_name='Monto')
            if data.empty:st.info('Sin movimientos fechados para graficar.')
            else:
                fig=px.bar(data,x='Monto' if horizontal else key,y=key if horizontal else 'Monto',orientation='h' if horizontal else 'v',color='Situación',barmode='group',color_discrete_map={'Egreso neto pagado':'#d69468','Por pagar':'#aa7bc4'},labels={'cc':'Centro de costo','Monto':'CLP'})
                fig.update_layout(height=350,margin=dict(l=0,r=10,t=35,b=10),paper_bgcolor='white',plot_bgcolor='white',font=dict(color='#62748a'),legend=dict(orientation='h',y=1.15,x=0,font=dict(size=10)),xaxis=dict(title=None),yaxis=dict(gridcolor='#edf2f7'))
                st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
    st.subheader('Libro de costos y estado de pago')
    cols=['id','Fecha','estado_pago','detalle','proveedor','cc','ccc','Monto','por_pagar','medio','boleta_factura','tipo_de_flujo','fila_publicada','fila_origen','revision']
    table=frame[[col for col in cols if col in frame]].sort_values('Fecha',ascending=False,na_position='last').rename(columns={'id':'ID','estado_pago':'Estado','detalle':'Trabajo','proveedor':'Proveedor','cc':'Centro','ccc':'Subcentro','por_pagar':'Por pagar','medio':'Medio de pago','fila_publicada':'Fila publicada','fila_origen':'Fila histórica','revision':'Revisión fuente'})
    st.dataframe(table,hide_index=True,width='stretch',column_config={'Fecha':st.column_config.DateColumn('Fecha registrada',format='DD/MM/YYYY'),'Monto':st.column_config.NumberColumn('Monto firmado · CLP',format='dollar'),'Por pagar':st.column_config.NumberColumn('Por pagar · CLP',format='dollar')})
    st.download_button('Descargar movimientos filtrados · CSV',table.to_csv(index=False).encode('utf-8-sig'),f'costos_{snapshot.revision}.csv','text/csv',key='costos_csv')
    pdf_cols=[c for c in ['ID','Fecha','Estado','Trabajo','Proveedor','Centro','Monto','Por pagar','Fila publicada'] if c in table]
    note=f'Año: {year} · Centro: {cat} · Estado: {state} · Búsqueda: {search or "Todos"}. Pagado afecta caja realizada; No pagado se presenta como compromiso separado.'
    pdf=build_table_pdf(table[pdf_cols],'Costos y compromisos de pago',snapshot.revision,snapshot.captured_at,URLS['costos'],note) if len(table) else b''
    st.download_button('Descargar movimientos filtrados · PDF',pdf,f'costos_{snapshot.revision}.pdf','application/pdf',key='costos_pdf',disabled=table.empty)
    st.markdown(f'[Consultar Costos · fuente publicada]({URLS["costos"]})')
