"""Análisis de cartera: métricas, tablas y gráficos derivados del mismo saldo."""
from html import escape
import pandas as pd

COLORS={'Gastos comunes':'#7357d9','Proyecto':'#3279d5','Mantención':'#df9635'}


def money(n):
    return '$'+f'{0 if abs(n)<.5 else n:,.0f}'.replace(',','.')


def percent(n):
    return f'{n:.1%}'.replace('.',',')


def analyze_portfolio(ledger,payments):
    """Una posición por parcela; cobertura ponderada sin compensar créditos."""
    work=ledger.copy()
    work['Aplicado']=work[['Cargo','Abonado']].clip(lower=0).min(axis=1)
    work['Base de cobertura']=work.Cargo.clip(lower=0)
    keys=work[['Parcela','Concepto']].drop_duplicates()
    selected=payments.merge(keys,on=['Parcela','Concepto'],how='inner')
    positions=work.groupby(['Parcela','Propietario'],as_index=False)[['Cargo','Abonado','Pendiente','A favor','Aplicado','Base de cobertura']].sum()
    dates=selected.groupby('Parcela').Fecha.max()
    positions['Último ingreso']=positions.Parcela.map(dates)
    positions['Cobertura']=positions.Aplicado.div(positions['Base de cobertura'].where(positions['Base de cobertura'].ne(0))).fillna(0)
    due=float(work.Pendiente.sum())
    positions['Participación']=positions.Pendiente/due if due else 0.
    for name in ['Gastos comunes','Proyecto','Mantención']:
        values=work[work.Concepto.eq(name)].groupby('Parcela').Pendiente.sum()
        positions[name]=positions.Parcela.map(values).fillna(0)
    positions['Situación']=positions.apply(lambda r:'Pendiente + crédito' if r.Pendiente>0 and r['A favor']>0 else 'Pendiente' if r.Pendiente>0 else 'A favor' if r['A favor']>0 else 'Sin pendiente',axis=1)
    if positions.empty: positions['Situación']=pd.Series(dtype=str)
    concepts=work.groupby('Concepto',as_index=False)[['Cargo','Abonado','Pendiente','A favor','Aplicado','Base de cobertura']].sum()
    concepts['Cobertura']=concepts.Aplicado.div(concepts['Base de cobertura'].where(concepts['Base de cobertura'].ne(0))).fillna(0)
    concepts['Exposición']=concepts.Pendiente.div(concepts['Base de cobertura'].where(concepts['Base de cobertura'].ne(0))).fillna(0)
    ranked=positions[positions.Pendiente.gt(0)].sort_values(['Pendiente','Parcela'],ascending=[False,True]).copy()
    ranked['Acumulado']=ranked.Pendiente.cumsum()/due if due else 0.
    dated=selected.dropna(subset=['Fecha']).copy()
    if len(dated):
        dated['Mes']=dated.Fecha.dt.to_period('M')
        monthly=dated.groupby('Mes').Abono.sum()
        monthly=monthly.reindex(pd.period_range(monthly.index.min(),monthly.index.max(),freq='M'),fill_value=0)
        monthly=monthly.rename_axis('Mes').reset_index(name='Abonos')
        monthly['Promedio móvil 3 meses']=monthly.Abonos.rolling(3,min_periods=1).mean()
        monthly['Mes']=monthly.Mes.astype(str)
    else:monthly=pd.DataFrame(columns=['Mes','Abonos','Promedio móvil 3 meses'])
    base=float(work['Base de cobertura'].sum())
    debtors=len(ranked)
    return {'positions':positions,'concepts':concepts,'ranked':ranked,'monthly':monthly,'payments':selected,
            'due':due,'credit':float(work['A favor'].sum()),'charges':float(work.Cargo.sum()),'base':base,
            'paid':float(work.Abonado.sum()),'applied':float(work.Aplicado.sum()),
            'coverage':float(work.Aplicado.sum()/base) if base else 0.,
            'uncovered':due/base if base else 0.,'net':due-float(work['A favor'].sum()),
            'debtors':debtors,'count':len(positions),'current':len(positions)-debtors,
            'debtor_share':debtors/len(positions) if len(positions) else 0.,
            'average':due/debtors if debtors else 0.,
            'top3':float(ranked.head(3).Pendiente.sum()/due) if due else 0.,
            'top5':float(ranked.head(5).Pendiente.sum()/due) if due else 0.}


def portfolio_table(positions,metrics):
    """Parcela y total por pagar contiguos y fijos; desglose y cuenta separados."""
    parts=['<div class="fin-table-shell"><div class="fin-table-scroll"><table class="fin-table" aria-label="Panorama financiero por parcela"><colgroup><col style="width:var(--fin-owner-width)"><col style="width:135px"><col span="10"></colgroup><thead><tr><th rowspan="2" class="fin-sticky">Parcela / propietario</th><th rowspan="2" class="fin-due-col">TOTAL POR PAGAR</th><th colspan="3">Desglose de lo pendiente · CLP</th><th colspan="3">Cuenta publicada · CLP</th><th colspan="3">Indicadores de seguimiento</th><th rowspan="2">Situación</th></tr><tr><th>Gastos comunes</th><th>Cámaras</th><th>Mantención</th><th>Cargo neto</th><th>Abonos</th><th>A favor</th><th>Cobertura</th><th>Peso en deuda</th><th>Último ingreso</th></tr></thead><tbody>']
    for _,r in positions.iterrows():
        tone='red' if r.Pendiente>0 else 'blue' if r['A favor']>0 else 'green'
        owner=escape(str(r.Propietario))
        parts.append(f'<tr><td class="fin-sticky"><div class="fin-owner"><span class="fin-unit">{int(r.Parcela):02d}</span><span><b title="{escape(str(r.Propietario),quote=True)}">{owner}</b><small>Parcela {int(r.Parcela)}</small></span></div></td><td class="fin-number fin-due-col {"is-pending" if r.Pendiente>0 else "is-clear"}">{money(r.Pendiente) if r.Pendiente else "—"}</td>')
        for field in ['Gastos comunes','Proyecto','Mantención','Cargo','Abonado','A favor']:
            cls='fin-credit' if field=='A favor' and r[field]>0 else 'fin-muted' if r[field]==0 else ''
            parts.append(f'<td class="fin-number {cls}">{money(r[field]) if r[field]!=0 else "—"}</td>')
        last=r['Último ingreso'].strftime('%d/%m/%Y') if pd.notna(r['Último ingreso']) else 'Sin fecha'
        parts.append(f'<td><div class="fin-coverage"><b>{percent(r.Cobertura)}</b><span><i style="width:{min(100,r.Cobertura*100):.2f}%"></i></span></div></td><td class="fin-number">{percent(r.Participación)}</td><td class="fin-date">{last}</td><td><span class="fin-status {tone}">{escape(r.Situación)}</span></td></tr>')
    m=metrics
    totals=[m['due']]+[positions[c].sum() for c in ['Gastos comunes','Proyecto','Mantención']]+[m['charges'],m['paid'],m['credit']]
    parts.append('<tr class="fin-total"><td class="fin-sticky">TOTAL DE LA SELECCIÓN</td>'+''.join(f'<td class="fin-number {"fin-due-col" if i==0 else ""}">{money(v)}</td>' for i,v in enumerate(totals))+f'<td>{percent(m["coverage"])}</td><td class="fin-number">{percent(1 if m["due"] else 0)}</td><td colspan="2">{m["count"]} parcelas</td></tr>')
    parts.append('</tbody></table></div><div class="fin-table-footer"><span>La columna roja muestra el total por pagar. Su desglose está a la derecha; los créditos se mantienen separados.</span><span>Cobertura ponderada por cargo positivo</span></div></div>')
    return ''.join(parts)


def financial_figures(metrics,window=12):
    import plotly.graph_objects as go
    def style(fig,height=330):
        fig.update_layout(height=height,margin=dict(l=8,r=18,t=22,b=18),font=dict(family='Inter, Arial',size=11,color='#687c91'),paper_bgcolor='white',plot_bgcolor='white',separators=',.',hoverlabel=dict(bgcolor='white',font_size=12),legend=dict(orientation='h',y=1.17,x=0,font=dict(size=10),traceorder='normal'),bargap=.35)
        fig.update_xaxes(showgrid=False,zeroline=False)
        fig.update_yaxes(gridcolor='#edf2f6',zeroline=False)
        return fig
    ranked=metrics['ranked']
    pareto=go.Figure()
    if len(ranked):
        labels=['P'+str(p) for p in ranked.Parcela]
        pareto.add_trace(go.Bar(x=labels,y=ranked.Pendiente,name='Saldo pendiente',marker=dict(color=['#104b63' if i<3 else '#81b9c6' for i in range(len(ranked))],cornerradius=4),customdata=ranked[['Propietario','Participación']],hovertemplate='%{x} · %{customdata[0]}<br>Pendiente: $%{y:,.0f}<br>Peso en cartera: %{customdata[1]:.1%}<extra></extra>'))
        pareto.add_trace(go.Scatter(x=labels,y=ranked.Acumulado*100,name='Participación acumulada',yaxis='y2',mode='lines+markers',line=dict(color='#8b67d6',width=2),marker=dict(size=5),hovertemplate='%{x}<br>Deuda acumulada: %{y:.1f}%<extra></extra>'))
    pareto=style(pareto)
    pareto.update_layout(yaxis=dict(title='Pendiente · CLP',tickformat=',.0f'),yaxis2=dict(title='Acumulado',overlaying='y',side='right',range=[0,108],ticksuffix='%',showgrid=False),margin=dict(l=8,r=28,t=35,b=20))
    concepts=metrics['concepts']
    concepts=concepts[concepts['Base de cobertura'].gt(0)]
    coverage=go.Figure()
    if len(concepts):
        coverage.add_trace(go.Bar(y=concepts.Concepto,x=concepts.Cobertura*100,name='Cubierto',orientation='h',marker=dict(color='#2c9a8a',cornerradius=4),text=[percent(v) for v in concepts.Cobertura],textposition='inside',insidetextanchor='middle',hovertemplate='%{y}<br>Cobertura: %{x:.1f}%<extra></extra>'))
        coverage.add_trace(go.Bar(y=concepts.Concepto,x=(1-concepts.Cobertura)*100,name='Sin cubrir',orientation='h',marker=dict(color='#e7a16a',cornerradius=4),customdata=concepts.Pendiente,text=[percent(1-v) if 1-v>=.12 else '' for v in concepts.Cobertura],textposition='inside',insidetextanchor='middle',textfont=dict(color='#6e3b18',size=11),hovertemplate='%{y}<br>Sin cubrir: %{x:.1f}%<br>Pendiente: $%{customdata:,.0f}<extra></extra>'))
    coverage=style(coverage)
    coverage.update_layout(barmode='stack',xaxis=dict(range=[0,112],ticksuffix='',tickmode='array',tickvals=[0,25,50,75,100],ticktext=['0%','25%','50%','75%','100%']),yaxis=dict(autorange='reversed',showgrid=False),margin=dict(l=8,r=12,t=35,b=20))
    coverage.update_layout(annotations=[dict(x=101,y=r.Concepto,text='<b>'+percent(1-r.Cobertura)+'</b>',showarrow=False,xanchor='left',font=dict(color='#a86631',size=11)) for _,r in concepts.iterrows() if 0<1-r.Cobertura<.12])
    bridge=go.Figure(go.Waterfall(x=['Cargos netos','Abonos','Créditos separados','Por cobrar'],measure=['absolute','relative','relative','total'],y=[metrics['charges'],-metrics['paid'],metrics['credit'],0],textposition='none',connector=dict(line=dict(color='#c5d4df',dash='dot')),increasing=dict(marker=dict(color='#8d70cf')),decreasing=dict(marker=dict(color='#67aaa2')),totals=dict(marker=dict(color='#103e58')),hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'))
    bridge=style(bridge)
    scale=max(abs(metrics[v]) for v in ['charges','paid','credit','due']) or 1.
    low=min(0,metrics['charges']-metrics['paid'])-.07*scale
    high=max(0,metrics['charges'],metrics['due'])+.18*scale
    marks=[('Cargos netos',metrics['charges']+.04*scale,money(metrics['charges']),'#42647b'),('Abonos',metrics['charges']-metrics['paid']/2,'−'+money(metrics['paid']),'#ffffff'),('Créditos separados',metrics['due']+.04*scale,'+'+money(metrics['credit']),'#7457ce'),('Por cobrar',metrics['due']+.04*scale,money(metrics['due']),'#244d66')]
    bridge.update_layout(showlegend=False,yaxis=dict(title='CLP',tickformat=',.2~s',range=[low,high]),margin=dict(l=8,r=12,t=20,b=22),annotations=[dict(x=x,y=y,text='<b>'+text+'</b>',showarrow=False,font=dict(size=10,color=color)) for x,y,text,color in marks])
    series=metrics['monthly'].tail(window) if window else metrics['monthly']
    trend=go.Figure()
    if len(series):
        trend.add_trace(go.Bar(x=series.Mes,y=series.Abonos,name='Abonos de caja',marker=dict(color='#3b83ad',cornerradius=4),hovertemplate='%{x}<br>Abonos: $%{y:,.0f}<extra></extra>'))
        trend.add_trace(go.Scatter(x=series.Mes,y=series['Promedio móvil 3 meses'],name='Promedio móvil 3 meses',mode='lines',line=dict(color='#936ed5',width=2.5),hovertemplate='%{x}<br>Promedio 3 meses: $%{y:,.0f}<extra></extra>'))
    trend=style(trend)
    trend.update_layout(yaxis=dict(title='CLP',tickformat=',.2~s'),margin=dict(l=8,r=12,t=35,b=22))
    if len(series):
        months=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
        labels=[months[int(v[5:7])-1]+' '+v[:4] for v in series.Mes]
        step=max(1,(len(series)+7)//8)
        trend.update_xaxes(type='category',tickmode='array',tickvals=series.Mes.iloc[::step],ticktext=labels[::step])
    return {'pareto':pareto,'coverage':coverage,'bridge':bridge,'trend':trend}


def cash_flow(snapshot):
    """Caja realizada por Estado y escenario separado de compromisos abiertos."""
    from condominio_data import parse_money,classify_costs
    income=snapshot.frames['ingresos'].copy()
    costs=classify_costs(snapshot.frames['costos'])
    income['Fecha']=pd.to_datetime(income.fecha,dayfirst=True,errors='coerce')
    costs['Fecha']=pd.to_datetime(costs.fecha,dayfirst=True,errors='coerce')
    income['Monto']=parse_money(income.abono)
    costs['Monto']=parse_money(costs.monto)
    paid=costs[costs.estado_pago.eq('Pagado')].copy()
    pending=costs[costs.estado_pago.eq('No pagado') & costs.por_pagar.gt(0)].copy()
    pending_total=float(pending.por_pagar.sum())
    pending_undated=float(pending.loc[pending.Fecha.isna(),'por_pagar'].sum())
    no_date=pd.concat([income.loc[income.Monto.notna()&income.Fecha.isna(),'Monto'],paid.loc[paid.Monto.notna()&paid.Fecha.isna(),'Monto']])
    a=income.dropna(subset=['Fecha','Monto']).copy();b=paid.dropna(subset=['Fecha','Monto']).copy();q=pending.dropna(subset=['Fecha']).copy()
    for frame in [a,b,q]:frame['Mes']=frame.Fecha.dt.to_period('M')
    inflows=a.groupby('Mes').Monto.sum().rename('Ingresos')
    outflows=(-b.loc[b.Monto.lt(0)].groupby('Mes').Monto.sum()).rename('Egresos')
    refunds=b.loc[b.Monto.ge(0)].groupby('Mes').Monto.sum().rename('Reintegros')
    commitments=q.groupby('Mes').por_pagar.sum().rename('Pendientes')
    result=pd.concat([inflows,outflows,refunds,commitments],axis=1).fillna(0).sort_index()
    # With no dated activity, retain an undated commitment as a separate current
    # scenario reference; its source still has no asserted payment date.
    if result.empty and pending_undated:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        month=pd.Period(datetime.now(ZoneInfo('America/Santiago')),freq='M')
        result=pd.DataFrame([[0.,0.,0.,0.]],index=pd.PeriodIndex([month]),columns=['Ingresos','Egresos','Reintegros','Pendientes'])
    metadata={'undated_count':len(no_date),'undated_net':float(no_date.sum()),'pending_total':pending_total,'pending_undated':pending_undated,'unclassified_count':int(costs.estado_pago.eq('Sin clasificar').sum())}
    cols=['Mes','Apertura','Ingresos','Reintegros','Egresos','Flujo neto','Cierre','Pendientes','Reserva acumulada','Cierre tras pendientes']
    if result.empty:return {'monthly':pd.DataFrame(columns=cols),**metadata}
    result=result.reindex(pd.period_range(result.index.min(),result.index.max(),freq='M'),fill_value=0)
    # Undated obligations affect the last scenario point without inventing a date.
    if pending_undated:result.loc[result.index[-1],'Pendientes']+=pending_undated
    result['Flujo neto']=result.Ingresos+result.Reintegros-result.Egresos
    result['Cierre']=result['Flujo neto'].cumsum()
    result['Apertura']=result.Cierre.shift(1,fill_value=0)
    result['Reserva acumulada']=result.Pendientes.cumsum()
    result['Cierre tras pendientes']=result.Cierre-result['Reserva acumulada']
    result=result.rename_axis('Mes').reset_index();result['Mes']=result.Mes.astype(str)
    return {'monthly':result[cols],**metadata}


def cash_window(flow,length=12):
    data=flow['monthly'].tail(length).copy() if length else flow['monthly'].copy()
    if data.empty:return {'data':data,'opening':0.,'closing':0.,'income':0.,'expenses':0.,'refunds':0.,'net':0.,'coverage':None,'pending':flow.get('pending_total',0.),'pending_window':0.,'adjusted_closing':0.,'adjusted_opening':0.}
    expenses=float(data.Egresos.sum());income=float(data.Ingresos.sum());refunds=float(data.Reintegros.sum())
    pending_before=float(data['Reserva acumulada'].iloc[0]-data.Pendientes.iloc[0])
    return {'data':data,'opening':float(data.Apertura.iloc[0]),'closing':float(data.Cierre.iloc[-1]),'income':income,'expenses':expenses,'refunds':refunds,'net':float(data['Flujo neto'].sum()),'coverage':(income+refunds)/expenses if expenses else None,'pending':flow.get('pending_total',0.),'pending_window':float(data.Pendientes.sum()),'adjusted_closing':float(data['Cierre tras pendientes'].iloc[-1]),'adjusted_opening':float(data.Apertura.iloc[0])-pending_before}


def cash_figure(window):
    import plotly.graph_objects as go
    d=window['data'];fig=go.Figure()
    fig.add_trace(go.Bar(x=d.Mes,y=d.Ingresos,name='Ingresos recibidos',marker=dict(color='#2b9b88',cornerradius=3),hovertemplate='%{x}<br>Ingresos: $%{y:,.0f}<extra></extra>'))
    if d.Reintegros.ne(0).any():fig.add_trace(go.Bar(x=d.Mes,y=d.Reintegros,name='Reintegros pagados',marker=dict(color='#88c9bc',cornerradius=3),hovertemplate='%{x}<br>Reintegros: $%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(x=d.Mes,y=-d.Egresos,name='Egresos pagados',marker=dict(color='#d69468',cornerradius=3),customdata=d.Egresos,hovertemplate='%{x}<br>Pagado: $%{customdata:,.0f}<extra></extra>'))
    has_pending=window.get('pending',0)>0
    if has_pending:fig.add_trace(go.Bar(x=d.Mes,y=-d.Pendientes,name='No pagado · compromiso',marker=dict(color='#aa7bc4',cornerradius=3),customdata=d.Pendientes,hovertemplate='%{x}<br>Pendiente de pagar: $%{customdata:,.0f}<br>Escenario según fecha registrada<extra></extra>'))
    fig.add_trace(go.Scatter(x=d.Mes,y=d.Cierre,name='Caja acumulada · pagados',yaxis='y2',mode='lines+markers',line=dict(color='#174e6c',width=3),marker=dict(size=5),customdata=d[['Apertura','Flujo neto']],hovertemplate='%{x}<br>Apertura: $%{customdata[0]:,.0f}<br>Flujo pagado: $%{customdata[1]:,.0f}<br>Caja pagada: $%{y:,.0f}<extra></extra>'))
    if has_pending:fig.add_trace(go.Scatter(x=d.Mes,y=d['Cierre tras pendientes'],name='Caja al pagar pendientes · escenario',yaxis='y2',mode='lines+markers',line=dict(color='#8b5ba9',width=2.6,dash='dash'),marker=dict(size=4),customdata=d['Reserva acumulada'],hovertemplate='%{x}<br>Compromisos acumulados: $%{customdata:,.0f}<br>Caja tras pago: $%{y:,.0f}<br>Escenario, no movimiento pagado<extra></extra>'))
    months=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];step=max(1,(len(d)+11)//12)
    fig.update_layout(height=430 if has_pending else 390,barmode='relative',bargap=.34,paper_bgcolor='white',plot_bgcolor='white',separators=',.',font=dict(family='Inter, Arial',size=11,color='#687c91'),margin=dict(l=8,r=35,t=65 if has_pending else 44,b=20),legend=dict(orientation='h',y=1.24 if has_pending else 1.14,x=0,traceorder='normal',font=dict(size=10)),xaxis=dict(type='category',tickmode='array',tickvals=d.Mes.iloc[::step],ticktext=[months[int(v[5:7])-1]+' '+v[:4] for v in d.Mes.iloc[::step]],showgrid=False),yaxis=dict(title='Egresos pagados y compromisos · CLP',gridcolor='#edf2f6',tickformat=',.2~s',zeroline=True,zerolinecolor='#bdcdd9'),yaxis2=dict(title='Caja pagada / escenario · CLP',overlaying='y',side='right',tickformat=',.2~s',showgrid=False),hovermode='x unified')
    return fig


def dashboard_headlines(snapshot,portfolio,as_of=None):
    """Caja global al corte diario y cobranza según los filtros de cartera."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from condominio_data import classify_costs,parse_money
    day=pd.Timestamp(as_of if as_of is not None else datetime.now(ZoneInfo('America/Santiago')).date()).normalize()
    result={'as_of':day,'receivable':float(portfolio['due']),'debtors':portfolio['debtors'],'count':portfolio['count'],'debtor_share':portfolio['debtor_share'],'cash_today':None,'income_today':None,'expenses_today':None,'payables':None,'cash_after_payables':None,'undated_count':0,'future_count':0}
    if snapshot is None:return result
    income=snapshot.frames['ingresos']
    costs=classify_costs(snapshot.frames['costos'])
    costs=costs[costs.estado_pago.eq('Pagado')]
    def recognized(frame,column):
        amounts=parse_money(frame[column]);dates=pd.to_datetime(frame.fecha,dayfirst=True,errors='coerce')
        valid=amounts.notna();future=valid & dates.gt(day);undated=valid & dates.isna()
        result['future_count']+=int(future.sum());result['undated_count']+=int(undated.sum())
        # Recognized entries without a date remain in cash, with a visible note.
        return float(amounts[valid & ~future].sum())
    income_today=recognized(income,'abono')
    expenses_today=-recognized(costs,'monto')
    cash=income_today-expenses_today
    payables=float(snapshot.totals['proveedores'])
    result.update(cash_today=cash,income_today=income_today,expenses_today=expenses_today,payables=payables,cash_after_payables=cash-payables)
    return result


def dashboard_kpis(portfolio,headlines):
    """Ocho indicadores compartidos por pantalla y PDF; caja al mismo corte diario."""
    m=portfolio;h=headlines
    fmt=lambda value:money(value) if value is not None else '—'
    committed=percent(h['payables']/h['cash_today']) if h['payables'] is not None and h['cash_today'] is not None and h['cash_today']>0 else '—'
    return [
        ('Obligaciones emitidas',money(m['charges']),'Según filtros · cargos con ajustes','#7457ce','OE'),
        ('Ingresos recibidos',fmt(h['income_today']),'Global · recibidos al corte de hoy','#16957f','IR'),
        ('Egresos netos pagados',fmt(h['expenses_today']),'Global · pagados menos reintegros','#d9803f','EP'),
        ('Proveedores por pagar',fmt(h['payables']),'Global · Costos No pagado','#8b5ba9','PP'),
        ('Obligaciones cubiertas',percent(m['coverage']) if m['base'] else '—','Según filtros · abonos / cargos positivos','#16957f','%'),
        ('Créditos de vecinos',money(m['credit']),'Según filtros · excedentes a favor','#5285bd','CV'),
        ('Concentración de la deuda',percent(m['top3']) if m['due'] else '—','Según filtros · tres mayores saldos','#7457ce','C3'),
        ('Caja comprometida con proveedores',committed,'Global · por pagar / caja actual','#b7474d','CC'),
    ]
