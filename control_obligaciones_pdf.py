"""Informe compartible de cobranza con gráficos vectoriales y tablas paginadas."""
from datetime import datetime
from html import escape
from io import BytesIO
from zoneinfo import ZoneInfo

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

NAVY = colors.HexColor('#073849')
TEAL = colors.HexColor('#087f91')
INK = colors.HexColor('#142a40')
MUTED = colors.HexColor('#657b8e')
LINE = colors.HexColor('#e2eaf0')
PALETTE = ['#8a3ffc', '#ed9a35', '#0f6bff', '#087f91', '#dd5263']
WIDTH = A4[0] - 72


def money(value):
    return '$' + f'{0 if abs(value) < .5 else value:,.0f}'.replace(',', '.')


class KpiGrid(Flowable):
    def __init__(self, items):
        super().__init__()
        self.items = items
        self.width, self.height = WIDTH, 154

    def draw(self):
        c = self.canv
        gap, columns = 9, 4
        width = (self.width - 3*gap)/columns
        for i, (label, value, sub, color) in enumerate(self.items):
            x, y = (i % 4)*(width+gap), self.height-72-(i//4)*80
            c.setFillColor(colors.white)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, width, 72, 7, fill=1, stroke=1)
            c.setFillColor(colors.HexColor(color))
            c.roundRect(x, y+11, 3, 50, 1.5, fill=1, stroke=0)
            c.setFillColor(MUTED)
            c.setFont('Helvetica', 8)
            c.drawString(x+11, y+54, label)
            c.setFillColor(INK)
            c.setFont('Helvetica-Bold', 17)
            c.drawString(x+11, y+29, value)
            c.setFont('Helvetica', 7)
            c.setFillColor(MUTED)
            c.drawString(x+11, y+13, sub)


class DebtCharts(Flowable):
    def __init__(self, summary, grouped):
        super().__init__()
        self.summary, self.grouped = summary, grouped
        self.width, self.height = WIDTH, 210

    def draw(self):
        c = self.canv
        left_w = 294
        for x, w in [(0, left_w), (left_w+12, WIDTH-left_w-12)]:
            c.setFillColor(colors.white)
            c.setStrokeColor(LINE)
            c.roundRect(x, 0, w, self.height, 8, stroke=1, fill=1)
        c.setFont('Helvetica-Bold', 10)
        c.setFillColor(INK)
        c.drawString(13, 190, 'Mayores saldos pendientes')
        c.drawString(left_w+25, 190, 'Deuda por concepto')
        rank = self.summary[self.summary.Pendiente > 0].nlargest(6, 'Pendiente')
        max_value = rank.Pendiente.max() if len(rank) else 0
        if rank.empty:
            c.setFont('Helvetica', 9)
            c.setFillColor(MUTED)
            c.drawString(15, 145, 'Sin saldos pendientes en la selección.')
        for i, (_, row) in enumerate(rank.iterrows()):
            y = 159-i*25
            c.setFont('Helvetica', 8)
            c.setFillColor(MUTED)
            c.drawString(13, y+2, f'Parcela {int(row.Parcela)}')
            c.setFillColor(colors.HexColor('#edf4f7'))
            c.roundRect(73, y, 136, 10, 3, stroke=0, fill=1)
            c.setFillColor(TEAL)
            c.roundRect(73, y, max(1,136*row.Pendiente/max_value), 10, 3, stroke=0, fill=1)
            c.setFont('Helvetica-Bold', 8)
            c.setFillColor(INK)
            c.drawRightString(left_w-13, y+1, money(row.Pendiente))
        pie = self.grouped[self.grouped.Pendiente > 0]
        total = pie.Pendiente.sum()
        cx, cy, radius = left_w+12+(WIDTH-left_w-12)/2, 123, 48
        start = 90
        if not total:
            c.setStrokeColor(LINE)
            c.setLineWidth(17)
            c.circle(cx, cy, radius-8, stroke=1, fill=0)
        for i, (_, row) in enumerate(pie.iterrows()):
            extent = -360*row.Pendiente/total
            c.setFillColor(colors.HexColor(PALETTE[i % len(PALETTE)]))
            c.wedge(cx-radius,cy-radius,cx+radius,cy+radius,start,extent,stroke=0,fill=1)
            start += extent
        c.setFillColor(colors.white)
        c.circle(cx,cy,32,stroke=0,fill=1)
        c.setFont('Helvetica-Bold', 10)
        c.setFillColor(INK)
        c.drawCentredString(cx,cy+1,money(total))
        c.setFont('Helvetica', 7)
        c.setFillColor(MUTED)
        c.drawCentredString(cx,cy-11,'POR COBRAR')
        # Legend wraps into two lines per concept; charts remain vector based.
        legend_y = 56
        for i, (_, row) in enumerate(pie.head(3).iterrows()):
            c.setFillColor(colors.HexColor(PALETTE[i % len(PALETTE)]))
            c.circle(left_w+27,legend_y+2,3,stroke=0,fill=1)
            c.setFillColor(MUTED)
            c.setFont('Helvetica', 8)
            label = str(row.Concepto)
            if len(label)>23: label=label[:22]+'…'
            c.drawString(left_w+35,legend_y,label)
            c.drawRightString(WIDTH-13,legend_y,f'{row.Pendiente/total:.1%}')
            legend_y -= 14
        if len(pie)>3:
            c.setFont('Helvetica',6)
            c.drawString(left_w+25,9,'Todos los conceptos se detallan en la tabla.')


def build_control_pdf(ledger, payments, annual, alerts, review, filters, urls, issued_at=None, revision=None, captured_at=None):
    """Render the exact selected ledger. Source quality notes retain global scope."""
    if ledger.empty:
        raise ValueError('No hay obligaciones para exportar con los filtros seleccionados.')
    issued_at = issued_at or datetime.now(ZoneInfo('America/Santiago'))
    summary = ledger.groupby(['Parcela','Propietario'],as_index=False)[['Cargo','Abonado','Pendiente','A favor']].sum()
    grouped = ledger.groupby('Concepto',as_index=False)[['Cargo','Abonado','Pendiente','A favor']].sum()
    keys = ledger[['Parcela','Concepto']].drop_duplicates()
    selected_payments = payments.merge(keys,on=['Parcela','Concepto'],how='inner')
    latest = selected_payments.Fecha.max()
    last_text = latest.strftime('%d/%m/%Y') if pd.notna(latest) else 'Sin abonos fechados'
    due, credit = ledger.Pendiente.sum(), ledger['A favor'].sum()
    charges = ledger.Cargo.clip(lower=0).sum()
    effective = ledger[['Cargo','Abonado']].min(axis=1).clip(lower=0).sum()
    coverage = effective/charges if charges else 0
    pending = int((summary.Pendiente>0).sum())
    current = len(summary)-pending
    styles = {
        'title': ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=26,leading=30,textColor=NAVY,spaceAfter=8),
        'h2': ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=INK,spaceBefore=12,spaceAfter=8,keepWithNext=True),
        'body': ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=13,textColor=MUTED,spaceAfter=8),
        'small': ParagraphStyle('small',fontName='Helvetica',fontSize=7.5,leading=10.5,textColor=MUTED,spaceAfter=6),
        'cell': ParagraphStyle('cell',fontName='Helvetica',fontSize=8,leading=10.5,textColor=INK),
        'head': ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=7.5,leading=10,textColor=colors.white),
        'number': ParagraphStyle('number',fontName='Helvetica',fontSize=8,leading=10.5,textColor=INK,alignment=TA_RIGHT),
    }
    def para(text,style='body'):
        return Paragraph(text,styles[style])
    def table(headers, rows, widths, money_from=None, total=False, compact=False):
        data=[[para(escape(str(h)),'head') for h in headers]]
        for row in rows:
            data.append([para(escape(str(cell)),'number' if money_from is not None and i>=money_from else 'cell') for i,cell in enumerate(row)])
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        commands=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),4.5 if compact else 6),('BOTTOMPADDING',(0,0),(-1,-1),4.5 if compact else 6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f8fb')]),('LINEBELOW',(0,0),(-1,0),0.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,LINE)]
        if total: commands.extend([('BACKGROUND',(0,-1),(-1,-1),colors.HexColor('#e7f2f4')),('LINEABOVE',(0,-1),(-1,-1),.8,TEAL)])
        t.setStyle(TableStyle(commands))
        return t
    filter_text = ' · '.join(f'{escape(str(k))}: {escape(str(v or "Todos"))}' for k,v in filters.items())
    story=[para('INFORME A LA COMUNIDAD','small'),para('Control de obligaciones','title'),para('Condominio Los Queltehues III · Cifras en pesos chilenos (CLP)'),para(f'Emitido: {issued_at:%d/%m/%Y %H:%M} · Hora de Santiago · Último ingreso seleccionado: {last_text}','small'),para(f'<b>Alcance:</b> {filter_text}','small'),para(f'Corte de datos: {escape(str(revision or "Sin identificador"))} · Consulta: {escape(str(captured_at or "Sin corte indicado"))}','small'),Spacer(1,8)]
    kpis=[('Total por cobrar',money(due),f'Parcelas con pendientes: {pending}','#de404c'),('Abonos registrados',money(ledger.Abonado.sum()),'Asignados por concepto','#008a65'),('Obligaciones',money(charges),'Cargos positivos publicados','#8a3ffc'),('Saldo a favor',money(credit),'Se muestra por separado','#0f6bff'),('Cobertura de cargos',f'{coverage:.1%}','Abonos aplicables / cargos','#087f91'),('Sin deuda en selección',str(current),f'Parcelas consideradas: {len(summary)}','#008a65'),('Con deuda en selección',str(pending),'Según filtros del informe','#d87917'),('Pendiente medio',money(due/pending if pending else 0),'Por parcela con deuda','#087f91')]
    story += [KpiGrid(kpis),Spacer(1,12),DebtCharts(summary,grouped),Spacer(1,10),para('Resumen por concepto','h2')]
    rows=[[r.Concepto,money(r.Cargo),money(r.Abonado),money(r.Pendiente),money(r['A favor'])] for _,r in grouped.iterrows()]
    rows.append(['TOTAL',money(ledger.Cargo.sum()),money(ledger.Abonado.sum()),money(due),money(credit)])
    story += [table(['Concepto','Cargo','Abonado','Pendiente','A favor'],rows,[WIDTH-4*87,87,87,87,87],1,True),Spacer(1,8),para('Los saldos a favor no compensan deudas de otras parcelas ni de otros conceptos. Los cargos corresponden al total publicado; este informe no certifica mora ni asigna vencimientos.','small'),PageBreak()]
    story += [para('Estado de las parcelas','title'),para('Consolidado por propietario de los conceptos incluidos en los filtros. Ordenado por saldo pendiente.'),para(f'{len(summary)} parcelas · Pendiente: {money(due)} · A favor: {money(credit)}','small')]
    rows=[]
    for _,r in summary.sort_values(['Pendiente','Parcela'],ascending=[False,True]).iterrows():
        rows.append([int(r.Parcela),r.Propietario,money(r.Cargo),money(r.Abonado),money(r.Pendiente),money(r['A favor'])])
    rows.append(['','TOTAL',money(ledger.Cargo.sum()),money(ledger.Abonado.sum()),money(due),money(credit)])
    story += [table(['Parcela','Propietario','Cargo','Abonado','Pendiente','A favor'],rows,[48,WIDTH-48-4*78,78,78,78,78],2,True),Spacer(1,10),para('Lectura de la cuenta: Pendiente es lo que falta pagar en cada concepto; A favor es el excedente de pagos de ese concepto. Una parcela puede tener ambos valores simultáneamente.','small'),PageBreak()]
    story += [para('Detalle de obligaciones','title'),para('Trazabilidad por parcela y concepto. Se muestran los mismos registros incluidos en la cartera filtrada.')]
    detail=ledger.sort_values(['Parcela','Concepto'])
    rows=[]
    for _,r in detail.iterrows():
        last = r['Último abono']
        concept_label = r.Concepto + (' · ' + str(r.get('Trabajo', '')) if r.Concepto == 'Mantención' and r.get('Trabajo', '') else '')
        rows.append([int(r.Parcela),concept_label,money(r.Cargo),money(r.Abonado),money(r.Pendiente),money(r['A favor']),last.strftime('%d/%m/%Y') if pd.notna(last) else 'Sin abonos'])
    rows.append(['','TOTAL',money(ledger.Cargo.sum()),money(ledger.Abonado.sum()),money(due),money(credit),''])
    story += [table(['Parcela','Concepto','Cargo','Abonado','Pendiente','A favor','Último abono'],rows,[48,WIDTH-48-4*73-66,73,73,73,73,66],2,True),Spacer(1,14)]
    # Keep the heading with the first paragraph on the final detail page.
    notes_start = len(story)
    story += [para('Criterios de lectura y calidad de datos','h2'),para('Las observaciones siguientes corresponden a las fuentes completas, aunque el informe tenga filtros activos.','small')]
    for note in alerts:
        story.append(para('• '+escape(note.replace('1 abonos no se pudieron', '1 abono no se pudo')),'small'))
    if not alerts: story.append(para('Los cargos y abonos se pudieron asignar sin observaciones.','small'))
    if not review.empty:
        story.append(para(f'Registros de abonos excluidos por datos incompletos: {len(review)}. Revisa el panel de conciliación antes de corregir la planilla.','small'))
    story.append(para('Gastos comunes: suma de los cargos anuales publicados por parcela. Proyectos: aporte esperado publicado por parcela; los abonos de cámaras ya están en Ingresos y no se suman dos veces. Mantención: cargo neto de ajustes por parcela y trabajo; los ajustes del auxiliar no son ingresos de caja. Abonos: cruce por parcela y concepto. Se excluyen registros sin monto, parcela válida o concepto reconocido.','small'))
    story.append(para('Para corregir o registrar pagos, actualiza la planilla y genera un nuevo informe. Las cuentas por pagar a proveedores se consultan por separado en la administración.','small'))
    source_links = []
    for key,label in [('obligaciones','Cobros GC por tramos'),('proyectos','Aportes esperados y auxiliar de cámaras'),('ingresos','Registro de abonos y propietarios'),('mantencion','Cargos y ajustes de mantención')]:
        if key in urls:
            source_links.append(f'<link href="{escape(urls[key],quote=True)}" color="#087f91">{label}</link>')
    story = story[:notes_start] + [KeepTogether(story[notes_start:notes_start+4])] + story[notes_start+4:]
    if 'Referencia del cargo' in ledger:
        story += [PageBreak(), para('Referencias del informe','title'), para('Identificación de los cargos que respaldan la cartera seleccionada. Los IDs se consultan en la hoja publicada; los números de fila corresponden a la publicación activa.')]
        story += [Paragraph('Fuentes: ' + ' · '.join(source_links), styles['small']), Spacer(1,8)]
        refs=ledger.groupby(['Fuente del cargo','Referencia del cargo'],as_index=False).Parcela.agg(lambda ps:', '.join(str(int(v)) for v in sorted(ps.unique())))
        rows=[[r['Fuente del cargo'],r['Referencia del cargo'],r.Parcela] for _,r in refs.iterrows()]
        story.append(table(['Hoja del cargo','ID / filas publicadas','Parcelas'],rows,[118,205,WIDTH-323],compact=True))
        story += [Spacer(1,10),para('Los abonos conservan su ID ING y la fila histórica en el detalle de Control de obligaciones. Los propietarios se identifican desde los últimos registros de Ingresos y Proyectos; no se consulta el archivo anterior.','small')]
    elif source_links:
        story += [para('Fuentes publicadas','h2'),Paragraph(' · '.join(source_links), styles['small'])]
    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,leftMargin=36,rightMargin=36,topMargin=47,bottomMargin=43,title='Control de obligaciones · Los Queltehues III',author='Administración · Condominio Los Queltehues III')
    def decorate(c,doc):
        c.saveState()
        c.setStrokeColor(LINE)
        c.line(36,A4[1]-30,A4[0]-36,A4[1]-30)
        c.setFont('Helvetica-Bold',7)
        c.setFillColor(NAVY)
        c.drawString(36,A4[1]-22,'LOS QUELTEHUES III')
        c.setFont('Helvetica',7)
        c.setFillColor(MUTED)
        c.drawRightString(A4[0]-36,A4[1]-22,'CONTROL DE OBLIGACIONES · CLP')
        c.line(36,31,A4[0]-36,31)
        c.drawString(36,20,f'Corte {revision or "publicado"} · Emisión {issued_at:%d/%m/%Y} · Según filtros indicados')
        c.drawRightString(A4[0]-36,20,f'Página {doc.page}')
        c.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    return buffer.getvalue()


def build_table_pdf(frame, title, revision='', captured_at='', source_url='', note=''):
    """Readable table export using the same report typography, palette and cut."""
    from reportlab.lib.pagesizes import landscape
    wide=len(frame.columns)>4
    page=landscape(A4) if wide else A4
    width=page[0]-72
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=page,leftMargin=36,rightMargin=36,topMargin=47,bottomMargin=43,title=title,author='Administración · Los Queltehues III')
    body=ParagraphStyle('export-body',fontName='Helvetica',fontSize=8,leading=11,textColor=INK,spaceAfter=8)
    heading=ParagraphStyle('export-title',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=NAVY,spaceAfter=10)
    head=ParagraphStyle('export-head',fontName='Helvetica-Bold',fontSize=8,leading=10,textColor=colors.white)
    def p(value,style=body):return Paragraph(escape(str(value)),style)
    story=[p(title,heading),p(f'Los Queltehues III · CLP · Consulta {captured_at} · Corte {revision}')]
    if note:story.append(p(note))
    if source_url:story.append(Paragraph(f'<link href="{escape(source_url,quote=True)}" color="#087f91">Consultar publicación de origen</link>',body))
    story.append(Spacer(1,10))
    data=[[p(col,head) for col in frame.columns]]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for col,val in zip(frame.columns,row):
            if pd.isna(val):val='Sin dato'
            elif isinstance(val,pd.Timestamp):val=val.strftime('%d/%m/%Y')
            elif col in ['Monto','Cargo','Abonado','Pendiente','A favor','Por pagar']:val=money(float(val))
            values.append(p(val))
        data.append(values)
    weights=[]
    for col in frame:
        weights.append(2.8 if col in ['Trabajo','Detalle','Propietario','Proveedor','Indicador'] else 1.6 if col in ['Centro','Clasificación','Valor'] else 1.)
    widths=[width*w/sum(weights) for w in weights]
    table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f8fb')]),('LINEBELOW',(0,0),(-1,-1),.3,LINE)]))
    story.append(table)
    def decorate(c,doc):
        c.saveState();c.setFont('Helvetica-Bold',7);c.setFillColor(NAVY);c.drawString(36,page[1]-23,'LOS QUELTEHUES III · ADMINISTRACIÓN')
        c.setStrokeColor(LINE);c.line(36,page[1]-31,page[0]-36,page[1]-31);c.line(36,31,page[0]-36,31)
        c.setFont('Helvetica',7);c.setFillColor(MUTED);c.drawString(36,20,f'Corte {revision} · Según filtros del informe');c.drawRightString(page[0]-36,20,f'Página {doc.page}');c.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    return buf.getvalue()
