"""Exportación vectorial del dashboard, hasta antes del detalle individual."""
from datetime import datetime
from html import escape
from io import BytesIO
from math import ceil
from zoneinfo import ZoneInfo
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable,SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from control_finance import analyze_portfolio,cash_flow,cash_window,dashboard_headlines,money,percent

PAGE=landscape(A4)
W=PAGE[0]-72
NAVY=colors.HexColor('#153e56')
INK=colors.HexColor('#264a62')
MUTED=colors.HexColor('#71869a')
LINE=colors.HexColor('#e2eaf1')
GREEN=colors.HexColor('#2b9b88')
ORANGE=colors.HexColor('#e7a16a')
PURPLE=colors.HexColor('#8b67d6')


def compact(v):
    if abs(v)>=1000000:return f'{v/1000000:.1f} M'.replace('.',',')
    if abs(v)>=1000:return f'{v/1000:.0f} mil'
    return f'{v:.0f}'


def month_label(value):
    months=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
    return months[int(value[5:7])-1]+' '+value[2:4]


class MetricGrid(Flowable):
    def __init__(self,items,width=W,columns=4):
        super().__init__();self.items=items;self.width=width;self.columns=columns;self.height=ceil(len(items)/columns)*75
    def draw(self):
        c=self.canv;gap=10;cw=(self.width-(self.columns-1)*gap)/self.columns
        for i,(label,value,note) in enumerate(self.items):
            x=(i%self.columns)*(cw+gap);y=self.height-(i//self.columns+1)*75+5
            c.setFillColor(colors.white);c.setStrokeColor(LINE);c.roundRect(x,y,cw,65,6,stroke=1,fill=1)
            c.setFillColor(MUTED);c.setFont('Helvetica',8);c.drawString(x+11,y+49,label)
            c.setFillColor(INK);c.setFont('Helvetica-Bold',19);c.drawString(x+11,y+25,value)
            c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawString(x+11,y+11,note)




class Chart(Flowable):
    def __init__(self,kind,m=None,data=None,width=W,height=275):
        super().__init__();self.kind=kind;self.m=m;self.data=data;self.width=width;self.height=height
    def draw(self):
        c=self.canv;w=self.width;h=self.height
        c.setStrokeColor(LINE);c.setFillColor(colors.white);c.roundRect(0,0,w,h,7,stroke=1,fill=1)
        titles={'pareto':'Concentración del pendiente por parcela','coverage':'Cobertura financiera por concepto','bridge':'Puente de conciliación de la cartera','trend':'Histórico mensual de abonos de caja','cash':'Flujo mensual y caja acumulada del condominio'}
        c.setFillColor(INK);c.setFont('Helvetica-Bold',10);c.drawString(13,h-22,titles[self.kind])
        if self.kind=='coverage':return self.coverage(c,w,h)
        if self.kind=='pareto':return self.pareto(c,w,h)
        if self.kind=='bridge':return self.bridge(c,w,h)
        if self.kind=='trend':return self.trend(c,w,h)
        if self.kind=='cash':return self.cash(c,w,h)
    def axes(self,c,x,y,w,h,lo,hi,right=False):
        for i in range(5):
            value=lo+(hi-lo)*i/4;yy=y+h*i/4
            c.setStrokeColor(LINE);c.setLineWidth(.4);c.line(x,yy,x+w,yy)
            c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawRightString(x-6,yy-2,compact(value))
        c.setStrokeColor(colors.HexColor('#bccfdd'));c.line(x,y,x+w,y)
    def legend(self,c,items,y):
        x=14
        for label,color in items:
            c.setFillColor(color);c.rect(x,y-2,7,7,stroke=0,fill=1);c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawString(x+11,y,label);x+=c.stringWidth(label,'Helvetica',7)+28
    def coverage(self,c,w,h):
        self.legend(c,[('Cubierto',GREEN),('Sin cubrir',ORANGE)],h-42)
        concepts=self.m['concepts'];concepts=concepts[concepts['Base de cobertura'].gt(0)]
        left=100;plotw=w-left-42
        if concepts.empty:return
        gap=min(60,(h-100)/max(1,len(concepts)));top=h-89
        for i,(_,r) in enumerate(concepts.iterrows()):
            y=top-i*gap;covered=float(r.Cobertura);uncovered=1-covered
            c.setFillColor(MUTED);c.setFont('Helvetica',8);c.drawRightString(left-8,y+7,str(r.Concepto))
            c.setFillColor(GREEN);c.rect(left,y,plotw*covered,23,stroke=0,fill=1)
            c.setFillColor(ORANGE);c.rect(left+plotw*covered,y,plotw*uncovered,23,stroke=0,fill=1)
            c.setFont('Helvetica-Bold',8)
            if covered>.12:c.setFillColor(colors.white);c.drawCentredString(left+plotw*covered/2,y+8,percent(covered))
            if uncovered>0:
                c.setFillColor(colors.HexColor('#81481d'))
                if uncovered>=.12:c.drawCentredString(left+plotw*(covered+uncovered/2),y+8,percent(uncovered))
                else:c.drawString(left+plotw+4,y+8,percent(uncovered))
        for n in [0,25,50,75,100]:
            c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawCentredString(left+plotw*n/100,22,str(n)+'%')
    def pareto(self,c,w,h):
        self.legend(c,[('Pendiente',NAVY),('Acumulado',PURPLE)],h-42)
        rank=self.m['ranked'];x,y,pw,ph=45,40,w-82,h-105
        if rank.empty:
            c.setFillColor(MUTED);c.setFont('Helvetica',9);c.drawString(20,h/2,'Sin saldos pendientes en la selección.');return
        maximum=float(rank.Pendiente.max())*1.15
        self.axes(c,x,y,pw,ph,0,maximum);step=pw/len(rank);points=[]
        for i,(_,r) in enumerate(rank.iterrows()):
            xx=x+(i+.5)*step
            c.setFillColor(NAVY if i<3 else colors.HexColor('#81b9c6'));c.rect(xx-step*.3,y,step*.6,ph*r.Pendiente/maximum,stroke=0,fill=1)
            c.setFillColor(MUTED);c.setFont('Helvetica',6.5);c.drawCentredString(xx,24,'P'+str(int(r.Parcela)))
            points.append((xx,y+ph*r.Acumulado))
        c.setStrokeColor(PURPLE);c.setLineWidth(1.5)
        for a,b in zip(points,points[1:]):c.line(*a,*b)
        for xx,yy in points:c.setFillColor(PURPLE);c.circle(xx,yy,1.8,stroke=0,fill=1)
        for n in [0,50,100]:c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawString(x+pw+5,y+ph*n/100-2,str(n)+'%')
    def bridge(self,c,w,h):
        m=self.m;x,y,pw,ph=48,42,w-65,h-96
        scale=max(abs(m[k]) for k in ['charges','paid','credit','due']) or 1
        lo=min(0,m['charges']-m['paid'])-.05*scale;hi=max(m['charges'],m['due'],0)+.13*scale
        self.axes(c,x,y,pw,ph,lo,hi)
        yy=lambda v:y+ph*(v-lo)/(hi-lo)
        before=0;steps=[('Cargos netos',0,m['charges'],NAVY),('Abonos',m['charges'],m['charges']-m['paid'],colors.HexColor('#67aaa2')),('Créditos',m['charges']-m['paid'],m['due'],PURPLE),('Por cobrar',0,m['due'],NAVY)]
        bw=pw/4*.64
        for i,(label,a,b,color) in enumerate(steps):
            xx=x+(i+.5)*pw/4;c.setFillColor(color);c.rect(xx-bw/2,min(yy(a),yy(b)),bw,max(.6,abs(yy(b)-yy(a))),stroke=0,fill=1)
            c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawCentredString(xx,25,label)
            amount=[m['charges'],m['paid'],m['credit'],m['due']][i]
            c.setFont('Helvetica-Bold',7)
            if i==1:c.setFillColor(colors.white);c.drawCentredString(xx,(yy(a)+yy(b))/2,'−'+money(amount))
            else:c.setFillColor(INK);c.drawCentredString(xx,max(yy(a),yy(b))+8,('+' if i==2 else '')+money(amount))
    def trend(self,c,w,h):
        self.legend(c,[('Abonos de caja',colors.HexColor('#3b83ad')),('Promedio 3 meses',PURPLE)],h-42)
        d=self.data;x,y,pw,ph=48,40,w-65,h-100
        if d.empty:return
        maximum=max(d.Abonos.max(),d['Promedio móvil 3 meses'].max(),1)*1.15
        self.axes(c,x,y,pw,ph,0,maximum);step=pw/len(d);points=[]
        every=max(1,ceil(len(d)/6))
        for i,(_,r) in enumerate(d.iterrows()):
            xx=x+(i+.5)*step;c.setFillColor(colors.HexColor('#3b83ad'));c.rect(xx-step*.33,y,step*.66,ph*r.Abonos/maximum,stroke=0,fill=1)
            points.append((xx,y+ph*r['Promedio móvil 3 meses']/maximum))
            if i%every==0:c.setFillColor(MUTED);c.setFont('Helvetica',6.5);c.drawCentredString(xx,24,month_label(r.Mes))
        c.setStrokeColor(PURPLE);c.setLineWidth(1.8)
        for a,b in zip(points,points[1:]):c.line(*a,*b)
    def cash(self,c,w,h):
        self.legend(c,[('Ingresos',GREEN),('Reintegros',colors.HexColor('#88c9bc')),('Pagado',colors.HexColor('#d69468')),('No pagado',colors.HexColor('#aa7bc4')),('Caja pagada',NAVY),('Caja tras pendientes',PURPLE)],h-42)
        d=self.data;x,y,pw,ph=53,38,w-106,h-98
        if d.empty:return
        lo=-max((d.Egresos+d.Pendientes).max(),1)*1.15;hi=max((d.Ingresos+d.Reintegros).max(),1)*1.15
        self.axes(c,x,y,pw,ph,lo,hi)
        yy=lambda v:y+ph*(v-lo)/(hi-lo)
        balance_min=float(min(d.Cierre.min(),d['Cierre tras pendientes'].min()));balance_max=float(max(d.Cierre.max(),d['Cierre tras pendientes'].max()))
        if balance_max==balance_min:balance_max=balance_min+max(abs(balance_min)*.1,1)
        balance_span=balance_max-balance_min;balance_max+=balance_span*.12;balance_min-=balance_span*.04
        cy=lambda v:y+ph*(v-balance_min)/(balance_max-balance_min)
        step=pw/len(d);points=[];scenario=[];every=max(1,ceil(len(d)/12))
        for i,(_,r) in enumerate(d.iterrows()):
            xx=x+(i+.5)*step;bw=step*.62
            for a,b,color in [(0,r.Ingresos,GREEN),(r.Ingresos,r.Ingresos+r.Reintegros,colors.HexColor('#88c9bc')),(0,-r.Egresos,colors.HexColor('#d69468')),(-r.Egresos,-r.Egresos-r.Pendientes,colors.HexColor('#aa7bc4'))]:
                c.setFillColor(color);c.rect(xx-bw/2,min(yy(a),yy(b)),bw,abs(yy(b)-yy(a)),stroke=0,fill=1)
            points.append((xx,cy(r.Cierre)));scenario.append((xx,cy(r['Cierre tras pendientes'])))
            if i%every==0:c.setFillColor(MUTED);c.setFont('Helvetica',6.5);c.drawCentredString(xx,22,month_label(r.Mes))
        c.setStrokeColor(NAVY);c.setLineWidth(2)
        for a,b in zip(points,points[1:]):c.line(*a,*b)
        for xx,yy_ in points:c.setFillColor(NAVY);c.circle(xx,yy_,2,stroke=0,fill=1)
        if d['Reserva acumulada'].gt(0).any():
            c.setStrokeColor(PURPLE);c.setLineWidth(1.7);c.setDash(4,2)
            for a,b in zip(scenario,scenario[1:]):c.line(*a,*b)
            c.setDash()
            for xx,yy_ in scenario:c.setFillColor(PURPLE);c.circle(xx,yy_,1.8,stroke=0,fill=1)
        for i in range(5):
            value=balance_min+(balance_max-balance_min)*i/4;c.setFont('Helvetica',7);c.setFillColor(MUTED);c.drawString(x+pw+6,y+ph*i/4-2,compact(value))
        c.setFont('Helvetica',6.5);c.drawString(x,y+ph+8,'Flujos mensuales · CLP');c.drawRightString(x+pw,y+ph+8,'Caja pagada / escenario · CLP')


def build_dashboard_pdf(ledger,payments,snapshot,filters,order='Mayor pendiente',cash_length=12,trend_length=12):
    if ledger.empty:raise ValueError('No hay obligaciones para exportar.')
    m=analyze_portfolio(ledger,payments)
    positions=m['positions']
    sort={'Número de parcela':(['Parcela'],[True]),'Menor cobertura':(['Cobertura','Pendiente'],[True,False]),'Mayor saldo a favor':(['A favor','Parcela'],[False,True])}.get(order,(['Pendiente','Parcela'],[False,True]))
    positions=positions.sort_values(sort[0],ascending=sort[1])
    flow=cash_flow(snapshot) if snapshot is not None else None
    window=cash_window(flow,cash_length) if flow is not None else None
    trend=m['monthly'].tail(trend_length) if trend_length else m['monthly']
    now=datetime.now(ZoneInfo('America/Santiago'))
    headlines=dashboard_headlines(snapshot,m,as_of=now.date())
    global_money=lambda value:money(value) if value is not None else '—'
    revision=snapshot.revision if snapshot else 'Sin corte';captured=snapshot.captured_at if snapshot else now.strftime('%d/%m/%Y %H:%M')
    styles={'title':ParagraphStyle('d-title',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=NAVY,spaceAfter=8),'h2':ParagraphStyle('d-h2',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=INK,spaceAfter=8),'body':ParagraphStyle('d-body',fontName='Helvetica',fontSize=8.5,leading=12,textColor=MUTED,spaceAfter=7),'small':ParagraphStyle('d-small',fontName='Helvetica',fontSize=7,leading=10,textColor=MUTED,spaceAfter=6),'cell':ParagraphStyle('d-cell',fontName='Helvetica',fontSize=7.2,leading=9.3,textColor=INK),'head':ParagraphStyle('d-head',fontName='Helvetica-Bold',fontSize=7,leading=9,textColor=colors.white)}
    p=lambda text,kind='body':Paragraph(text,styles[kind])
    def table(headers,rows,widths):
        t=Table([[p(escape(str(v)),'head') for v in headers]]+[[p(escape(str(v)),'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fb')]),('LINEBELOW',(0,0),(-1,-1),.3,LINE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));return t
    from condominio_data import PUBLICATION_URL
    from control_pdf_design import CommunityCover,ExposureSummary,panorama_table
    story=[CommunityCover(m,headlines,filters,now,W,PUBLICATION_URL if snapshot else None)]
    half=(W-14)/2
    # A fixed-size readable block per page preserves all panorama columns and rows.
    for start in range(0,len(positions),12):
        block=positions.iloc[start:start+12]
        story += [PageBreak(),p('01 · Panorama financiero por parcela','title'),p(f'Orden: {escape(order)} · Registros {start+1}–{start+len(block)} de {len(positions)}. Los créditos no se compensan con el pendiente.','small')]
        final=start+len(block)==len(positions)
        t=panorama_table(block,positions,m,W,final)
        story += [t,Spacer(1,8),p('La columna roja es el total por pagar de cada parcela. — indica sin saldo. Los créditos a favor no se compensan con la deuda. Montos en CLP.','small')]
    if window is not None:
        d=window['data'];period=f'{d.Mes.iloc[0]} a {d.Mes.iloc[-1]}' if len(d) else 'Sin movimientos fechados'
        story += [PageBreak(),p('02 · Flujo de caja del condominio','title'),p('ALCANCE GLOBAL · Costos Pagado y escenario No pagado según fecha registrada. No aplica filtros de cobranza. Ventana: '+period,'small'),MetricGrid([('Caja de apertura',money(window['opening']),'Saldo anterior a la ventana'),('Ingresos recibidos',money(window['income']),'Libro Ingresos · ventana de caja'),('Egresos pagados',money(window['expenses']),'Solo Estado Pagado'),('Proveedores por pagar',money(window['pending']),'Costos · Estado No pagado'),('Caja pagada al cierre',money(window['closing']),'Solo movimientos realizados'),('Caja al pagar pendientes',money(window['adjusted_closing']),'Escenario: caja pagada menos compromisos')],columns=3),Spacer(1,8),Chart('cash',data=d,height=245),Spacer(1,9),p(f'Flujo pagado: {money(window["net"])} · Reintegros: {money(window["refunds"])} · Pendientes en la ventana: {money(window["pending_window"])}.','small'),p('Caja pagada = apertura + ingresos + reintegros − egresos Pagado. Escenario = caja pagada − Costos No pagado acumulados según fecha registrada. No se suma el auxiliar Deudas proveedores. Referencia inicial cero; no es cartola bancaria certificada.','small')]
        if flow.get('pending_undated',0):story.append(p(f'Pendientes sin fecha: {money(flow["pending_undated"])}; se incluyen solo en el último punto del escenario sin asignar una fecha real de pago.','small'))
        if flow['undated_count']:story.append(p(f'{flow["undated_count"]} movimientos con monto y sin fecha quedan fuera del gráfico; su efecto neto es {money(flow["undated_net"])}.','small'))
    story += [PageBreak(),p('03 · Conciliación y comportamiento de abonos','title'),p('Cartera filtrada: relación entre cargos, abonos y créditos; histórico de abonos recibidos.')]
    pair=Table([[Chart('bridge',m,width=half),Chart('trend',data=trend,width=half)]],colWidths=[W/2,W/2]);pair.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story += [pair,Spacer(1,12),p(f'<b>Control de saldo:</b> {money(m["charges"])} − {money(m["paid"])} + {money(m["credit"])} = {money(m["due"])}.','body')]
    if len(trend):story.append(p(f'<b>Histórico seleccionado:</b> {trend.Mes.iloc[0]} a {trend.Mes.iloc[-1]} · {money(trend.Abonos.sum())} recibidos. Meses sin abonos: cero. Promedio móvil: tres meses; al inicio usa meses disponibles.','body'))
    story.append(p('Las ventanas temporales cambian únicamente sus gráficos; los KPI y la cartera mantienen los filtros principales. El histórico de abonos no reconstruye vencimientos ni antigüedad de deuda.','small'))
    story += [PageBreak(),ExposureSummary(m,W)]
    buf=BytesIO();doc=SimpleDocTemplate(buf,pagesize=PAGE,leftMargin=36,rightMargin=36,topMargin=43,bottomMargin=36,title='Dashboard financiero · Control de obligaciones',author='Administración · Los Queltehues III')
    def decorate(c,doc):
        c.saveState();c.setFont('Helvetica-Bold',7);c.setFillColor(NAVY);c.drawString(36,PAGE[1]-22,'LOS QUELTEHUES III · CONTROL FINANCIERO');c.setStrokeColor(LINE);c.line(36,PAGE[1]-30,PAGE[0]-36,PAGE[1]-30);c.line(36,28,PAGE[0]-36,28);c.setFont('Helvetica',7);c.setFillColor(MUTED);c.drawString(36,17,f'Corte {revision} · Emisión {now:%d/%m/%Y} · CLP');c.drawRightString(PAGE[0]-36,17,f'Página {doc.page}');c.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    return buf.getvalue()
