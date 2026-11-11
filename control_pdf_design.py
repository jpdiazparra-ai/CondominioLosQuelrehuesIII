"""Diseño editorial del informe a la comunidad y lectura de cuentas por parcela."""
from html import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable,Paragraph,Table,TableStyle
from control_finance import money,percent,dashboard_kpis

NAVY=colors.HexColor('#153e56')
INK=colors.HexColor('#264a62')
MUTED=colors.HexColor('#71869a')
LINE=colors.HexColor('#d8e4ec')
RED=colors.HexColor('#b7474d')
GREEN=colors.HexColor('#237963')
SOFT=colors.HexColor('#f3f7fa')


class CommunityCover(Flowable):
    """Primera lectura: liquidez disponible, compromisos y situación de cobro."""
    def __init__(self,m,headlines,filters,issued,width,source_url=None):
        super().__init__();self.m=m;self.headlines=headlines;self.filters=filters;self.issued=issued;self.source_url=source_url;self.width=width;self.height=480
    def draw(self):
        c=self.canv;m=self.m;h=self.headlines;H=self.height;W=self.width
        fmt=lambda v:money(v) if v is not None else '—'
        c.setFillColor(MUTED);c.setFont('Helvetica-Bold',7);c.drawString(0,H-10,'INFORME A LA COMUNIDAD / LOS QUELTEHUES III')
        c.setFillColor(NAVY);c.setFont('Helvetica-Bold',29);c.drawString(0,H-45,'Tesorería y cobranza')
        c.setFillColor(MUTED);c.setFont('Helvetica',10);c.drawString(0,H-64,'Estado de la caja, compromisos de pago y cuentas de las parcelas')
        c.setFillColor(SOFT);c.roundRect(W-128,H-58,128,52,6,stroke=0,fill=1)
        c.setFillColor(MUTED);c.setFont('Helvetica-Bold',6.6);c.drawString(W-116,H-22,'CORTE DE INFORMACIÓN')
        months=['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
        c.setFillColor(NAVY);c.setFont('Helvetica-Bold',16);c.drawString(W-116,H-44,f'{self.issued.day:02d} {months[self.issued.month-1]} {self.issued.year}')
        fs=ParagraphStyle('cover-filter',fontName='Helvetica',fontSize=7.2,leading=10,textColor=MUTED)
        filtered=' · '.join(escape(str(k))+': '+escape(str(v)) for k,v in self.filters.items())
        para=Paragraph(filtered,fs);_,ph=para.wrap(W,26);para.drawOn(c,0,H-91-ph+10)
        gap=10;cw=(W-3*gap)/4;y=H-207
        cells=[('Caja al día de hoy',fmt(h['cash_today']),f'Pagado al {h["as_of"]:%d/%m/%Y}',NAVY,colors.white),('Parcelas con pendientes',f'{m["debtors"]} / {m["count"]}',percent(m['debtor_share'])+' de la selección',SOFT,NAVY),('Cuentas por cobrar',money(m['due']),'Pendientes de las parcelas seleccionadas',colors.HexColor('#fff1f1'),RED),('Caja tras pagar proveedores',fmt(h['cash_after_payables']),'Caja actual menos pagos pendientes',colors.HexColor('#eaf5ef') if h['cash_after_payables'] is not None and h['cash_after_payables']>=0 else colors.HexColor('#fff1f1'),GREEN if h['cash_after_payables'] is not None and h['cash_after_payables']>=0 else RED)]
        for i,(label,value,note,bg,tone) in enumerate(cells):
            x=i*(cw+gap);c.setFillColor(bg);c.setStrokeColor(LINE);c.roundRect(x,y,cw,91,7,fill=1,stroke=0 if i==0 else 1)
            c.setFillColor(colors.HexColor('#b9d4e1') if i==0 else MUTED);c.setFont('Helvetica-Bold',7.5);c.drawString(x+12,y+72,label)
            c.setFillColor(tone);c.setFont('Helvetica-Bold',25);c.drawString(x+12,y+37,value)
            c.setFillColor(colors.HexColor('#b9d4e1') if i==0 else MUTED);c.setFont('Helvetica',6.5);c.drawString(x+12,y+16,note)
        panelw=(W-16)/2;py=H-378;panelh=153
        for x in [0,panelw+16]:
            c.setFillColor(colors.white);c.setStrokeColor(LINE);c.roundRect(x,py,panelw,panelh,7,stroke=1,fill=1)
        c.setFillColor(NAVY);c.setFont('Helvetica-Bold',12);c.drawString(14,py+130,'Cómo se distribuye la caja')
        for label,value,yy,tone in [('Caja actual',fmt(h['cash_today']),py+106,INK),('Pagos comprometidos a proveedores','−'+fmt(h['payables']),py+84,RED)]:
            c.setFillColor(MUTED);c.setFont('Helvetica',8.5);c.drawString(14,yy,label);c.setFillColor(tone);c.setFont('Helvetica-Bold',11);c.drawRightString(panelw-14,yy,value)
        c.setStrokeColor(LINE);c.line(14,py+72,panelw-14,py+72)
        c.setFillColor(GREEN if h['cash_after_payables'] is not None and h['cash_after_payables']>=0 else RED);c.setFont('Helvetica-Bold',9);c.drawString(14,py+52,'Saldo después de pagar');c.setFont('Helvetica-Bold',18);c.drawRightString(panelw-14,py+49,fmt(h['cash_after_payables']))
        cash=h['cash_today'];payables=h['payables']
        if cash is not None and cash>0 and payables is not None:
            reserved=min(1,max(0,payables/cash));barw=panelw-28
            c.setFillColor(RED);c.rect(14,py+25,barw*reserved,8,stroke=0,fill=1)
            c.setFillColor(GREEN);c.rect(14+barw*reserved,py+25,barw*(1-reserved),8,stroke=0,fill=1)
            c.setFont('Helvetica',6.6);c.setFillColor(MUTED);c.drawString(14,py+10,percent(payables/cash)+' de la caja comprometida')
            remaining='Brecha: '+money(payables-cash) if payables>cash else percent((cash-payables)/cash)+' disponible'
            c.drawRightString(panelw-14,py+10,remaining)
        rx=panelw+16
        c.setFillColor(NAVY);c.setFont('Helvetica-Bold',12);c.drawString(rx+14,py+130,'Situación de cobranza')
        c.setFillColor(MUTED);c.setFont('Helvetica',8);c.drawString(rx+14,py+108,'Cobertura de cargos');c.drawString(rx+panelw-120,py+108,'Por cobrar')
        c.setFillColor(GREEN);c.setFont('Helvetica-Bold',28);c.drawString(rx+14,py+78,percent(m['coverage']) if m['base'] else '—')
        c.setFillColor(RED);c.setFont('Helvetica-Bold',24);c.drawString(rx+panelw-120,py+79,percent(m['uncovered']) if m['base'] else '—')
        c.setFillColor(colors.HexColor('#f6e2e2'));c.roundRect(rx+14,py+61,panelw-28,7,3,stroke=0,fill=1)
        c.setFillColor(GREEN);c.rect(rx+14,py+61,(panelw-28)*min(1,max(0,m['coverage'])),7,stroke=0,fill=1)
        c.setFillColor(INK);c.setFont('Helvetica-Bold',9);c.drawString(rx+14,py+40,'Pendiente de recaudar: '+money(m['due']))
        c.setFillColor(MUTED);c.setFont('Helvetica',7);c.drawString(rx+14,py+24,'Los abonos se aplican a la obligación que corresponde.');c.drawString(rx+14,py+11,'Los créditos se muestran separados de los saldos por pagar.')
        # Ocho KPI secundarios, en dos filas compactas con el mismo orden de pantalla.
        for i,(label,value,note,tone,icon) in enumerate(dashboard_kpis(m,h)):
            x=(i%4)*(W+8)/4;y=12+(1-i//4)*43;cw=(W-24)/4
            c.setFillColor(SOFT);c.roundRect(x,y,cw,39,4,stroke=0,fill=1)
            c.setFillColor(MUTED);c.setFont('Helvetica',6.7);c.drawString(x+8,y+29,label)
            c.setFillColor(colors.HexColor(tone));c.setFont('Helvetica-Bold',11.2);c.drawString(x+8,y+15,value)
            c.setFillColor(MUTED);c.setFont('Helvetica',5.8);c.drawString(x+8,y+5,note)
        c.setFillColor(MUTED);c.setFont('Helvetica',6.4);c.drawString(0,0,'Caja, ingresos, egresos y proveedores: globales. Cobranza: según filtros. El saldo no suma cobros futuros.')
        if self.source_url:
            c.setFillColor(GREEN);c.drawRightString(W,0,'Consultar fuente publicada');c.linkURL(self.source_url,(W-105,-2,W,9),relative=1,thickness=0)


def panorama_table(block,positions,metrics,width,final):
    """El total por pagar queda junto a la parcela, en rojo y con columna propia."""
    cell=ParagraphStyle('account-cell',fontName='Helvetica',fontSize=7.4,leading=10,textColor=INK)
    head=ParagraphStyle('account-head',fontName='Helvetica-Bold',fontSize=7,leading=9,textColor=colors.white)
    number=ParagraphStyle('account-number',parent=cell,alignment=TA_RIGHT)
    due=ParagraphStyle('account-due',parent=number,fontName='Helvetica-Bold',fontSize=10,textColor=RED)
    credit=ParagraphStyle('account-credit',parent=number,textColor=colors.HexColor('#387db6'))
    zero=ParagraphStyle('account-zero',parent=number,textColor=colors.HexColor('#a0afbc'))
    owner=ParagraphStyle('account-owner',parent=cell,fontSize=7.8,leading=10.5)
    h1=['Parcela / propietario','TOTAL POR PAGAR','Desglose de lo pendiente','','','Cuenta publicada','','','Seguimiento','','','Situación']
    h2=['','','Gastos comunes','Cámaras','Mantención','Cargo neto','Abonos','A favor','Cobertura','Peso deuda','Últ. ingreso','']
    data=[[Paragraph(v,head) for v in h1],[Paragraph(v,head) for v in h2]]
    for _,r in block.iterrows():
        last=r['Último ingreso'].strftime('%d/%m/%Y') if pd.notna(r['Último ingreso']) else 'Sin fecha'
        row=[Paragraph(f'<font size="10"><b>{int(r.Parcela)}</b></font><br/>'+escape(str(r.Propietario)),owner),Paragraph(money(r.Pendiente) if r.Pendiente else '—',due if r.Pendiente else zero)]
        for field in ['Gastos comunes','Proyecto','Mantención','Cargo','Abonado','A favor']:
            row.append(Paragraph(money(r[field]) if r[field] else '—',credit if field=='A favor' and r[field] else number if r[field] else zero))
        row += [Paragraph(percent(r.Cobertura),number),Paragraph(percent(r.Participación),number),Paragraph(last,cell),Paragraph(escape(r.Situación),cell)]
        data.append(row)
    if final:
        m=metrics;sums=[m['due']]+[positions[c].sum() for c in ['Gastos comunes','Proyecto','Mantención']]+[m['charges'],m['paid'],m['credit']]
        data.append([Paragraph('<b>TOTAL</b>',cell)]+[Paragraph(money(v),due if i==0 else number) for i,v in enumerate(sums)]+[Paragraph(percent(m['coverage']),number),Paragraph('100%' if m['due'] else '0%',number),Paragraph('',cell),Paragraph(str(m['count'])+' parcelas',cell)])
    widths=[128,80,62,58,62,62,64,58,48,40,57,51];widths=[v*width/sum(widths) for v in widths]
    t=Table(data,colWidths=widths,repeatRows=2)
    commands=[('BACKGROUND',(0,0),(-1,1),NAVY),('BACKGROUND',(1,0),(1,1),RED),('SPAN',(0,0),(0,1)),('SPAN',(1,0),(1,1)),('SPAN',(2,0),(4,0)),('SPAN',(5,0),(7,0)),('SPAN',(8,0),(10,0)),('SPAN',(11,0),(11,1)),('ROWBACKGROUNDS',(0,2),(-1,-1),[colors.white,colors.HexColor('#f2f6fa')]),('BACKGROUND',(1,2),(1,-1),colors.HexColor('#fff0f1')),('LINEBELOW',(0,1),(-1,-1),.55,colors.HexColor('#d6e1e9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5.5),('BOTTOMPADDING',(0,0),(-1,-1),5.5)]
    for col in [0,4,7,10]:commands.append(('LINEAFTER',(col,0),(col,-1),.6,colors.HexColor('#cedce6')))
    commands.append(('LINEAFTER',(1,0),(1,-1),.8,colors.HexColor('#e5b3b8')))
    if final:commands += [('BACKGROUND',(0,-1),(-1,-1),colors.HexColor('#e7f1f5')),('BACKGROUND',(1,-1),(1,-1),colors.HexColor('#f8dfe3')),('LINEABOVE',(0,-1),(-1,-1),1,NAVY)]
    t.setStyle(TableStyle(commands))
    return t


class ExposureSummary(Flowable):
    """Cierre analítico: concentración y cobertura de la cartera seleccionada."""
    def __init__(self,m,width):
        super().__init__();self.m=m;self.width=width;self.height=480
    def draw(self):
        c=self.canv;m=self.m;W=self.width
        def text(x,y,value,size=8,tone=INK,bold=False,right=False):
            c.setFillColor(tone);c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
            (c.drawRightString if right else c.drawString)(x,y,value)
        text(0,470,'04 / ANÁLISIS DE LA CARTERA SELECCIONADA',7,MUTED,True)
        text(0,438,'Exposición y concentración',26,NAVY,True)
        text(0,418,'Dónde se concentra lo pendiente y qué conceptos requieren seguimiento.',9,MUTED)
        gap=12;cw=(W-2*gap)/3
        cards=[('TOTAL PENDIENTE',money(m['due']),str(m['debtors'])+' parcelas con saldo por cobrar',RED),('TRES MAYORES SALDOS',percent(m['top3']),money(m['ranked'].head(3).Pendiente.sum())+' de la cartera pendiente',NAVY),('CINCO MAYORES SALDOS',percent(m['top5']),money(m['ranked'].head(5).Pendiente.sum())+' de la cartera pendiente',NAVY)]
        for i,(label,value,note,tone) in enumerate(cards):
            x=i*(cw+gap);c.setFillColor(colors.HexColor('#fff1f1') if i==0 else SOFT);c.roundRect(x,320,cw,77,6,stroke=0,fill=1)
            text(x+13,379,label,7,MUTED,True);text(x+13,350,value,24,tone,True);text(x+13,332,note,7,MUTED)
        left=W*.52;right=W-left-16;rx=left+16
        for x,width in [(0,left),(rx,right)]:
            c.setFillColor(colors.white);c.setStrokeColor(LINE);c.roundRect(x,35,width,270,6,stroke=1,fill=1)
        text(14,284,'Distribución por parcela',12,NAVY,True)
        text(14,268,'Monto pendiente / participación en la selección',7,MUTED)
        rank=m['ranked'];entries=[('Parcela '+str(int(r.Parcela)),float(r.Pendiente)) for _,r in rank.head(8).iterrows()]
        if len(rank)>8:entries.append(('Otras '+str(len(rank)-8)+' parcelas',float(rank.iloc[8:].Pendiente.sum())))
        maximum=max([v for _,v in entries],default=1) or 1
        if not entries:text(14,230,'Sin saldos pendientes en esta selección.',9,GREEN)
        for i,(label,amount) in enumerate(entries):
            y=242-i*21
            if i%2==0:c.setFillColor(SOFT);c.rect(8,y-7,left-16,21,fill=1,stroke=0)
            text(14,y,label,7.5,INK,i<3)
            bx=104;bw=left-218
            c.setFillColor(colors.HexColor('#f6e2e2'));c.roundRect(bx,y-1,bw,6,2,fill=1,stroke=0)
            c.setFillColor(RED if i<3 else colors.HexColor('#91aabd'));c.rect(bx,y-1,bw*amount/maximum,6,fill=1,stroke=0)
            text(left-48,y,money(amount),8,RED,True,True)
            text(left-12,y,percent(amount/m['due']) if m['due'] else '—',7,MUTED,right=True)
        text(rx+14,284,'Cobertura por concepto',12,NAVY,True)
        text(rx+14,268,'Abonos aplicados / cargos positivos',7,MUTED)
        concepts=m['concepts']
        for i,(_,r) in enumerate(concepts.iterrows()):
            y=241-i*62
            text(rx+14,y,str(r.Concepto),9,INK,True)
            text(W-14,y,money(r.Pendiente)+' por cobrar',8,RED,True,True)
            base=float(r['Base de cobertura']);covered=min(1,max(0,float(r.Cobertura))) if base else 0
            bx=rx+14;bw=right-28
            c.setFillColor(colors.HexColor('#e7a16a'));c.roundRect(bx,y-22,bw,10,3,fill=1,stroke=0)
            if covered:c.setFillColor(GREEN);c.rect(bx,y-22,bw*covered,10,fill=1,stroke=0)
            text(bx,y-36,'Cubierto: '+(percent(covered) if base else '—'),7,GREEN,True)
            text(W-14,y-36,'Sin cubrir: '+(percent(1-covered) if base else '—'),7,RED,True,True)
        text(rx+14,48,'Verde: aplicado a cargos. Naranja: sin cubrir.',7,MUTED)
        text(0,17,'Lectura de cobranza: importes de la selección activa. Los créditos a favor se mantienen separados.',7,MUTED)
        text(0,3,'La concentración describe el saldo actual; no representa antigüedad ni vencimiento de las obligaciones.',7,MUTED)
