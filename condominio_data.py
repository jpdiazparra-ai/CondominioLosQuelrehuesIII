"""Fuente única, adaptación de las ocho hojas y conciliación del condominio."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import StringIO
from zoneinfo import ZoneInfo
import re
import unicodedata

import pandas as pd

PUBLICATION_ID = '2PACX-1vQrn8w2pA3tkwgrRzyT2DxUHhDMQeECSSFnvjR-ve2x1l-Oiv1iYBPq4azmY-F7iQ'
PUBLICATION_URL = f'https://docs.google.com/spreadsheets/d/e/{PUBLICATION_ID}/pub?output=csv'
SHEETS = {
    'resumen': ('Resumen', '1279265217'),
    'ingresos': ('Ingresos', '1555162048'),
    'obligaciones': ('Cobros GC', '498984169'),
    'proyectos': ('Proyectos', '965577812'),
    'mantencion': ('Mantención parcelas', '1147757578'),
    'costos': ('Costos', '1064134694'),
    'proveedores': ('Deudas proveedores', '707622201'),
    'control': ('Origen y control', '2089994970'),
}
URLS = {key: f'{PUBLICATION_URL}&single=true&gid={gid}' for key, (_, gid) in SHEETS.items()}
# Propietarios no es una hoja operativa de la nueva publicación.
URLS['propietarios'] = URLS['ingresos']


def normalize(value):
    value = unicodedata.normalize('NFKD', str(value).strip()).encode('ascii','ignore').decode().lower()
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+','_',value)).strip('_')


def parse_money(series):
    def parse(value):
        if pd.isna(value): return float('nan')
        if isinstance(value, (int,float)): return float(value)
        text = str(value).strip()
        if text in ('','-','—','–'): return float('nan')
        negative = text.startswith('(') and text.endswith(')')
        text = re.sub(r'[^0-9,\.\-]','',text)
        if ',' in text:
            text = text.replace('.','').replace(',','.')
        elif text.count('.') > 1 or re.fullmatch(r'-?\d{1,3}(\.\d{3})+',text):
            text = text.replace('.','')
        try:
            number=float(text)
            return -abs(number) if negative else number
        except ValueError: return float('nan')
    return series.map(parse).astype(float)


def classify_costs(frame):
    """Estado controla realización; No pagado se mantiene como compromiso separado."""
    result=frame.copy()
    source=next((c for c in ['estado','estado_de_pago','estado_pago'] if c in result),None)
    if source is None:
        result['estado_pago']='Pagado'
    else:
        states=result[source].map(normalize)
        mapping={'pagado':'Pagado','pagada':'Pagado','no_pagado':'No pagado','no_pagada':'No pagado'}
        result['estado_pago']=states.map(mapping).fillna('Sin clasificar')
    amount=parse_money(result['monto'])
    result['monto_pagado']=amount.where(result.estado_pago.eq('Pagado'),0.)
    result['por_pagar']=(-amount).clip(lower=0).where(result.estado_pago.eq('No pagado'),0.)
    return result


def extract_table(raw, required, aliases=None, ids=False):
    header = None
    for idx,row in raw.iterrows():
        names = [normalize(v) if pd.notna(v) else '' for v in row]
        if set(required).issubset(names):
            header=idx; break
    if header is None:
        raise ValueError('No se encontró encabezado con columnas: '+', '.join(required))
    seen={}; columns=[]
    for name in names:
        n=seen.get(name,0);seen[name]=n+1
        columns.append(name if not n else f'{name}_{n}')
    frame=raw.iloc[header+1:].copy();frame.columns=columns
    frame=frame.loc[:,[c for c in frame if c and not c.startswith('_')]]
    frame['fila_publicada']=frame.index+1
    if ids: frame=frame[frame.id.notna() & frame.id.astype(str).str.match(r'^(ING|COS|CAM|MAN|PRO)-\d+$')]
    frame=frame.dropna(how='all').reset_index(drop=True)
    return frame.rename(columns=aliases or {})


@dataclass
class Snapshot:
    frames: dict
    raw: dict
    ledger: pd.DataFrame
    payments: pd.DataFrame
    annual: pd.DataFrame
    alerts: list
    review: pd.DataFrame
    checks: pd.DataFrame
    captured_at: str
    revision: str
    totals: dict

    def legacy_balances(self):
        gc=self.ledger[self.ledger.Concepto.eq('Gastos comunes')]
        out=gc[['Parcela','Cargo','Abonado','Pendiente','A favor']].rename(columns={'Parcela':'parcela','Cargo':'gc_total','Abonado':'pagado','Pendiente':'pendiente_pos','A favor':'saldo_favor'}).copy()
        out['pendiente']=out.gc_total-out.pagado
        for concept,col in [('Mantención','Mantención'),('Proyecto','Pendiente Proyecto')]:
            balances=self.ledger[self.ledger.Concepto.eq(concept)].groupby('Parcela').Pendiente.sum()
            out[col]=out.parcela.map(balances).fillna(0)
        return out


def load_snapshot(reader=None):
    reader = reader or pd.read_csv
    raw={key:reader(url,header=None,dtype=str) for key,url in URLS.items() if key!='propietarios'}
    return build_snapshot(raw)


def build_snapshot(raw):
    f={}
    f['ingresos']=extract_table(raw['ingresos'],['id','monto_recibido','clasificacion'],{'monto_recibido':'abono','clasificacion':'cc','medio_de_pago':'medio'},True)
    f['costos']=extract_table(raw['costos'],['id','monto_firmado','centro_de_costo'],{'fecha_de_pago':'fecha','monto_firmado':'monto','centro_de_costo':'cc','subcentro':'ccc','detalle_del_trabajo':'detalle','medio_de_pago':'medio'},True)
    has_cost_status=any(c in f['costos'] for c in ['estado','estado_de_pago','estado_pago'])
    f['costos']=classify_costs(f['costos'])
    f['tramos']=extract_table(raw['obligaciones'],['ano','tramo','total_del_tramo'])
    f['tramos']=f['tramos'][pd.to_numeric(f['tramos'].ano,errors='coerce').notna()].copy()
    tramos=f['tramos'];tramos['cargo']=parse_money(tramos.total_del_tramo)
    # A year can contain several legitimate tariff segments.
    annual=tramos.groupby('ano',as_index=False).cargo.sum(min_count=1).rename(columns={'cargo':'gc'})
    f['obligaciones']=annual
    f['mantencion']=extract_table(raw['mantencion'],['id','trabajo','monto_firmado'],{'monto_firmado':'monto','trabajo':'obs'},True)
    f['proyectos']=extract_table(raw['proyectos'],['id','aporte_esperado','abono_registrado'],{'aporte_esperado':'monto'},True)
    # Read only the main table: adjacent auxiliary summaries are never added.
    f['proveedores']=extract_table(raw['proveedores'],['id','presupuesto','pendiente'],{'motivo_proveedor':'motivo','abono_registrado':'abono','pendiente':'pendiente_a_proveedor'},True)
    f['control']=extract_table(raw['control'],['control','resultado_calculado','referencia_del_archivo'])
    control=f['control'];control=control[control.resultado_calculado.notna()]
    published={normalize(r.control):parse_money(pd.Series([r.resultado_calculado])).iloc[0] for _,r in control.iterrows()}
    for key in ('ingresos','costos','proyectos','mantencion'):
        f[key]['parcela_num']=pd.to_numeric(f[key].get('parcela',pd.Series(index=f[key].index,dtype=float)),errors='coerce')
    income=f['ingresos']; projects=f['proyectos']; maintenance=f['mantencion']
    income['abono_fuente'] = income.abono
    income['abono'] = income.abono.replace(r'^\s*[-–—]\s*$', '0', regex=True)
    # Names are derived only from the new source, never from the old publication.
    owners=pd.concat([income[['parcela','propietario','fecha']],projects[['parcela','propietario','fecha']]],ignore_index=True)
    owners['fecha_norm']=pd.to_datetime(owners.fecha,dayfirst=True,errors='coerce')
    owners['n_parcela']=pd.to_numeric(owners.parcela,errors='coerce')
    owners=owners[owners.n_parcela.between(17,36)&owners.propietario.notna()].sort_values('fecha_norm',na_position='first').drop_duplicates('n_parcela',keep='last')
    f['propietarios']=owners[['n_parcela','propietario']].rename(columns={'propietario':'nombre'})
    alerts=[]
    for key in ('ingresos','costos','proyectos','mantencion','proveedores'):
        if f[key].id.duplicated().any(): raise ValueError(f'{SHEETS[key][0]}: hay IDs duplicados; corrige la fuente antes de calcular.')
    if tramos[['ano','tramo']].duplicated().any():raise ValueError('Cobros GC contiene tramos duplicados.')
    if tramos.cargo.isna().any():raise ValueError('Cobros GC contiene un tramo sin monto válido.')
    calculated_tramos=parse_money(tramos.cuota_mensual_por_parcela)*pd.to_numeric(tramos.meses_por_cobrar,errors='coerce')
    if ((calculated_tramos-tramos.cargo).abs()>.01).any():alerts.append('Hay tramos GC cuyo total no coincide con meses × cuota; se utiliza el total publicado y se requiere revisión.')
    if projects.parcela_num.duplicated().any():raise ValueError('Proyecto cámaras contiene parcelas repetidas; revisa la fuente.')
    project_lookup=projects.set_index('parcela_num')
    from control_obligaciones import build_control
    # Project receipts remain in Ingresos; auxiliary collections never enter cash.
    unique_charges=parse_money(projects.monto).dropna().unique()
    configuration=pd.DataFrame([{'cc':'Proyecto','monto':0,'obs':'Cámaras','total':parse_money(projects.monto).sum()}])
    ledger,payments,annual_view,base_alerts,review=build_control(annual,income,f['propietarios'],maintenance,configuration,parse_money)
    for idx,row in ledger[ledger.Concepto.eq('Proyecto')].iterrows():
        parcel=row.Parcela
        if parcel not in project_lookup.index:
            alerts.append(f'Parcela {parcel}: sin cargo de cámaras publicado; revisar.')
            amount=0.
        else:
            amount=parse_money(pd.Series([project_lookup.loc[parcel,'monto']])).iloc[0]
            if pd.isna(amount):raise ValueError(f'Proyecto cámaras parcela {parcel}: aporte esperado sin monto.')
        paid=row.Abonado; balance=amount-paid
        ledger.loc[idx,['Cargo','Pendiente','A favor','Estado']]=[amount,max(balance,0),max(-balance,0),'Pendiente' if balance>0 else 'A favor' if balance<0 else 'Al día']
    # Mantención conserva el grano parcela + trabajo: un crédito de un trabajo
    # no oculta una deuda de otro. Los ajustes del auxiliar no son ingresos de caja.
    maintenance['monto_num'] = parse_money(maintenance.monto)
    if payments.Concepto.eq('Mantención').any():
        raise ValueError('Hay pagos de mantención en Ingresos sin vínculo al trabajo. Asigna el ajuste al trabajo en Mantención parcelas para evitar doble aplicación.')
    names=f['propietarios'].set_index('n_parcela').nombre.to_dict()
    maintenance_rows=[]
    for (parcel,work),group in maintenance.groupby(['parcela_num','obs'],dropna=False):
        if pd.isna(parcel) or parcel not in range(17,37):continue
        amount=group.monto_num.sum(min_count=1)
        if pd.isna(amount):continue
        last=pd.to_datetime(group.loc[group.monto_num.lt(0),'fecha'],dayfirst=True,errors='coerce').max()
        maintenance_rows.append({'Parcela':int(parcel),'Propietario':names.get(parcel,'Sin propietario'),'Concepto':'Mantención','Trabajo':str(work),'Cargo':float(amount),'Abonado':0.,'Pendiente':max(float(amount),0),'A favor':max(-float(amount),0),'Estado':'Pendiente' if amount>0 else 'A favor' if amount<0 else 'Al día','Último abono':last})
    ledger=ledger[~ledger.Concepto.eq('Mantención')].copy()
    ledger['Trabajo']=''
    ledger=pd.concat([ledger,pd.DataFrame(maintenance_rows)],ignore_index=True)
    f['proyecto_config']=pd.DataFrame([{'cc':'Proyecto','monto':unique_charges[0] if len(unique_charges)==1 else 0,'obs':'Cámaras','total':parse_money(projects.monto).sum()}])
    alerts+=base_alerts
    for key,moneycol in [('costos','monto'),('mantencion','monto')]:
        invalid=parse_money(f[key][moneycol]).isna()
        if invalid.any():alerts.append(f'{SHEETS[key][0]}: {int(invalid.sum())} registros sin monto válido, excluidos de las sumas.')
        undated=pd.to_datetime(f[key].fecha,dayfirst=True,errors='coerce').isna()
        if undated.any():
            if key=='costos':alerts.append(f'Costos: {int(undated.sum())} registros sin fecha válida; Pagado afecta la caja total y No pagado se mantiene como compromiso, con advertencia en el gráfico.')
            else:alerts.append(f'{SHEETS[key][0]}: {int(undated.sum())} movimientos sin fecha válida; sus montos sí se incluyen en el saldo.')
    maintenance['fecha_norm']=pd.to_datetime(maintenance.fecha,dayfirst=True,errors='coerce')
    # Auxiliary project collection is a comparison, never a second cash receipt.
    received=income.assign(amount=parse_money(income.abono))
    all_costs=f['costos'].assign(amount=parse_money(f['costos'].monto))
    costs=all_costs[all_costs.estado_pago.eq('Pagado')]
    payable_total=float(all_costs.por_pagar.sum())
    if has_cost_status:
        unknown=all_costs.estado_pago.eq('Sin clasificar')
        if unknown.any():alerts.append(f'Costos: {int(unknown.sum())} estados no reconocidos o vacíos. Se excluyen de caja pagada y compromisos hasta clasificarlos.')
        pending_refunds=all_costs.estado_pago.eq('No pagado') & all_costs.amount.gt(0)
        if pending_refunds.any():alerts.append(f'Costos: {int(pending_refunds.sum())} reintegros positivos No pagado no se cuentan como cobro realizado ni como deuda a proveedores.')
    providers=f['proveedores']; budget=parse_money(providers.presupuesto); paid=parse_money(providers.abono).fillna(0)
    provider_pending=parse_money(providers.pendiente_a_proveedor).fillna(0)
    if ((budget-paid-provider_pending).abs()>.01).any():alerts.append('Proveedores: presupuesto menos abonos difiere del pendiente publicado; revisar el registro.')
    project_paid=payments[payments.Concepto.eq('Proyecto')].groupby('Parcela').Abono.sum()
    aux_paid=parse_money(projects.abono_registrado).fillna(0)
    for idx,row in projects.iterrows():
        if abs(project_paid.get(row.parcela_num,0)-aux_paid.loc[idx])>.01:alerts.append(f'Cámaras parcela {int(row.parcela_num)}: el auxiliar no coincide con Ingresos. La cobranza usa Ingresos.')
    totals={'ingresos':received.amount.sum(),'costos_firmados':costs.amount.sum(),'egresos_netos':-costs.amount.sum(),'proveedores':payable_total if has_cost_status else provider_pending.sum(),'proveedores_auxiliar':provider_pending.sum(),'costos_pendientes':payable_total,'costos_registrados_firmados':all_costs.amount.sum(),'gc_parcela':annual.gc.sum(),'mantencion':parse_money(maintenance.monto).sum(),'proyecto_cargo':parse_money(projects.monto).sum(),'proyecto_abono':payments.loc[payments.Concepto.eq('Proyecto'),'Abono'].sum(),'proyecto_pendiente':ledger.loc[ledger.Concepto.eq('Proyecto'),'Pendiente'].sum()}
    totals['caja']=totals['ingresos']+totals['costos_firmados'];totals['caja_tras_proveedores']=totals['caja']-totals['proveedores'];totals['pendiente']=ledger.Pendiente.sum()
    checks=[]
    mappings=[('Ingresos con monto','ingresos'),('Costos netos con signo','costos_firmados'),('Saldo calculado','caja'),('Total de cobros GC por parcela','gc_parcela'),('Cámaras: aporte esperado','proyecto_cargo'),('Cámaras: recaudado','proyecto_abono'),('Cámaras: pendiente','proyecto_pendiente'),('Mantención parcelas: saldo neto','mantencion'),('Proveedores: pendiente','proveedores_auxiliar')]
    for label,key in mappings:
        ref=published.get(normalize(label),float('nan'));value=totals[key];delta=value-ref
        checks.append({'Control':label,'Calculado':value,'Publicado':ref,'Diferencia':delta,'Estado':'Coincide' if pd.notna(ref) and abs(delta)<.01 else 'Revisar','Fuente':'Origen y control'})
    if has_cost_status:
        delta=totals['proveedores']-totals['proveedores_auxiliar']
        checks.append({'Control':'Por pagar en Costos vs auxiliar proveedores','Calculado':totals['proveedores'],'Publicado':totals['proveedores_auxiliar'],'Diferencia':delta,'Estado':'Coincide' if abs(delta)<.01 else 'Revisar','Fuente':'Costos / Deudas proveedores'})
        if abs(delta)>.01:alerts.append(f'El Estado No pagado de Costos es la fuente operativa de proveedores por pagar. Difiere del auxiliar Deudas proveedores en ${abs(delta):,.0f}; no se suman ambos saldos.')
    if any(c['Estado']=='Revisar' for c in checks):alerts.append('Hay conciliaciones con diferencia o sin referencia publicada. Revisa Fuentes y control.')
    review_cols = [c for c in ['id','fila_origen','fila_publicada','fecha','parcela','cc','abono_fuente','revision'] if c in income]
    review = income.loc[review.index, review_cols].copy()
    ledger['Fuente del cargo']=ledger.Concepto.map({'Gastos comunes':'Cobros GC','Proyecto':'Proyectos','Mantención':'Mantención parcelas'})
    ledger['Referencia del cargo']=ledger.apply(lambda r:', '.join(str(v) for v in (tramos.fila_publicada if r.Concepto=='Gastos comunes' else projects.loc[projects.parcela_num.eq(r.Parcela),'id'] if r.Concepto=='Proyecto' else maintenance.loc[maintenance.parcela_num.eq(r.Parcela) & maintenance.obs.eq(r.Trabajo),'id'])),axis=1)
    payments['Fuente']='Ingresos'
    revision=sha256(''.join(raw[k].to_csv(index=False,header=False) for k in sorted(raw)).encode()).hexdigest()[:12]
    return Snapshot(f,raw,ledger,payments,annual_view,list(dict.fromkeys(alerts)),review,pd.DataFrame(checks),datetime.now(ZoneInfo('America/Santiago')).strftime('%d/%m/%Y %H:%M'),revision,totals)
