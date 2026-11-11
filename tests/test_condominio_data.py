import unittest
from copy import deepcopy
import pandas as pd
from condominio_data import build_snapshot, parse_money


def raw_table(headers, rows):
    return pd.DataFrame([['Título']+[None]*(len(headers)-1),[None]*len(headers),[None]*len(headers),headers]+rows)


def fixture():
    income=[];projects=[]
    for i,p in enumerate(range(17,37),1):
        income += [[f'ING-{i:04}',i,'01/01/2026',p,f'Propietario {p}','$100','Gasto Común','Transferencia',None], [f'ING-{i+20:04}',i+20,'01/01/2026',p,f'Propietario {p}','$75','Proyecto','Transferencia',None]]
        projects += [[f'CAM-{i:03}',i,'01/01/2026',p,f'Propietario {p}','$75','$75']]
    checks=[['Ingresos con monto',3500],['Costos netos con signo',-1400],['Saldo calculado',2100],['Total de cobros GC por parcela',100],['Cámaras: aporte esperado',1500],['Cámaras: recaudado',1500],['Cámaras: pendiente',0],['Mantención parcelas: saldo neto',30],['Proveedores: pendiente',60]]
    return {
        'ingresos':raw_table(['ID','Fila origen','Fecha','Parcela','Propietario','Monto recibido','Clasificación','Medio de pago','Revisión'],income),
        'proyectos':raw_table(['ID','Fila origen','Fecha','Parcela','Propietario','Aporte esperado','Abono registrado'],projects),
        'obligaciones':raw_table(['Año','Tramo','Meses por cobrar','Cuota mensual por parcela','Total del tramo'],[[2026,'Tramo 1',5,'$10','$50'],[2026,'Tramo 2',5,'$10','$50']]),
        'mantencion':raw_table(['ID','Fila origen','Fecha','Parcela','Trabajo','Tipo de movimiento','Monto firmado'],[['MAN-001',1,'01/01/2026',22,'Trabajo A','Cargo','$40'],['MAN-002',2,'02/01/2026',22,'Trabajo A','Abono','($10)']]),
        'costos':raw_table(['ID','Fila origen','Fecha de pago','Monto firmado','Centro de costo'],[['COS-001',1,'01/01/2026','($1.500)','Jardín'],['COS-002',2,'02/01/2026','$100','Jardín']]),
        'proveedores':raw_table(['ID','Fila origen','Motivo / proveedor','Presupuesto','Abono registrado','Pendiente'],[['PRO-001',1,'Proveedor','$100','$40','$60']]),
        'control':raw_table(['Control','Resultado calculado','Referencia del archivo'],[[a,b,'Referencia'] for a,b in checks]),
        'resumen':pd.DataFrame([['Resumen',None]]),
    }


class DataModelTests(unittest.TestCase):
    def test_cash_signs_and_auxiliary_not_counted_twice(self):
        s=build_snapshot(fixture())
        self.assertEqual(s.totals['caja'],2100)
        self.assertEqual(s.totals['ingresos'],3500)
        self.assertEqual(s.totals['proveedores'],60)
        self.assertEqual(s.totals['caja_tras_proveedores'],2040)
        self.assertTrue(s.checks.Estado.eq('Coincide').all())

    def test_multiple_segments_and_balances_reconcile(self):
        s=build_snapshot(fixture())
        self.assertEqual(s.totals['gc_parcela'],100)
        self.assertEqual(s.ledger.Pendiente.sum(),30)
        self.assertTrue(((s.ledger.Cargo-s.ledger.Abonado)-(s.ledger.Pendiente-s.ledger['A favor'])).abs().lt(.01).all())
        self.assertEqual(s.legacy_balances()[['pendiente_pos','Mantención','Pendiente Proyecto']].sum().sum(),30)

    def test_maintenance_credit_does_not_hide_other_work_debt(self):
        raw=fixture()
        raw['mantencion'].loc[5,4]='Trabajo B'
        s=build_snapshot(raw)
        m=s.ledger[s.ledger.Concepto.eq('Mantención')]
        self.assertEqual(m.Pendiente.sum(),40)
        self.assertEqual(m['A favor'].sum(),10)
        self.assertEqual(s.totals['mantencion'],30)

    def test_project_uses_actual_charge_per_parcel(self):
        raw=fixture();raw['proyectos'].loc[4,5]='$80'
        s=build_snapshot(raw)
        self.assertEqual(s.ledger.loc[s.ledger.Parcela.eq(17)&s.ledger.Concepto.eq('Proyecto'),'Pendiente'].iloc[0],5)
        self.assertEqual(s.totals['ingresos'],3500)
        self.assertTrue(s.checks.Estado.eq('Revisar').any())

    def test_missing_income_is_flagged_and_dash_is_zero(self):
        raw=fixture();raw['ingresos'].loc[4,5]='-';raw['ingresos'].loc[5,5]=None
        s=build_snapshot(raw)
        self.assertEqual(len(s.review),1)
        self.assertEqual(len(s.payments),39)
        self.assertEqual(s.totals['ingresos'],3325)

    def test_duplicate_ids_stop_calculation(self):
        raw=fixture();raw['ingresos'].loc[5,0]=raw['ingresos'].loc[4,0]
        with self.assertRaisesRegex(ValueError,'IDs duplicados'):build_snapshot(raw)

    def test_snapshot_revision_is_deterministic_and_changes_with_content(self):
        raw=fixture();s=build_snapshot(raw)
        self.assertEqual(s.revision,build_snapshot(deepcopy(raw)).revision)
        raw['resumen'].iloc[0,0]='Cambio'
        self.assertNotEqual(s.revision,build_snapshot(raw).revision)

    def test_currency_parser_preserves_numeric_and_accounting_sign(self):
        result=parse_money(pd.Series(['($25.000)','$1.234,00',10000.,'-',0])).fillna(99).tolist()
        self.assertEqual(result,[-25000,1234,10000,99,0])


if __name__=='__main__':unittest.main()
