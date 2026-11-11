import unittest
from copy import deepcopy
import pandas as pd
from condominio_data import build_snapshot
from control_finance import cash_flow,cash_window,cash_figure
from test_condominio_data import fixture


def status_fixture():
    raw=fixture();costs=raw['costos'].copy();costs[5]=None;costs.loc[3,5]='Estado';costs.loc[4:,5]='Pagado'
    costs.loc[len(costs)]=['COS-003',3,'01/02/2026','($60)','Proyecto','No pagado']
    costs.loc[len(costs)]=['COS-004',4,'01/01/2026','($40)','Jardín',' no PAGADO ']
    raw['costos']=costs
    return raw


class PaymentStatusTests(unittest.TestCase):
    def test_payables_use_cost_state_and_do_not_sum_supplier_auxiliary(self):
        s=build_snapshot(status_fixture())
        self.assertEqual(s.totals['proveedores'],100)
        self.assertEqual(s.totals['proveedores_auxiliar'],60)
        self.assertEqual(s.totals['costos_firmados'],-1400)
        self.assertEqual(s.totals['costos_registrados_firmados'],-1500)
        self.assertEqual(s.totals['caja'],2100)
        self.assertEqual(s.totals['caja_tras_proveedores'],2000)
        self.assertTrue(s.checks.Estado.eq('Revisar').any())

    def test_unpaid_bars_and_adjusted_line_do_not_change_real_cash(self):
        s=build_snapshot(status_fixture());flow=cash_flow(s);w=cash_window(flow,0)
        self.assertEqual(w['expenses'],1500)
        self.assertEqual(w['pending'],100)
        self.assertEqual(w['closing'],2100)
        self.assertEqual(w['adjusted_closing'],2000)
        self.assertEqual(flow['monthly'].Pendientes.tolist(),[40,60])
        fig=cash_figure(w)
        traces={trace.name:trace for trace in fig.data}
        self.assertEqual(traces['Caja acumulada · pagados'].y[-1],2100)
        self.assertEqual(traces['Caja al pagar pendientes · escenario'].y[-1],2000)
        self.assertEqual(sum(traces['No pagado · compromiso'].y),-100)

    def test_partial_window_preserves_reserves_from_prior_months(self):
        w=cash_window(cash_flow(build_snapshot(status_fixture())),1)
        self.assertEqual(w['pending_window'],60)
        self.assertEqual(w['opening'],2100)
        self.assertEqual(w['adjusted_opening'],2060)
        self.assertEqual(w['adjusted_opening']+w['net']-w['pending_window'],w['adjusted_closing'])
        self.assertEqual(w['pending'],100)

    def test_marking_paid_releases_reserve_and_books_cash_only_once(self):
        raw=status_fixture();before=build_snapshot(raw)
        raw['costos'].loc[6:,5]='Pagado';after=build_snapshot(raw)
        self.assertEqual(after.totals['caja'],2000)
        self.assertEqual(after.totals['proveedores'],0)
        self.assertEqual(cash_window(cash_flow(before),0)['adjusted_closing'],cash_window(cash_flow(after),0)['closing'])

    def test_unpaid_without_date_stays_payable_and_is_flagged(self):
        raw=status_fixture();raw['costos'].loc[7,2]=None
        s=build_snapshot(raw);flow=cash_flow(s);w=cash_window(flow,0)
        self.assertEqual(flow['pending_undated'],40)
        self.assertEqual(flow['undated_count'],0)
        self.assertEqual(w['pending'],100)
        self.assertEqual(w['adjusted_closing'],2000)
        self.assertTrue(pd.isna(s.frames['costos'].loc[s.frames['costos'].id.eq('COS-004'),'fecha'].iloc[0]))

    def test_unknown_state_does_not_silently_book_payment(self):
        raw=status_fixture();raw['costos'].loc[4,5]='Por revisar'
        s=build_snapshot(raw)
        self.assertEqual(s.totals['costos_firmados'],100)
        self.assertEqual(s.totals['caja'],3600)
        self.assertTrue(any('estados no reconocidos' in alert for alert in s.alerts))

    def test_unpaid_positive_refund_is_not_treated_as_provider_debt(self):
        raw=status_fixture();raw['costos'].loc[5,5]='No pagado'
        s=build_snapshot(raw)
        self.assertEqual(s.totals['proveedores'],100)
        self.assertEqual(s.totals['caja'],2000)
        self.assertTrue(any('reintegros positivos No pagado' in a for a in s.alerts))


if __name__=='__main__':unittest.main()
