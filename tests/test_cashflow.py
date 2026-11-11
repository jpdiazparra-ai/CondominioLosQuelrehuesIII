import unittest
from copy import deepcopy
import pandas as pd
from condominio_data import build_snapshot
from control_finance import analyze_portfolio,financial_figures,cash_flow,cash_window,cash_figure
from control_dashboard_pdf import build_dashboard_pdf
from test_condominio_data import fixture


class CashFlowTests(unittest.TestCase):
    def setUp(self):self.snapshot=build_snapshot(fixture())

    def test_real_cash_excludes_charges_auxiliary_and_supplier_commitments(self):
        flow=cash_flow(self.snapshot);w=cash_window(flow,0)
        self.assertEqual(w['income'],3500)
        self.assertEqual(w['expenses'],1500)
        self.assertEqual(w['refunds'],100)
        self.assertEqual(w['closing'],2100)
        self.assertEqual(w['opening']+w['income']+w['refunds']-w['expenses'],w['closing'])
        self.assertEqual(w['closing'],self.snapshot.totals['caja'])

    def test_window_keeps_opening_and_fills_months_without_movements(self):
        s=deepcopy(self.snapshot)
        s.frames['ingresos'].loc[0,'fecha']='01/03/2026'
        flow=cash_flow(s)
        self.assertEqual(flow['monthly'].Mes.tolist(),['2026-01','2026-02','2026-03'])
        self.assertEqual(flow['monthly']['Flujo neto'].iloc[1],0)
        w=cash_window(flow,1)
        self.assertEqual(w['opening'],2000)
        self.assertEqual(w['closing'],2100)
        self.assertEqual(w['net'],100)

    def test_undated_amount_is_flagged_and_reconciles_to_total_cash(self):
        s=deepcopy(self.snapshot);s.frames['ingresos'].loc[0,'fecha']=None
        flow=cash_flow(s);w=cash_window(flow,0)
        self.assertEqual(flow['undated_count'],1)
        self.assertEqual(flow['undated_net'],100)
        self.assertEqual(w['closing']+flow['undated_net'],s.totals['caja'])

    def test_orange_segments_have_inside_or_outside_percentages(self):
        m=analyze_portfolio(self.snapshot.ledger,self.snapshot.payments)
        fig=financial_figures(m)['coverage']
        self.assertIn('100,0%',list(fig.data[1].text))
        self.assertTrue(all('x' in a and 'y' in a for a in fig.layout.annotations))
        ledger=self.snapshot.ledger.copy()
        ix=ledger.Concepto.eq('Proyecto')
        ledger.loc[ix,'Abonado']*=.95
        ledger.loc[ix,'Pendiente']=ledger.loc[ix,'Cargo']-ledger.loc[ix,'Abonado']
        fig=financial_figures(analyze_portfolio(ledger,self.snapshot.payments))['coverage']
        self.assertTrue(any('5,0%' in a.text for a in fig.layout.annotations))

    def test_cash_figure_has_distinct_flow_and_balance_axes(self):
        fig=cash_figure(cash_window(cash_flow(self.snapshot),0))
        self.assertEqual(fig.data[-1].yaxis,'y2')
        self.assertEqual(fig.data[-1].y[-1],2100)
        self.assertTrue(all(v<=0 for v in fig.data[-2].y))

    def test_dashboard_pdf_is_generated_for_full_and_filtered_portfolio(self):
        for ledger in [self.snapshot.ledger,self.snapshot.ledger[self.snapshot.ledger.Parcela.eq(22)]]:
            pdf=build_dashboard_pdf(ledger,self.snapshot.payments,self.snapshot,{'Concepto':'Todos'},cash_length=0,trend_length=0)
            self.assertTrue(pdf.startswith(b'%PDF-'))
            self.assertGreater(len(pdf),10000)


if __name__=='__main__':unittest.main()
