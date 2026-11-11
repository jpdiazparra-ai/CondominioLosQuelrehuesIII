import unittest
import pandas as pd
from condominio_data import build_snapshot
from control_finance import analyze_portfolio, financial_figures, portfolio_table
from test_condominio_data import fixture


class PortfolioAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.snapshot=build_snapshot(fixture())

    def test_bridge_reconciles_without_compensating_credits(self):
        ledger=self.snapshot.ledger.copy()
        mask=ledger.Parcela.eq(17)&ledger.Concepto.eq('Gastos comunes')
        ledger.loc[mask,['Abonado','A favor']]=[120,20]
        m=analyze_portfolio(ledger,self.snapshot.payments)
        self.assertEqual(m['due'],30)
        self.assertEqual(m['credit'],20)
        self.assertEqual(m['net'],10)
        self.assertEqual(m['charges']-m['paid']+m['credit'],m['due'])
        self.assertAlmostEqual(m['coverage'],3500/3530)

    def test_coverage_is_weighted_and_not_mean_of_parcel_rates(self):
        m=analyze_portfolio(self.snapshot.ledger,self.snapshot.payments)
        self.assertAlmostEqual(m['coverage'],3500/3530)
        self.assertEqual(m['debtors'],1)
        self.assertEqual(m['top3'],1)
        self.assertEqual(m['average'],30)

    def test_multiple_work_rows_do_not_duplicate_payments(self):
        raw=fixture();raw['mantencion'].loc[5,4]='Trabajo B'
        s=build_snapshot(raw)
        m=analyze_portfolio(s.ledger,s.payments)
        self.assertEqual(len(m['payments']),40)
        self.assertEqual(m['payments'].Abono.sum(),3500)
        self.assertEqual(m['positions'].loc[lambda d:d.Parcela.eq(22),'Mantención'].iloc[0],40)

    def test_months_without_payments_are_zero_and_rolling_is_actual(self):
        payments=self.snapshot.payments.copy()
        payments['Fecha']=pd.Timestamp('2026-01-01')
        payments.loc[payments.index[-1],'Fecha']=pd.Timestamp('2026-03-01')
        m=analyze_portfolio(self.snapshot.ledger,payments)
        self.assertEqual(m['monthly'].Mes.tolist(),['2026-01','2026-02','2026-03'])
        self.assertEqual(m['monthly'].Abonos.iloc[1],0)
        self.assertAlmostEqual(m['monthly']['Promedio móvil 3 meses'].iloc[-1],3500/3)

    def test_filter_scopes_payments_and_empty_results_are_supported(self):
        selected=self.snapshot.ledger[self.snapshot.ledger.Parcela.eq(22)]
        m=analyze_portfolio(selected,self.snapshot.payments)
        self.assertEqual(len(m['payments']),2)
        self.assertEqual(m['paid'],175)
        empty=analyze_portfolio(selected.iloc[0:0],self.snapshot.payments)
        self.assertEqual(empty['count'],0)
        self.assertEqual(empty['coverage'],0)
        self.assertTrue(empty['monthly'].empty)

    def test_table_preserves_separate_debt_credit_and_escapes_names(self):
        m=analyze_portfolio(self.snapshot.ledger,self.snapshot.payments)
        p=m['positions'].copy();p.loc[0,'Propietario']='<script>unsafe</script>'
        result=portfolio_table(p,m)
        self.assertIn('TOTAL DE LA SELECCIÓN',result)
        self.assertIn('&lt;script&gt;',result)
        self.assertNotIn('<script>',result)
        self.assertNotIn('control-map',result)

    def test_chart_percentages_and_bridge_equal_analytics(self):
        m=analyze_portfolio(self.snapshot.ledger,self.snapshot.payments)
        figs=financial_figures(m)
        self.assertEqual(set(figs),{'pareto','coverage','bridge','trend'})
        self.assertEqual(list(figs['bridge'].data[0].y),[3530,-3500,0,0])
        self.assertEqual(figs['pareto'].data[1].y[-1],100)
        self.assertEqual(figs['coverage'].layout.xaxis.range,(0,112))


if __name__=='__main__':unittest.main()
