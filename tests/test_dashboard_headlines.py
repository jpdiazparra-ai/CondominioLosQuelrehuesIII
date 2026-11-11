import unittest
from condominio_data import build_snapshot
from control_finance import analyze_portfolio,dashboard_headlines
from test_payment_status import status_fixture


class DashboardHeadlineTests(unittest.TestCase):
    def test_cash_and_payables_are_global_receivables_follow_filter(self):
        s=build_snapshot(status_fixture())
        full=analyze_portfolio(s.ledger,s.payments)
        selected=analyze_portfolio(s.ledger[s.ledger.Parcela.eq(17)],s.payments)
        a=dashboard_headlines(s,full,as_of='2026-11-11')
        b=dashboard_headlines(s,selected,as_of='2026-11-11')
        self.assertEqual(a['cash_today'],2100)
        self.assertEqual(a['payables'],100)
        self.assertEqual(a['cash_after_payables'],2000)
        self.assertEqual(a['receivable'],30)
        self.assertEqual(b['receivable'],0)
        self.assertEqual(b['count'],1)
        self.assertEqual(a['cash_today'],b['cash_today'])
        self.assertEqual(a['payables'],b['payables'])

    def test_cash_today_excludes_future_book_movements(self):
        raw=status_fixture()
        raw['ingresos'].loc[len(raw['ingresos'])]=['ING-0099',99,'01/01/2027',17,'Propietario 17','$500','Gasto Común','Transferencia',None]
        raw['costos'].loc[len(raw['costos'])]=['COS-005',5,'01/01/2027','($200)','Jardín','Pagado']
        s=build_snapshot(raw)
        h=dashboard_headlines(s,analyze_portfolio(s.ledger,s.payments),as_of='2026-11-11')
        self.assertEqual(s.totals['caja'],2400)
        self.assertEqual(h['cash_today'],2100)
        self.assertEqual(h['cash_after_payables'],2000)
        self.assertEqual(h['future_count'],2)

    def test_recognized_undated_payments_are_included_and_identified(self):
        raw=status_fixture();raw['ingresos'].loc[4,2]=None;raw['costos'].loc[4,2]=None
        s=build_snapshot(raw)
        h=dashboard_headlines(s,analyze_portfolio(s.ledger,s.payments),as_of='2026-11-11')
        self.assertEqual(h['cash_today'],s.totals['caja'])
        self.assertEqual(h['undated_count'],2)


if __name__=='__main__':unittest.main()
