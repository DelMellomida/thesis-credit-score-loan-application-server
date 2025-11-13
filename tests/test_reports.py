import pytest
import asyncio
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.asyncio
async def test_report_service_applicants_over_time(monkeypatch):
    # Patch LoanApplication.aggregate to simulate MongoDB responses
    class FakeAgg:
        def __init__(self, result):
            self._result = result

        async def to_list(self):
            return self._result

    # Simulate grouped, totals and status results
    grouped = [{"_id": "2025-11-01", "count": 5}, {"_id": "2025-11-02", "count": 3}]
    total = [{"_id": None, "total": 8}]
    status = [{"_id": "Approved", "count": 4}, {"_id": "Pending", "count": 4}]

    # monkeypatch aggregate to return different results for successive calls
    results = [grouped, total, status]

    async def fake_aggregate(pipeline):
        return FakeAgg(results.pop(0))

    import app.services.report_service as rs
    import app.database.models.loan_application_model as lam

    monkeypatch.setattr(lam, 'LoanApplication', lam.LoanApplication)
    monkeypatch.setattr(lam.LoanApplication, 'aggregate', staticmethod(fake_aggregate))

    # call the service
    report = await rs.report_service.applicants_over_time(
        start_date=__import__('datetime').datetime(2025,11,1),
        end_date=__import__('datetime').datetime(2025,11,2),
        group_by='day'
    )

    assert report['totals']['total'] == 8
    assert len(report['raw']) == 2


def test_reports_route_applicants(monkeypatch):
    # Patch the auth dependency to allow access and the report_service
    async def fake_current_user():
        return {"id": "test-user"}

    def fake_report(sd, ed, group_by='day', status=None, loan_officer_id=None):
        return {
            'labels': ['2025-11-01','2025-11-02'],
            'series': [5,3],
            'totals': {'total': 8, 'by_status': {'approved':4,'pending':4,'denied':0,'cancelled':0}},
            'raw': [{'group':'2025-11-01','count':5},{'group':'2025-11-02','count':3}]
        }

    from app.api.reports_routes import router as reports_router
    app.dependency_overrides = {}
    from app.core.auth_dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"id":"test-user"}

    import app.services.report_service as rs
    async def fake_report_async(sd, ed, group_by='day', status=None, loan_officer_id=None):
        return fake_report(sd, ed, group_by, status, loan_officer_id)

    monkeypatch.setattr(rs, 'report_service', rs.report_service)
    monkeypatch.setattr(rs.report_service, 'applicants_over_time', fake_report_async)

    client = TestClient(app)
    resp = client.get('/reports/applicants?start_date=2025-11-01&end_date=2025-11-02&group_by=day')
    assert resp.status_code == 200
    data = resp.json()
    assert data['totals']['total'] == 8
    assert data['labels'][0] == '2025-11-01'
