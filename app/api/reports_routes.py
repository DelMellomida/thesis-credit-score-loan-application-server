from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
import io
import csv
import logging

from app.services.report_service import report_service
from app.services.loan_service import loan_application_service
from app.core.auth_dependencies import get_current_user

router = APIRouter(prefix="/reports", tags=["Reports"])

logger = logging.getLogger(__name__)


def _parse_date_range(s: Optional[str]):
    """
    Parse an ISO date string (YYYY-MM-DD) and return a tuple of
    (start_utc, end_utc) datetimes corresponding to the full day in
    Asia/Manila timezone converted to UTC. Returns None on parse failure.
    """
    if not s:
        return None
    try:
        # parse date part only (datetime.fromisoformat on YYYY-MM-DD gives midnight naive)
        d = datetime.fromisoformat(s).date()

        tz = ZoneInfo('Asia/Manila')

        local_start = datetime.combine(d, time.min).replace(tzinfo=tz)
        local_end = datetime.combine(d, time.max).replace(tzinfo=tz)

        start_utc = local_start.astimezone(timezone.utc)
        end_utc = local_end.astimezone(timezone.utc)

        return start_utc, end_utc
    except Exception:
        return None


@router.get("/applicants")
async def applicants_report(
    start_date: str = Query(..., description="ISO start date, e.g. 2025-11-01"),
    end_date: str = Query(..., description="ISO end date, e.g. 2025-11-30"),
    group_by: str = Query('day', regex="^(day|month|year)$"),
    status: Optional[str] = Query(None),
    loan_officer_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    sd_range = _parse_date_range(start_date)
    ed_range = _parse_date_range(end_date)
    if not sd_range or not ed_range:
        raise HTTPException(status_code=400, detail="Invalid start_date or end_date; use ISO format YYYY-MM-DD")

    # Use start of start_date (PHT) -> UTC, and end of end_date (PHT) -> UTC
    sd = sd_range[0]
    ed = ed_range[1]

    try:
        result = await report_service.applicants_over_time(sd, ed, group_by=group_by, status=status, loan_officer_id=loan_officer_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Error generating applicants report: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate report")


@router.get("/applicants/list")
async def applicants_list(
    start_date: str = Query(...),
    end_date: str = Query(...),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=1000),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    loan_officer_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    # Delegate to existing loan service's get_loan_applications by computing skip/limit and date filter via search.
    try:
        skip = (page - 1) * limit
        # Reuse loan service - it doesn't accept date range yet; this endpoint provides a simple paginated list by querying DB directly
        from app.database.models.loan_application_model import LoanApplication
        from bson import ObjectId

        sd_range = _parse_date_range(start_date)
        ed_range = _parse_date_range(end_date)
        if not sd_range or not ed_range:
            raise HTTPException(status_code=400, detail="Invalid dates")

        sd = sd_range[0]
        ed = ed_range[1]

        query = {"timestamp": {"$gte": sd, "$lte": ed}}
        if status and status.lower() != 'all':
            query['status'] = status.title()
        if loan_officer_id:
            query['loan_officer_id'] = loan_officer_id
        if search:
            query['$text'] = {"$search": search}

        total = await LoanApplication.find(query).count()
        apps = await LoanApplication.find(query).sort([("timestamp", -1)]).skip(skip).limit(limit).to_list()

        data = []
        for app in apps:
            data.append({
                "application_id": str(app.application_id),
                "timestamp": app.timestamp.isoformat(),
                "status": app.status,
                "applicant_name": app.applicant_info.full_name if app.applicant_info else None,
                "contact_number": app.applicant_info.contact_number if app.applicant_info else None,
            })

        return {
            "data": data,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit if limit > 0 else 1
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error listing applicants: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list applications")


@router.get("/applicants/export")
async def applicants_export(
    start_date: str = Query(...),
    end_date: str = Query(...),
    group_by: str = Query('day', regex="^(day|month|year)$"),
    status: Optional[str] = Query(None),
    loan_officer_id: Optional[str] = Query(None),
    format: str = Query('csv', regex="^(csv|pdf)$"),
    current_user: dict = Depends(get_current_user)
):
    sd_range = _parse_date_range(start_date)
    ed_range = _parse_date_range(end_date)
    if not sd_range or not ed_range:
        raise HTTPException(status_code=400, detail="Invalid start_date or end_date; use ISO format YYYY-MM-DD")

    sd = sd_range[0]
    ed = ed_range[1]

    if format == 'csv':
        try:
            report = await report_service.applicants_over_time(sd, ed, group_by=group_by, status=status, loan_officer_id=loan_officer_id)

            buffer = io.StringIO()
            writer = csv.writer(buffer)
            # header
            writer.writerow(["group", "count"])
            for row in report.get('raw', []):
                writer.writerow([row.get('group'), row.get('count')])

            buffer.seek(0)
            return StreamingResponse(buffer, media_type='text/csv', headers={
                'Content-Disposition': f'attachment; filename="applicants-report-{sd.date().isoformat()}_{ed.date().isoformat()}.csv"'
            })
        except Exception as e:
            logger.exception("CSV export failed: %s", e)
            raise HTTPException(status_code=500, detail="Failed to generate CSV")

    else:
        # PDF generation not implemented server-side yet
        raise HTTPException(status_code=501, detail="PDF export not implemented; use client print view")
