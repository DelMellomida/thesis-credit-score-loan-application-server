from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import calendar
import logging

from app.database.models.loan_application_model import LoanApplication

logger = logging.getLogger(__name__)


def _group_format(group_by: str) -> Dict[str, Any]:
    # Returns a MongoDB $dateToString spec for grouping
    if group_by == 'month':
        return {"$dateToString": {"format": "%Y-%m", "date": "$timestamp", "timezone": "Asia/Manila"}}
    if group_by == 'year':
        return {"$dateToString": {"format": "%Y", "date": "$timestamp", "timezone": "Asia/Manila"}}
    # default -> day
    return {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp", "timezone": "Asia/Manila"}}


class ReportService:
    """Service that produces aggregated reports for loan applications."""

    async def applicants_over_time(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = 'day',
        status: Optional[str] = None,
        loan_officer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            # Normalize incoming datetimes to UTC naive datetimes for Mongo comparisons
            try:
                sd_utc = start_date.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                sd_utc = start_date
            try:
                ed_utc = end_date.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                ed_utc = end_date

            # For monthly/year grouping, expand the provided start/end to full month/year
            # so that monthly returns whole-month buckets even when user selects partial dates.
            if group_by == 'month':
                sd_norm = sd_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                last_day = calendar.monthrange(ed_utc.year, ed_utc.month)[1]
                ed_norm = ed_utc.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)
            elif group_by == 'year':
                sd_norm = sd_utc.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                ed_norm = ed_utc.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
            else:
                sd_norm = sd_utc
                ed_norm = ed_utc

            match: Dict[str, Any] = {
                "timestamp": {"$gte": sd_norm, "$lte": ed_norm}
            }

            logger.debug("Applicants report match filter: %s", match)

            if status and status.lower() != 'all':
                match["status"] = status.title()

            if loan_officer_id:
                match["loan_officer_id"] = loan_officer_id

            group_id = _group_format(group_by)

            pipeline = [
                {"$match": match},
                {"$group": {"_id": group_id, "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]

            grouped = await LoanApplication.aggregate(pipeline).to_list()

            # totals and status breakdown
            totals_pipeline = [
                {"$match": match},
                {"$group": {"_id": None, "total": {"$sum": 1}}}
            ]
            total_result = await LoanApplication.aggregate(totals_pipeline).to_list()
            total = total_result[0]["total"] if total_result else 0
            logger.debug("Applicants report total_result: %s", total_result)

            status_pipeline = [
                {"$match": match},
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_result = await LoanApplication.aggregate(status_pipeline).to_list()

            by_status = {"approved": 0, "denied": 0, "pending": 0, "cancelled": 0}
            for r in status_result:
                key = (r.get("_id") or "Pending").lower()
                by_status[key] = r.get("count", 0)

            # format grouped data and ensure contiguous labels (fill zeros for missing groups)
            # grouped _id will be the formatted date string per _group_format
            raw_map = { (r.get("_id")): r.get("count", 0) for r in grouped }

            def _iter_groups(sd: datetime, ed: datetime, gb: str):
                # yields formatted group labels between sd and ed inclusive
                if gb == 'day':
                    cur = sd.date()
                    endd = ed.date()
                    while cur <= endd:
                        yield cur.strftime("%Y-%m-%d")
                        from datetime import timedelta
                        cur = cur + timedelta(days=1)
                elif gb == 'month':
                    y = sd.year
                    m = sd.month
                    end_y = ed.year
                    end_m = ed.month
                    while (y < end_y) or (y == end_y and m <= end_m):
                        yield f"{y:04d}-{m:02d}"
                        m += 1
                        if m > 12:
                            m = 1
                            y += 1
                else:  # year
                    start_year = sd.year
                    end_year = ed.year
                    # ensure at least 5-year span ending at end_year
                    if (end_year - start_year + 1) < 5:
                        start_year = end_year - 4
                    for yy in range(start_year, end_year + 1):
                        yield f"{yy:04d}"

            labels = []
            series = []
            raw = []
            for label in _iter_groups(start_date, end_date, group_by):
                count = raw_map.get(label, 0)
                labels.append(label)
                series.append(count)
                raw.append({"group": label, "count": count})

            return {
                "labels": labels,
                "series": series,
                "totals": {"total": total, "by_status": by_status},
                "raw": raw,
            }

        except Exception as e:
            logger.exception("Failed to build applicant report: %s", e)
            raise


report_service = ReportService()
