from typing import Dict, Any, List, Optional
from datetime import datetime
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
            match: Dict[str, Any] = {
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }

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

            status_pipeline = [
                {"$match": match},
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_result = await LoanApplication.aggregate(status_pipeline).to_list()

            by_status = {"approved": 0, "denied": 0, "pending": 0, "cancelled": 0}
            for r in status_result:
                key = (r.get("_id") or "Pending").lower()
                by_status[key] = r.get("count", 0)

            # format grouped data
            raw = [{"group": r.get("_id"), "count": r.get("count", 0)} for r in grouped]
            labels = [r["group"] for r in raw]
            series = [r["count"] for r in raw]

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
