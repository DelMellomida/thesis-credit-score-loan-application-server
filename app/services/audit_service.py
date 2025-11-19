import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.database.models.audit_log_model import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    """Simple service to create and read audit logs stored in MongoDB using Beanie."""

    async def create_audit(self, *, action: str, actor: Optional[str] = None, acted: Optional[str] = None, status: str = "successful", timestamp: Optional[datetime] = None) -> AuditLog:
        try:
            payload = {
                "action": action,
                "actor": actor,
                "acted": acted,
                "status": status,
                "timestamp": timestamp or datetime.now()
            }

            try:
                audit = AuditLog(**payload)
            except Exception as ve:
                # More detailed logging for validation errors
                logger.error(f"AuditLog validation failed when creating audit (payload={payload}): {ve}")
                raise

            await audit.insert()
            return audit
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            raise

    async def get_audits(self, skip: int = 0, limit: int = 50, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            query = {}
            if filters:
                # Accept direct equality filters for action, actor, acted, status
                if "action" in filters and filters["action"]:
                    query["action"] = filters["action"]
                if "actor" in filters and filters["actor"]:
                    query["actor"] = filters["actor"]
                if "acted" in filters and filters["acted"]:
                    query["acted"] = filters["acted"]
                if "status" in filters and filters["status"]:
                    query["status"] = filters["status"]
                # Date range: start_date and/or end_date should be datetime objects
                if "start_date" in filters or "end_date" in filters:
                    ts_query = {}
                    if filters.get("start_date"):
                        ts_query["$gte"] = filters.get("start_date")
                    if filters.get("end_date"):
                        ts_query["$lte"] = filters.get("end_date")
                    if ts_query:
                        query["timestamp"] = ts_query

            total = await AuditLog.find(query).count()
            docs = await AuditLog.find(query).sort("-timestamp").skip(skip).limit(limit).to_list()

            results: List[Dict[str, Any]] = []
            for d in docs:
                results.append({
                    "id": str(d.id),
                    "action": d.action,
                    "actor": d.actor,
                    "acted": d.acted,
                    "timestamp": d.timestamp.isoformat() if d.timestamp else None,
                    "status": d.status,
                })

            return {"data": results, "total": total, "skip": skip, "limit": limit}
        except Exception as e:
            logger.error(f"Failed to query audit logs: {e}")
            raise


audit_service = AuditService()
