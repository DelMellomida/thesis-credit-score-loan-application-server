from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi import status
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from app.services.audit_service import audit_service, AuditService

router = APIRouter(prefix="/audits", tags=["Audits"])

logger = logging.getLogger(__name__)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_audit(payload: Dict[str, Any]):
    """Create an audit log entry.

    Expected payload: { action, actor, acted?, status? }
    """
    try:
        action = payload.get("action")
        actor = payload.get("actor")
        acted = payload.get("acted")
        status_val = payload.get("status") or "successful"

        if not action or not actor:
            raise HTTPException(status_code=400, detail="'action' and 'actor' are required")

        audit = await audit_service.create_audit(action=action, actor=actor, acted=acted, status=status_val)
        return {
            "id": str(audit.id),
            "action": audit.action,
            "actor": audit.actor,
            "acted": audit.acted,
            "timestamp": audit.timestamp.isoformat(),
            "status": audit.status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating audit log: {e}")
        raise HTTPException(status_code=500, detail="Failed to create audit log")


@router.get("/", status_code=200)
async def list_audits(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=1000),
    action: Optional[str] = Query(default=None),
    actor: Optional[str] = Query(default=None),
    acted: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(default=None, description="YYYY-MM-DD")
):
    try:
        filters: Dict[str, Any] = {}
        if action:
            filters["action"] = action
        if actor:
            filters["actor"] = actor
        if acted:
            filters["acted"] = acted
        if status:
            filters["status"] = status

        # parse start/end dates to datetimes if provided
        if start_date:
            try:
                sd = datetime.strptime(start_date, "%Y-%m-%d")
                filters["start_date"] = sd
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid start_date format, expected YYYY-MM-DD")
        if end_date:
            try:
                ed = datetime.strptime(end_date, "%Y-%m-%d")
                # set to end of day
                ed = ed.replace(hour=23, minute=59, second=59, microsecond=999999)
                filters["end_date"] = ed
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid end_date format, expected YYYY-MM-DD")

        result = await audit_service.get_audits(skip=skip, limit=limit, filters=filters)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing audits: {e}")
        raise HTTPException(status_code=500, detail="Failed to list audit logs")
