import logging
from supabase import create_client, Client
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """Create and return a Supabase client using values from `settings`.

    Raises a clear RuntimeError if required configuration is missing.
    """
    supabase_url = settings.SUPABASE_URL
    supabase_key = settings.SUPABASE_SERVICE_ROLE or getattr(settings, 'SUPABASE_ANON_PUBLIC', None)

    if not supabase_url:
        logger.error("SUPABASE_URL is not configured")
        raise RuntimeError("Supabase configuration missing: set SUPABASE_URL environment variable")

    if not supabase_key:
        logger.error("No Supabase key configured (SUPABASE_SERVICE_ROLE or SUPABASE_ANON_PUBLIC)")
        raise RuntimeError("Supabase configuration missing: set SUPABASE_SERVICE_ROLE or SUPABASE_ANON_PUBLIC environment variable")

    # If using anon key instead of service role, log a warning about reduced privileges
    if settings.SUPABASE_SERVICE_ROLE is None and getattr(settings, 'SUPABASE_ANON_PUBLIC', None):
        logger.warning("SUPABASE_SERVICE_ROLE not set; falling back to SUPABASE_ANON_PUBLIC (reduced privileges)")

    # Debug: log masked values to help diagnose environment issues without exposing secrets
    try:
        masked_url = supabase_url.split('://')[-1]
    except Exception:
        masked_url = supabase_url
    masked_key = f"{supabase_key[:4]}...{supabase_key[-4:]}" if supabase_key and len(supabase_key) > 8 else "<hidden>"
    logger.debug(f"Using Supabase URL host: {masked_url} and key: {masked_key}")

    return create_client(supabase_url, supabase_key)