import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Credit Score and Loan Recommendation"
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CLIENT_URL: str = os.getenv("CLIENT_URL", "http://localhost:3000")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE: str = os.getenv("SUPABASE_SERVICE_ROLE")
    SUPABASE_ANON_PUBLIC: str = os.getenv("SUPABASE_ANON_PUBLIC")
    # TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID")
    # TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN")
    # TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER")
    PHILSMS_API_TOKEN: str = os.getenv("PHILSMS_API_TOKEN")
    PHILSMS_SENDER_ID: str = os.getenv("PHILSMS_SENDER_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Debug: Print the loaded JWT_SECRET_KEY (masked for safety)
if settings.JWT_SECRET_KEY:
    def _mask_secret(val: str) -> str:
        if not val:
            return "<missing>"
        if len(val) <= 8:
            return "*" * len(val)
        return f"{val[:4]}...{val[-4:]}"

    def _mask_url(url: str) -> str:
        # Keep scheme and host, but strip credentials and path for safety
        try:
            from urllib.parse import urlparse, urlunparse
            p = urlparse(url)
            netloc = p.hostname or ""
            if p.port:
                netloc = f"{netloc}:{p.port}"
            return f"{p.scheme}://{netloc}"
        except Exception:
            return _mask_secret(url)

    print("DEBUG: Loaded Supabase settings (redacted):",
          {
              "SUPABASE_URL": _mask_url(settings.SUPABASE_URL) if settings.SUPABASE_URL else None,
              "SUPABASE_SERVICE_ROLE": _mask_secret(settings.SUPABASE_SERVICE_ROLE),
              "SUPABASE_ANON_PUBLIC": _mask_secret(settings.SUPABASE_ANON_PUBLIC),
          })
    print(f"DEBUG: Loaded JWT_SECRET_KEY: {_mask_secret(settings.JWT_SECRET_KEY)}")
else:
    print("DEBUG: JWT_SECRET_KEY is missing or empty!")