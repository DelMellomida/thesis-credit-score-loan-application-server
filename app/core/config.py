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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Debug: Print the loaded JWT_SECRET_KEY (masked for safety)
if settings.JWT_SECRET_KEY:
    print(f"DEBUG: Loaded Supabase URL and Keys {settings.SUPABASE_URL=}, {settings.SUPABASE_SERVICE_ROLE=}, {settings.SUPABASE_ANON_PUBLIC=}")
    print(f"DEBUG: Loaded JWT_SECRET_KEY: {settings.JWT_SECRET_KEY[:4]}...{'*' * (len(settings.JWT_SECRET_KEY)-8)}...{settings.JWT_SECRET_KEY[-4:]}")
else:
    print("DEBUG: JWT_SECRET_KEY is missing or empty!")