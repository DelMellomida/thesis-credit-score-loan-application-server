import motor.motor_asyncio
from beanie import init_beanie
from app.database.models import User, LoanApplication, ApplicationDocument
from app.core import Settings  # Use your existing import
import logging
import certifi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database instance
database = None

async def init_db():
    global database
    try:
        # Get MongoDB configuration from Settings
        mongodb_uri = Settings.MONGODB_URI
        mongodb_db_name = Settings.MONGODB_DB_NAME
        
        logger.info(f"Attempting to connect to MongoDB at: {mongodb_uri}")
        logger.info(f"Database name: {mongodb_db_name}")
        
        # Validate that we have the required Settings
        if not mongodb_uri:
            logger.error("MONGODB_URI is not set in environment variables")
            raise ValueError("MONGODB_URI is not set in environment variables")
        if not mongodb_db_name:
            logger.error("MONGODB_DB_NAME is not set in environment variables")
            raise ValueError("MONGODB_DB_NAME is not set in environment variables")
        
        # Create MongoDB client optimized for Render
        logger.info("Creating MongoDB client with Render-optimized TLS configuration...")
        client = motor.motor_asyncio.AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=10000,  # Increased for Render's cold starts
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            tlsCAFile=certifi.where(),
            retryWrites=True,
            w='majority',
            maxPoolSize=10,  # Limit connection pool for Render
            minPoolSize=1
        )
        
        # Test the connection
        logger.info("Testing MongoDB connection...")
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        
        # Get database
        logger.info("Getting database instance...")
        database = client[mongodb_db_name]
        
        # Initialize Beanie with document models
        logger.info("Initializing Beanie with document models...")
        await init_beanie(database, document_models=[User, LoanApplication, ApplicationDocument])
        logger.info("Beanie initialized successfully!")
        
        return database
        
    except ImportError as e:
        logger.error(f"Import error - missing dependency: {str(e)}")
        raise RuntimeError(f"Missing dependency: {str(e)}") from e
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise RuntimeError(f"Configuration error: {str(e)}") from e
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Full error details: {repr(e)}")
        # Don't mask the original error - let it bubble up
        raise e

def get_database():
    """Get the initialized database instance"""
    if database is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return database