"""Health check and diagnostics routes for the NL2SQL API.

Provides liveness and readiness probes:
  - Docker / Kubernetes container orchestrators can ping `/health` to verify service readiness.
  - Verifies that PostgreSQL is reachable and executing queries before routing traffic.
"""

from fastapi import APIRouter
from loguru import logger
from sqlalchemy import text

from nl2sql.api.models import HealthResponse
from nl2sql.database.postgresql import PostgreSQLConnector

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe testing API readiness and PostgreSQL connectivity.
    
    Executes a minimal test query (`SELECT 1`) against the PostgreSQL database.
    If the connection succeeds and returns 1, the service is deemed healthy.
    """
    try:
        # Connect to PostgreSQL using database configuration
        db_connector = PostgreSQLConnector(config_path="configs/database.yml")

        # Execute lightweight ping query
        with db_connector.engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]

        # Verify expected response
        if test_value == 1:
            logger.info("✅ Health check passed - database connected")
            return HealthResponse(
                status="healthy",
                database_connected=True,
                message="Service is healthy and database is connected",
            )
        else:
            logger.error("❌ Health check failed - unexpected database response")
            return HealthResponse(
                status="unhealthy",
                database_connected=False,
                message="Database connection test failed",
            )

    except Exception as e:
        # Catch connection timeouts, authentication failures, or network issues
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            message=f"Database connection error: {e!s}",
        )

