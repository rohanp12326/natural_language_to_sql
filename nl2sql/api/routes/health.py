"""Health check and diagnostics routes for the NL2SQL API.

Provides liveness and readiness probes:
  - Docker / Kubernetes container orchestrators can ping `/health` to verify service readiness.
  - Verifies that PostgreSQL is reachable and executing queries before routing traffic.
"""

import asyncio

from fastapi import APIRouter
from loguru import logger
from sqlalchemy import text

from nl2sql.api.models import HealthResponse
from nl2sql.database.postgresql import PostgreSQLConnector

router = APIRouter()

_health_connector: PostgreSQLConnector | None = None


def get_health_connector() -> PostgreSQLConnector:
    """Get or create singleton PostgreSQL connector for health checks."""
    global _health_connector
    if _health_connector is None:
        _health_connector = PostgreSQLConnector(config_path="configs/database.yml")
    return _health_connector


def _ping_database() -> bool:
    """Execute lightweight ping query against PostgreSQL."""
    connector = get_health_connector()
    with connector.engine.connect() as conn:
        result = conn.execute(text("SELECT 1 as test"))
        test_value = result.fetchone()[0]
        return test_value == 1


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe testing API readiness and PostgreSQL connectivity.

    Executes a minimal test query (`SELECT 1`) against the PostgreSQL database.
    If the connection succeeds and returns 1, the service is deemed healthy.
    """
    try:
        is_connected = await asyncio.to_thread(_ping_database)

        if is_connected:
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
