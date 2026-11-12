"""Entry point 5: start the REST service, after checking that everything it needs is on disk.

Run from the Project folder:  python run_api_service.py
Interactive documentation (Swagger UI) is then at http://localhost:8081/docs
"""

import os
import sys

from config import config

REQUIRED_BEFORE_STARTING = [
    (config.AMENDMENT_MODEL_FILE, "run_model_training.py"),
    (config.EXPOSURE_MODEL_FILE, "run_model_training.py"),
    (config.ISSUE_MODEL_FILE, "run_model_training.py"),
    (config.AMENDMENT_SCHEMA_FILE, "run_model_training.py"),
    (config.EXPOSURE_SCHEMA_FILE, "run_model_training.py"),
    (config.OPEN_CAMPAIGNS_FILE, "the supplied data folder"),
    (config.BOOKED_SPOTS_FILE, "the supplied data folder"),
    (config.PROGRAMME_SCHEDULE_FILE, "the supplied data folder"),
    (config.OPEN_CAMPAIGN_NOTES_FILE, "the supplied data folder"),
    (config.VECTOR_STORE_DIRECTORY, "run_knowledge_indexing.py"),
    (config.APPLICATION_DATABASE_FILE, "run_database_setup.py"),
]


def main() -> int:
    """Start the service, or name every missing path and what produces it."""
    missing = [
        (path, producer) for path, producer in REQUIRED_BEFORE_STARTING if not os.path.exists(path)
    ]
    if missing:
        print("The service cannot start.")
        for path, producer in missing:
            print(f"   {path} is missing; it comes from {producer}")
        return 1

    import uvicorn  # imported only once everything the service loads is known to exist

    from api.routes import service_application

    print(
        f"Starting {config.API_TITLE} on port {config.API_PORT}. "
        f"Interactive documentation: {config.API_BASE_URL}/docs"
    )
    uvicorn.run(service_application, host=config.API_HOST, port=config.API_PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
