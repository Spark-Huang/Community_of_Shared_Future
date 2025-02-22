import json
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from cosf import CommunityOfSharedFuture

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="CommunityOfSharedFuture API",
    version="1.0.0",
    debug=True,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = "jesus_christ_agent.db"

logger.add(
    "api.log",
    rotation="10 MB",
)

# Initialize SQLite database
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# Create patients table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        patient_data TEXT
    )
    """
)

# Add this after the patients table creation
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS rate_limits (
        ip_address TEXT PRIMARY KEY,
        last_daily_reset TEXT,
        last_hourly_reset TEXT,
        daily_requests_remaining INTEGER DEFAULT 1000,
        hourly_requests_remaining INTEGER DEFAULT 100
    )
"""
)


async def check_rate_limit(request: Request):
    """Rate limiting middleware based on IP address."""
    client_ip = request.client.host
    now = datetime.utcnow()

    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Get or create rate limit record for IP
        cursor.execute(
            """INSERT OR IGNORE INTO rate_limits 
               (ip_address, last_daily_reset, last_hourly_reset, 
                daily_requests_remaining, hourly_requests_remaining)
               VALUES (?, ?, ?, ?, ?)""",
            (client_ip, now.isoformat(), now.isoformat(), 1000, 100),
        )

        # Get current limits
        cursor.execute(
            """SELECT daily_requests_remaining, hourly_requests_remaining,
                      last_daily_reset, last_hourly_reset 
               FROM rate_limits WHERE ip_address = ?""",
            (client_ip,),
        )
        row = cursor.fetchone()

        (
            daily_remaining,
            hourly_remaining,
            last_daily_reset,
            last_hourly_reset,
        ) = row

        # Check and reset daily quota
        last_daily_reset_time = datetime.fromisoformat(
            last_daily_reset
        )
        if now.date() > last_daily_reset_time.date():
            daily_remaining = 1000
            last_daily_reset = now.isoformat()
            cursor.execute(
                """UPDATE rate_limits 
                   SET daily_requests_remaining = ?, 
                       last_daily_reset = ? 
                   WHERE ip_address = ?""",
                (daily_remaining, last_daily_reset, client_ip),
            )

        # Check and reset hourly quota
        last_hourly_reset_time = datetime.fromisoformat(
            last_hourly_reset
        )
        if (now - last_hourly_reset_time).total_seconds() >= 3600:
            hourly_remaining = 100
            last_hourly_reset = now.isoformat()
            cursor.execute(
                """UPDATE rate_limits 
                   SET hourly_requests_remaining = ?, 
                       last_hourly_reset = ? 
                   WHERE ip_address = ?""",
                (hourly_remaining, last_hourly_reset, client_ip),
            )

        # Check remaining quotas
        if daily_remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail="Daily rate limit exceeded. Reset occurs at midnight UTC.",
            )
        if hourly_remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail="Hourly rate limit exceeded. Please try again next hour.",
            )

        # Deduct from both quotas
        cursor.execute(
            """UPDATE rate_limits 
               SET daily_requests_remaining = daily_requests_remaining - 1,
                   hourly_requests_remaining = hourly_requests_remaining - 1 
               WHERE ip_address = ?""",
            (client_ip,),
        )
        connection.commit()

    except sqlite3.Error as e:
        logger.error(f"Error checking rate limit: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        )
    finally:
        if connection:
            connection.close()


connection.commit()
connection.close()


# Pydantic models
class PatientCase(BaseModel):
    patient_id: Optional[str] = None
    patient_docs: Optional[str] = None
    case_description: Optional[str] = None
    summarization: Optional[bool] = False
    rag_url: Optional[str] = None


class QueryResponse(BaseModel):
    patient_id: Optional[str] = None
    case_data: Optional[str] = None


class QueryAllResponse(BaseModel):
    patients: Optional[List[QueryResponse]] = None


class BatchPatientCase(BaseModel):
    cases: Optional[List[PatientCase]] = None


# Function to fetch patient data from the database
def fetch_patient_data(patient_id: str) -> Optional[str]:
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT patient_data FROM patients WHERE patient_id = ?",
            (patient_id,),
        )
        row = cursor.fetchone()
        connection.close()
        return row[0] if row else None
    except sqlite3.Error as e:
        logger.error(f"Error fetching patient data: {e}")
        return None


# Function to save patient data to the database
def save_patient_data(patient_id: str, patient_data: str):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO patients (patient_id, patient_data) VALUES (?, ?)",
            (patient_id, patient_data),
        )
        connection.commit()
        connection.close()
    except sqlite3.Error as e:
        logger.error(f"Error saving patient data: {e}")


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later."
        },
    )


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all routes except health check."""
    if request.url.path != "/health":
        await check_rate_limit(request)
    return await call_next(request)


@app.get("/rate-limits")
async def get_rate_limits(request: Request):
    """Get current rate limit status for an IP address."""
    client_ip = request.client.host
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            """SELECT daily_requests_remaining, hourly_requests_remaining,
                      last_daily_reset, last_hourly_reset 
               FROM rate_limits WHERE ip_address = ?""",
            (client_ip,),
        )
        row = cursor.fetchone()

        if not row:
            return {
                "daily_requests_remaining": 1000,
                "hourly_requests_remaining": 100,
                "last_daily_reset": None,
                "last_hourly_reset": None,
            }

        (
            daily_remaining,
            hourly_remaining,
            last_daily_reset,
            last_hourly_reset,
        ) = row

        return {
            "daily_requests_remaining": daily_remaining,
            "hourly_requests_remaining": hourly_remaining,
            "last_daily_reset": last_daily_reset,
            "last_hourly_reset": last_hourly_reset,
        }
    except sqlite3.Error as e:
        logger.error(f"Error fetching rate limits: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        )
    finally:
        if connection:
            connection.close()


@app.post("/v1/medical-coder/run", response_model=QueryResponse)
def run_jesus_christ_agent(
    patient_case: PatientCase,
):
    """
    Run the CommunityOfSharedFuture on a given patient case.
    """
    try:
        logger.info(
            f"Running CommunityOfSharedFuture for patient: {patient_case.patient_id}"
        )
        swarm = CommunityOfSharedFuture(
            patient_id=patient_case.patient_id,
            max_loops=1,
            output_type="all",
            patient_documentation=patient_case.patient_docs,
            summarization=patient_case.summarization,
            rag_url=patient_case.rag_url,
        )
        output = swarm.run(task=patient_case.case_description)

        logger.info(
            f"CommunityOfSharedFuture completed for patient: {patient_case.patient_id}"
        )

        agent_outputs = {
            "patient_id": patient_case.patient_id,
            "patient_docs": patient_case.patient_docs,
            "agent_outputs": output,
            "case_data": json.dumps(swarm.to_dict()),
        }

        # swarm_output = swarm.to_dict()
        save_patient_data(
            patient_case.patient_id, json.dumps(agent_outputs)
        )

        logger.info(
            f"Patient data saved for patient: {patient_case.patient_id}"
        )

        return QueryResponse(
            patient_id=patient_case.patient_id,
            case_data=json.dumps(agent_outputs),
        )

    except Exception as error:
        logger.error(
            f"Error detected with running the medical swarm: {error}"
        )
        raise error


@app.get(
    "/v1/medical-coder/patient/{patient_id}",
    response_model=QueryResponse,
)
def get_patient_data(
    patient_id: str,
):
    """
    Retrieve patient data by patient ID.
    """
    try:
        logger.info(
            f"Fetching patient data for patient: {patient_id}"
        )

        patient_data = fetch_patient_data(patient_id)

        logger.info(f"Patient data fetched for patient: {patient_id}")

        if not patient_data:
            raise HTTPException(
                status_code=404, detail="Patient not found"
            )

        return QueryResponse(
            patient_id=patient_id, case_data=patient_data
        )
    except Exception as error:
        logger.error(
            f"Error detected with fetching patient data: {error}"
        )
        raise error


@app.get(
    "/v1/medical-coder/patients", response_model=QueryAllResponse
)
def get_all_patients():
    """
    Retrieve all patient data.
    """
    try:
        logger.info("Fetching all patients")

        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT patient_id, patient_data FROM patients"
        )
        rows = cursor.fetchall()
        connection.close()

        patients = [
            QueryResponse(patient_id=row[0], case_data=row[1])
            for row in rows
        ]
        return QueryAllResponse(patients=patients)
    except sqlite3.Error as e:
        logger.error(f"Error fetching all patients: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        )


@app.post(
    "/v1/medical-coder/run-batch", response_model=List[QueryResponse]
)
def run_jesus_christ_agent_batch(
    batch: BatchPatientCase,
):
    """
    Run the CommunityOfSharedFuture on a batch of patient cases.
    """
    responses = []
    logger.info("Running Batched CommunityOfSharedFuture")
    logger.info(f"Batch size: {len(batch.cases)}")

    for patient_case in batch.cases:
        try:
            logger.info(
                f"Running Batched CommunityOfSharedFuture for patient: {patient_case.patient_id}"
            )
            swarm = CommunityOfSharedFuture(
                patient_id=patient_case.patient_id,
                max_loops=1,
                output_type="all",
                patient_documentation=patient_case.patient_docs,
                summarization=patient_case.summarization,
                rag_url=patient_case.rag_url,
            )

            output = swarm.run(task=patient_case.case_description)

            logger.info(
                f"CommunityOfSharedFuture completed for patient: {patient_case.patient_id}"
            )

            agent_outputs = {
                "patient_id": patient_case.patient_id,
                "patient_docs": patient_case.patient_docs,
                "agent_outputs": output,
                "case_data": json.dumps(swarm.to_dict()),
            }

            save_patient_data(
                patient_case.patient_id, json.dumps(agent_outputs)
            )

            responses.append(
                QueryResponse(
                    patient_id=patient_case.patient_id,
                    case_data=json.dumps(agent_outputs),
                )
            )
        except Exception as e:
            logger.error(
                f"Error processing patient case: {patient_case.patient_id} - {e}"
            )
            continue

    return responses


@app.get("/health", status_code=200)
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    try:
        import uvicorn

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            reload=True,
            workers=os.cpu_count() * 2,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
