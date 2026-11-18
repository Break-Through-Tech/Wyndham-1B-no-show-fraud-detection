
from io import BytesIO
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
import math
import numbers
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# Allow imports from dashboard_backend
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import predict_fraud

app = FastAPI(title="Wyndham Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)

def read_upload(file: UploadFile, contents: bytes) -> pd.DataFrame:
    name = (file.filename or "").lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(BytesIO(contents), dtype={
                "confirmation_number": "string",
                "member_number": "string",
            })

        if name.endswith(".xlsx"):
            return pd.read_excel(BytesIO(contents), dtype={
                "confirmation_number": "string",
                "member_number": "string",
            })

        raise HTTPException(
            status_code=400,
            detail="Upload a CSV or XLSX file.",
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {file.filename}: {error}",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    reservations: UploadFile = File(...),
    stays: UploadFile = File(...),
):
    # Dataset identity is stable across identical re-uploads, independent of filenames.
    reservation_bytes = reservations.file.read()
    stay_bytes = stays.file.read()
    digest = hashlib.sha256()
    for content in (reservation_bytes, stay_bytes):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    dataset_id = digest.hexdigest()[:24]
    reservation_data = read_upload(reservations, reservation_bytes)
    stay_data = read_upload(stays, stay_bytes)

    # Check required columns
    reservation_columns = {
        "confirmation_number",
        "member_number",
        "booking_timestamp",
        "check_in_date",
        "check_out_date",
        "member_enrollment_date",
        "rate_code",
        "booking_channel",
        "room_revenue",
        "points_redeemed",
    }

    stay_columns = {
        "confirmation_number",
        "qualification_code",
        "points_earned",
    }

    if not reservation_columns.issubset(reservation_data.columns):
        missing = sorted(
            reservation_columns - set(reservation_data.columns)
        )
        raise HTTPException(
            status_code=400,
            detail=f"Reservations missing columns: {missing}",
        )

    if not stay_columns.issubset(stay_data.columns):
        missing = sorted(
            stay_columns - set(stay_data.columns)
        )
        raise HTTPException(
            status_code=400,
            detail=f"Stays missing columns: {missing}",
        )

    # Prevent duplicate or missing reservation IDs
    for name, data in [
        ("Reservations", reservation_data),
        ("Stays", stay_data),
    ]:
        if (
            data["confirmation_number"].isna().any()
            or data["confirmation_number"].duplicated().any()
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{name} has missing or duplicate confirmation numbers.",
            )

    # Join stay outcomes to reservation records
    combined = reservation_data.merge(
        stay_data[
            [
                "confirmation_number",
                "qualification_code",
                "points_earned",
            ]
        ],
        on="confirmation_number",
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != len(reservation_data):
        raise HTTPException(
            status_code=400,
            detail="Some reservations do not have matching stay records.",
        )

    if combined.empty:
        raise HTTPException(
            status_code=400,
            detail="No matching reservations found.",
        )

    # Prepare features and run Model 2
    try:
        results = predict_fraud(combined)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Data preparation failed: {error}",
        )

    flagged = results[
        results["fraud_prediction"] == 1
    ].sort_values("fraud_risk_score", ascending=False)

    # Join prediction identifiers back to the uploaded reservation/stay data.
    # This preserves the trained feature pipeline and exposes only fields
    # genuinely supplied in the files (not inferred property/geography/SHAP).
    detail_fields = [
        "confirmation_number", "booking_timestamp", "check_in_date",
        "check_out_date", "rate_code", "booking_channel",
        "qualification_code", "points_earned", "points_redeemed",
        "room_revenue", "member_enrollment_date",
    ]
    details = combined[detail_fields].set_index("confirmation_number")

    def safe_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, numbers.Number):
            number = float(value)
            return number if math.isfinite(number) else None
        return str(value)

    # Chart is a count of flagged reservations by *check-in date* in the
    # uploaded batch, not an assertion about production monitoring history.
    checkin_dates = pd.to_datetime(combined["check_in_date"], errors="coerce").dt.normalize()
    latest = checkin_dates.max()
    daily_trend = []
    if pd.notna(latest):
        flagged_ids = set(flagged["confirmation_number"].astype(str))
        flags_by_day = combined.loc[
            combined["confirmation_number"].astype(str).isin(flagged_ids),
            ["check_in_date"],
        ].copy()
        flags_by_day["day"] = pd.to_datetime(
            flags_by_day["check_in_date"], errors="coerce"
        ).dt.normalize()
        counts = flags_by_day.groupby("day").size()
        for day in pd.date_range(end=latest, periods=30, freq="D"):
            daily_trend.append({
                "date": day.date().isoformat(),
                "flagged_count": int(counts.get(day, 0)),
            })

    # Return flagged reservations and their uploaded context for the dashboard.
    return {
        "dataset_id": dataset_id,
        "total_reservations": len(results),
        "flagged_reservations": len(flagged),
        "flagged_rate": round(
            len(flagged) / len(results) * 100, 4
        ),
        "daily_trend": daily_trend,
        "predictions": [
            {
                "confirmation_number": str(row.confirmation_number),
                "member_number": (
                    None if pd.isna(row.member_number)
                    else str(row.member_number)
                ),
                "fraud_risk_score": round(
                    float(row.fraud_risk_score), 6
                ),
                "fraud_prediction": int(row.fraud_prediction),
                **{
                    field: safe_value(details.loc[str(row.confirmation_number), field])
                    for field in detail_fields
                    if field != "confirmation_number"
                },
            }
            for row in flagged.itertuples()
        ],
    }


# Local prototype case store. Review decisions never modify model outputs.
# WYNDHAM_REVIEW_DB is only meant for tests; the default is the real store.
DB_PATH = Path(os.environ.get("WYNDHAM_REVIEW_DB") or Path(__file__).resolve().parent / "review_cases.sqlite3")
CLOSED_STATUSES = ("cleared", "escalated", "confirmed_fraud")


@contextmanager
def db_connection():
    # isolation_level=None -> we control transactions explicitly (BEGIN IMMEDIATE),
    # and the connection is always closed (sqlite3's own `with` only commits).
    connection = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        connection.execute("""CREATE TABLE IF NOT EXISTS case_reviews (
            dataset_id TEXT NOT NULL,
            confirmation_number TEXT NOT NULL,
            status TEXT NOT NULL,
            notes_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, confirmation_number)
        )""")
        yield connection
    finally:
        connection.close()


def parse_notes(raw):
    try:
        notes = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return notes if isinstance(notes, list) else []


class CaseUpdate(BaseModel):
    status: Literal["flagged", "reviewing", "cleared", "escalated", "confirmed_fraud"] | None = None
    note: str | None = Field(default=None, max_length=5000)


@app.get("/cases/{dataset_id}")
def list_cases(dataset_id: str):
    with db_connection() as connection:
        rows = connection.execute(
            "SELECT confirmation_number, status, notes_json, updated_at FROM case_reviews WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchall()
    return {"cases": {key: {"status": status, "notes": parse_notes(notes), "updated_at": updated}
                      for key, status, notes, updated in rows}}


@app.patch("/cases/{dataset_id}/{confirmation_number:path}")
def update_case(dataset_id: str, confirmation_number: str, change: CaseUpdate):
    if not dataset_id or len(dataset_id) > 64 or not confirmation_number or len(confirmation_number) > 200:
        raise HTTPException(status_code=400, detail="Invalid case identifier")
    note = change.note.strip() if change.note else ""
    if change.status is None and not note:
        raise HTTPException(status_code=400, detail="Provide a status or a note")
    with db_connection() as connection:
        # Take the write lock before reading so concurrent saves cannot lose notes.
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT status, notes_json FROM case_reviews WHERE dataset_id = ? AND confirmation_number = ?",
                (dataset_id, confirmation_number),
            ).fetchone()
            previous_status = row[0] if row else "flagged"
            notes = parse_notes(row[1]) if row else []
            status = change.status or previous_status
            if change.status in CLOSED_STATUSES and previous_status != "reviewing":
                raise HTTPException(status_code=409, detail="Start or reopen review before closing the case")
            if change.status in CLOSED_STATUSES and not note:
                raise HTTPException(status_code=400, detail="A decision rationale is required")
            now = datetime.now(timezone.utc).isoformat()
            if note:
                notes.append({"text": note, "created_at": now})
            connection.execute("""INSERT INTO case_reviews(dataset_id,confirmation_number,status,notes_json,updated_at)
                VALUES(?,?,?,?,?) ON CONFLICT(dataset_id,confirmation_number) DO UPDATE SET
                status=excluded.status, notes_json=excluded.notes_json, updated_at=excluded.updated_at""",
                (dataset_id, confirmation_number, status, json.dumps(notes), now),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return {"case": {"status": status, "notes": notes, "updated_at": now}}
