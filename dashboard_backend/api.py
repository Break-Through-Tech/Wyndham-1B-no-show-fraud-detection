
from io import BytesIO
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
    allow_methods=["GET", "POST"],
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
    # Read uploaded files
    reservation_data = read_upload(
        reservations, reservations.file.read()
    )
    stay_data = read_upload(
        stays, stays.file.read()
    )

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

    # Return flagged reservations for the dashboard
    return {
        "total_reservations": len(results),
        "flagged_reservations": len(flagged),
        "flagged_rate": round(
            len(flagged) / len(results) * 100, 4
        ),
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
            }
            for row in flagged.itertuples()
        ],
    }
