import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "predictions.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pregnancies INTEGER NOT NULL,
                glucose REAL NOT NULL,
                blood_pressure REAL NOT NULL,
                skin_thickness REAL NOT NULL,
                insulin REAL NOT NULL,
                bmi REAL NOT NULL,
                diabetes_pedigree_function REAL NOT NULL,
                age INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                risk TEXT NOT NULL,
                message TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction(payload: dict[str, Any], result: dict[str, Any]) -> int:
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age,
                prediction,
                risk,
                message,
                recommendation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["Pregnancies"],
                payload["Glucose"],
                payload["BloodPressure"],
                payload["SkinThickness"],
                payload["Insulin"],
                payload["BMI"],
                payload["DiabetesPedigreeFunction"],
                payload["Age"],
                result["prediction"],
                result["risk"],
                result["message"],
                result["recommendation"],
                created_at,
            ),
        )
        return int(cursor.lastrowid)


def list_predictions(limit: int = 20):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age,
                prediction,
                risk,
                message,
                recommendation,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
