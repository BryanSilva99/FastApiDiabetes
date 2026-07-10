from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import get_settings


def get_db_path() -> Path:
    return get_settings().database_path


def get_connection():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
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
                probability REAL DEFAULT 0,
                risk_percentage REAL DEFAULT 0,
                threshold REAL DEFAULT 0.5,
                message TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_name TEXT DEFAULT 'unknown',
                model_version TEXT DEFAULT 'unknown',
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_columns(connection)


def _ensure_columns(connection: sqlite3.Connection) -> None:
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()}
    migrations = {
        "probability": "ALTER TABLE predictions ADD COLUMN probability REAL DEFAULT 0",
        "risk_percentage": "ALTER TABLE predictions ADD COLUMN risk_percentage REAL DEFAULT 0",
        "threshold": "ALTER TABLE predictions ADD COLUMN threshold REAL DEFAULT 0.5",
        "model_name": "ALTER TABLE predictions ADD COLUMN model_name TEXT DEFAULT 'unknown'",
        "model_version": "ALTER TABLE predictions ADD COLUMN model_version TEXT DEFAULT 'unknown'",
    }

    for column, statement in migrations.items():
        if column not in columns:
            connection.execute(statement)


def database_available() -> bool:
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def save_prediction(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
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
                probability,
                risk_percentage,
                threshold,
                message,
                recommendation,
                model_name,
                model_version,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                result["probability"],
                result["risk_percentage"],
                result["threshold"],
                result["message"],
                result["recommendation"],
                result["model_name"],
                result["model_version"],
                created_at,
            ),
        )
        return {"id": int(cursor.lastrowid), "created_at": created_at}


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
                probability,
                risk_percentage,
                threshold,
                message,
                recommendation,
                model_name,
                model_version,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_predictions() -> int:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM predictions")
        return int(cursor.rowcount)
