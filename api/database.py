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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                alias TEXT,
                age INTEGER,
                preferred_theme TEXT NOT NULL DEFAULT 'system',
                text_scale REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
        "updated_at": "ALTER TABLE predictions ADD COLUMN updated_at TEXT",
    }

    for column, statement in migrations.items():
        if column not in columns:
            connection.execute(statement)

    connection.execute("UPDATE predictions SET updated_at = created_at WHERE updated_at IS NULL")


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
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                created_at,
            ),
        )
        return {"id": int(cursor.lastrowid), "created_at": created_at, "updated_at": created_at}


def _prediction_select_sql(where_clause: str = "") -> str:
    return f"""
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
            created_at,
            updated_at
        FROM predictions
        {where_clause}
    """


def list_predictions(limit: int = 20):
    with get_connection() as connection:
        rows = connection.execute(
            _prediction_select_sql("ORDER BY id DESC LIMIT ?"),
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_prediction(prediction_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            _prediction_select_sql("WHERE id = ?"),
            (prediction_id,),
        ).fetchone()

    return dict(row) if row else None


def update_prediction(prediction_id: int, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    updated_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE predictions
            SET
                pregnancies = ?,
                glucose = ?,
                blood_pressure = ?,
                skin_thickness = ?,
                insulin = ?,
                bmi = ?,
                diabetes_pedigree_function = ?,
                age = ?,
                prediction = ?,
                risk = ?,
                probability = ?,
                risk_percentage = ?,
                threshold = ?,
                message = ?,
                recommendation = ?,
                model_name = ?,
                model_version = ?,
                updated_at = ?
            WHERE id = ?
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
                updated_at,
                prediction_id,
            ),
        )

    if cursor.rowcount == 0:
        return None

    return get_prediction(prediction_id)


def delete_prediction(prediction_id: int) -> bool:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        return cursor.rowcount > 0


def delete_predictions() -> int:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM predictions")
        return int(cursor.rowcount)


def get_profile() -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, alias, age, preferred_theme, text_scale, created_at, updated_at
            FROM profile
            WHERE id = 1
            """
        ).fetchone()

    return dict(row) if row else None


def save_profile(payload: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO profile (id, alias, age, preferred_theme, text_scale, created_at, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("alias"),
                payload.get("age"),
                payload["preferred_theme"],
                payload["text_scale"],
                now,
                now,
            ),
        )

    profile = get_profile()
    if profile is None:
        raise RuntimeError("No se pudo guardar el perfil")
    return profile


def update_profile(payload: dict[str, Any]) -> dict[str, Any] | None:
    current = get_profile()
    if current is None:
        return None

    updated = {
        "alias": payload.get("alias", current["alias"]),
        "age": payload.get("age", current["age"]),
        "preferred_theme": payload.get("preferred_theme", current["preferred_theme"]),
        "text_scale": payload.get("text_scale", current["text_scale"]),
    }
    updated_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE profile
            SET alias = ?, age = ?, preferred_theme = ?, text_scale = ?, updated_at = ?
            WHERE id = 1
            """,
            (
                updated["alias"],
                updated["age"],
                updated["preferred_theme"],
                updated["text_scale"],
                updated_at,
            ),
        )

    return get_profile()


def delete_profile() -> bool:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM profile WHERE id = 1")
        return cursor.rowcount > 0
