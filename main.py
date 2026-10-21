import os

import psycopg
from fastapi import FastAPI, HTTPException

app = FastAPI(title="A2A Invoice Agent")


@app.get("/")
def home():
    return {"message": "A2A Invoice Agent is running!"}


@app.get("/db-check")
def check_database():
    database_url = os.getenv("POSTGRES_URL")

    if not database_url:
        raise HTTPException(
            status_code=500,
            detail="Database configuration is missing",
        )

    try:
        with psycopg.connect(
            database_url,
            connect_timeout=5,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()

        return {"database": "connected", "test": result[0]}

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed",
        )
