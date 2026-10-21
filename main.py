from fastapi import FastAPI

app = FastAPI(title="A2A Invoice Agent")


@app.get("/")
def home():
    return {"message": "A2A Invoice Agent is running!"}