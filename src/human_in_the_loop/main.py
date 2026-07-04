"""Main application entry point."""
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from human_in_the_loop.api.router import router
from human_in_the_loop.config.settings import get_settings

load_dotenv()

def create_app() -> FastAPI:
    app = FastAPI(title="Human-in-the-Loop API", description="Approval gates, review queues, active learning", version="1.0.0")
    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("human_in_the_loop.main:app", host=settings.api.host, port=settings.api.port, reload=settings.api.reload)
