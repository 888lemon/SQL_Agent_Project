from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.config.settings import load_settings, validate_required_settings
from app.db.factory import DBFactory
from app.core.cache import ProSemanticCache
from app.core.agent import SQLProAgent
from app.api.endpoints import build_query_router
from app.api.schemas import QueryRequest

# 加载环境变量
load_dotenv()

app = FastAPI()

settings = load_settings()
validate_required_settings(settings)

# 初始化组件
db = DBFactory.get_handler(settings.db_type, settings.db_path)
cache = ProSemanticCache(
    model_name=settings.embedding_model_name,
    threshold=settings.cache_threshold,
    ttl_seconds=settings.cache_ttl_seconds,
    max_entries=settings.cache_max_entries,
)
agent = SQLProAgent(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    db_handler=db,
    cache_engine=cache
)
app.include_router(build_query_router(agent))

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "db_type": settings.db_type,
        "llm_configured": settings.has_valid_api_key,
        "last_error": agent.last_error,
    }

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        question = query.resolved_question()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return agent.ask(question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
