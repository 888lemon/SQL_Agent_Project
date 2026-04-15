from fastapi import APIRouter, HTTPException
from app.api.schemas import QueryRequest


def build_query_router(agent) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["query"])

    @router.post("/query")
    async def handle_query(request: QueryRequest):
        try:
            question = request.resolved_question()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return agent.ask(question)

    return router
