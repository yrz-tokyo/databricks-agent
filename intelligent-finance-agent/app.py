"""
app.py - Intelligent Finance Agent FastAPI Backend

Databricks Apps として動作するバックエンド。
Model Serving エンドポイントを呼び出してチャット応答を返す。

開発ガイドライン (CLAUDE.md セクション7):
- WorkspaceClient を使用（Databricks Apps が DATABRICKS_TOKEN を自動注入）
- asyncio.to_thread() で同期 SDK 呼び出しをラップ
- SSE (Server-Sent Events) でストリーミングレスポンス
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "finance-agent-endpoint")
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet")

app = FastAPI(
    title="Intelligent Finance Agent API",
    description="IR資料を使った経営質問応答とインサイト生成",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# Request / Response Models
# ========================================


class ChatRequest(BaseModel):
    question: str
    endpoint_name: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    rag_response: str
    rag_sources: list
    insight_response: str
    success: bool


# ========================================
# Agent 呼び出しロジック
# ========================================


def _query_serving_endpoint(question: str, endpoint_name: str) -> str:
    """
    Model Serving エンドポイントを呼び出す。
    DATABRICKS_TOKEN 環境変数（Databricks Secrets 経由）を使って認証する。
    """
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise ValueError("DATABRICKS_TOKEN is not set")

    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messages": [{"role": "user", "content": question}]
    }

    logger.info(f"Calling endpoint: {url}")
    resp = http_requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"Response keys: {list(data.keys())}")

    # OpenAI 互換フォーマット: {"choices": [{"message": {"content": "..."}}]}
    if "choices" in data and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content", "")
        if content:
            return content

    # pyfunc / ChatAgent フォーマット: {"messages": [...]}
    if "messages" in data and data["messages"]:
        for msg in reversed(data["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    return content

    # predictions フォーマット: {"predictions": [...]}
    if "predictions" in data and data["predictions"]:
        pred = data["predictions"][0]
        if isinstance(pred, dict):
            messages = pred.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content:
                        return content
        return str(pred)

    logger.warning(f"Unexpected response format: {data}")
    return "応答を取得できませんでした"


def _generate_insight(rag_content: str, question: str, llm_endpoint: str) -> str:
    """
    RAG 回答を元に経営インサイトを生成する。
    Foundation Model API を直接呼び出す。
    """
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise ValueError("DATABRICKS_TOKEN is not set")

    insight_prompt = f"""以下の質問とIR資料からの回答を分析し、経営インサイトを提供してください。

質問: {question}

IR資料からの回答:
{rag_content}

以下の観点で経営インサイトを生成してください:

1. **主要なトレンドと変化**
   - データから読み取れる傾向・前年比分析

2. **潜在的なリスクと機会**
   - 懸念事項・成長の機会

3. **実行可能な提言**
   - 具体的なアクションアイテム

具体的な数値やファクトを引用し、根拠のあるインサイトを提供してください。"""

    url = f"{host}/serving-endpoints/{llm_endpoint}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "あなたは経営コンサルタントです。"},
            {"role": "user", "content": insight_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }

    resp = http_requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "choices" in data and data["choices"]:
        return data["choices"][0].get("message", {}).get("content", "")
    return "インサイトを生成できませんでした"


async def run_finance_agent_streaming(
    question: str,
    endpoint_name: str,
) -> AsyncGenerator[dict, None]:
    """
    Model Serving エンドポイントを呼び出し、SSE イベントとして結果を返す。
    """
    # Step 1: 接続確認
    yield {"type": "step", "id": "connect", "content": "エージェントエンドポイントに接続中...", "status": "running"}
    await asyncio.sleep(0.3)
    yield {"type": "step", "id": "connect", "content": "エージェントエンドポイントに接続中...", "status": "done"}

    # Step 2: RAG（エンドポイント呼び出し）
    yield {"type": "step", "id": "search", "content": "IR文書をベクター検索中...", "status": "running"}
    yield {"type": "rag", "status": "running"}

    rag_content = ""
    try:
        rag_content = await asyncio.to_thread(
            _query_serving_endpoint, question, endpoint_name
        )
        yield {"type": "step", "id": "search", "content": "IR文書をベクター検索中...", "status": "done"}
        yield {"type": "step", "id": "generate", "content": "LLMによる回答を生成しました", "status": "done"}
        yield {"type": "rag", "status": "complete", "content": rag_content, "sources": []}
    except Exception as e:
        logger.error(f"Serving endpoint error: {e}")
        yield {"type": "step", "id": "search", "content": "IR文書をベクター検索中...", "status": "error"}
        yield {"type": "error", "content": f"エラーが発生しました: {str(e)}"}
        yield {"type": "done"}
        return

    # Step 3: インサイト生成
    yield {"type": "step", "id": "insight", "content": "経営インサイトを生成中...", "status": "running"}
    yield {"type": "insight", "status": "running"}

    try:
        insight_content = await asyncio.to_thread(
            _generate_insight, rag_content, question, LLM_ENDPOINT_NAME
        )
        yield {"type": "step", "id": "insight", "content": "経営インサイトを生成しました", "status": "done"}
        yield {"type": "insight", "status": "complete", "content": insight_content}
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        yield {"type": "step", "id": "insight", "content": "インサイト生成中...", "status": "error"}
        yield {"type": "insight", "status": "complete", "content": f"インサイト生成に失敗しました: {str(e)}"}

    yield {"type": "done"}


# ========================================
# API Endpoints
# ========================================


@app.get("/api")
async def root():
    return {
        "message": "Intelligent Finance Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat_stream": "/api/chat/stream (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "intelligent-finance-agent"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    ストリーミングチャットエンドポイント（SSE）

    Model Serving エンドポイントを呼び出し、SSE 形式でレスポンスを返す。
    """
    logger.info(f"Chat stream request: {request.question}")

    endpoint_name = request.endpoint_name or SERVING_ENDPOINT_NAME

    async def event_generator():
        try:
            async for event in run_finance_agent_streaming(
                question=request.question,
                endpoint_name=endpoint_name,
            ):
                event_data = json.dumps(event, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
        except Exception as e:
            logger.error(f"Error in event_generator: {e}")
            error_event = json.dumps(
                {"type": "error", "content": f"エラーが発生しました: {str(e)}"},
                ensure_ascii=False,
            )
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非ストリーミングチャットエンドポイント（テスト用）"""
    logger.info(f"Chat request: {request.question}")

    endpoint_name = request.endpoint_name or SERVING_ENDPOINT_NAME

    try:
        content = await asyncio.to_thread(
            _query_serving_endpoint, request.question, endpoint_name
        )
        return ChatResponse(
            question=request.question,
            rag_response=content,
            rag_sources=[],
            insight_response="",
            success=True,
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# Static Files（ビルド済みフロントエンド - Vite build）
# ========================================

frontend_path = Path(__file__).parent / "frontend" / "build"
if frontend_path.exists():
    # Vite の出力構造: build/assets/*.js, build/assets/*.css
    assets_path = frontend_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        from fastapi.responses import HTMLResponse
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text())
        raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
