from fastapi import FastAPI, Request, Response
import httpx
import os
import uvicorn

# =========================
# Config
# =========================

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")   # Do not include /v1
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# =========================
# App
# =========================

app = FastAPI()


# =========================
# Universal OpenAI Proxy
# =========================

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def openai_proxy(path: str, request: Request):
    """
    Universal OpenAI-compatible proxy.
    Supports chat.completions, responses, models, embeddings, etc.
    """

    # Target URL
    target_url = f"{LLM_BASE_URL}/v1/{path}"

    # Copy headers
    headers = dict(request.headers)

    # Override authorization
    headers["authorization"] = f"Bearer {LLM_API_KEY}"
    headers.pop("host", None)

    # Read body
    body = await request.body()

    # Forward request
    async with httpx.AsyncClient(timeout=300) as client:
        upstream_resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

    # -------------------------
    # IMPORTANT: header cleanup
    # -------------------------
    resp_headers = dict(upstream_resp.headers)

    # httpx already decompressed the body
    # Do NOT forward these headers
    resp_headers.pop("content-encoding", None)
    resp_headers.pop("content-length", None)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=resp_headers.get("content-type"),
    )

def run_server(host="0.0.0.0", port=8001):
    """
    Run the OpenAI proxy server.
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # python -m structai.openai_server
    run_server()
