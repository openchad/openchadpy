from fastapi import Request, Response, HTTPException
import httpx
import time
import logging
# Configure logging
logger = logging.getLogger(__name__)
# In-memory cache: { url: { data, expires, link, content_type, redirected, final_url } }
cache = {}
# Cache duration: 10 minutes (in seconds)
CACHE_TTL = 60 * 10

async def proxy_handler(request: Request):
    target_url = request.query_params.get("url")
    if not target_url:
        return Response(content='{"error": "Missing ?url= parameter"}', status_code=400, media_type="application/json")
    method = request.method.upper()
    # ✅ Only cache GET requests
    if method == "GET":
        cached = cache.get(target_url)
        now = time.time()
        if cached and cached["expires"] > now:
            logger.info(f"🟢 Cache hit: {target_url}")
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": cached["content_type"] or "application/json",
                "X-Cache": "HIT",
                "X-Upstream-Redirected": str(cached["redirected"]).lower(),
            }
            if cached["final_url"]:
                headers["X-Upstream-Final-Url"] = cached["final_url"]
            if cached["link"]:
                headers["Link"] = cached["link"]
            return Response(content=cached["data"], headers=headers)
    logger.info(f"🟡 Fetching fresh: {target_url}")
    # Prepare headers to forward
    forwarded_headers = dict(request.headers)
    forwarded_headers.pop("host", None)
    forwarded_headers.pop("origin", None)
    forwarded_headers.pop("content-length", None) # dynamic content length might change
    forwarded_headers.pop("accept-encoding", None) # Let httpx handle encoding/decoding
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        try:
            body = await request.body() if method not in ["GET", "HEAD"] else None
            resp = await client.request(
                method,
                target_url,
                headers=forwarded_headers,
                content=body
            )
            data = resp.content # bytes
            content_type = resp.headers.get("content-type", "application/json")
            link = resp.headers.get("link")
            redirected = resp.history and len(resp.history) > 0 or str(resp.url) != target_url
            final_url = str(resp.url) if resp.url else None
            # 💾 Cache only successful GET responses
            if method == "GET" and resp.status_code >= 200 and resp.status_code < 300:
                cache[target_url] = {
                    "data": data,
                    "expires": time.time() + CACHE_TTL,
                    "link": link,
                    "content_type": content_type,
                    "redirected": redirected,
                    "final_url": final_url,
                }
            response_headers = {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": content_type,
                "X-Cache": "MISS",
                "X-Upstream-Redirected": str(redirected).lower(),
            }
            if final_url:
                response_headers["X-Upstream-Final-Url"] = final_url
            if link:
                response_headers["Link"] = link
            return Response(
                content=data,
                status_code=resp.status_code,
                headers=response_headers
            )
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return Response(content=f'{{"error": "{str(e)}"}}', status_code=500, media_type="application/json")
