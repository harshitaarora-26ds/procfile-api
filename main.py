"""
Orders API

Implements:
  1. Idempotent order creation      - POST /orders  (Idempotency-Key header)
  2. Cursor-based pagination        - GET  /orders   (limit, cursor query params)
  3. Per-client rate limiting       - X-Client-Id header, 17 requests / 10s window
  4. CORS enabled for all origins

In-memory storage only (single-process); no external database required.
"""

import time
import itertools
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

TOTAL_ORDERS = 59          # fixed catalog size for pagination
RATE_LIMIT = 17            # max requests
RATE_WINDOW_SECONDS = 10   # per this many seconds
DEFAULT_PAGE_LIMIT = 10

app = FastAPI(title="Orders API")

# --------------------------------------------------------------------------
# In-memory state
# --------------------------------------------------------------------------

_order_id_counter = itertools.count(1)
_orders_by_idempotency_key: dict[str, dict] = {}

# Fixed catalog of orders used for the pagination endpoint.
_catalog = [
    {"id": i, "name": f"Order {i}", "status": "confirmed"}
    for i in range(1, TOTAL_ORDERS + 1)
]

# client_id -> deque of request timestamps within the current window
_client_buckets: dict[str, deque] = defaultdict(deque)


# --------------------------------------------------------------------------
# Rate limiting middleware
# --------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Buckets requests per X-Client-Id header. Requests without this header
    are not rate limited. Uses a sliding-window log so exactly RATE_LIMIT
    requests are allowed in any trailing RATE_WINDOW_SECONDS window.
    CORS preflight (OPTIONS) requests are always passed through untouched.

    NOTE: this middleware must be added to the app BEFORE CORSMiddleware
    (i.e. CORSMiddleware must be the outermost layer). Starlette wraps
    middleware in the reverse order they're added, so whichever is added
    last runs first on the way in / last on the way out. If this
    middleware were outermost, its short-circuited 429 responses would
    never pass through CORSMiddleware and would be missing CORS headers
    entirely — which browsers then refuse to let JS read cross-origin,
    surfacing as a generic "Failed to fetch" on the client.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        client_id = request.headers.get("x-client-id")

        if client_id:
            now = time.monotonic()
            bucket = _client_buckets[client_id]

            # drop timestamps outside the window
            while bucket and now - bucket[0] > RATE_WINDOW_SECONDS:
                bucket.popleft()

            if len(bucket) >= RATE_LIMIT:
                retry_after = max(1, int(RATE_WINDOW_SECONDS - (now - bucket[0])) + 1)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "rate limit exceeded",
                        "limit": RATE_LIMIT,
                        "window_seconds": RATE_WINDOW_SECONDS,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            bucket.append(now)

        return await call_next(request)


# Add RateLimitMiddleware first, then CORSMiddleware last, so CORSMiddleware
# ends up as the OUTERMOST layer and adds headers to every response —
# including RateLimitMiddleware's short-circuited 429s.
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "service": "orders-api"}


@app.options("/{rest_of_path:path}")
async def cors_preflight_fallback(rest_of_path: str):
    """
    Explicit catch-all OPTIONS handler so CORS preflight requests always
    get a clean 200 response (with CORS headers added by CORSMiddleware),
    regardless of whether a route defines OPTIONS itself.
    """
    return Response(status_code=200)


@app.post("/orders", status_code=201)
async def create_order(
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """
    Creates a new order. If an Idempotency-Key header is supplied and has
    been seen before, the previously created order is returned unchanged
    (no duplicate is created) with a 200 status. New orders return 201.
    """
    if idempotency_key:
        existing = _orders_by_idempotency_key.get(idempotency_key)
        if existing is not None:
            return JSONResponse(status_code=200, content=existing)

    order_id = next(_order_id_counter)
    order = {"id": str(order_id), "status": "created"}

    if idempotency_key:
        _orders_by_idempotency_key[idempotency_key] = order

    return JSONResponse(status_code=201, content=order)


@app.get("/orders")
async def list_orders(limit: int = DEFAULT_PAGE_LIMIT, cursor: Optional[str] = None):
    """
    Cursor-based pagination over a fixed set of TOTAL_ORDERS orders.
    - `limit` caps the number of items returned per page.
    - `cursor` is an opaque token from a previous response's next_cursor;
      omit it (or pass None) to fetch the first page.
    Repeatedly following next_cursor with a fixed limit visits every order
    exactly once, with no gaps or repeats, until next_cursor is null.

    The items list is returned under both `items` and `orders` keys, and
    the pagination token under both `next_cursor` and `next`, since
    consumers may look for either name.
    """
    if limit < 1:
        limit = 1

    if not cursor:
        start = 1
    else:
        try:
            start = int(cursor)
        except ValueError:
            start = 1
        if start < 1:
            start = 1

    end = min(start + limit - 1, TOTAL_ORDERS)

    if start > TOTAL_ORDERS:
        page_items: list[dict] = []
        next_cursor = None
    else:
        page_items = _catalog[start - 1 : end]
        next_start = end + 1
        next_cursor = str(next_start) if next_start <= TOTAL_ORDERS else None

    return {
        "items": page_items,
        "orders": page_items,
        "next_cursor": next_cursor,
        "next": next_cursor,
        "total": TOTAL_ORDERS,
    }
