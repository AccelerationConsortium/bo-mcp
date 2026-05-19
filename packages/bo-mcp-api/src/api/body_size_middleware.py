"""ASGI middleware enforcing a maximum request-body size.

The earlier header-only preflight (``Content-Length`` check) only
guarded the well-behaved client. Three transport shapes can still
slip past it:

* a chunked transfer-encoding request advertises no length;
* a client deliberately under-states ``Content-Length`` to lie about
  body size;
* a ``multipart/...`` content type sent to a non-upload JSON route
  would be exempted by a loose multipart skip.

This middleware closes all three by (a) restricting the multipart
exemption to a configured allow-list of upload paths, (b) running a
``Content-Length`` preflight that short-circuits the fast path
without reading the body, and (c) draining the request body in the
middleware itself, counting bytes against the cap and rejecting
with HTTP 413 once exceeded. The drained body is replayed to the
downstream app via a synthetic ``receive`` callable so the route
handler sees the same bytes it would otherwise.

Buffering the body adds latency proportional to body size, but is
bounded by the configured cap (5 MiB by default for JSON routes).
Doing the check in middleware is the only way to reject the body
*before* FastAPI's body parser runs — FastAPI catches any exception
raised by ``receive`` and converts it to a generic 400, which would
mask the real cause.

References
----------
* RFC 9110 §15.5.14 (413 Content Too Large)
* ASGI specification, "Receive" / "Send" event shapes
  https://asgi.readthedocs.io/en/latest/specs/index.html
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

# ASGI message types are plain dicts; alias them for readability.
ASGIScope = dict[str, Any]
ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]

# HTTP methods that may carry a request body large enough to warrant
# the cap. GET / HEAD / DELETE / OPTIONS / TRACE are body-less by
# convention (RFC 9110); draining ``receive`` on those would consume
# the ``http.request`` sentinel the downstream app needs to detect
# client disconnects on streaming responses and produces spurious
# "No response returned" failures.
_BODY_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})


class BodySizeLimitMiddleware:
    """Reject HTTP request bodies above ``max_body_size`` bytes.

    Parameters
    ----------
    app:
        The downstream ASGI application.
    max_body_size:
        Inclusive byte limit. A request whose body exceeds this is
        rejected with HTTP 413.
    upload_paths:
        Paths that are *exempt* from this cap because they apply
        their own per-route streaming reader (the result-upload
        endpoint, primarily). Matched as a suffix check so the
        legacy ``/api/...`` alias is covered alongside the
        versioned ``/api/v1/...`` mount.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int,
        upload_paths: Iterable[str],
    ) -> None:
        self.app = app
        self.max_body_size = max_body_size
        self.upload_paths = tuple(upload_paths)

    async def __call__(self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        """Dispatch the ASGI call, enforcing the body-size cap on HTTP."""
        if scope.get("type") != "http" or self._is_exempt(scope):
            await self.app(scope, receive, send)
            return

        if self._declared_length_exceeds(scope):
            await self._send_413(send, self._declared_length(scope))
            return

        # Only methods that carry a body get the drain-and-replay
        # treatment. Body-less methods (GET, HEAD, ...) flow straight
        # through so the downstream ``receive`` channel keeps its
        # original semantics — replaying a synthetic ``http.disconnect``
        # under a ``StreamingResponse`` confuses the disconnect-
        # listener task and surfaces as "No response returned".
        if scope.get("method", "").upper() not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        body, oversize_total = await self._drain_body(receive)
        if oversize_total is not None:
            await self._send_413(send, oversize_total)
            return

        replay = _make_body_replay(body)
        await self.app(scope, replay, send)

    def _is_exempt(self, scope: ASGIScope) -> bool:
        """Skip the cap for upload endpoints (they cap themselves)."""
        path = scope.get("path", "")
        return any(path.endswith(suffix) for suffix in self.upload_paths)

    def _declared_length(self, scope: ASGIScope) -> int:
        """Return the parsed ``Content-Length`` header, or ``-1`` if missing/invalid."""
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return -1
        return -1

    def _declared_length_exceeds(self, scope: ASGIScope) -> bool:
        """Header preflight: True when the advertised length is over the cap."""
        declared = self._declared_length(scope)
        return declared > self.max_body_size

    async def _drain_body(self, receive: ASGIReceive) -> tuple[bytes, int | None]:
        """Read the entire request body, stopping at the cap.

        Returns ``(body, oversize_total)`` — when ``oversize_total``
        is not ``None`` the body was cut off because the running
        total crossed :attr:`max_body_size`; the caller must then
        emit a 413 instead of invoking the downstream app.
        """
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.request":
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.max_body_size:
                    return b"", total
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            elif message_type == "http.disconnect":
                # Client gave up mid-request. Return what we have;
                # the downstream app will see an empty body.
                break
            else:
                # Unknown message type — pass an empty body downstream.
                break
        return b"".join(chunks), None

    async def _send_413(self, send: ASGISend, observed: int) -> None:
        """Emit a minimal 413 response without invoking the downstream app."""
        body = json.dumps(
            {
                "detail": (
                    f"Request body of {observed} bytes exceeds the {self.max_body_size}-byte limit."
                )
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _make_body_replay(body: bytes) -> ASGIReceive:
    """Build a ``receive`` callable that replays a pre-buffered body once.

    The downstream app reads the body via ``receive``; we have
    already consumed it, so we re-emit it as a single
    ``http.request`` message and then signal "no more body".
    Subsequent calls return ``http.disconnect`` to mirror Starlette's
    behaviour for an idempotent reader.
    """
    sent = False

    async def replay() -> ASGIMessage:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return replay
