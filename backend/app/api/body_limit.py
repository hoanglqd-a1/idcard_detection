"""Bound multipart request bodies even without a Content-Length header."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope['type'] != 'http' or scope['method'] != 'POST' or scope['path'] != '/api/v1/analyze':
            return await self.app(scope, receive, send)
        # The bounded buffer prevents multipart parsing from accepting an
        # arbitrarily large chunked upload before endpoint validation runs.
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.max_bytes:
                response = JSONResponse({'error': {'code': 'file_too_large',
                    'message': 'The uploaded request exceeds the size limit.'}},
                    status_code=413, headers={'Cache-Control': 'no-store'})
                return await response(scope, receive, send)
            if not message.get('more_body', False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        await self.app(scope, replay, send)
