import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .api.routes import router
from .api.body_limit import BodyLimitMiddleware
from .config import Settings
from .inference.pipeline import IDCardPipeline, Pipeline
from .schemas.analysis import ErrorDetail, ErrorResponse
from .services.analysis import AnalysisError

logger = logging.getLogger(__name__)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(body.model_dump(), status_code=status, headers={'Cache-Control': 'no-store'})


def create_app(settings: Settings | None = None,
               pipeline_factory: Callable[[Settings], Pipeline] = IDCardPipeline) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup clearly on missing or incompatible assets.
        app.state.pipeline = pipeline_factory(settings)
        yield
        app.state.pipeline = None

    app = FastAPI(title='CardScope API', version='1.0.0', lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = None
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)

    @app.middleware('http')
    async def upload_guard(request: Request, call_next):
        length = request.headers.get('content-length')
        if length:
            try:
                oversized = int(length) > settings.max_upload_bytes + 64 * 1024
            except ValueError:
                return error_response(400, 'invalid_request', 'Invalid Content-Length header.')
            if oversized:
                return error_response(413, 'file_too_large', 'The uploaded request exceeds the size limit.')
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.exception_handler(AnalysisError)
    async def analysis_error(request: Request, exc: AnalysisError):
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(422, 'invalid_request', 'Upload one image in the multipart field named file.')

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error_response(exc.status_code, 'http_error', str(exc.detail))

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        logger.exception('Image analysis failed', exc_info=exc)
        return error_response(500, 'inference_error', 'Analysis failed. Please retry or use another image.')

    app.include_router(router)
    return app


app = create_app()
