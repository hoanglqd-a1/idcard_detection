from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..schemas.analysis import AnalyzeResponse, ErrorResponse, HealthResponse
from ..services.analysis import AnalysisError, analyze_image

router = APIRouter(prefix='/api/v1')


@router.get('/health', response_model=HealthResponse)
def health(request: Request):
    pipeline = getattr(request.app.state, 'pipeline', None)
    result = HealthResponse(status='ready' if pipeline else 'not_ready',
                            models_loaded=pipeline is not None,
                            template_count=len(pipeline.templates) if pipeline else 0)
    return JSONResponse(result.model_dump(), status_code=200 if pipeline else 503)


@router.post('/analyze', response_model=AnalyzeResponse,
             responses={code: {'model': ErrorResponse} for code in (400, 413, 415, 422, 500, 503)})
async def analyze(request: Request, file: Annotated[UploadFile, File()]):
    settings = request.app.state.settings
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise AnalysisError(503, 'not_ready', 'The inference service is not ready.')
    try:
        data = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if len(data) > settings.max_upload_bytes:
        raise AnalysisError(413, 'file_too_large', 'The uploaded file exceeds the size limit.')
    if not data:
        raise AnalysisError(422, 'invalid_image', 'The uploaded file is empty.')
    return await run_in_threadpool(analyze_image, data, pipeline, settings)
