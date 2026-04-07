"""
감정인식 FastAPI 백엔드

엔드포인트:
  GET  /api/health              서버·모델 상태
  GET  /api/models              모델 목록 + 성능 지표
  POST /api/analyze             이미지(파일) → 단일 모델 감정 분석
  POST /api/analyze/compare     이미지(파일) → 전체 모델 비교 분석
  POST /api/analyze/base64      base64 이미지 → 단일 모델 (웹캠용)
  GET  /api/pipeline/{name}     시각화 이미지 제공

Usage:
  python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
"""
import base64
import logging
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server.predictor import ModelManager, PIPELINE_IMAGES, BASE_DIR, detect_and_crop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Face EMG Emotion API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

manager = ModelManager()


@app.on_event('startup')
async def startup():
    manager.load_all()
    logger.info(f'서버 준비. 로드 모델: {list(manager.predictors.keys())}')


# ── 헬스체크 ──────────────────────────────────────────────────────────────────

@app.get('/api/health')
def health():
    import torch
    return {
        'status':        'ok',
        'loaded_models': list(manager.predictors.keys()),
        'device':        str(next(iter(manager.predictors.values())).device)
                         if manager.predictors else 'none',
        'cuda':          torch.cuda.is_available(),
    }


# ── 모델 목록 + 성능 지표 ─────────────────────────────────────────────────────

@app.get('/api/models')
def get_models():
    return {'models': manager.available_models()}


# ── 이미지 공통 전처리 ────────────────────────────────────────────────────────

def _decode_image(contents: bytes) -> np.ndarray:
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail='이미지 디코딩 실패')
    # 긴 변 1280px 제한
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return img_bgr


# ── 단일 모델 분석 ────────────────────────────────────────────────────────────

@app.post('/api/analyze')
async def analyze(
    file:     UploadFile = File(...),
    model_id: str        = Form(default='densenet121'),
):
    """
    이미지 업로드 → 지정 모델로 감정 분석.
    Returns: emotion, confidence, scores, face_b64, face_detected, infer_ms
    """
    import traceback as tb
    try:
        if model_id not in manager.predictors:
            loaded = list(manager.predictors.keys())
            if not loaded:
                raise HTTPException(status_code=503, detail='로드된 모델 없음')
            model_id = loaded[0]

        contents = await file.read()
        logger.info(f'analyze: model={model_id} file={file.filename} size={len(contents)}')
        img_bgr = _decode_image(contents)
        logger.info(f'  decoded: shape={img_bgr.shape}')
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)
        logger.info(f'  face: bbox={bbox} face_shape={face_rgb.shape}')

        result = manager.predict_one(model_id, face_rgb)
        if result is None:
            raise HTTPException(status_code=503, detail=f'모델 추론 실패: {model_id}')

        return {
            **result,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
            'bbox':          bbox,
            'model_id':      model_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'analyze 500:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


# ── 전체 모델 비교 분석 ───────────────────────────────────────────────────────

@app.post('/api/analyze/compare')
async def analyze_compare(file: UploadFile = File(...)):
    """
    이미지 업로드 → 로드된 모든 모델로 비교 분석.
    Returns: results (모델별 예측), face_b64, face_detected
    """
    import traceback as tb
    try:
        contents = await file.read()
        logger.info(f'compare: file={file.filename} size={len(contents)}')
        img_bgr = _decode_image(contents)
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)

        results = manager.predict_all(face_rgb)
        if not results:
            raise HTTPException(status_code=503, detail='로드된 모델 없음')

        return {
            'results':       results,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
            'bbox':          bbox,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'compare 500:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


# ── base64 분석 (웹캠용) ──────────────────────────────────────────────────────

@app.post('/api/analyze/base64')
async def analyze_base64(payload: dict):
    """
    payload: { image_b64, model_id (optional), compare (optional bool) }
    """
    image_b64 = payload.get('image_b64', '')
    model_id  = payload.get('model_id', 'densenet121')
    compare   = payload.get('compare', False)

    if not image_b64:
        raise HTTPException(status_code=400, detail='image_b64 없음')

    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    img_bytes = base64.b64decode(image_b64)
    img_bgr   = _decode_image(img_bytes)

    bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)

    if compare:
        results = manager.predict_all(face_rgb)
        return {
            'results':       results,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
        }
    else:
        if model_id not in manager.predictors:
            loaded = list(manager.predictors.keys())
            if not loaded:
                raise HTTPException(status_code=503, detail='로드된 모델 없음')
            model_id = loaded[0]

        result = manager.predict_one(model_id, face_rgb)
        return {
            **result,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
            'model_id':      model_id,
        }


# ── 파이프라인 시각화 이미지 ──────────────────────────────────────────────────

@app.get('/api/pipeline/{name}')
def get_pipeline_image(name: str):
    """
    name: edge_samples | gradcam_samples | class_gradcam | tsne | comparison
    """
    if name not in PIPELINE_IMAGES:
        raise HTTPException(status_code=404, detail=f'알 수 없는 이미지: {name}')
    path = os.path.join(BASE_DIR, PIPELINE_IMAGES[name])
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f'이미지 없음: {path}')
    return FileResponse(path, media_type='image/png')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server.main:app', host='0.0.0.0', port=8000, reload=False)
