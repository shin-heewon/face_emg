"""
Vercel Python 서버리스 진입점.

Vercel은 api/ 디렉토리의 Python 파일을 서버리스 함수로 실행.
FastAPI ASGI 앱을 그대로 노출.

주의: PyTorch 모델(.pth)은 Vercel 250MB 제한으로 직접 배포 불가.
      모델 경로는 환경변수 MODEL_BASE_DIR 또는 외부 스토리지로 오버라이드.
"""
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.main import app  # noqa: F401 — Vercel이 `app`을 ASGI 앱으로 인식
