import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field

from common.schemas import SimulationDataSchema, FGIDataSchema

# ---------------------------------------------------------
# 2. 통합 비동기 ContextManager
# ---------------------------------------------------------
class ContextManagerAsync:
    """단일 매니저로 시뮬레이션, FGI, 일반 프롬프트 렌더링을 모두 비동기로 처리합니다."""

    GCS_PATTERN = re.compile(r'^gs://([^/]+)/(.*)$')

    def __init__(self):
        self.storage_client = storage.Client()
        self._template_cache: Dict[str, Template] = {}

    def _parse_gcs_path(self, path: str) -> Tuple[str, str]:
        match = self.GCS_PATTERN.match(path)
        if not match:
            raise ValueError(f"잘못된 GCS 경로 형식입니다: {path}")
        return match.groups()

    # 💡 1. async def로 변경하고 GCS 다운로드 네트워크 I/O 병목 제거
    async def _get_template(self, path: str) -> Template:
        if path in self._template_cache:
            return self._template_cache[path]

        bucket_name, blob_path = self._parse_gcs_path(path)
        blob = self.storage_client.bucket(bucket_name).blob(blob_path)

        # 💡 동기식 GCS 다운로드 함수를 asyncio.to_thread로 감싸 메인 이벤트 루프 방어
        content = await asyncio.to_thread(blob.download_as_text, encoding='utf-8')

        template = Template(content)
        self._template_cache[path] = template
        return template

    # 💡 2. async def로 변경하여 상위 파이프라인과 비동기 정렬
    async def render_single_template(self, template_path: str, **kwargs) -> str:
        """단일 템플릿 렌더링 (예: Promotion 정보 추출용)"""
        template = await self._get_template(template_path)
        # Jinja2 렌더링은 가벼운 문자열 치환 연산이므로 동기로 바로 처리합니다.
        return template.render(**kwargs)

    # 💡 3. async def로 변경 및 동시성(Concurreny) 극대화
    async def build_prompt(self, sys_path: str, user_path: str, data: BaseModel) -> Tuple[str, str]:
        """Pydantic 스키마를 받아 시스템/유저 프롬프트 쌍을 비동기로 생성 (시뮬레이션 & FGI 공용)"""

        # 💡 두 개의 템플릿 다운로드 태스크를 생성하여 asyncio.gather로 동시에 병렬 처리합니다.
        # 캐싱이 안 되어 있어도 네트워크 대기 시간이 최소화됩니다.
        sys_template_task = self._get_template(sys_path)
        user_template_task = self._get_template(user_path)

        sys_template, user_template = await asyncio.gather(sys_template_task, user_template_task)

        data_dict = data.model_dump(by_alias=True)

        system_instruction = sys_template.render(MS=data_dict.get('MS', ''))
        user_prompt = user_template.render(**data_dict)

        return system_instruction, user_prompt
