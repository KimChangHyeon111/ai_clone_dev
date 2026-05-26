import re
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field

from common.schemas import SimulationDataSchema, FGIDataSchema

class ContextManager:
    """단일 매니저로 시뮬레이션, FGI, 일반 프롬프트 렌더링을 모두 처리합니다."""

    GCS_PATTERN = re.compile(r'^gs://([^/]+)/(.*)$')

    def __init__(self):
        self.storage_client = storage.Client()
        self._template_cache: Dict[str, Template] = {}

    def _parse_gcs_path(self, path: str) -> Tuple[str, str]:
        """
        GCS 경로 문자열에서 버킷 이름과 오브젝트 경로를 분리합니다.

        Args:
            path (str): 'gs://bucket_name/path/to/blob' 형식의 경로.

        Returns:
            Tuple[str, str]: (버킷 이름, 오브젝트 경로)

        Raises:
            ValueError: 경로 형식이 올바르지 않은 경우 발생.
        """

        match = self.GCS_PATTERN.match(path)
        if not match:
            raise ValueError(f"잘못된 GCS 경로 형식입니다: {path}")
        return match.groups()

    def _get_template(self, path: str) -> Template:
        """
        GCS에서 템플릿 파일을 읽어 Jinja2 Template 객체로 반환합니다.
        이미 로드된 경로는 캐시를 활용합니다.

        Args:
            path (str): GCS 템플릿 파일 경로.

        Returns:
            Template: 컴파일된 Jinja2 템플릿 객체.

        Raises:
            RuntimeError: GCS 다운로드 중 오류 발생 시 발생.
        """
        if path in self._template_cache:
            return self._template_cache[path]

        bucket_name, blob_path = self._parse_gcs_path(path)
        blob = self.storage_client.bucket(bucket_name).blob(blob_path)
        content = blob.download_as_text(encoding='utf-8')

        template = Template(content)
        self._template_cache[path] = template
        return template

    def render_single_template(self, template_path: str, **kwargs) -> str:
        """단일 템플릿 렌더링 (예: Promotion 정보 추출용)"""
        template = self._get_template(template_path)
        return template.render(**kwargs)

    def build_prompt(self, sys_path: str, user_path: str, data: BaseModel) -> Tuple[str, str]:
        """Pydantic 스키마를 받아 시스템/유저 프롬프트 쌍을 생성 (시뮬레이션 & FGI 공용)"""
        sys_template = self._get_template(sys_path)
        user_template = self._get_template(user_path)

        data_dict = data.model_dump(by_alias=True)

        system_instruction = sys_template.render(MS=data_dict.get('MS', ''))
        user_prompt = user_template.render(**data_dict)

        return system_instruction, user_prompt
