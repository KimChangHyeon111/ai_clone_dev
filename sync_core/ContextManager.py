import re
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field

from common.schemas import SimulationDataSchema, FGIDataSchema

class ContextManager:
    """단일 매니저로 시뮬레이션, FGI, 일반 프롬프트 렌더링을 모두 처리합니다."""
    def __init__(self):
        self._template_cache: Dict[str, Template] = {}

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

        # 로컬 파일 읽기
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

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
