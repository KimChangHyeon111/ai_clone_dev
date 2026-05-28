from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------
# 1. 입력 데이터 스키마 정의 (기존 데이터 구조 완벽 유지)
# ---------------------------------------------------------
class SimulationDataSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    ms: str = Field(..., min_length=5, description='페르소나의 핵심 마인드셋', alias='MS')
    ml: Optional[List[Any]] = Field(default=[], description='관련 상품 거래 이력', alias='ML')
    pixel: Optional[List[Any]] = Field(default=[], description='고객 마이크로 어트리뷰트', alias='PIXEL')
    promo_info: str = Field(..., description="타겟 상품의 프로모션 정보", alias='PROMOTION_INFO')

class FGIDataSchema(SimulationDataSchema):
    conversation_history: str = Field(default='', description='이전 대화 맥락', alias='CONVERSATION_HISTORY')
    user_input: str = Field(..., description='현재 인터뷰어의 질문 또는 입력', alias='USER_INPUT')

class MultiAgentFGIDataSchema(FGIDataSchema):
    current_name: str = Field(default='', description='현재 답변 중인 에이전트 이름', alias='CURRENT_NAME')
    other_names: str = Field(default='', description='참여 중인 다른 에이전트들의 이름 목록', alias='OTHER_NAMES')
    # 💡 [핵심 부활] 프롬프트 하드코딩을 없애기 위해 Jinja 템플릿으로 직행할 변수들
    last_moderator_msg: str = Field(default='', description='직전 모더레이터 질문', alias='LAST_MODERATOR_MSG')
    past_summary: str = Field(default='', description='과거 대화 요약본', alias='PAST_SUMMARY')