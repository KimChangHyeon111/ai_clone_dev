"""
LangGraph FGI 그래프의 전역 State 정의와 메모리 리듀서.

- ``FGIState``: 그래프 전체에서 공유되는 단일 상태 스키마.
- ``AgentSummaryTask`` / ``DialogueSummaryTask``: 병렬 요약 노드(Send)로 넘기는
  에이전트별 독립 입력 페이로드.
- ``update_agent_memories``: 여러 병렬 노드가 동시에 내보내는 agent_memories
  부분 업데이트를 충돌 없이 병합하는 리듀서.
"""
from typing import TypedDict, Annotated, Dict, Any, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# =====================================================================
# 병렬 요약 노드로 전달되는 독립 Task 객체
# =====================================================================
class AgentSummaryTask(TypedDict):
    """ML/PIXEL 장기기억 요약 노드(agent_summarizer)로 보내는 단일 작업.

    Attributes:
        agent_name: 요약 대상 에이전트.
        summary_type: "ml" 또는 "pixel".
        past_summary: 기존 누적 요약본.
        recent_data: 이번에 요약할 신규 항목 리스트.
    """
    agent_name: str
    summary_type: str
    past_summary: str
    recent_data: list


class DialogueSummaryTask(TypedDict):
    """1인칭 대화 요약 노드(dialogue_summarizer)로 보내는 단일 작업.

    Attributes:
        agent_name: 요약 주체 에이전트("나" 시점).
        messages_to_summarize: 요약 대상 메시지들.
        agent_ms: 해당 에이전트의 핵심 마인드셋(MS).
        last_internal_state: 직전 감정/관심도/속마음 요약 문자열.
        past_summary: 기존 누적 대화 요약본.
    """
    agent_name: str
    messages_to_summarize: list[Any]
    agent_ms: str
    last_internal_state: str
    past_summary: str


# =====================================================================
# agent_memories 리듀서 (병렬 노드들의 부분 업데이트를 병합)
# =====================================================================
def update_agent_memories(left: Dict[str, dict], right: Dict[str, dict]) -> Dict[str, dict]:
    """agent_memories 부분 업데이트를 병합하는 LangGraph 리듀서.

    에이전트 단위로 깊은 병합한다. ``recent_ml``/``recent_pixel``은 빈 리스트가
    오면 초기화(요약 후 비우기), 아니면 중복 없이 누적한다. 요약본 필드
    (``past_*_summary``, ``past_dialogue_summary``)는 들어온 값으로 덮어쓴다.

    Args:
        left: 기존 상태.
        right: 노드가 새로 내보낸 부분 업데이트.

    Returns:
        병합된 agent_memories 딕셔너리.
    """
    if not left:
        left = {}
    if not right:
        return left
    res = {k: v.copy() for k, v in left.items()}

    for agent_name, updates in right.items():
        if agent_name not in res:
            res[agent_name] = {
                "past_ml_summary": "", "recent_ml": [],
                "past_pixel_summary": "", "recent_pixel": [],
                "past_dialogue_summary": "",
                "needs_summary": False, "last_summarized_msg_id": None
            }

        agent_mem = res[agent_name]

        if "recent_ml" in updates and len(updates["recent_ml"]) == 0:
            agent_mem["recent_ml"] = []
        elif "recent_ml" in updates:
            for item in updates["recent_ml"]:
                if item not in agent_mem["recent_ml"]:
                    agent_mem["recent_ml"].append(item)

        if "recent_pixel" in updates and len(updates["recent_pixel"]) == 0:
            agent_mem["recent_pixel"] = []
        elif "recent_pixel" in updates:
            for item in updates["recent_pixel"]:
                if item not in agent_mem["recent_pixel"]:
                    agent_mem["recent_pixel"].append(item)

        if "past_ml_summary" in updates: agent_mem["past_ml_summary"] = updates["past_ml_summary"]
        if "past_pixel_summary" in updates: agent_mem["past_pixel_summary"] = updates["past_pixel_summary"]
        if "past_dialogue_summary" in updates: agent_mem["past_dialogue_summary"] = updates["past_dialogue_summary"]

    return res


# =====================================================================
# 그래프 전역 State
# =====================================================================
class FGIState(TypedDict):
    """그래프 전역 상태. 실시간 FGI와 대량 시뮬레이션 필드를 함께 담는다.

    ``messages``는 add_messages 리듀서로 누적/삭제되고, ``agent_memories``는
    update_agent_memories 리듀서로 에이전트별 병합된다. ``sim_*`` 필드는
    @SIMULATE 분기에서만 채워진다.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    global_past_summary: str
    global_last_summarized_id: str | None
    agent_names: list[str]
    chat_mode: str
    last_moderator_msg: str
    targeted_agent: str | None
    last_agent: str | None
    display_msg: str
    next_target: str
    promo_info: str
    agent_profiles: Dict[str, dict]
    agent_memories: Annotated[Dict[str, dict], update_agent_memories]
    pending_dialogue_messages: list[Any]  # 엣지로 메시지를 넘겨주기 위한 임시 저장소
    preprocessor_status: str   # "PROCEED", "CACHE_HIT", "BLOCKED"
    postprocessor_status: str  # "SAFE", "BLOCKED"
    cached_response: Optional[str]
    intent: str
    # --- Mass Simulation 전용 필드 ---
    sim_target_df: Optional[Any]   # 시뮬레이션 대상 Polars DataFrame
    sim_job_name: Optional[str]    # Gemini Batch Job 이름
    sim_job_id: Optional[str]      # 내부 추적용 job id
    sim_status: str                # 진행 상태 문자열 (로깅/디버그용)
