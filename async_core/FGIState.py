from typing import TypedDict, Annotated, Dict, Any, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# =====================================================================
# 병렬 요약 노드로 전달되는 독립 Task 객체
# =====================================================================
class AgentSummaryTask(TypedDict):
    agent_name: str
    summary_type: str
    past_summary: str
    recent_data: list


class DialogueSummaryTask(TypedDict):
    agent_name: str
    messages_to_summarize: list[Any]
    agent_ms: str
    last_internal_state: str
    past_summary: str


# =====================================================================
# agent_memories 리듀서 (병렬 노드들의 부분 업데이트를 병합)
# =====================================================================
def update_agent_memories(left: Dict[str, dict], right: Dict[str, dict]) -> Dict[str, dict]:
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
