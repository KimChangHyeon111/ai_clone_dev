"""
대화/장기기억 요약 관련 LangGraph 노드와 라우터.

토큰 한도를 넘으면 오래된 메시지를 잘라(trigger) 에이전트별 1인칭 요약으로
압축하고(dialogue_summarizer), 동적 검색으로 쌓인 ML/PIXEL이 임계치를 넘으면
Map-Reduce 방식으로 병렬 요약한다(agent_summarizer).

흐름: dialogue_summary_trigger → (route_to_dialogue_summarizers) →
      dialogue_summarizer ×N → memory_sync → (route_to_summaries) → agent_summarizer ×N
"""
import json
from typing import Dict, Any

from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from common.schemas import DataSummarySchema, DialogueSummarySchema
from common.utils import debug_node
from async_core.FGIState import FGIState, DialogueSummaryTask, AgentSummaryTask


# =====================================================================
# 1. 대화 요약 조건 검사 및 메시지 삭제 (트리거 노드)
# =====================================================================
def create_dialogue_summary_trigger(max_tokens: int = 500, debug=False):
    """미요약 대화가 길어지면 요약을 트리거하는 노드를 생성한다.

    마지막 요약 지점 이후 누적 대화의 추정 토큰이 ``max_tokens``를 넘고 메시지가
    충분히 쌓이면, 가장 최근 1건만 남기고 나머지를 메인 State에서 삭제
    (RemoveMessage)한 뒤 요약 대상으로 임시 보관(pending)한다.

    Args:
        max_tokens: 요약을 시작할 누적 토큰 임계값.
        debug: 트리거 로그 출력 여부.

    Returns:
        조건 충족 시 메시지를 잘라 pending_dialogue_messages로 넘기는 노드 함수.
    """
    def _estimate_tokens(text: str) -> int:
        """글자 수 기반 대략적 토큰 추정(≈ len/4)."""
        return len(text) // 4 if text else 0

    @debug_node
    def dialogue_summary_trigger_node(state: FGIState) -> Dict[str, Any]:
        """누적 대화량을 보고 요약 트리거 여부를 판단한다."""
        all_messages = state.get("messages", [])
        global_last_id = state.get("global_last_summarized_id")

        start_idx = 0
        if global_last_id:
            for i, m in enumerate(all_messages):
                if m.id == global_last_id:
                    start_idx = i + 1
                    break

        unsummarized_messages = all_messages[start_idx:]
        current_dialogue_str = "\n".join([f"{getattr(m, 'name', 'Unknown')}: {m.content}" for m in unsummarized_messages])

        if _estimate_tokens(current_dialogue_str) > max_tokens and len(unsummarized_messages) > 4:
            messages_to_summarize = unsummarized_messages[:-1]
            delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

            if debug: print(" ⚙️ [시스템] 대화 요약 조건 충족. 병렬 요약기 호출을 준비합니다.")

            return {
                "global_last_summarized_id": messages_to_summarize[-1].id,
                "messages": delete_messages,                          # 메인 State에서 영구 삭제
                "pending_dialogue_messages": messages_to_summarize    # 엣지로 넘기기 위한 임시 저장소
            }

        return {"pending_dialogue_messages": []}
    return dialogue_summary_trigger_node


# =====================================================================
# 2. 에이전트별 요약 노드로 Task 분배 (조건부 엣지)
# =====================================================================
def route_to_dialogue_summarizers(state: FGIState):
    """요약 대상이 있으면 에이전트별 요약 노드로 fan-out하는 조건부 엣지.

    pending 메시지가 있으면 각 에이전트마다 전용 페이로드(MS·직전 내면상태·기존
    요약 포함)를 만들어 ``dialogue_summarizer``로 Send 한다. 없으면 곧장
    ``memory_sync``로 보낸다.

    Returns:
        Send 리스트(병렬 분배) 또는 문자열 "memory_sync".
    """
    pending = state.get("pending_dialogue_messages", [])
    if not pending:
        return "memory_sync"  # 요약할 게 없으면 바로 동기화 노드로 패스

    sends = []
    for agent_name in state.get("agent_names", []):
        agent_ms = state.get("agent_profiles", {}).get(agent_name, {}).get("ms", "정보 없음")

        last_state_str = "정보 없음"
        for m in reversed(pending):
            if getattr(m, 'name', None) == agent_name:
                int_state = m.additional_kwargs.get("internal_state")
                if int_state:
                    emo = int_state.get("emotion", "N/A")
                    interest = int_state.get("interest_level", "N/A")
                    reasoning = int_state.get("reasoning", "N/A")
                    last_state_str = f"이전 감정: {emo}\n이전 관심도: {interest}/100\n이전 속마음: {reasoning}"
                break

        past_summary = state.get("agent_memories", {}).get(agent_name, {}).get("past_dialogue_summary", "")

        sends.append(Send("dialogue_summarizer", {
            "agent_name": agent_name,
            "messages_to_summarize": pending,
            "agent_ms": agent_ms,
            "last_internal_state": last_state_str,
            "past_summary": past_summary
        }))
    return sends


# =====================================================================
# 3. 1인칭 주관적 요약 처리 (개인화 노드)
# =====================================================================
def create_dialogue_summarizer_node(summary_llm, ctx_manager, dialogue_sys_path: str, dialogue_user_path: str, debug=False):
    """에이전트 1인칭 시점의 대화 요약 노드를 생성한다.

    대상 메시지를 "[나]/[진행자]/[다른 참여자]" 시점으로 라벨링하고, 해당
    에이전트의 MS·직전 내면상태·기존 요약과 함께 요약 LLM에 넣어 주관적 기억을
    갱신한다(실패 시 기존 요약 유지). 에이전트별로 병렬 실행된다.

    Args:
        summary_llm: 요약용 LangChain 챗모델.
        ctx_manager: 프롬프트 렌더링용 ContextManagerAsync.
        dialogue_sys_path / dialogue_user_path: 대화 요약 템플릿 경로.
        debug: 로그 출력 여부.

    Returns:
        DialogueSummaryTask를 받아 past_dialogue_summary를 갱신하는 노드 함수.
    """
    @debug_node
    async def dialogue_summarizer_node(state: DialogueSummaryTask) -> Dict[str, Any]:
        """대상 대화를 1인칭으로 재구성해 누적 기억 요약을 갱신한다."""
        agent_name = state["agent_name"]
        pending = state["messages_to_summarize"]

        personalized_dialogue = []
        for m in pending:
            if not m.content.strip(): continue
            speaker = getattr(m, 'name', 'Unknown')
            if speaker == agent_name:
                personalized_dialogue.append(f"[나] {speaker}: {m.content}")
            elif speaker in ["Moderator", "Interviewer"]:
                personalized_dialogue.append(f"[진행자] Moderator: {m.content}")
            else:
                personalized_dialogue.append(f"[다른 참여자] {speaker}: {m.content}")
        new_data_str = "\n".join(personalized_dialogue)

        data = DialogueSummarySchema(
            agent_name=agent_name,
            ms=state["agent_ms"],
            last_internal_state=state["last_internal_state"],
            past_summary=state["past_summary"] if state["past_summary"] else "없음",
            new_data=new_data_str
        )

        sys_inst, final_user_prompt = await ctx_manager.build_prompt(dialogue_sys_path, dialogue_user_path, data)

        try:
            res = await summary_llm.ainvoke([SystemMessage(content=sys_inst), HumanMessage(content=final_user_prompt)])
            new_summary = res.content.strip()
            if debug: print(f" 💾 [{agent_name}] 페르소나 및 속마음이 반영된 1인칭 기억 요약 완료.")
        except Exception as e:
            if debug: print(f"⚠️ {agent_name} 대화 요약 실패: {e}")
            new_summary = state["past_summary"]

        return {
            "agent_memories": {
                agent_name: {
                    "past_dialogue_summary": new_summary
                }
            }
        }
    return dialogue_summarizer_node


# =====================================================================
# 4. 병렬 요약 동기화 (Sync 노드)
# =====================================================================
def create_memory_sync_node(preprocessor):
    """병렬 대화 요약을 합류시키는 동기화 노드를 생성한다.

    fan-out 된 dialogue_summarizer들이 다시 모이는 join 지점. 임시 보관용
    pending_dialogue_messages를 비워 다음 턴을 깨끗이 시작한다. (캐시 저장은
    persona_id 정합성을 위해 customer_node로 이전되어 여기서는 하지 않는다.)

    Args:
        preprocessor: 호출부 시그니처 호환을 위해 유지(현재 미사용).

    Returns:
        pending_dialogue_messages를 초기화하는 노드 함수.
    """
    @debug_node
    async def memory_sync_node(state: FGIState) -> Dict[str, Any]:
        """병렬 요약 합류 지점. 임시 메시지 버퍼를 비운다."""
        # 캐시 저장 로직은 customer_node로 이전됨 (persona_id 정합성 확보)
        return {"pending_dialogue_messages": []}
    return memory_sync_node


# =====================================================================
# 5. ML / PIXEL 장기 기억 요약 (Map-Reduce)
# =====================================================================
def create_agent_summarizer_node(summary_llm, ctx_manager, ml_sys: str, ml_user: str, pixel_sys: str, pixel_user: str):
    """ML/PIXEL 장기기억을 압축하는 요약 노드를 생성한다(Map-Reduce의 Map).

    summary_type에 따라 ML 또는 PIXEL 템플릿을 골라 신규 데이터를 기존 요약과
    합쳐 재요약하고, 처리한 recent 버퍼는 비운다.

    Args:
        summary_llm: 요약용 LangChain 챗모델.
        ctx_manager: 프롬프트 렌더링용 ContextManagerAsync.
        ml_sys / ml_user / pixel_sys / pixel_user: 각 요약 템플릿 경로.

    Returns:
        AgentSummaryTask를 받아 past_{ml,pixel}_summary를 갱신하고 recent를
        비우는 노드 함수.
    """
    @debug_node
    async def agent_summarizer_node(state: AgentSummaryTask) -> Dict[str, Any]:
        """신규 ML/PIXEL을 기존 요약과 합쳐 재요약하고 버퍼를 비운다."""
        agent_name = state["agent_name"]
        stype = state["summary_type"]

        data = DataSummarySchema(
            agent_name=agent_name,
            past_summary=state["past_summary"] if state["past_summary"] else "없음",
            new_data=json.dumps(state["recent_data"], ensure_ascii=False, default=str)
        )

        sys_path, user_path = (ml_sys, ml_user) if stype == "ml" else (pixel_sys, pixel_user)
        sys_inst, final_user_prompt = await ctx_manager.build_prompt(sys_path, user_path, data)

        res = await summary_llm.ainvoke([SystemMessage(content=sys_inst), HumanMessage(content=final_user_prompt)])

        return {
            "agent_memories": {
                agent_name: {
                    f"past_{stype}_summary": res.content.strip(),
                    f"recent_{stype}": []
                }
            }
        }
    return agent_summarizer_node


def create_route_to_summaries(max_context_items: int):
    """recent 버퍼가 임계치를 넘은 에이전트를 요약 노드로 fan-out하는 라우터를 생성한다.

    Args:
        max_context_items: ML/PIXEL 각각에 대해 요약을 트리거할 누적 개수 임계값.

    Returns:
        조건을 넘긴 (에이전트, 타입)마다 agent_summarizer로 Send를 만드는 조건부
        엣지 함수. 대상이 없으면 "__end__".
    """
    def route_to_summaries(state: FGIState):
        agent_memories = state.get("agent_memories", {})
        sends = []

        for agent_name, memory in agent_memories.items():
            recent_ml = memory.get("recent_ml", [])
            recent_pixel = memory.get("recent_pixel", [])

            if len(recent_ml) >= max_context_items:
                sends.append(Send("agent_summarizer", {
                    "agent_name": agent_name, "summary_type": "ml",
                    "past_summary": memory.get("past_ml_summary", ""), "recent_data": recent_ml
                }))

            if len(recent_pixel) >= max_context_items:
                sends.append(Send("agent_summarizer", {
                    "agent_name": agent_name, "summary_type": "pixel",
                    "past_summary": memory.get("past_pixel_summary", ""), "recent_data": recent_pixel
                }))

        return sends if sends else "__end__"
    return route_to_summaries
