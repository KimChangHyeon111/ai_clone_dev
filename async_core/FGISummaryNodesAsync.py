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
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4 if text else 0

    @debug_node
    def dialogue_summary_trigger_node(state: FGIState) -> Dict[str, Any]:
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
    @debug_node
    async def dialogue_summarizer_node(state: DialogueSummaryTask) -> Dict[str, Any]:
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
    @debug_node
    async def memory_sync_node(state: FGIState) -> Dict[str, Any]:
        # 캐시 저장 로직은 customer_node로 이전됨 (persona_id 정합성 확보)
        return {"pending_dialogue_messages": []}
    return memory_sync_node


# =====================================================================
# 5. ML / PIXEL 장기 기억 요약 (Map-Reduce)
# =====================================================================
def create_agent_summarizer_node(summary_llm, ctx_manager, ml_sys: str, ml_user: str, pixel_sys: str, pixel_user: str):
    @debug_node
    async def agent_summarizer_node(state: AgentSummaryTask) -> Dict[str, Any]:
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
