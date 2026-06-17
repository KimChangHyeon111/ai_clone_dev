import re
import json
import uuid
import asyncio
import numpy as np
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from common.schemas import FGIResponse
from common.utils import debug_node, clean_json_string
from async_core.FGIState import FGIState
from async_core.InputPreprocessorAsync import InputPreprocessorAsync

# SIMULATE / 시뮬레이션 입력 시 대량 시뮬레이션 모드로 분기
MASS_SIM_TRIGGER = re.compile(r'@(SIMULATE|MASS|시뮬레이션|시뮬)\b', re.IGNORECASE)


# =====================================================================
# 입력 전처리 노드 (Guardrail + Intent Router)
# =====================================================================
def create_preprocessor_node(preprocessor: InputPreprocessorAsync):
    @debug_node
    async def preprocessor_node(state: FGIState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"preprocessor_status": "ERROR"}

        last_msg = messages[-1].content
        # user_id는 적절한 식별자(예: thread_id)를 사용
        user_id = "default_user"
        intent = "MASS_SIMULATION" if MASS_SIM_TRIGGER.search(last_msg) else "REALTIME_FGI"

        # 캐시 조회는 에이전트 노드에서 persona_id 기준으로 처리하므로 여기선 끈다.
        result = await preprocessor.process_request(user_id, intent, last_msg, check_cache=False)

        if result["status"] == "BLOCKED":
            return {
                "preprocessor_status": "BLOCKED",
                "messages": [AIMessage(content=f"⚠️ {result['reason']}: 안전한 입력을 부탁드립니다.")]
            }

        if result["status"] == "CACHE_HIT":
            return {
                "preprocessor_status": "CACHE_HIT",
                "cached_response": result["response"],
                "messages": [AIMessage(content=result["response"], name="Semantic_Cache")]
            }

        return {
            "preprocessor_status": "PROCEED",
            "intent": result.get("intent", intent)
        }
    return preprocessor_node


# =====================================================================
# 라우터 노드 (모드 판별 + 타겟 선정)
# =====================================================================
def create_router_node(customer_names: list):
    @debug_node
    def router_node(state: FGIState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_msg_obj = messages[-1] if messages else None
        last_message = last_msg_obj.content.strip() if last_msg_obj else ""
        input_upper = last_message.upper()

        chat_mode = state.get("chat_mode", "SEQUENTIAL")
        targeted_agent = state.get("targeted_agent")
        last_agent = state.get("last_agent")

        core_question = state.get("last_moderator_msg", "최근에 논의하던 주제")
        if not core_question:
            core_question = "최근에 논의하던 주제"

        match = re.search(r"앞서 드린 질문\('(.+?)'\)에 대해", core_question)
        if match:
            core_question = match.group(1)

        display_msg = ""
        target = None

        def get_next_agent(current_last):
            if not customer_names: return ""
            if not current_last or current_last not in customer_names: return customer_names[0]
            idx = customer_names.index(current_last)
            return customer_names[(idx + 1) % len(customer_names)]

        target_candidate = f"Customer_{input_upper}"
        is_silence_or_shortcut = False

        if target_candidate in customer_names and len(last_message) == 1:
            chat_mode = "TARGETED"
            targeted_agent = target_candidate
            target = target_candidate
            is_silence_or_shortcut = True
        elif last_message == '':
            target = targeted_agent if (chat_mode == "TARGETED" and targeted_agent) else get_next_agent(last_agent)
            is_silence_or_shortcut = True
        else:
            if '@FREE_TALK' in input_upper:
                chat_mode = "FREE_TALK"
                targeted_agent = None
                target = get_next_agent(last_agent)
                display_msg = re.sub(r'@FREE_TALK', '[자유 토론]', last_message, flags=re.IGNORECASE).strip()
            elif '@ALL' in input_upper or '@모두' in input_upper:
                chat_mode = "SEQUENTIAL"
                targeted_agent = None
                target = get_next_agent(last_agent)
                display_msg = re.sub(r'@ALL|@모두', '', last_message, flags=re.IGNORECASE).strip()
            else:
                mentioned = False
                for name in customer_names:
                    char = name.split('_')[1] if '_' in name else name
                    if re.search(fr'@{char}(?![a-zA-Z])', last_message, re.IGNORECASE):
                        chat_mode = "TARGETED"
                        targeted_agent = name
                        target = name
                        display_msg = re.sub(fr'@{char}(?![a-zA-Z])', f'@{name}', last_message, flags=re.IGNORECASE).strip()
                        mentioned = True
                        break
                if not mentioned:
                    target = targeted_agent if (chat_mode == "TARGETED" and targeted_agent) else get_next_agent(last_agent)
                    display_msg = last_message

            if display_msg and display_msg != "[자유 토론]":
                core_question = display_msg

        if is_silence_or_shortcut:
            if chat_mode != "FREE_TALK":
                display_msg = f"{target}님, 앞서 드린 질문('{core_question}')에 대해 어떻게 생각하시나요?"
            else:
                display_msg = "[자유 토론]"

        updates = {
            "chat_mode": chat_mode,
            "targeted_agent": targeted_agent,
            "display_msg": display_msg,
            "next_target": target,
            "last_moderator_msg": core_question
        }

        if last_msg_obj:
            final_content = display_msg if display_msg else last_message
            modified_msg = HumanMessage(
                content=final_content,
                name=last_msg_obj.name,
                id=last_msg_obj.id,
                additional_kwargs={"target": target if target else "All"}
            )
            updates["messages"] = [modified_msg]

        return updates
    return router_node


# =====================================================================
# 동적 컨텍스트 검색 (Retriever)
# =====================================================================
def retrieve_dynamic_context(query_vector: np.ndarray, embeddings: np.ndarray, raw_data: list, threshold: float = 0.7) -> list:
    if not raw_data or embeddings is None or len(raw_data) == 0: return []
    similarities = np.dot(embeddings, query_vector.T).flatten()
    sorted_indices = similarities.argsort()[::-1]
    valid_indices = [idx for idx in sorted_indices if similarities[idx] >= threshold]
    if sorted_indices.size > 0 and sorted_indices[0] not in valid_indices:
        valid_indices.insert(0, sorted_indices[0])
    return [raw_data[i] for i in valid_indices]


def create_retriever_node(embedder_model):
    @debug_node
    async def retriever_node(state: FGIState) -> Dict[str, Any]:
        agent_name = state.get("next_target")
        if not agent_name or agent_name == "__end__": return {}

        display_msg = state.get("display_msg", "[침묵]")
        all_messages = state.get("messages", [])

        if display_msg in ["[자유 토론]", "[침묵]", ""] or "어떻게 생각하시나요?" in display_msg:
            embed_query = all_messages[-2].content if len(all_messages) >= 2 else "일반적인 관심사"
        else:
            embed_query = display_msg

        query_vector = await asyncio.to_thread(embedder_model.encode, [embed_query], normalize_embeddings=True)
        query_vector = query_vector.astype(np.float32)

        agent_profile = state.get("agent_profiles", {}).get(agent_name, {})

        dynamic_ml = retrieve_dynamic_context(query_vector, agent_profile.get("ml_embeddings"), agent_profile.get("full_ml_history", []), 0.7)
        dynamic_pixel = retrieve_dynamic_context(query_vector, agent_profile.get("pixel_embeddings"), agent_profile.get("full_pixel", []), 0.7)

        return {
            "agent_memories": {
                agent_name: {
                    "recent_ml": dynamic_ml,
                    "recent_pixel": dynamic_pixel
                }
            }
        }
    return retriever_node


# =====================================================================
# 고객(페르소나) 에이전트 노드
# =====================================================================
def create_customer_node_async(
        agent_name: str, customer_names: list, base_data, ctx_manager, llm_backend,
        sys_tmpl_path: str, user_tmpl_path: str, free_sys_tmpl_path: str, free_user_tmpl_path: str,
        inputpreprocessor, debug: bool = False,
):
    structured_llm = llm_backend.with_structured_output(FGIResponse)

    @debug_node
    async def customer_node(state: FGIState) -> Dict[str, Any]:
        display_msg = state.get("display_msg", "[침묵]")
        chat_mode = state.get("chat_mode", "SEQUENTIAL")
        all_messages = state.get("messages", [])

        # 💡 [캐시 조회] 라우터 이후라 타겟 에이전트가 확정됨 → persona_id=agent_name
        cache_query = (state.get("last_moderator_msg") or "").strip()
        use_cache = (chat_mode != "FREE_TALK") and bool(cache_query)

        if use_cache:
            cached_resp, _sim = await inputpreprocessor.cache.check_cache(agent_name, cache_query)
            if cached_resp:
                if debug: print(f"⚡ [캐시 HIT] {agent_name} (query='{cache_query[:20]}...')")
                return {
                    "messages": [AIMessage(
                        content=cached_resp,
                        name=agent_name,
                        additional_kwargs={
                            "target": "All",
                            "internal_state": {
                                "reasoning": "Semantic cache hit",
                                "interest_level": 50, "emotion": "캐시", "target_listener": None
                            },
                            "from_cache": True,
                        },
                        id=str(uuid.uuid4())
                    )],
                    "last_agent": agent_name
                }

        if not base_data.ms:
            if debug:
                print(f"🚫 [Early Exit] {agent_name}의 핵심 페르소나(MS) 데이터가 없어 LLM 호출을 차단합니다.")
            return {
                "messages": [AIMessage(
                    content="[시스템] 저는 현재 할당된 페르소나 데이터가 부족하여 답변을 드릴 수 없습니다. 다른 분의 의견을 먼저 들어보시죠.",
                    name=agent_name,
                    additional_kwargs={
                        "target": "Group" if chat_mode == "FREE_TALK" else "All",
                        "internal_state": {
                            "reasoning": "데이터 결측으로 인한 시스템 자동 방어 로직 작동",
                            "interest_level": 0, "emotion": "무응답", "target_listener": None
                        }
                    },
                    id=str(uuid.uuid4())
                )],
                "last_agent": agent_name
            }

        temp_data = base_data.model_copy(deep=True)
        temp_data.promo_info = state.get("promo_info", "정보 없음")

        memories = state.get("agent_memories", {}).get(agent_name, {})
        past_ml_summary = memories.get("past_ml_summary", "")
        past_pixel_summary = memories.get("past_pixel_summary", "")

        recent_ml = memories.get("recent_ml", [])
        recent_pixel = memories.get("recent_pixel", [])

        temp_data.ml = base_data.ml.copy() if isinstance(base_data.ml, list) else []
        if past_ml_summary: temp_data.ml.append({"[누적 과거 이력 요약]": past_ml_summary})
        temp_data.ml.extend(recent_ml)

        temp_data.pixel = base_data.pixel.copy() if isinstance(base_data.pixel, list) else []
        if past_pixel_summary: temp_data.pixel.append({"[누적 과거 특성 요약]": past_pixel_summary})
        temp_data.pixel.extend(recent_pixel)

        last_state_str = ""
        for m in reversed(all_messages):
            if getattr(m, 'name', None) == agent_name:
                int_state = m.additional_kwargs.get("internal_state")
                if int_state:
                    emo = int_state.get("emotion", "N/A")
                    interest = int_state.get("interest_level", "N/A")
                    reasoning = int_state.get("reasoning", "N/A")
                    last_state_str = f"이전 대화에서의 내 감정: {emo}\n이전 대화에서의 상품 관심도: {interest}/100\n이전 대화에서의 내 속마음: {reasoning}"
                break

        temp_data.last_internal_state = last_state_str

        past_summary = memories.get("past_dialogue_summary", "")
        last_summarized_id = state.get("global_last_summarized_id")
        history_str = ""

        start_idx = 0
        if last_summarized_id:
            for i, m in enumerate(all_messages):
                if m.id == last_summarized_id:
                    start_idx = i + 1
                    break

        past_messages = all_messages[start_idx:-1] if len(all_messages) > start_idx else []

        for m in past_messages:
            speaker = getattr(m, 'name', None) or "Moderator"
            target_val = m.additional_kwargs.get("target", "All")

            target_str = ""
            if target_val not in ["All", "Group", None]:
                target_str = f" [👉 {target_val} 지목]"

            if speaker in ["Moderator", "Interviewer"]:
                history_str += f"Moderator{target_str}: {m.content}\n"
            elif speaker == agent_name:
                history_str += f"Your Past Answer ({agent_name}){target_str}: {m.content}\n"
            else:
                history_str += f"Other Participant ({speaker}){target_str}: {m.content}\n"

        temp_data.conversation_history = history_str.strip()
        temp_data.past_summary = past_summary
        temp_data.last_moderator_msg = state.get("last_moderator_msg", "")
        temp_data.user_input = display_msg
        temp_data.current_name = agent_name
        temp_data.other_names = ", ".join([n for n in customer_names if n != agent_name])

        active_sys_path = free_sys_tmpl_path if chat_mode == "FREE_TALK" else sys_tmpl_path
        active_user_path = free_user_tmpl_path if chat_mode == "FREE_TALK" else user_tmpl_path

        sys_inst, final_user_prompt = await ctx_manager.build_prompt(
            sys_path=active_sys_path, user_path=active_user_path, data=temp_data
        )

        if debug:
            print(f"\n{'='*20} 🐛 [DEBUG: {agent_name} PROMPT START] {'='*20}")
            print(f"\n[SYSTEM INSTRUCTION]\n{sys_inst}")
            print(f"\n[USER PROMPT]\n{final_user_prompt}")
            print(f"\n{'='*65}\n")

        try:
            result = await structured_llm.ainvoke([SystemMessage(content=sys_inst), HumanMessage(content=final_user_prompt)])
            final_content = result.response
            final_internal_state = result.internal_state.model_dump()

        except Exception as e:
            if debug: print(f"⚠️ [경고] {agent_name} 구조화 파싱 실패. Fallback 가동: {e}")
            try:
                raw_result = await llm_backend.ainvoke([SystemMessage(content=sys_inst), HumanMessage(content=final_user_prompt)])
                cleaned_text = clean_json_string(raw_result.content)
                try:
                    parsed = json.loads(cleaned_text)
                    final_content = parsed.get("response", cleaned_text).strip()
                except json.JSONDecodeError:
                    final_content = cleaned_text.strip()

                final_internal_state = {
                    "reasoning": "파싱 에러로 인해 Raw Text Fallback이 작동했습니다.",
                    "interest_level": 50, "emotion": "당황", "target_listener": None
                }
            except Exception as fatal_e:
                print(f"❌ [치명적 에러] {agent_name} API 호출 완전 실패: {fatal_e}")
                final_content = "음... 갑자기 생각이 잘 안 나네요. 다른 분 의견 먼저 들어볼 수 있을까요?"
                final_internal_state = {
                    "reasoning": "LLM API 호출 완전 실패",
                    "interest_level": 0, "emotion": "침묵", "target_listener": None
                }

        if use_cache and final_content:
            await inputpreprocessor.cache.add_to_cache(agent_name, cache_query, final_content)

        return {
            "messages": [AIMessage(
                content=final_content,
                name=agent_name,
                additional_kwargs={
                    "target": "Group" if chat_mode == "FREE_TALK" else "All",
                    "internal_state": final_internal_state
                },
                id=str(uuid.uuid4())
            )],
            "last_agent": agent_name
        }
    return customer_node


# =====================================================================
# 출력 후처리 노드 (Output Guardrail)
# =====================================================================
def create_postprocessor_node(postprocessor):
    @debug_node
    async def postprocessor_node(state: FGIState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"postprocessor_status": "SAFE"}

        last_msg = messages[-1]
        # 에이전트(AI)가 생성한 메시지만 검사 (시스템 메시지 등 패스)
        if not isinstance(last_msg, AIMessage) or getattr(last_msg, "name", "") == "Semantic_Cache":
            return {"postprocessor_status": "SAFE"}

        intent = state.get("intent", "REALTIME_FGI")

        result = await postprocessor.process_output(intent, last_msg.content)

        # BLOCKED 상태인 경우 기존 메시지 덮어쓰기 (경고 메시지로 내용 교체)
        if result.get("status") == "BLOCKED":
            reason = result.get("reason", "알 수 없는 이유")
            warning_message = f"⚠️ {reason}: 안전한 입력을 부탁드립니다."

            modified_msg = AIMessage(
                content=warning_message,
                name=last_msg.name,
                id=last_msg.id,  # 동일한 ID로 위험한 원본 메시지를 완전히 덮어쓰기
                additional_kwargs={
                    **last_msg.additional_kwargs,
                    "blocked_reason": reason
                }
            )
            return {"messages": [modified_msg], "postprocessor_status": "BLOCKED"}

        return {"postprocessor_status": "SAFE"}
    return postprocessor_node
