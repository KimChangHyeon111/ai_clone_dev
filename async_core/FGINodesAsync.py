"""
실시간 FGI 흐름의 LangGraph 노드 팩토리 모음.

각 ``create_*`` 함수는 노드가 의존하는 객체(전처리기/LLM/임베더 등)를 클로저로
잡아둔 뒤, 실제 그래프에 등록할 노드 함수를 반환하는 팩토리다. 노드 함수는
``FGIState``를 받아 부분 업데이트(dict)를 돌려주는 LangGraph 규약을 따른다.

흐름: preprocessor → router → retriever → customer(에이전트) → postprocessor
"""
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
    """입력 가드레일 + 인텐트 분류 노드를 생성한다.

    Args:
        preprocessor: 가드레일/캐시/인텐트 라우팅을 담당하는 InputPreprocessorAsync.

    Returns:
        마지막 사용자 메시지를 검사해 ``preprocessor_status``
        ("PROCEED"/"BLOCKED"/"CACHE_HIT"/"ERROR")와 ``intent``를 정하는 노드 함수.
        시맨틱 캐시 조회는 에이전트 노드(persona_id 기준)에서 처리하므로 여기선 끈다.
    """
    @debug_node
    async def preprocessor_node(state: FGIState) -> Dict[str, Any]:
        """마지막 입력을 가드레일에 통과시키고 FGI/대량 시뮬레이션 인텐트를 판별한다."""
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
    """대화 모드 판별 + 다음 응답 에이전트 선정 노드를 생성한다.

    진행자 입력을 해석해 모드(SEQUENTIAL/TARGETED/FREE_TALK)와 타겟을 정한다.
    지원 문법: ``@A``(단일 지목), ``@ALL``/``@모두``(전체 순차),
    ``@FREE_TALK``(자유 토론), 빈 입력(침묵 → 다음/지목 에이전트).

    Args:
        customer_names: 그래프에 등록된 에이전트 이름 목록(예: ["Customer_A", ...]).

    Returns:
        모드/타겟/표시 메시지(display_msg)와, 라우팅 메타데이터를 실은
        재작성 HumanMessage를 State에 반영하는 노드 함수.
    """
    @debug_node
    def router_node(state: FGIState) -> Dict[str, Any]:
        """입력을 파싱해 chat_mode·next_target·display_msg를 결정한다."""
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
    """질의 벡터와 코사인 유사도가 임계값 이상인 원본 항목들을 반환한다.

    유사도 내림차순으로 정렬해 임계값을 넘는 항목을 모으되, 하나도 못 넘으면
    최소한 Top-1은 강제로 포함시켜 빈 컨텍스트를 방지한다.

    Args:
        query_vector: 정규화된 질의 임베딩 (shape: [1, d] 또는 [d]).
        embeddings: 후보 항목들의 임베딩 행렬 (shape: [N, d]).
        raw_data: 임베딩과 인덱스가 일치하는 원본 데이터 리스트.
        threshold: 채택 유사도 하한(기본 0.7).

    Returns:
        선택된 원본 항목 리스트(유사도 높은 순). 입력이 비면 빈 리스트.
    """
    if not raw_data or embeddings is None or len(raw_data) == 0: return []
    similarities = np.dot(embeddings, query_vector.T).flatten()
    sorted_indices = similarities.argsort()[::-1]
    valid_indices = [idx for idx in sorted_indices if similarities[idx] >= threshold]
    if sorted_indices.size > 0 and sorted_indices[0] not in valid_indices:
        valid_indices.insert(0, sorted_indices[0])
    return [raw_data[i] for i in valid_indices]


def create_retriever_node(embedder_model):
    """타겟 에이전트의 동적 장기기억(ML/PIXEL) 검색 노드를 생성한다.

    현재 질문을 임베딩해, 해당 에이전트의 전체 이력/특성 중 의미적으로 관련된
    항목만 골라 ``recent_ml`` / ``recent_pixel``로 메모리에 채운다.

    Args:
        embedder_model: 질의 임베딩에 쓰는 SentenceTransformer(에이전트 프로필
            임베딩과 동일 공간이어야 함).

    Returns:
        next_target 에이전트의 동적 컨텍스트를 검색해 agent_memories를 갱신하는 노드 함수.
    """
    @debug_node
    async def retriever_node(state: FGIState) -> Dict[str, Any]:
        """현재 질문 기준으로 타겟 에이전트의 관련 ML/PIXEL을 검색한다."""
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
    """단일 페르소나(고객) 에이전트 노드를 생성한다.

    페르소나 데이터(MS/ML/PIXEL) + 누적 요약 + 동적 검색 결과 + 대화 이력으로
    프롬프트를 구성해 구조화 응답(FGIResponse)을 생성한다. 처리 순서:
      1) 시맨틱 캐시 조회(persona_id=agent_name) — 적중 시 LLM 호출 생략
      2) MS 결측 시 조기 방어 응답
      3) 프롬프트 빌드 → structured_llm 호출 (실패 시 raw text fallback)
      4) 정상 응답을 같은 키로 캐시에 저장

    Args:
        agent_name: 이 노드의 에이전트 이름.
        customer_names: 전체 참여자 이름(다른 참여자 표기에 사용).
        base_data: 이 에이전트의 기본 페르소나 스키마(MultiAgentFGIDataSchema).
        ctx_manager: 프롬프트 렌더링용 ContextManagerAsync.
        llm_backend: LangChain 챗모델(구조화 출력 + ainvoke 지원).
        sys_tmpl_path / user_tmpl_path: 일반 FGI 시스템/유저 템플릿 경로.
        free_sys_tmpl_path / free_user_tmpl_path: 자유토론 모드 템플릿 경로.
        inputpreprocessor: 시맨틱 캐시(check/add) 접근용.
        debug: 프롬프트/캐시 로그 출력 여부.

    Returns:
        에이전트 발화 AIMessage(+internal_state)를 State에 추가하는 노드 함수.
    """
    structured_llm = llm_backend.with_structured_output(FGIResponse)

    @debug_node
    async def customer_node(state: FGIState) -> Dict[str, Any]:
        """페르소나 컨텍스트로 응답을 생성한다(캐시 우선, 실패 시 fallback)."""
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
    """출력 가드레일 노드를 생성한다.

    직전 에이전트 발화를 검열해, 위험하면 동일 ID의 경고 메시지로 덮어써
    원본을 완전히 대체한다(시스템 메시지·캐시 응답은 검사 제외).

    Args:
        postprocessor: 출력 검열을 담당하는 OutputPostprocessorAsync.

    Returns:
        ``postprocessor_status``("SAFE"/"BLOCKED")를 정하고 필요 시 메시지를
        교체하는 노드 함수.
    """
    @debug_node
    async def postprocessor_node(state: FGIState) -> Dict[str, Any]:
        """직전 AI 발화를 검열하고 위험 시 경고로 덮어쓴다."""
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
