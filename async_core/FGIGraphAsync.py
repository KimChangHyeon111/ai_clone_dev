import uuid
import asyncio
import numpy as np
import polars as pl

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from common.schemas import MultiAgentFGIDataSchema
from common import paths
from async_core.ContextManagerAsync import ContextManagerAsync
from async_core.InputPreprocessorAsync import InputPreprocessorAsync
from async_core.OutputPostprocessorAsync import OutputPostprocessorAsync
from async_core.FGIState import FGIState
from async_core.FGINodesAsync import (
    create_preprocessor_node,
    create_router_node,
    create_retriever_node,
    create_customer_node_async,
    create_postprocessor_node,
)
from async_core.FGISummaryNodesAsync import (
    create_dialogue_summary_trigger,
    route_to_dialogue_summarizers,
    create_dialogue_summarizer_node,
    create_memory_sync_node,
    create_agent_summarizer_node,
    create_route_to_summaries,
)
from async_core.MassSimulationNodesAsync import register_mass_simulation


# =====================================================================
# 라우팅 함수 (그래프 조립 로직)
# =====================================================================
def route_after_preprocess(state: FGIState):
    status = state.get("preprocessor_status")
    if status == "BLOCKED" or status == "CACHE_HIT":
        return END
    if state.get("intent") == "MASS_SIMULATION":
        return "mass_data_loader"   # 대량 시뮬레이션 첫 노드로
    return "router"                 # 기존 FGI 흐름


def route_after_postprocess(state: FGIState):
    status = state.get("postprocessor_status", "SAFE")
    if status == "BLOCKED":
        return END  # 차단 시 즉시 대화 턴 종료
    return "dialogue_summary_trigger"


# =====================================================================
# 그래프 생성 (Build)
# =====================================================================
def build_fgi_graph(
    data_loader, promo_item: str, promo_info: str, embedder_model, srg_keys: list,
    df_pd_de_tot: pl.DataFrame, df_pixel: pl.DataFrame, main_llm, summary_llm, context_manager,
    inputpreprocessor: InputPreprocessorAsync,
    outputpostprocessor: OutputPostprocessorAsync,
    genai_client, storage_client, bucket_name: str,   # Mass Simulation용
    max_context_items: int = 5, max_tokens: int = 500, debug: bool = False
):
    ctx_manager = context_manager

    alphabet_mapping = {srg_key: chr(65 + i) for i, srg_key in enumerate(srg_keys)}
    customer_names = [f"Customer_{alphabet_mapping[key]}" for key in srg_keys]

    workflow = StateGraph(FGIState)

    # --- 노드 추가 ---
    workflow.add_node("preprocessor", create_preprocessor_node(inputpreprocessor))
    workflow.add_node("router", create_router_node(customer_names))
    workflow.add_node("retriever", create_retriever_node(embedder_model))
    workflow.add_node("postprocessor", create_postprocessor_node(outputpostprocessor))

    workflow.add_node("dialogue_summary_trigger", create_dialogue_summary_trigger(max_tokens=max_tokens, debug=debug))
    workflow.add_node("dialogue_summarizer", create_dialogue_summarizer_node(
        summary_llm=summary_llm, ctx_manager=ctx_manager,
        dialogue_sys_path=paths.DIALOGUE_SUMMARY_SYSTEM,
        dialogue_user_path=paths.DIALOGUE_SUMMARY_USER,
        debug=debug
    ))
    workflow.add_node("memory_sync", create_memory_sync_node(inputpreprocessor))
    workflow.add_node("agent_summarizer", create_agent_summarizer_node(
        summary_llm=summary_llm, ctx_manager=ctx_manager,
        ml_sys=paths.ML_SUMMARY_SYSTEM, ml_user=paths.ML_SUMMARY_USER,
        pixel_sys=paths.PIXEL_SUMMARY_SYSTEM, pixel_user=paths.PIXEL_SUMMARY_USER
    ))

    global_agent_profiles = {}

    for srg_key in srg_keys:
        short_char = alphabet_mapping[srg_key]
        agent_name = f"Customer_{short_char}"

        raw_profile = data_loader.get_fgi_profile(clone_srg_key=srg_key, df_pd_de_tot=df_pd_de_tot, df_pixel=df_pixel)

        ms_data = raw_profile.get("ms", "해당 고객의 구체적인 마인드셋 정보가 없습니다.")
        other_names_str = ", ".join([n for n in customer_names if n != agent_name])

        base_data = MultiAgentFGIDataSchema(
            ms=ms_data, ml=raw_profile.get("ml", []), pixel=raw_profile.get('pixel', []),
            promo_info=raw_profile.get("promo_info", promo_info),
            conversation_history="", user_input="", current_name=agent_name,
            other_names=other_names_str, is_langgraph=True
        )

        global_agent_profiles[agent_name] = {
            "ms": ms_data,
            "ml_embeddings": raw_profile.get("ml_embeddings", np.array([])),
            "full_ml_history": raw_profile.get("full_ml", []),
            "pixel_embeddings": raw_profile.get("pixel_embeddings", np.array([])),
            "full_pixel": raw_profile.get("full_pixel", [])
        }

        workflow.add_node(agent_name, create_customer_node_async(
            agent_name, customer_names, base_data, ctx_manager, main_llm,
            paths.MULTI_FGI_SYSTEM, paths.MULTI_FGI_USER,
            paths.FREE_TALK_SYSTEM, paths.FREE_TALK_USER,
            inputpreprocessor, debug
        ))

    # --- 엣지 연결 (분리된 아키텍처 흐름) ---
    workflow.add_edge(START, "preprocessor")
    workflow.add_conditional_edges(
        "preprocessor",
        route_after_preprocess,
        {
            END: END,                                   # 차단/캐시 적중 → 종료
            "router": "router",                         # REALTIME_FGI → FGI 흐름
            "mass_data_loader": "mass_data_loader",     # MASS_SIMULATION → 시뮬레이션 진입
        }
    )
    workflow.add_conditional_edges(
        "router",
        lambda state: "__end__" if not state.get("next_target") or state.get("next_target") == "__end__" else "retriever",
        {"retriever": "retriever", "__end__": END}
    )
    workflow.add_conditional_edges("retriever", lambda state: state.get("next_target"), {name: name for name in customer_names})

    for name in customer_names:
        workflow.add_edge(name, "postprocessor")

    workflow.add_conditional_edges(
        "postprocessor",
        route_after_postprocess,
        {
            END: END,
            "dialogue_summary_trigger": "dialogue_summary_trigger"
        }
    )

    workflow.add_conditional_edges(
        "dialogue_summary_trigger",
        route_to_dialogue_summarizers,
        {"dialogue_summarizer": "dialogue_summarizer", "memory_sync": "memory_sync"}
    )
    workflow.add_edge("dialogue_summarizer", "memory_sync")
    workflow.add_conditional_edges(
        "memory_sync",
        create_route_to_summaries(max_context_items),
        {"agent_summarizer": "agent_summarizer", "__end__": END}
    )
    workflow.add_edge("agent_summarizer", END)

    # --- Mass Simulation 서브그래프 등록 ---
    register_mass_simulation(
        workflow,
        genai_client=genai_client,
        storage_client=storage_client,
        bucket_name=bucket_name,
        ctx_manager=ctx_manager,
        data_loader=data_loader,
        df_pd_de_tot=df_pd_de_tot,
        # ms_top_m_percent=0.1, ml_top_n=200,  # 대상 너무 크면 검색 범위 제한
    )

    memory = MemorySaver()
    fgi_app = workflow.compile(checkpointer=memory)

    return fgi_app, global_agent_profiles, customer_names


# =====================================================================
# 실행 루프 (Run)
# =====================================================================
async def run_fgi_simulation_async(
    fgi_app, global_agent_profiles: dict, customer_names: list, promo_info: str, debug: bool = False
):
    print("=" * 60)
    print("🎤 [비동기(Async) FGI 시뮬레이션 환경 - LangGraph Ver] 🎤")
    print("  * 지원 모드: 1:1 집중(A입력), 멘션(@A), 전체순차(@ALL), 자유토론(@FREE_TALK)")
    if debug: print("🐛 [디버그 모드 ON] 프롬프트 로그가 출력됩니다.")
    print("=" * 60)

    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_memories = {name: {"past_ml_summary": "", "recent_ml": [], "past_pixel_summary": "", "recent_pixel": [], "past_dialogue_summary": ""} for name in customer_names}

    fgi_app.update_state(thread_config, {
        "agent_names": customer_names, "chat_mode": "SEQUENTIAL", "targeted_agent": None,
        "last_agent": None, "display_msg": "", "next_target": "", "last_moderator_msg": "",
        "promo_info": promo_info, "agent_memories": initial_memories,
        "agent_profiles": global_agent_profiles, "global_past_summary": "", "global_last_summarized_id": None,
        "pending_dialogue_messages": []
    })

    while True:
        user_input = await asyncio.to_thread(input, "\n👤 진행자(당신): ")
        user_input = user_input.strip()

        if user_input.lower() in ['quit', 'exit']: break

        payload = {"messages": [HumanMessage(content=user_input, name="Moderator", id=str(uuid.uuid4()))]}
        print("-" * 60)

        async for event in fgi_app.astream(payload, config=thread_config, stream_mode="updates"):
            for node_name, state_update in event.items():
                if not state_update: continue

                if node_name == "preprocessor" and "messages" in state_update:
                    print(f"\n 🛡️ [시스템 가드레일/캐시]:\n   {state_update['messages'][-1].content}")
                elif node_name == "router":
                    if debug: print(f" ⚙️ [시스템] 모드: {state_update.get('chat_mode', '유지')} | 다음 타겟: {state_update.get('next_target', '종료')}")
                elif node_name == "retriever":
                    if debug: print(f" 🔎 [검색기] {fgi_app.get_state(thread_config).values.get('next_target', '에이전트')}의 동적 기억 검색 완료.")
                elif node_name.startswith("Customer_") and "messages" in state_update:
                    if debug: print(f"\n 🤖 [원본 생성 대기 중 - {node_name}]:\n   {state_update['messages'][-1].content}")

                elif node_name == "postprocessor":
                    if state_update.get("postprocessor_status") == "BLOCKED":
                        safe_msg = state_update['messages'][-1]
                        reason = safe_msg.additional_kwargs.get("blocked_reason")
                        print(f"\n 🛡️ [출력 가드레일 작동 - {safe_msg.name}] ({reason}):\n   {safe_msg.content}")
                    elif state_update.get("postprocessor_status") == "SAFE":
                        current_state = fgi_app.get_state(thread_config).values
                        safe_msg = current_state["messages"][-1]
                        print(f"\n 🤖 [{safe_msg.name}]:\n   {safe_msg.content}")

                elif node_name in ("mass_data_loader", "simulation_runner"):
                    print(f" 🏭 [Mass Sim] {node_name}: {state_update.get('sim_status', '')}")

                elif node_name == "agent_summarizer" and "agent_memories" in state_update:
                    for agent in state_update["agent_memories"].keys():
                        print(f" 💾 [시스템] {agent} 동적 병렬 요약 완료 (Map-Reduce).")

            current_state = fgi_app.get_state(thread_config).values
            msg_count = len(current_state.get("messages", []))
            if debug: print(f"👉 [체크] 현재 DB에 남은 메시지 개수: {msg_count}개")
