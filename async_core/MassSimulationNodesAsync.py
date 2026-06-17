import asyncio
from typing import Optional, Dict, Any

from langgraph.graph import END

from sync_core.DataLoader import DataLoader
from async_core.SimulationEngineAsync import SimulationEngineAsync
from async_core.FGIState import FGIState
from common.utils import debug_node
from common import paths


# =====================================================================
# Mass Simulation 1단계: 시뮬레이션 대상 고객 DF 생성
# =====================================================================
def create_mass_data_loader_node(data_loader: DataLoader, df_pd_de_tot,
                                 ms_top_n: Optional[int] = None,
                                 ms_top_m_percent: Optional[float] = None,
                                 ml_top_n: Optional[int] = None,
                                 ml_top_m_percent: Optional[float] = None,
                                 join_type: str = "left"):
    @debug_node
    async def mass_data_loader_node(state: FGIState) -> Dict[str, Any]:
        # 💡 FAISS(GPU)·BM25는 무겁고 '동기'인 연산이라, 그대로 두면 이벤트 루프가 멈춤.
        #    세 호출을 한 덩어리(_build)로 묶어 워커 스레드로 오프로딩.
        def _build():
            data_loader.find_relevant_ms_faiss(top_n=ms_top_n, top_m_percent=ms_top_m_percent)
            data_loader.find_relevant_ml(top_n=ml_top_n, top_m_percent=ml_top_m_percent)
            return data_loader.get_mass_simulation_df(df_pd_de_tot, join_type=join_type)

        target_df = await asyncio.to_thread(_build)
        return {
            "sim_target_df": target_df,
            "sim_status": f"TARGET_READY({target_df.height})",
        }
    return mass_data_loader_node


# =====================================================================
# Mass Simulation 2단계: Gemini Batch Job 제출
# =====================================================================
def create_simulation_runner_node(sim_engine: SimulationEngineAsync, bucket_name: str):
    @debug_node
    async def simulation_runner_node(state: FGIState) -> Dict[str, Any]:
        target_df = state.get("sim_target_df")

        # 대상이 비었으면 배치를 던질 이유가 없으니 조기 종료 (빈 Job 생성 방지)
        if target_df is None or target_df.is_empty():
            return {"sim_status": "NO_TARGET", "sim_job_name": None, "sim_job_id": None}

        # run_batch: JSONL 병렬 렌더링 → GCS 업로드 → Batch Job 생성 (전부 네이티브 async)
        job_name, job_id = await sim_engine.run_batch(
            df=target_df,
            promotion_info=state.get("promo_info", ""),
            bucket_name=bucket_name,
        )
        return {"sim_job_name": job_name, "sim_job_id": job_id, "sim_status": "BATCH_SUBMITTED"}
    return simulation_runner_node


# =====================================================================
# Mass Simulation 서브그래프 등록
#   campaign 추출 / 폴링 / 검증은 그래프 밖(동기 BatchResultValidator)에서 처리한다.
#   여기서는 "대상 선정 → Batch Job 제출"까지만 담당한다.
# =====================================================================
def register_mass_simulation(
    workflow, *,
    genai_client, storage_client, bucket_name, ctx_manager,
    data_loader, df_pd_de_tot,
    sim_sys_tmpl_path: str = paths.SIMULATION_SYSTEM,
    sim_user_tmpl_path: str = paths.SIMULATION_USER,
    model_name: str = "gemini-2.5-flash-lite",
    ms_top_n=None, ms_top_m_percent=None, ml_top_n=None, ml_top_m_percent=None,
    join_type: str = "left",
):
    sim_engine = SimulationEngineAsync(
        genai_client=genai_client, storage_client=storage_client,
        context_manager=ctx_manager,
        sys_tmpl_path=sim_sys_tmpl_path, user_tmpl_path=sim_user_tmpl_path,
        model_name=model_name,
    )

    workflow.add_node("mass_data_loader",
                      create_mass_data_loader_node(data_loader, df_pd_de_tot,
                                                   ms_top_n, ms_top_m_percent,
                                                   ml_top_n, ml_top_m_percent, join_type))
    workflow.add_node("simulation_runner",
                      create_simulation_runner_node(sim_engine, bucket_name))

    workflow.add_edge("mass_data_loader", "simulation_runner")
    workflow.add_edge("simulation_runner", END)   # 💡 여기서 끝. Job 제출까지만.
