"""
대량 시뮬레이션(@SIMULATE) 분기의 LangGraph 노드와 서브그래프 등록.

이 그래프는 "대상 선정 → Gemini Batch Job 제출"까지만 담당한다. 배치는 수십 분
걸릴 수 있어 폴링/검증/캠페인 메타 추출은 그래프 밖(동기 BatchResultValidator)에서
별도로 처리한다 — 일부러 분리한 설계다.
"""
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
    """시뮬레이션 대상 고객 DF를 만드는 노드를 생성한다.

    프로모션과 관련된 MS(FAISS)·ML(BM25)을 검색해 대상 DataFrame을 조립한다.
    세 호출 모두 무거운 동기 연산이라 워커 스레드로 오프로딩한다.

    Args:
        data_loader: 검색/조립을 수행하는 DataLoader.
        df_pd_de_tot: 거래 상세 polars DataFrame.
        ms_top_n / ms_top_m_percent: MS 검색 범위 제한(개수/비율).
        ml_top_n / ml_top_m_percent: ML 검색 범위 제한(개수/비율).
        join_type: 대상 조립 시 조인 방식("left"/"inner").

    Returns:
        sim_target_df와 sim_status를 채우는 노드 함수.
    """
    @debug_node
    async def mass_data_loader_node(state: FGIState) -> Dict[str, Any]:
        """관련 MS/ML을 검색해 대량 시뮬레이션 대상 DF를 만든다."""
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
    """Gemini Batch Job을 제출하는 노드를 생성한다.

    대상 DF를 JSONL로 렌더링→GCS 업로드→Batch Job 생성한다. 대상이 비면 빈
    Job 생성을 막기 위해 조기 종료한다.

    Args:
        sim_engine: 배치 렌더링/제출을 담당하는 SimulationEngineAsync.
        bucket_name: 입출력에 쓰는 GCS 버킷 이름.

    Returns:
        sim_job_name/sim_job_id/sim_status를 채우는 노드 함수.
    """
    @debug_node
    async def simulation_runner_node(state: FGIState) -> Dict[str, Any]:
        """대상 DF로 Gemini Batch Job을 제출한다(빈 대상이면 스킵)."""
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
    """대량 시뮬레이션 서브그래프(mass_data_loader → simulation_runner → END)를 등록한다.

    SimulationEngineAsync를 구성하고 두 노드를 workflow에 추가/연결한다. 진입
    엣지(preprocessor → mass_data_loader)는 호출하는 빌더 쪽에서 연결한다.

    Args:
        workflow: 노드를 추가할 StateGraph.
        genai_client / storage_client / bucket_name: 배치 제출용 GCP 자원.
        ctx_manager: 시뮬레이션 프롬프트 렌더링용 ContextManagerAsync.
        data_loader / df_pd_de_tot: 대상 선정용.
        sim_sys_tmpl_path / sim_user_tmpl_path: 시뮬레이션 템플릿 경로(기본 paths).
        model_name: 배치에 쓸 Gemini 모델명.
        ms_top_n/ms_top_m_percent/ml_top_n/ml_top_m_percent/join_type: 대상 검색 옵션.
    """
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
