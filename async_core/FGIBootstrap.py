"""
FGI LangGraph 부트스트랩 (Colab 셀 재사용형 / Vertex AI)

Colab에 이미 만들어 둔 객체들(임베더, LLM, 가드레일 래퍼, DataFrame)을
'인자로 받아서' 나머지 조립 → 필수 사전작업 → 그래프 빌드 → 실행까지 담당한다.

build_fgi_graph 호출 전에 반드시 처리해야 하는 것들을 여기서 책임진다.
  1) data_loader.find_relevant_ms_faiss / find_relevant_ml  (안 하면 get_fgi_profile ValueError)
  2) await inputpreprocessor.initialize()                    (Redis 시맨틱 캐시 인덱스 셋업)

Colab 사용 예시는 파일 맨 아래 주석 참고.
"""
import asyncio
from typing import Any, Optional, List

from google import genai
from google.cloud import storage

from sync_core.DataLoader import DataLoader
from async_core.ContextManagerAsync import ContextManagerAsync
from async_core.FGIGraphAsync import build_fgi_graph, run_fgi_simulation_async


# =====================================================================
# 조립 (Assemble): 사전작업 + 그래프 빌드
# =====================================================================
async def assemble_fgi(
    *,
    # --- 이미 Colab에 떠 있다고 가정하는 객체들 ---
    embedder_model: Any,          # SentenceTransformer (DataLoader/retriever 공유)
    main_llm: Any,                # LangChain 챗모델 (with_structured_output / ainvoke)
    summary_llm: Any,             # 요약용 LangChain 챗모델
    inputpreprocessor: Any,       # 이미 생성된 InputPreprocessorAsync 인스턴스
    outputpostprocessor: Any,     # 이미 생성된 OutputPostprocessorAsync 인스턴스
    df_pd_de_tot: Any,            # polars DataFrame (거래 상세)
    df_pixel: Any,                # polars DataFrame (픽셀/마이크로 어트리뷰트)

    # --- 데이터/캠페인 설정 ---
    ms_table_path: str,
    ml_table_path: str,
    promo_item: str,
    promo_info: str,
    srg_keys: List[str],

    # --- GCP (Vertex AI + GCS, 대량 시뮬레이션용) ---
    gcp_project: str,
    gcp_location: str,
    bucket_name: str,

    # --- 검색 범위 제한 (대상 너무 크면 사용) ---
    ms_top_n: Optional[int] = None,
    ms_top_m_percent: Optional[float] = None,
    ml_top_n: Optional[int] = None,
    ml_top_m_percent: Optional[float] = None,

    # --- 옵션 ---
    initialize_preprocessor: bool = True,   # 노트북에서 이미 initialize() 했으면 False
    max_context_items: int = 5,
    max_tokens: int = 500,
    debug: bool = False,
):
    """노트북에 떠 있는 객체들을 받아 그래프를 조립해 반환한다.

    DataLoader/ContextManagerAsync를 만들고, build_fgi_graph의 전제조건인
    find_relevant_ms_faiss/find_relevant_ml과 Redis 캐시 initialize()를 처리한
    뒤, Vertex AI genai/GCS 클라이언트를 구성해 그래프를 빌드한다.

    Args:
        embedder_model: DataLoader/retriever가 공유할 SentenceTransformer.
        main_llm / summary_llm: 응답용 / 요약용 LangChain 챗모델.
        inputpreprocessor / outputpostprocessor: 이미 생성된 가드레일 래퍼.
        df_pd_de_tot / df_pixel: 거래 상세 / 픽셀 polars DataFrame.
        ms_table_path / ml_table_path: DataLoader가 읽을 parquet 경로.
        promo_item / promo_info / srg_keys: 캠페인 설정과 대상 고객 키.
        gcp_project / gcp_location / bucket_name: Vertex AI + GCS 자원.
        ms_top_n/ms_top_m_percent/ml_top_n/ml_top_m_percent: 검색 범위 제한.
        initialize_preprocessor: 노트북에서 이미 initialize() 했으면 False.
        max_context_items / max_tokens / debug: 그래프 튜닝/디버그 옵션.

    Returns:
        (fgi_app, global_agent_profiles, customer_names, promo_info) 튜플.
    """
    # 1) 컨텍스트 매니저 (가벼움)
    context_manager = ContextManagerAsync()

    # 2) DataLoader — 임베더 공유 주입
    data_loader = DataLoader(
        model=embedder_model,
        ms_table_path=ms_table_path,
        ml_table_path=ml_table_path,
        promo_item=promo_item,
        promo_info=promo_info,
    )

    # 3) ⚠️ build 전에 반드시: 관련 MS/ML 필터링 (get_fgi_profile 전제조건)
    #    FAISS·BM25는 동기 무거운 연산이라 워커 스레드로 오프로딩.
    def _filter():
        data_loader.find_relevant_ms_faiss(top_n=ms_top_n, top_m_percent=ms_top_m_percent)
        data_loader.find_relevant_ml(top_n=ml_top_n, top_m_percent=ml_top_m_percent)
    await asyncio.to_thread(_filter)

    # 4) Redis 시맨틱 캐시 인덱스 셋업 (인덱스가 이미 있으면 내부에서 스킵)
    if initialize_preprocessor:
        await inputpreprocessor.initialize()

    # 5) GCP 클라이언트 (Vertex AI + GCS) — 대량 시뮬레이션 배치용
    genai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
    storage_client = storage.Client(project=gcp_project)

    # 6) 그래프 빌드
    fgi_app, global_agent_profiles, customer_names = build_fgi_graph(
        data_loader=data_loader,
        promo_item=promo_item,
        promo_info=promo_info,
        embedder_model=embedder_model,
        srg_keys=srg_keys,
        df_pd_de_tot=df_pd_de_tot,
        df_pixel=df_pixel,
        main_llm=main_llm,
        summary_llm=summary_llm,
        context_manager=context_manager,
        inputpreprocessor=inputpreprocessor,
        outputpostprocessor=outputpostprocessor,
        genai_client=genai_client,
        storage_client=storage_client,
        bucket_name=bucket_name,
        max_context_items=max_context_items,
        max_tokens=max_tokens,
        debug=debug,
    )

    return fgi_app, global_agent_profiles, customer_names, promo_info


# =====================================================================
# 메인 진입점: 조립 후 대화형 루프 실행
# =====================================================================
async def main(**kwargs):
    """assemble_fgi로 그래프를 만든 뒤 대화형 루프를 실행하는 진입점.

    모든 키워드 인자는 assemble_fgi로 그대로 전달된다. Colab에서는
    ``await main(...)``로 호출한다(asyncio.run은 실행 중 루프와 충돌).
    """
    debug = kwargs.get("debug", False)
    fgi_app, global_agent_profiles, customer_names, promo_info = await assemble_fgi(**kwargs)
    await run_fgi_simulation_async(
        fgi_app, global_agent_profiles, customer_names, promo_info, debug=debug
    )


# 독립 스크립트로 실행할 때만 사용 (Colab에서는 아래 주석처럼 await main(...) 호출).
if __name__ == "__main__":
    raise SystemExit(
        "이 모듈은 Colab/노트북에서 'await main(...)'로 실행하세요. "
        "독립 스크립트로 쓰려면 모델/데이터/래퍼 로딩을 추가한 뒤 asyncio.run(main(...))로 감싸세요."
    )


# =====================================================================
# 📓 Colab 사용 예시 (셀에 복붙, 본인 변수명으로 교체)
# ---------------------------------------------------------------------
# Colab은 이미 이벤트 루프가 돌고 있어서 asyncio.run()이 막힌다.
# 셀에서 그냥 'await main(...)' 으로 호출하면 된다.
#
#   from async_core.FGIBootstrap import main
#
#   await main(
#       embedder_model=embedder,             # 이미 로드된 SentenceTransformer
#       main_llm=llm,                        # 이미 로드된 LangChain 챗모델
#       summary_llm=summary_llm,
#       inputpreprocessor=inputpreprocessor, # 이미 생성한 InputPreprocessorAsync
#       outputpostprocessor=outputpostprocessor,  # 이미 생성한 OutputPostprocessorAsync
#       df_pd_de_tot=df_pd_de_tot,
#       df_pixel=df_pixel,
#       ms_table_path="gs_or_local/ms.parquet",
#       ml_table_path="gs_or_local/ml.parquet",
#       promo_item="귀리 음료",
#       promo_info="...캠페인 전문...",
#       srg_keys=["key1", "key2", "key3"],
#       gcp_project="my-gcp-project",
#       gcp_location="us-central1",
#       bucket_name="dt_test_data",
#       debug=True,
#   )
#
#   # 노트북에서 inputpreprocessor.initialize()를 이미 했다면:
#   #   await main(..., initialize_preprocessor=False)
# =====================================================================
