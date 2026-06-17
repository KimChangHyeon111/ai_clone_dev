"""
FGI LangGraph 부트스트랩 (Colab 셀 재사용형 / Vertex AI)

Colab에 이미 로드돼 있는 무거운 객체들(임베더, 가드레일 모델, LLM, DataFrame)을
'인자로 받아서' 래퍼 조립 → 필수 사전작업 → 그래프 빌드 → 실행까지 담당한다.

build_fgi_graph 호출 전에 반드시 처리해야 하는 두 가지를 여기서 책임진다.
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
from async_core.InputPreprocessorAsync import InputPreprocessorAsync
from async_core.OutputPostprocessorAsync import OutputPostprocessorAsync, OutputGuardrail
from async_core.FGIGraphAsync import build_fgi_graph, run_fgi_simulation_async
from common import paths


# =====================================================================
# 조립 (Assemble): 래퍼 객체 생성 + 필수 사전작업 + 그래프 빌드
# =====================================================================
async def assemble_fgi(
    *,
    # --- 이미 Colab에 로드돼 있다고 가정하는 무거운 객체들 ---
    embedder_model: Any,            # SentenceTransformer (DataLoader/retriever/cache 공유)
    main_llm: Any,                  # LangChain 챗모델 (with_structured_output / ainvoke)
    summary_llm: Any,               # 요약용 LangChain 챗모델
    injection_classifier: Any,      # InputGuardrail: 프롬프트 인젝션 분류 pipeline
    toxic_pipeline: Any,            # OutputGuardrail: 한국어 toxic 분류 pipeline
    safe_tokenizer: Any,            # OutputGuardrail: safe 판정 causal LM 토크나이저
    safe_model: Any,                # OutputGuardrail: safe 판정 causal LM
    pii_pipeline: Any,              # OutputGuardrail: PII NER pipeline
    df_pd_de_tot: Any,              # polars DataFrame (거래 상세)
    df_pixel: Any,                  # polars DataFrame (픽셀/마이크로 어트리뷰트)

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

    # --- 인프라 ---
    redis_host: str = "localhost",
    forbidden_words_path: str = "forbidden_words.txt",

    # --- 검색 범위 제한 (대상 너무 크면 사용) ---
    ms_top_n: Optional[int] = None,
    ms_top_m_percent: Optional[float] = None,
    ml_top_n: Optional[int] = None,
    ml_top_m_percent: Optional[float] = None,

    # --- 그래프 튜닝 ---
    max_context_items: int = 5,
    max_tokens: int = 500,
    debug: bool = False,
):
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

    # 4) 입력 전처리기 (가드레일 + 시맨틱 캐시) + Redis 인덱스 셋업
    inputpreprocessor = InputPreprocessorAsync(
        guardrail_classifier=injection_classifier,
        embedder=embedder_model,
        redis_host=redis_host,
    )
    await inputpreprocessor.initialize()

    # 5) 출력 후처리기 (가드레일)
    output_guardrail = OutputGuardrail(
        toxic_pipeline=toxic_pipeline,
        safe_tokenizer=safe_tokenizer,
        safe_model=safe_model,
        pii_pipeline=pii_pipeline,
        forbidden_words_path=forbidden_words_path,
    )
    outputpostprocessor = OutputPostprocessorAsync(guardrail=output_guardrail)

    # 6) GCP 클라이언트 (Vertex AI + GCS) — 대량 시뮬레이션 배치용
    genai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
    storage_client = storage.Client(project=gcp_project)

    # 7) 그래프 빌드
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
    debug = kwargs.get("debug", False)
    fgi_app, global_agent_profiles, customer_names, promo_info = await assemble_fgi(**kwargs)
    await run_fgi_simulation_async(
        fgi_app, global_agent_profiles, customer_names, promo_info, debug=debug
    )


# 독립 스크립트로 실행할 때만 사용 (Colab에서는 아래 주석처럼 await main(...) 호출).
if __name__ == "__main__":
    raise SystemExit(
        "이 모듈은 Colab/노트북에서 'await main(...)'로 실행하세요. "
        "독립 스크립트로 쓰려면 모델/데이터 로딩을 추가한 뒤 asyncio.run(main(...))로 감싸세요."
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
#       embedder_model=embedder,            # 이미 로드된 SentenceTransformer
#       main_llm=llm,                       # 이미 로드된 LangChain 챗모델
#       summary_llm=summary_llm,
#       injection_classifier=injection_clf, # 이미 로드된 pipeline
#       toxic_pipeline=toxic_clf,
#       safe_tokenizer=safe_tok,
#       safe_model=safe_lm,
#       pii_pipeline=pii_clf,
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
#       redis_host="localhost",
#       forbidden_words_path="forbidden_words.txt",
#       debug=True,
#   )
# =====================================================================
