# AI Clone Development - System Architecture Map

> **Harness Engineering Map**: 본 문서는 시스템의 모듈간 의존성, 데이터 파이프라인 흐름, 비동기/동기 처리 계층을 한눈에 파악하기 위한 아키텍처 지도(Map)입니다.
> 프로젝트 개요·실행법·그래프 흐름은 [`README.md`](./README.md)를 참고하세요.

## `async_core/` — LangGraph FGI 그래프 (현재 메인)
LangGraph 기반 멀티 에이전트 FGI/대량 시뮬레이션 그래프를 구성하는 모듈 그룹입니다.
흐름: `preprocessor → router → retriever → customer → postprocessor → (요약)`,
대량은 `mass_data_loader → simulation_runner`.

* **`FGIGraphAsync.py`** — 그래프 조립·라우팅·실행의 진입 모듈
  * `route_after_preprocess()`: 전처리 결과로 분기 (END / router / mass_data_loader)
  * `route_after_postprocess()`: 출력 검열 결과로 분기 (END / dialogue_summary_trigger)
  * `build_fgi_graph()`: 모든 노드/엣지를 배선해 컴파일된 앱 반환 (MemorySaver 체크포인터)
  * `run_fgi_simulation_async()`: 진행자 대화형 루프 + astream 스트리밍 출력
* **`FGIBootstrap.py`** — Colab 재사용형 부트스트랩
  * `assemble_fgi()`: 래퍼 조립 + `find_relevant_*` 선행 + `initialize()` + Vertex/GCS 클라이언트 + 그래프 빌드
  * `main()`: 조립 후 실행 루프 (`await main(...)`)
* **`FGIState.py`** — 그래프 상태와 리듀서
  * `FGIState` (TypedDict): 그래프 전역 상태 (messages, agent_memories, sim_* 등)
  * `AgentSummaryTask`, `DialogueSummaryTask` (TypedDict): 병렬 요약 노드(Send) 페이로드
  * `update_agent_memories()`: agent_memories 부분 업데이트 병합 리듀서
* **`FGINodesAsync.py`** — 실시간 FGI 노드 팩토리
  * `create_preprocessor_node()`: 입력 가드레일 + 인텐트(FGI/MASS) 판별
  * `create_router_node()`: 모드(@ALL/@A/@FREE_TALK/침묵) 파싱 및 타겟 선정
  * `retrieve_dynamic_context()`: 코사인 유사도 기반 동적 컨텍스트 선별
  * `create_retriever_node()`: 타겟 에이전트의 관련 ML/PIXEL 임베딩 검색
  * `create_customer_node_async()`: 페르소나 발화 생성 (시맨틱 캐시 우선 → 구조화 LLM → fallback)
  * `create_postprocessor_node()`: 출력 가드레일, 위험 발화 경고 덮어쓰기
* **`FGISummaryNodesAsync.py`** — 계층적 메모리 요약 노드
  * `create_dialogue_summary_trigger()`: 토큰 한도 초과 시 메시지 절단 + 요약 트리거
  * `route_to_dialogue_summarizers()`: 에이전트별 1인칭 요약으로 fan-out (Send)
  * `create_dialogue_summarizer_node()`: 1인칭 시점 주관적 대화 요약
  * `create_memory_sync_node()`: 병렬 요약 합류(join) 지점
  * `create_agent_summarizer_node()`: ML/PIXEL 장기기억 Map-Reduce 요약
  * `create_route_to_summaries()`: recent 버퍼 임계 초과 에이전트로 fan-out
* **`MassSimulationNodesAsync.py`** — 대량 시뮬레이션 노드 (Job 제출까지)
  * `create_mass_data_loader_node()`: FAISS·BM25 검색으로 대상 DF 생성 (워커 스레드 오프로딩)
  * `create_simulation_runner_node()`: Gemini Batch Job 제출 (빈 대상 시 스킵)
  * `register_mass_simulation()`: 위 두 노드를 서브그래프로 등록 (→ END)

## `async_core/` — 비동기 컴포넌트
그래프 노드들이 의존하는 비동기 처리 컴포넌트입니다.

* **`ContextManagerAsync.py`**
  * `ContextManagerAsync` (Class): 로컬 템플릿 비동기 로딩 및 Jinja2 프롬프트 렌더링
    * `_get_template()`: 로컬 파일 비동기 읽기 + 캐싱 (`asyncio.to_thread`)
    * `render_single_template()`: 단일 템플릿 렌더링
    * `build_prompt()`: Pydantic 스키마 기반 시스템/유저 프롬프트 비동기 병렬 생성
* **`InputPreprocessorAsync.py`** — 입력 통제 계층
  * `InputGuardrail` (Class): SLM 기반 프롬프트 인젝션 탐지
  * `SemanticCacheRedisAsync` (Class): Redis 비동기 시맨틱 캐시 (장애 격리, persona_id 단위)
  * `ExecutionRouter` (Class): 인텐트 기반 파이프라인 라우팅
  * `InputPreprocessorAsync` (Class): 위를 통합 — `process_request()`(check_cache 옵션), `save_to_cache()`, `initialize()`
* **`OutputPostprocessorAsync.py`** — 출력 통제 계층
  * `OutputGuardrail` (Class): 독성/유해(safe LM)/PII/금칙어 검열 (슬라이딩 윈도우)
  * `OutputPostprocessorAsync` (Class): `process_output()`로 검열 결과(SAFE/BLOCKED) 반환
* **`SimulationEngineAsync.py`**
  * `SimulationEngineAsync` (Class): 대규모 데이터 → Gemini Batch 비동기 연동
    * `_prepare_payload()`: 단일 Row → JSONL Payload 비동기 파싱
    * `_build_jsonl_buffer()`: 전체 DF 병렬 렌더링
    * `run_batch()`: GCS 업로드 + Batch Job 비동기 생성
* **`BatchResultValidatorAsync.py`** — 그래프 밖 결과 수집/검증
  * `PromotionMeta` (Class): 캠페인 메타(가격·앵커) 추출 (Ground Truth)
  * `BatchResultValidatorAsync` (Class): `poll_and_ingest()`(완료 폴링+수집), `validate_all()`(룰 검증 + LLM 심판)

## `async_core/` — 레거시 (LangGraph 이전 클래스 구조)
LangGraph 도입 이전의 오케스트레이션 구현. 현재 메인 경로는 아니지만 보존됩니다.

* **`MultiAgentOrchestratorAsync.py`**
  * `_SharedMemoryRouterAsync`, `MultiAgentOrchestratorAsync` (Class): 다중 에이전트 대화 조정
    * `broadcast()`, `_get_next_agent()`, `_parse_intent_and_target()`, `_retrieve_dynamic_context()`, `process_turn()`
* **`PersonaAgentAsync.py`**
  * `PersonaAgentAsync` (Class): 단일 비동기 페르소나 에이전트 — `process_interviewer_turn()`
* **`HybridMemoryManagerAsync.py`**
  * `Turn` (Pydantic Model), `HybridMemoryManagerAsync` (Class): 대화/동적 컨텍스트 메모리 관리
    * `add_interaction()`, `_manage_memory_window()`, `_update_summary_batch()`, `add_dynamic_context()`, `get_combined_ml()`, `get_combined_pixel()`

## `sync_core/` — 동기 처리 코어
단일 스레드 순차 처리 버전. `DataLoader`는 비동기 그래프에서도 그대로 사용됩니다.

* **`DataLoader.py`** (그래프에서도 사용)
  * `DataLoader` (Class): 프로모션/고객 데이터 로딩·검색·필터링 파이프라인
    * `ms_table`, `ml_table`, `retriever`, `encoded_promo` (Property): 지연 로딩(Lazy)
    * `find_relevant_ms_faiss()`: FAISS 기반 마인드셋(MS) 벡터 검색
    * `find_relevant_ml()`: BM25 기반 유관 상품이력(ML) 검색 (Kiwi 하이브리드 토크나이즈)
    * `get_fgi_profile()`: 단일 에이전트용 프로필(+장기기억 임베딩) 생성
    * `get_mass_simulation_df()`: 대량 시뮬레이션용 데이터프레임 생성
* **`ContextManager.py` / `BatchResultValidator.py` / `SimulationEngine.py` / `MultiAgentOrchestrator.py` / `PersonaAgent.py` / `HybridMemoryManager.py`**
  * 각 `*Async` 버전의 동기 구현 (메서드 구조 동일, API 대기 시 블로킹).

## `common/` — 공통 유틸리티 및 스키마
시스템 전반에서 공통 의존하는 데이터 구조·헬퍼·경로 상수입니다.

* **`schemas.py`** — Pydantic 입출력 스키마
  * `SimulationDataSchema`: 시뮬레이션 기초 데이터 (MS / ML / PIXEL / PROMOTION_INFO)
  * `FGIDataSchema`: FGI 맥락 추가 (CONVERSATION_HISTORY, USER_INPUT)
  * `MultiAgentFGIDataSchema`: 다중 에이전트 라우팅 메타데이터 포함
  * `DataSummarySchema`: ML/PIXEL 요약 입력
  * `DialogueSummarySchema`: 대화 요약 입력 (MS, LAST_INTERNAL_STATE 포함)
  * `InternalState`: 에이전트 내면 상태(속마음) — LLM 구조화 출력
  * `FGIResponse`: 내면 상태 + 실제 발화 (그래프 응답 스키마)
* **`paths.py`**
  * 프롬프트 템플릿 경로 상수 (`__file__` 기준 절대경로 — CWD 비의존)
* **`utils.py`**
  * `clean_json_string()`: LLM 응답의 Markdown 찌꺼기 제거
  * `debug_node()`: LangGraph 노드 진입/완료/에러 로깅 데코레이터 (동기/비동기 겸용)
* **`Prompts/`**
  * FGI/요약/시뮬레이션/심판/캠페인추출용 `*.md` Jinja2 템플릿
