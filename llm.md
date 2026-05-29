# AI Clone Development - System Architecture Map

> **Harness Engineering Map**: 본 문서는 시스템의 모듈간 의존성, 데이터 파이프라인 흐름, 비동기/동기 처리 계층을 한눈에 파악하기 위한 아키텍처 지도(Map)입니다.
## `async_core/` (비동기 처리 코어 모듈)
비동기(I/O, 네트워크 대기 최소화) 파이프라인 처리를 담당하는 모듈 그룹입니다.

* **`ContextManagerAsync.py`**
  * `ContextManagerAsync` (Class): GCS 템플릿 비동기 로딩 및 Jinja2 프롬프트 렌더링 
    * `__init__()`: GCS 스토리지 클라이언트 및 캐시 초기화
    * `_parse_gcs_path()`: GCS 경로 파싱
    * `_get_template()`: 템플릿 비동기 다운로드 및 캐싱 (`asyncio.to_thread` 활용)
    * `render_single_template()`: 단일 템플릿 렌더링
    * `build_prompt()`: Pydantic 스키마 기반 시스템/유저 프롬프트 비동기 병렬 생성
* **`HybridMemoryManagerAsync.py`**
  * `Turn` (Pydantic Model): 단일 대화 단위 스키마
  * `HybridMemoryManagerAsync` (Class): 비동기 기반 대화 및 동적 컨텍스트(ML/PIXEL) 메모리 관리
    * `__init__()`: 메모리 한도 설정 및 Gemini 모델 초기화
    * `_count_tokens()`: 비동기 토큰 카운팅
    * `get_formatted_history()`: 대화 기록 포맷팅 (동기)
    * `add_interaction()`: 신규 대화 추가 및 토큰 한도 체크
    * `_manage_memory_window()`: 토큰 한도 초과 시 메모리 윈도우 슬라이싱
    * `_update_summary_batch()`: LLM 활용 대화 이력 비동기 압축/요약
    * `add_dynamic_context()`: ML/PIXEL 데이터 누적 및 요약 트리거
    * `_summarize_context()`: 누적된 동적 정보 백그라운드 요약
    * `get_combined_ml()`, `get_combined_pixel()`: 원본과 요약본 병합
* **`MultiAgentOrchestratorAsync.py`**
  * `_SharedMemoryRouterAsync` (Class): 비동기 환경용 공유 메모리 브로드캐스트 라우터
  * `MultiAgentOrchestratorAsync` (Class): 다중 에이전트 간의 대화 오케스트레이션
    * `__init__()`: 에이전트 그룹, 프롬프트 경로, 라우터 초기화
    * `broadcast()`: 전체 에이전트 메모리에 대화 동시 전파 (`asyncio.gather` 활용)
    * `_get_next_agent()`: 순차 대화 시 다음 타자 지정
    * `_parse_intent_and_target()`: 발화 인텐트 파싱 (@멘션, 자유토론 등)
    * `_retrieve_dynamic_context()`: 코사인 유사도 기반 동적 컨텍스트 벡터 검색 (Numpy 연산)
    * `process_turn()`: 사용자 입력 파싱, 컨텍스트 주입, LLM 비동기 호출 및 결과 브로드캐스트 파이프라인
* **`PersonaAgentAsync.py`**
  * `PersonaAgentAsync` (Class): 개별 페르소나를 부여받은 단일 비동기 에이전트
    * `__init__()`: 프로필(MS, ML, PIXEL) 및 임베딩 로드, Gemini Client 초기화
    * `process_interviewer_turn()`: 인터뷰어 질의에 대한 프롬프트 생성, 비동기 LLM 호출 및 응답 반환
* **`SimulationEngineAsync.py`**
  * `SimulationEngineAsync` (Class): 대규모 고객 데이터 비동기 시뮬레이션 및 Batch API 연동
    * `__init__()`: 클라이언트 및 컨텍스트 매니저 주입
    * `_prepare_payload()`: 단일 Row 데이터를 Gemini Batch API용 JSONL 포맷으로 비동기 파싱
    * `_build_jsonl_buffer()`: 전체 데이터프레임을 JSONL 버퍼로 병렬 렌더링
    * `run_batch()`: GCS 업로드 및 Gemini Batch Job 비동기 생성

## `sync_core/` (동기 처리 코어 모듈)
단일 스레드 기반으로 순차 처리를 진행하는 코어 파이프라인 모듈 그룹입니다.

* **`ContextManager.py`**
  * `ContextManager` (Class): GCS 템플릿 로드 및 동기식 프롬프트 렌더링 (`Async` 버전과 메서드 구조 동일)
* **`HybridMemoryManager.py`**
  * `Turn` (Pydantic Model)
  * `HybridMemoryManager` (Class): 동기식 대화 이력 및 동적 컨텍스트 요약 관리 (`Async` 버전과 메서드 구조 동일)
* **`MultiAgentOrchestrator.py`**
  * `_SharedMemoryRouter` (Class): 동기식 라우터
  * `MultiAgentOrchestrator` (Class): 동기식 다중 에이전트 FGI 조정 (벡터 검색 및 프롬프트 주입 동기 처리)
* **`PersonaAgent.py`**
  * `PersonaAgent` (Class): 동기식 개별 페르소나 에이전트 (API 호출 대기 블로킹 발생)
* **`SimulationEngine.py`**
  * `SimulationEngine` (Class): 대규모 시뮬레이션을 위한 데이터 버퍼링 및 Batch 작업 요청 (Polars `iter_rows` 활용 메모리 최적화)
* **`DataLoader.py`**
  * `DataLoader` (Class): 프로모션 및 고객 데이터 로딩, 검색, 필터링 파이프라인
    * `__init__()`: 경로 및 기본 재료 주입
    * `ms_table`, `ml_table`, `retriever`, `encoded_promo` (Property): 메모리 최적화를 위한 지연 로딩(Lazy Loading) 프로퍼티
    * `find_relevant_ms_faiss()`: FAISS GPU 기반 마인드셋(MS) 벡터 검색
    * `_yield_hybrid_tokenize()`: Kiwi 형태소 분석기 기반 하이브리드 토크나이징 (제너레이터)
    * `find_relevant_ml()`: BM25 기반 유관 상품 이력(ML) 검색
    * `get_fgi_profile()`: 검색된 결과를 종합하여 단일 에이전트용 프로필 딕셔너리 생성 (장기 기억용 벡터 포함)
    * `get_mass_simulation_df()`: 필터링된 데이터를 조인하여 대규모 시뮬레이션용 데이터프레임 생성 (Polars 네이티브 JSON 직렬화 적용)

## `common/` (공통 유틸리티 및 스키마)
시스템 전반에서 공통으로 의존하는 데이터 구조 및 헬퍼 함수 모음입니다.

* **`schemas.py`**
  * Pydantic 구조화 템플릿 (입출력 정합성 보장)
  * `SimulationDataSchema`: 시뮬레이션 기초 데이터 (MS, ML 구조화 리스트, PIXEL 구조화 리스트, PROMOTION_INFO)
  * `FGIDataSchema`: FGI 추가 맥락 데이터 (CONVERSATION_HISTORY, USER_INPUT 포함)
  * `MultiAgentFGIDataSchema`: 다중 에이전트 라우팅 메타데이터 포함
  * `MemorySummaryDataSchema`: 메모리 요약 추적용 스키마
* **`utils.py`**
  * `clean_json_string()`: LLM 응답에 포함된 Markdown 찌꺼기(```json 등)를 정규식으로 제거하는 헬퍼 함수
