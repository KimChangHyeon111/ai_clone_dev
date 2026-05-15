import io
import json
import uuid
import polars as pl
from google.cloud import storage

# GCP SDK
import vertexai
from google import genai
from google.genai import types
from vertexai.generative_models import GenerativeModel, GenerationConfig

class SimulationEngine:
    def __init__(
        self,
        genai_client: genai.Client,
        storage_client: storage.Client,
        context_manager, # 통합 ContextManager
        sys_tmpl_path: str,  # [추가됨] 시뮬레이션용 시스템 프롬프트 경로
        user_tmpl_path: str, # [추가됨] 시뮬레이션용 유저 프롬프트 경로
        model_name: str = "gemini-2.5-flash-lite"
    ):
        # 3. SDK 초기화 파편화 개선: 외부에서 생성된 클라이언트를 주입받아 사용
        self.client = genai_client
        self.storage_client = storage_client
        self.ctx = context_manager

        # [추가됨] 템플릿 경로를 인스턴스 변수로 저장
        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path

        self.model_name = model_name

    def _prepare_payload(self, row: dict, promotion_info: str) -> dict:
        raw_ms = row.get('ms_core_mindset') or row.get('ms') or ""
        raw_ml = row.get('ml_transaction_history') or row.get('ml') or "[]"
        parsed_ml = json.loads(raw_ml) if isinstance(raw_ml, str) else raw_ml
        srg_key = str(row.get('srg_key', 'unknown'))

        valid_data = SimulationDataSchema(
            ms=raw_ms,
            ml=parsed_ml,
            promo_info=promotion_info
        )

        # [수정됨] 통합 ContextManager의 build_prompt 스펙에 맞춰 경로 전달
        sys_inst, user_content = self.ctx.build_prompt(
            sys_path=self.sys_tmpl_path,
            user_path=self.user_tmpl_path,
            data=valid_data
        )

        return {
            "custom_id": srg_key,
            "request": {
                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                "system_instruction": {"parts": [{"text": sys_inst}]},
                "generation_config": {
                    "temperature": 0.1,
                    "response_mime_type": "application/json"
                }
            }
        }

    def run_batch(self, df: pl.DataFrame, promotion_info: str, bucket_name: str):
        job_id = f"sim_{uuid.uuid4().hex[:8]}"
        gcs_input_uri = f"gs://{bucket_name}/batch_input/{job_id}.jsonl"
        gcs_output_uri = f"gs://{bucket_name}/batch_output/{job_id}/"

        # 2. 로컬 디스크 의존성 제거: /tmp 파일 대신 메모리 버퍼(io.StringIO) 사용
        buffer = io.StringIO()

        # 1. Polars 성능 이점 향상: to_dicts() 전체 변환 대신 iter_rows로 메모리 사용량 최소화 및 속도 개선
        for row in df.iter_rows(named=True):
            payload = self._prepare_payload(row, promotion_info)
            buffer.write(json.dumps(payload, ensure_ascii=False) + "\n")

        # 메모리 버퍼 데이터를 GCS로 직접 업로드
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(f"batch_input/{job_id}.jsonl")
        blob.upload_from_string(buffer.getvalue(), content_type="application/jsonl")

        buffer.close()

        batch_job = self.client.batches.create(
            model=self.model_name,
            src=gcs_input_uri,
            config=types.CreateBatchJobConfig(dest=gcs_output_uri)
        )
        # 튜플로 반환 (job_name, job_id)
        return batch_job.name, job_id

class MassSimulationManager:
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        sys_tmpl_path: str,
        user_tmpl_path: str,
        location: str = "global"
    ):
        self.bucket_name = bucket_name

        # 3. 최상위 매니저에서 클라이언트들을 단 한 번만 초기화
        vertexai.init(project=project_id, location=location)
        self.genai_client = genai.Client(vertexai=True, project=project_id, location=location)
        self.storage_client = storage.Client(project=project_id)

        sim_ctx = ContextManager()

        # 초기화된 클라이언트를 Engine에 주입
        self.engine = SimulationEngine(
            genai_client=self.genai_client,
            storage_client=self.storage_client,
            context_manager=sim_ctx,
            sys_tmpl_path=sys_tmpl_path,
            user_tmpl_path=user_tmpl_path
        )

    def launch_campaign(self, target_df: pl.DataFrame, promotion_info: str) -> str:
        if target_df.is_empty():
            raise ValueError("시뮬레이션 대상 고객이 없습니다.")
        return self.engine.run_batch(target_df, promotion_info, self.bucket_name)

class PersonaAgent:
    def __init__(
        self,
        srg_key: str,
        fgi_profile: dict,
        sys_tmpl_path: str,
        user_tmpl_path: str,
        location: str = "global",
        context_manager = None
    ):
        self.srg_key = srg_key

        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.ctx = context_manager or ContextManager()

        self.memory = HybridMemoryManager()
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

        self.profile = {
            "ms_core_mindset": fgi_profile.get("ms", ""),
            "ml_transaction_history": fgi_profile.get("ml", [])
        }
        self.is_valid = bool(self.profile["ms_core_mindset"])

    def process_interviewer_turn(self, interviewer_input: str, promo_info: str = "프로모션 정보 없음") -> str:
        if not self.is_valid:
            return "[시스템 알림] 데이터가 부족합니다."

        valid_data = FGIDataSchema(
            ms=self.profile["ms_core_mindset"],
            ml=self.profile["ml_transaction_history"],
            promo_info=promo_info,
            conversation_history=self.memory.get_formatted_history(),
            user_input=interviewer_input
        )

        sys_inst, user_content = self.ctx.build_prompt(
            sys_path=self.sys_tmpl_path,
            user_path=self.user_tmpl_path,
            data=valid_data
        )

        prompt = f"Interviewer: {user_content}\nCustomer: "

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.6
            )
        )
        response_text = response.text.strip()
        self.memory.add_interaction(interviewer_input, response_text)

        return response_text
