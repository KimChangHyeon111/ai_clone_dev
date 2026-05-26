import io
import json
import uuid
import asyncio
import polars as pl
from typing import List, Tuple

# GCP SDK
from google import genai
from google.genai import types
from google.cloud import storage

# 💡 비동기로 개편된 ContextManagerAsync 스키마 임포트
from async_core.ContextManagerAsync import ContextManagerAsync, SimulationDataSchema

class SimulationEngineAsync:
    def __init__(
        self,
        genai_client: genai.Client,
        storage_client: storage.Client,
        context_manager,
        sys_tmpl_path: str,
        user_tmpl_path: str,
        model_name: str = "gemini-2.5-flash-lite"
    ):
        self.client = genai_client
        self.storage_client = storage_client
        # 💡 외부에서 ContextManagerAsync 인스턴스가 주입된다고 가정합니다.
        self.ctx = context_manager

        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.model_name = model_name

    # 💡 1. async def로 변경하여 ContextManagerAsync와 호환
    async def _prepare_payload(self, row: dict, promotion_info: str) -> dict:
        """단일 행(Row) 데이터를 JSONL Payload 구조로 파싱합니다."""
        raw_ms = row.get('ms_core_mindset') or row.get('ms') or ""
        raw_ml = row.get('ml_transaction_history') or row.get('ml') or "[]"
        parsed_ml = json.loads(raw_ml) if isinstance(raw_ml, str) else raw_ml
        srg_key = str(row.get('srg_key', 'unknown'))

        valid_data = SimulationDataSchema(
            ms=raw_ms,
            ml=parsed_ml,
            promo_info=promotion_info
        )

        # 💡 [핵심] await를 통한 네이티브 비동기 프롬프트 생성
        sys_inst, user_content = await self.ctx.build_prompt(
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

    # 💡 2. 전체 버퍼 생성 로직을 비동기로 개편
    async def _build_jsonl_buffer(self, df: pl.DataFrame, promotion_info: str) -> str:
        # Step 1: 전체 Row에 대한 Task 리스트 생성 및 병렬 실행 (초고속 프롬프트 렌더링)
        tasks = [self._prepare_payload(row, promotion_info) for row in df.iter_rows(named=True)]
        payloads = await asyncio.gather(*tasks)

        # Step 2: CPU 집약적인 직렬화 및 문자열 결합 로직 분리
        def _compile_buffer():
            buffer = io.StringIO()
            for payload in payloads:
                buffer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            result = buffer.getvalue()
            buffer.close()
            return result

        # Step 3: 순수 CPU 연산만 워커 쓰레드로 던져서 이벤트 루프 방어
        return await asyncio.to_thread(_compile_buffer)

    async def run_batch(self, df: pl.DataFrame, promotion_info: str, bucket_name: str) -> Tuple[str, str]:
        job_id = f"sim_{uuid.uuid4().hex[:8]}"
        gcs_input_uri = f"gs://{bucket_name}/batch_input/{job_id}.jsonl"
        gcs_output_uri = f"gs://{bucket_name}/batch_output/{job_id}/"

        # 💡 3. to_thread(self._build_jsonl_buffer) 대신 직접 await 호출
        payload_data = await self._build_jsonl_buffer(df, promotion_info)

        # GCS 업로드 네트워크 I/O 비동기화 (이 부분은 GCS 라이브러리가 동기식이므로 유지)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(f"batch_input/{job_id}.jsonl")

        await asyncio.to_thread(
            blob.upload_from_string,
            payload_data,
            content_type="application/jsonl"
        )

        # Gemini Batch API 호출 비동기화
        try:
            batch_job = await self.client.aio.batches.create(
                model=self.model_name,
                src=gcs_input_uri,
                config=types.CreateBatchJobConfig(dest=gcs_output_uri)
            )
        except AttributeError:
            batch_job = await asyncio.to_thread(
                self.client.batches.create,
                model=self.model_name,
                src=gcs_input_uri,
                config=types.CreateBatchJobConfig(dest=gcs_output_uri)
            )

        return batch_job.name, job_id
