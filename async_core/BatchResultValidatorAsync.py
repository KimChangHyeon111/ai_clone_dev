import json
import asyncio
import re
import polars as pl
from kiwipiepy import Kiwi
from google.cloud import storage
from google.genai import Client, types
from pydantic import BaseModel, Field
from typing import List, Set, Tuple, Literal, Union

# 💡 비동기로 개편된 ContextManagerAsync 스키마 임포트
from async_core.ContextManagerAsync import ContextManagerAsync
from common.utils import clean_json_string

# =====================================================================
# 1. 데이터 스키마 정의 (기존과 동일)
# =====================================================================
class SimulationResultSchema(BaseModel):
    reasoning: str
    decision: Literal["ACCEPT", "REJECT", "HOLD"]
    probability_score: float = Field(ge=0.0, le=1.0)
    primary_reason: str
    feedback_keyword: Union[str, List[str]]
    willingness_to_pay: int
    improvement_suggestion: str

class PromoExtractionSchema(BaseModel):
    price: int = Field(description="프로모션 기준 가격 (숫자)")
    promo_anchors: List[str] = Field(description="상품명, 혜택, 핵심 가치 등을 모두 포함한 통합 키워드 리스트")

class JudgeResultSchema(BaseModel):
    is_valid: bool
    judge_reason: str

class JudgeDataSchema(BaseModel):
    price: int
    promo_info: str
    decision: str
    probability_score: float
    willingness_to_pay: int
    primary_reason: str
    reasoning: str
    logic_error_reasons: List[str]

# =====================================================================
# 2. 프로모션 메타 관리 (Ground Truth) - 비동기 업그레이드
# =====================================================================
class PromotionMeta:
    def __init__(self, price: int, promo_anchors: List[str], promo_info: str = ''):
        self.price = price
        self.anchors: Set[str] = set(promo_anchors)
        self.common_nouns = {
            "성향", "패턴", "경향", "관심사", "페르소나", "기호", "니즈", "충성도", "로열티", "속성",
            "프로모션", "혜택", "리워드", "오퍼", "베네핏", "특전", "어드밴티지", "바우처", "켐페인", "소구",
            "이력", "이용", "구매", "소비", "결제", "히스토리", "지출", "내역"
        }
        self.anchors = self.anchors | self.common_nouns
        self.promo_info = promo_info

    # 💡 1. async 메서드로 변경
    @classmethod
    async def extract(cls, client: Client, context_manager: ContextManagerAsync, tmpl_path: str, promo_info: str, model_name:str = "gemini-2.5-flash-lite") -> 'PromotionMeta':
        # 💡 비동기 렌더링 대기
        prompt = await context_manager.render_single_template(tmpl_path, promo_info=promo_info)

        # 💡 비동기 API 호출
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromoExtractionSchema,
                temperature=0.0
            )
        )
        clean_txt = clean_json_string(response.text)
        data = json.loads(clean_txt)
        return cls(price=data.get("price", 0), promo_anchors=data.get("promo_anchors", []), promo_info=promo_info)

# =====================================================================
# 3. 통합 검증 및 수집 클래스 - 비동기 업그레이드
# =====================================================================
class BatchResultValidatorAsync:
    def __init__(
            self,
            genai_client: Client,
            storage_client: storage.Client,
            bucket_name: str,
            context_manager: ContextManagerAsync,
            judge_sys_tmpl_path: str,
            judge_user_tmpl_path: str,
            model_name:str = "gemini-2.5-flash-lite"
    ):
        self.client = genai_client
        self.storage_client = storage_client
        self.bucket = self.storage_client.bucket(bucket_name)

        print("Kiwi 형태소 분석기 로드 중...")
        self.kiwi = Kiwi() # Kiwi는 CPU 연산이므로 초기화 유지
        self.model_name = model_name

        self.context_manager = context_manager
        self.judge_sys_tmpl_path = judge_sys_tmpl_path
        self.judge_user_tmpl_path = judge_user_tmpl_path

    # --- [수집부] ---
    async def poll_and_ingest(self, google_job_name: str, custom_sim_id: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        print(f"👀 작업 감시 및 수집 시작: {custom_sim_id}")

        # 💡 2. 가장 위험했던 time.sleep(60)을 비동기 sleep으로 변경
        while True:
            try:
                job_status = await self.client.aio.batches.get(name=google_job_name)
            except AttributeError:
                # SDK 버전에 따라 batches.get의 aio 지원이 안 될 경우 대비
                job_status = await asyncio.to_thread(self.client.batches.get, name=google_job_name)

            state = str(job_status.state)
            if "SUCCEEDED" in state: break
            if any(err in state for err in ["FAILED", "CANCELLED"]): raise RuntimeError(f"Job Failed: {state}")

            await asyncio.sleep(60)

        base_prefix = f"batch_output/{custom_sim_id}/"

        # 💡 3. GCS 파일 다운로드 및 파싱(I/O + CPU 연산)을 통째로 워커 쓰레드로 오프로딩
        def _fetch_and_parse():
            blobs = self.bucket.list_blobs(prefix=base_prefix)
            valid, error = [], []
            for blob in blobs:
                if not blob.name.endswith(".jsonl"): continue
                with blob.open("r") as f:
                    for line in f:
                        res = self._parse_line(line)
                        if res.get("is_error"): error.append(res)
                        else: valid.append(res)
            return valid, error

        valid_records, error_records = await asyncio.to_thread(_fetch_and_parse)
        return pl.DataFrame(valid_records), pl.DataFrame(error_records)

    def _parse_line(self, line: str) -> dict:
        try:
            data = json.loads(line)
            custom_id = data.get('custom_id', 'unknown')
            raw_txt = data['response']['candidates'][0]['content']['parts'][0]['text']

            clean_txt = clean_json_string(raw_txt)
            payload = json.loads(clean_txt)

            validated = SimulationResultSchema(**payload)
            return {"srg_key": custom_id, "is_error": False, **validated.model_dump()}
        except Exception as e:
            return {"srg_key": "unknown", "is_error": True, "error_message": str(e)}

    # --- [검증부] ---
    # 💡 4. validate_all을 async def로 변경하여 await 체인 연결
    async def validate_all(self, df: pl.DataFrame, meta: PromotionMeta) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if df.is_empty(): return pl.DataFrame(), pl.DataFrame()

        # 1차 룰 기반 필터링 (Polars 연산이므로 동기 호출 유지 - 아주 빠름)
        logic_valid_df, suspicious_df = self._run_rule_check(df, meta)

        # 💡 2차 LLM 심사 (비동기 병렬 대기)
        rescued_df, confirmed_error_df = await self._run_llm_judge(suspicious_df, meta)

        if not rescued_df.is_empty():
            clean_rescued = rescued_df.drop(["logic_error_reasons", "llm_is_valid", "llm_judge_reason"])
            clean_valid = logic_valid_df.drop(['logic_error_reasons'])
            final_df = pl.concat([clean_valid, clean_rescued], how="vertical")
        else:
            final_df = logic_valid_df

        return final_df, confirmed_error_df

    def _run_rule_check(self, df: pl.DataFrame, meta: PromotionMeta) -> Tuple[pl.DataFrame, pl.DataFrame]:
        # (기존 코드와 완벽히 동일 - CPU 단일 쓰레드 연산)
        df = df.with_columns([
            ((pl.col("decision") == "ACCEPT") & (pl.col("probability_score") < 0.5)).alias("err_score"),
            ((pl.col("decision") == "ACCEPT") & (pl.col("willingness_to_pay") < (meta.price * 0.7))).alias("err_wtp")
        ])

        df = df.with_columns(
            pl.col("primary_reason").fill_null('').map_elements(
                lambda x: list({t.form for t in self.kiwi.tokenize(x) if t.tag.startswith('N')} & meta.anchors),
                return_dtype=pl.List(pl.String)
            ).alias("matched_concepts")
        ).with_columns(
            (pl.col("matched_concepts").list.len() == 0).alias("err_context_mismatch")
        )

        err_cols = ["err_score", "err_wtp", "err_context_mismatch"]
        df = df.with_columns(
            pl.any_horizontal(pl.col(err_cols)).alias("is_suspicious")
        ).with_columns(
            pl.struct(err_cols).map_elements(
                lambda x: [k for k, v in x.items() if v],
                return_dtype=pl.List(pl.String)
            ).alias("logic_error_reasons")
        )

        return df.filter(~pl.col("is_suspicious")).drop(err_cols + ["is_suspicious", "matched_concepts"]), \
               df.filter(pl.col("is_suspicious")).drop(err_cols + ["is_suspicious", "matched_concepts"])

    # 💡 5. 개별 심사 작업을 비동기 단일 함수로 쪼갬
    async def _judge_single_row(self, row: dict, meta: PromotionMeta) -> dict:
        judge_data = JudgeDataSchema(
            price=meta.price,
            promo_info=meta.promo_info,
            decision=row.get('decision', 'UNKNOWN'),
            probability_score=row.get('probability_score', 0.0),
            willingness_to_pay=row.get('willingness_to_pay', 0),
            primary_reason=row.get('primary_reason', ''),
            reasoning=row.get('reasoning', ''),
            logic_error_reasons=row.get('logic_error_reasons', [])
        )

        # 비동기 프롬프트 렌더링
        sys_inst, user_content = await self.context_manager.build_prompt(
            sys_path=self.judge_sys_tmpl_path,
            user_path=self.judge_user_tmpl_path,
            data=judge_data
        )

        try:
            # 비동기 API 통신
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    response_mime_type="application/json",
                    response_schema=JudgeResultSchema,
                    temperature=0.0
                )
            )

            clean_txt = clean_json_string(response.text)
            result = json.loads(clean_txt)
            row['llm_is_valid'] = result.get('is_valid', False)
            row['llm_judge_reason'] = result.get('judge_reason', '판결 사유 누락')

        except Exception as e:
            print(f"⚠️ 심사 API 에러: {e}")
            row['llm_is_valid'] = False
            row['llm_judge_reason'] = f"API Failed: {str(e)}"

        return row

    # 💡 6. asyncio.gather를 통한 병렬 심사 처리! (100건도 수 초 내에 완료)
    async def _run_llm_judge(self, suspicious_df: pl.DataFrame, meta: PromotionMeta) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if suspicious_df.is_empty():
            return pl.DataFrame(), pl.DataFrame()

        print(f"⚖️ 2차 LLM 판사 심사 시작... (총 {suspicious_df.height}건 병렬 처리 중)")

        # 태스크 리스트 생성 및 병렬 대기
        tasks = [self._judge_single_row(row, meta) for row in suspicious_df.to_dicts()]
        results = await asyncio.gather(*tasks)

        res_df = pl.DataFrame(results)
        rescued_df = res_df.filter(pl.col("llm_is_valid") == True)
        confirmed_error_df = res_df.filter(pl.col("llm_is_valid") == False)

        return rescued_df, confirmed_error_df
