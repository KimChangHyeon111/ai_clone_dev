import json
import time
import re
import polars as pl
from kiwipiepy import Kiwi
from google.cloud import storage
from google.genai import Client, types
from pydantic import BaseModel, Field
from typing import List, Set, Tuple, Literal, Union

# =====================================================================
# 1. 데이터 스키마 정의
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

# [추가됨] ContextManager로 넘겨줄 심사용 데이터 스키마
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
# 2. 프로모션 메타 관리 (Ground Truth)
# =====================================================================
class PromotionMeta:
    def __init__(
            self,
            price: int,
            promo_anchors: List[str],
            promo_info: str = ''
        ):
        self.price = price
        self.anchors: Set[str] = set(promo_anchors)
        self.common_nouns = {
            "성향", "패턴", "경향", "관심사", "페르소나", "기호", "니즈", "충성도", "로열티", "속성",  # 고객 특성
            "프로모션", "혜택", "리워드", "오퍼", "베네핏", "특전", "어드밴티지", "바우처", "켐페인", "소구" # 혜택
            "이력", "이용", "구매", "소비", "결제", "히스토리", "지출", "내역" # 이용
        }
        self.anchors = self.anchors | self.common_nouns
        self.promo_info = promo_info


    @classmethod
    def extract(cls, client: Client, context_manager, tmpl_path: str, promo_info: str, model_name:str = "gemini-2.5-flash-lite") -> 'PromotionMeta':
        # 여기서 시스템이 없는게 약간 이상하네. 이것도 추가해야 할듯.
        prompt = context_manager.render_single_template(tmpl_path, promo_info=promo_info)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromoExtractionSchema,
                temperature=0.0
            )
        )
        data = json.loads(response.text)
        return cls(price=data.get("price", 0), promo_anchors=data.get("promo_anchors", []), promo_info=promo_info)

# =====================================================================
# 3. 통합 검증 및 수집 클래스
# =====================================================================
class BatchResultValidator:
    def __init__(
            self,
            genai_client: Client,
            storage_client: storage.Client,
            bucket_name: str,
            context_manager,           # [수정됨] 외부에서 주입받음
            judge_sys_tmpl_path: str,  # [수정됨] 시스템 템플릿 경로
            judge_user_tmpl_path: str,  # [수정됨] 유저 템플릿 경로
            model_name:str = "gemini-2.5-flash-lite"
    ):
        self.client = genai_client
        self.storage_client = storage_client
        self.bucket = self.storage_client.bucket(bucket_name)

        print("Kiwi 형태소 분석기 로드 중...")
        self.kiwi = Kiwi()
        self.model_name = model_name

        # [수정됨] 의존성 주입된 변수 저장 (기존 하드코딩 프롬프트 삭제)
        self.context_manager = context_manager
        self.judge_sys_tmpl_path = judge_sys_tmpl_path
        self.judge_user_tmpl_path = judge_user_tmpl_path

    # --- [수집부] ---
    def poll_and_ingest(self, google_job_name: str, custom_sim_id: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        print(f"👀 작업 감시 및 수집 시작: {custom_sim_id}")
        while True:
            job_status = self.client.batches.get(name=google_job_name)
            state = str(job_status.state)
            if "SUCCEEDED" in state: break
            if any(err in state for err in ["FAILED", "CANCELLED"]): raise RuntimeError(f"Job Failed: {state}")
            time.sleep(60)

        base_prefix = f"batch_output/{custom_sim_id}/"
        blobs = self.bucket.list_blobs(prefix=base_prefix)
        valid_records, error_records = [], []

        for blob in blobs:
            if not blob.name.endswith(".jsonl"): continue
            with blob.open("r") as f:
                for line in f:
                    res = self._parse_line(line)
                    if res.get("is_error"): error_records.append(res)
                    else: valid_records.append(res)

        return pl.DataFrame(valid_records), pl.DataFrame(error_records)

    def _parse_line(self, line: str) -> dict:
        try:
            data = json.loads(line)
            custom_id = data.get('custom_id', 'unknown')
            raw_txt = data['response']['candidates'][0]['content']['parts'][0]['text']
            clean_txt = re.sub(r'^```json\n|^```|```$', '', raw_txt.strip(), flags=re.MULTILINE).strip()
            payload = json.loads(clean_txt)
            validated = SimulationResultSchema(**payload)
            return {"srg_key": custom_id, "is_error": False, **validated.model_dump()}
        except Exception as e:
            return {"srg_key": "unknown", "is_error": True, "error_message": str(e)}

    # --- [검증부] ---
    def validate_all(self, df: pl.DataFrame, meta: PromotionMeta) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if df.is_empty(): return pl.DataFrame(), pl.DataFrame()

        # 1차 룰 기반 필터링
        logic_valid_df, suspicious_df = self._run_rule_check(df, meta)

        # 2차 LLM 심사
        rescued_df, confirmed_error_df = self._run_llm_judge(suspicious_df, meta)

        # 스키마 불일치 방지를 위해 컬럼 정리 후 병합
        if not rescued_df.is_empty():
            clean_rescued = rescued_df.drop(["logic_error_reasons", "llm_is_valid", "llm_judge_reason"])
            clean_valid = logic_valid_df.drop(['logic_error_reasons'])
            final_df = pl.concat([clean_valid, clean_rescued], how="vertical")
        else:
            final_df = logic_valid_df

        return final_df, confirmed_error_df

    def _run_rule_check(self, df: pl.DataFrame, meta: PromotionMeta) -> Tuple[pl.DataFrame, pl.DataFrame]:
        # 1-A. 수치/결정 논리 검증
        df = df.with_columns([
            ((pl.col("decision") == "ACCEPT") & (pl.col("probability_score") < 0.5)).alias("err_score"),
            ((pl.col("decision") == "ACCEPT") & (pl.col("willingness_to_pay") < (meta.price * 0.7))).alias("err_wtp")
        ])

        # 1-B. 맥락 불일치 탐지
        df = df.with_columns(
            pl.col("primary_reason").fill_null('').map_elements(
                lambda x: list({t.form for t in self.kiwi.tokenize(x) if t.tag.startswith('N')} & meta.anchors),
                return_dtype=pl.List(pl.String)
            ).alias("matched_concepts")
        ).with_columns(
            (pl.col("matched_concepts").list.len() == 0).alias("err_context_mismatch")
        )

        # 1-C. 통합 에러 판정
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

    def _run_llm_judge(self, suspicious_df: pl.DataFrame, meta: PromotionMeta) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if suspicious_df.is_empty():
            return pl.DataFrame(), pl.DataFrame()

        print(f"⚖️ 2차 LLM 판사 심사 시작... (총 {suspicious_df.height}건)")
        results = []

        for row in suspicious_df.to_dicts():
            # 1. Pydantic 스키마에 맞춰 데이터 매핑
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

            # 2. [수정됨] ContextManager를 활용한 프롬프트 동적 생성
            sys_inst, user_content = self.context_manager.build_prompt(
                sys_path=self.judge_sys_tmpl_path,
                user_path=self.judge_user_tmpl_path,
                data=judge_data
            )

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,  # 렌더링된 유저 프롬프트 주입
                    config=types.GenerateContentConfig(
                        system_instruction=sys_inst,  # 렌더링된 시스템 프롬프트 주입
                        response_mime_type="application/json",
                        response_schema=JudgeResultSchema,
                        temperature=0.0
                    )
                )

                result = json.loads(response.text)
                row['llm_is_valid'] = result.get('is_valid', False)
                row['llm_judge_reason'] = result.get('judge_reason', '판결 사유 누락')

            except Exception as e:
                print(f"⚠️ 심사 API 에러: {e}")
                row['llm_is_valid'] = False
                row['llm_judge_reason'] = f"API Failed: {str(e)}"

            results.append(row)

        res_df = pl.DataFrame(results)

        rescued_df = res_df.filter(pl.col("llm_is_valid") == True)
        confirmed_error_df = res_df.filter(pl.col("llm_is_valid") == False)

        return rescued_df, confirmed_error_df
