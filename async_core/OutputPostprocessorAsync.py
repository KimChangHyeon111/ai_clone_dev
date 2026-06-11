import os
import re
import asyncio
import polars as pl
from transformers import pipeline
from typing import Tuple, Dict, Any, List, Union, Optional
import torch

# =====================================================================
# 1. 출력 통제 계층 (입력과 동일한 DeBERTa 파이프라인 공유)
# =====================================================================
class OutputGuardrail:
    """TOXIC / UNSAFE OUTPUT 수정"""

    def __init__(
        self,
        toxic_pipeline,
        safe_tokenizer,
        safe_model,
        pii_pipeline,
        untoxic_threshold: float = 0.8,
        safe_threshold: float = 0.8,
        pii_threshold: float = 0.8,
        target_pii_labels: list = ['private_person',  # 이름
            'private_phone',   # 전화번호
            'private_email',   # 이메일
            'account_number',  # 계좌번호, 주민번호 등 고유 식별번호
            'private_url'      # 개인 URL 주소
        ],
        forbidden_words_path: str = "forbidden_words.txt",
    ):
        self.toxic_pipeline = toxic_pipeline
        self.toxic_tokenizer = getattr(self.toxic_pipeline, "tokenizer", None)
        self.untoxic_threshold = untoxic_threshold

        self.safe_model = safe_model
        self.safe_tokenizer = safe_tokenizer
        self.safe_threshold = safe_threshold

        self.pii_pipeline = pii_pipeline
        self.pii_tokenizer = getattr(self.pii_pipeline, "tokenizer", None)
        self.pii_threshold = pii_threshold
        self.target_pii_labels = target_pii_labels

        self.forbidden_words_path = forbidden_words_path
        self.forbidden_pattern: Optional[re.Pattern] = None
        self._load_forbidden_words()

        special_tokens_ids = list(self.safe_tokenizer.added_tokens_decoder.keys())[-10:]
        self.category_ids = [
            [special_tokens_ids[i], special_tokens_ids[i+1]] for i in range(0, len(special_tokens_ids), 2)
        ]
        self.category_names = ["Crime", "Manipulation", "Privacy", "Sexual", "Violence"]


    def _load_forbidden_words(self):
        forbidden_words = []
        if os.path.exists(self.forbidden_words_path):
            with open(self.forbidden_words_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        forbidden_words.append(word)
        if forbidden_words:
            sorted_words = sorted(forbidden_words, key=len, reverse=True)
            self.forbidden_pattern = re.compile('|'.join(map(re.escape, sorted_words)), re.IGNORECASE)

    def _filter_forbidden_words(self, text: str) -> str:
        if self.forbidden_pattern:
            return self.forbidden_pattern.sub(lambda m: '*' * len(m.group()), text)
        return text

    def _check_safety(self, text:str) -> tuple[bool, str]:
        if not text.strip():
            return True, text, "EMPTY_TEXT"

        # ==============================================================
        # 1단계: TOXIC 검사 (한국어 악플/혐오 특화)
        # ==============================================================
        toxic_results = self.toxic_pipeline(text)[0]
        for res in toxic_results:
            # 'clean' 라벨이 아니고, 기준치 초과시 즉시 차단
                if res['label'] != 'clean' and res['score'] > self.untoxic_threshold:
                    return False, text, f"TOXIC 차단 [{res['label']}] (위험도: {res['score']:.2f})"

        # ==============================================================
        # 2단계: UNSAFE 검사 (문맥 기반 위험 정보)
        # ==============================================================
        inputs = self.safe_tokenizer(text, return_tensors="pt").to(self.safe_model.device)

        with torch.no_grad(): # 기울기 계산 비활성화 (메모리 절약 및 속도 향상)
            outputs = self.safe_model(**inputs)

        # 2. 가장 마지막 토큰의 Logit(확률값 생성 전 단계) 추출
        next_token_logits = outputs.logits[0, -1, :]

        # 3. 5가지 카테고리별로 Safe vs Unsafe 확률 비교
        for i, (safe_id, unsafe_id) in enumerate(self.category_ids):
            # 해당 카테고리의 Safe vs Unsafe 토큰 두 개만의 로짓을 가져옴
            pair_logits = next_token_logits[[safe_id, unsafe_id]]

            # Softmax로 두 값의 합이 1이 되도록 확률 변환
            probs = torch.nn.functional.softmax(pair_logits, dim=-1)
            unsafe_prob = probs[1].item() # Unsafe 토큰의 확률

            # Unsafe 확률이 기준치(threshold) 이상이면 즉시 차단
            if unsafe_prob >= self.safe_threshold:
                return False, text, f"UNSAFE 차단 [{self.category_names[i]}] (위험도: {unsafe_prob:.2f})"
        # ==============================================================
        # 3단계: PII 탐지 (FrameByFrame/korean-pii-e5-base)
        # ==============================================================
        # 참고: pipeline 초기화 시 aggregation_strategy="simple" 옵션을 적용했다고 가정한 구조입니다.
        pii_results = self.pii_pipeline(text)

        for res in pii_results:
            # 설정한 임계값 초과, 차단 대상 라벨에 포함되는 경우 즉시 차단
            if res['entity_group'] in self.target_pii_labels and res['score'] > self.pii_threshold:
                return False, text, f"PII 차단 [{res['entity_group']}] (확률: {res['score']:.2f})"

        return True, text, "SAFE"

    def process_text(self, text: str) -> Tuple[bool, str, str]:
        if not isinstance(text, str) or not text.strip():
            return True, text, "EMPTY_OR_NON_STRING"

        # 1차: 금칙어 포함 여부 확인 후 즉시 차단
        if self.forbidden_pattern and self.forbidden_pattern.search(text):
            matched = self.forbidden_pattern.search(text).group()
            return False, text, f"금칙어 차단 [{matched}]"

        clean_text = text.strip()

        # 2차: 주입받은 DeBERTa 파이프라인으로 딥러닝 문맥 검열 (슬라이딩 윈도우 적용)
        try:
            # 💡 DeBERTa 최대 토큰(512)을 넘지 않도록 보수적으로 400자 윈도우 설정
            # 문맥 단절 방지를 위해 100자씩 겹치도록(Overlap) 청크 분할
            window_size = 400
            overlap = 100
            stride = window_size - overlap

            # 토크나이저가 서로 생긴게 다르므로 그냥 글자 수 기준으로 작업
            chunks = [clean_text[i:i+window_size] for i in range(0, max(1, len(clean_text)), stride)]

            for chunk in chunks:
                if not chunk.strip():
                    continue

                is_safe, _, reason = self._check_safety(chunk)
                if not is_safe:
                    return False, clean_text, reason
            return True, clean_text, "SAFE"

        except Exception as e:
            print(f"⚠️ OutputGuardrail DeBERTa 추론 에러: {e}")
            return False, text, 'OutputGuardrail DeBERTa 추론 에러'

# =====================================================================
# 2. 통합 출력 후처리 계층 (OutputPostprocessorAsync)
# =====================================================================
class OutputPostprocessorAsync:
    def __init__(self, guardrail: OutputGuardrail):
        self.guardrail = guardrail

    async def process_output(self, intent: str, raw_result: Union[str, Tuple[pl.DataFrame, pl.DataFrame]]) -> Dict[str, Any]:
        # 💡 to_thread를 통해 DeBERTa의 무거운 추론이 비동기 루프를 멈추지 않게 방어
        if intent == "REALTIME_FGI":
            return await asyncio.to_thread(self._process_fgi_result, raw_result)
        return {"status": "ERROR", "message": f"지원하지 않는 Intent 입니다: {intent}"}

    def _process_fgi_result(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text: return {"status": "ERROR", "message": "생성된 답변이 없습니다."}

        is_safe, safe_text, reason = self.guardrail.process_text(raw_text)

        if not is_safe:
            return {"status": "BLOCKED", "data": {"response": safe_text}, "reason": reason}
        return {"status": "SUCCESS", "data": {"response": safe_text}, "reason": reason}
