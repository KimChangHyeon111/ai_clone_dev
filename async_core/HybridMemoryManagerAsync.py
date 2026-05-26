import json
from typing import List, Optional, Any
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

class Turn(BaseModel):
    speaker: str
    text: str
    target: str = "All"

class HybridMemoryManagerAsync: # 비동기 버전 클래스
    def __init__(
        self,
        max_tokens: int = 2000,
        max_context_items: int = 5,
        project_id: str = None,
        location: str = "global"
    ):
        self.max_tokens = max_tokens
        self.max_context_items = max_context_items

        # --- 대화(Turn) 메모리 ---
        self.past_summary: str = ""
        self.recent_turns: List[Turn] = []
        self._current_token_count: int = 0
        self._summary_token_count: int = 0

        # --- 동적 컨텍스트(ML/PIXEL) 메모리 ---
        self.past_ml_summary: str = ""
        self.recent_ml: List[Any] = []

        self.past_pixel_summary: str = ""
        self.recent_pixel: List[Any] = []

        if project_id:
            vertexai.init(project=project_id, location=location)
        else:
            vertexai.init(location=location)

        self.gemini_model = GenerativeModel("gemini-2.5-flash-lite")

    # 1. 토큰 카운트 비동기화
    async def _count_tokens(self, text: str) -> int:
        if not text: return 0
        response = await self.gemini_model.count_tokens_async(text)
        return response.total_tokens

    # 2. 순수 대화 기록 포맷팅은 CPU 연산이므로 동기 유지
    def get_formatted_history(self, current_name: str = None) -> str:
        history_str = ""
        if self.past_summary:
            history_str += f"<PAST_SUMMARY>\n{self.past_summary}\n</PAST_SUMMARY>\n"
        if self.recent_turns:
            history_str += "<RECENT_TURNS_CONTEXT>\n"
            for turn in self.recent_turns:
                if current_name:
                    if turn.speaker == "Moderator":
                        history_str += f"Moderator: {turn.text}\n"
                    elif turn.speaker == current_name:
                        history_str += f"Your Past Answer ({current_name}): {turn.text}\n"
                    else:
                        history_str += f"Other Participant ({turn.speaker}): {turn.text}\n"
                else:
                    history_str += f"{turn.speaker}: {turn.text}\n"
            history_str += "</RECENT_TURNS_CONTEXT>"
        return history_str.strip()

    # 3. 대화 상호작용 추가 비동기화
    async def add_interaction(self, speaker: str, text: str, target: str = "All"):
        self.recent_turns.append(Turn(speaker=speaker, text=text, target=target))
        # await 추가
        token_added = await self._count_tokens(f"{speaker}: {text}\n")
        self._current_token_count += token_added

        if (self._summary_token_count + self._current_token_count) > self.max_tokens:
            await self._manage_memory_window()

    # 4. 메모리 윈도우 관리 비동기화
    async def _manage_memory_window(self):
        if len(self.recent_turns) <= 2: return
        chunk_size = max(1, len(self.recent_turns) // 2)
        turns_to_summarize = self.recent_turns[:chunk_size]

        # await 추가
        success = await self._update_summary_batch(turns_to_summarize)
        if success:
            self.recent_turns = self.recent_turns[chunk_size:]
            self._summary_token_count = await self._count_tokens(self.past_summary) if self.past_summary else 0
            self._current_token_count = await self._count_tokens("".join([f"{t.speaker}: {t.text}\n" for t in self.recent_turns]))

    # 5. 대화 배치 요약 비동기화 (generate_content -> generate_content_async)
    async def _update_summary_batch(self, turns: List[Turn]) -> bool:
        new_dialogue = "\n".join([f"{t.speaker}: {t.text}" for t in turns])
        prompt = f"""다음은 대화 내용입니다. 기존 요약본과 새로운 대화를 바탕으로 핵심을 요약해주세요.
[기존 요약]\n{self.past_summary if self.past_summary else '없음'}
[추가 대화]\n{new_dialogue}"""
        try:
            # generate_content_async 호출 및 await 적용
            response = await self.gemini_model.generate_content_async(prompt)
            self.past_summary = response.text.strip()
            return True
        except Exception as e:
            print(f"⚠️ 대화 요약 API 호출 실패 (데이터 유지): {e}")
            return False

    # 6. 동적 컨텍스트 누적 및 요약 트리거 비동기화
    async def add_dynamic_context(self, dynamic_ml: list, dynamic_pixel: list):
        for m in dynamic_ml:
            if m not in self.recent_ml:
                self.recent_ml.append(m)

        for p in dynamic_pixel:
            if p not in self.recent_pixel:
                self.recent_pixel.append(p)

        if len(self.recent_ml) >= self.max_context_items:
            await self._summarize_context("ML")
        if len(self.recent_pixel) >= self.max_context_items:
            await self._summarize_context("PIXEL")

    # 7. 유관 정보(ML/PIXEL) 요약 비동기화
    async def _summarize_context(self, target: str):
        if target == "ML":
            prompt = f"다음 구매 이력을 한 문단으로 요약하세요.\n[기존 요약]: {self.past_ml_summary}\n[추가 이력]: {json.dumps(self.recent_ml, ensure_ascii=False)}"
            try:
                response = await self.gemini_model.generate_content_async(prompt)
                self.past_ml_summary = response.text.strip()
                self.recent_ml = []
            except Exception as e:
                print(f"⚠️ ML 정보 요약 API 호출 실패 (기억 유지): {e}")

        elif target == "PIXEL":
            prompt = f"다음 성향(PIXEL) 데이터를 한 문단으로 요약하세요.\n[기존 요약]: {self.past_pixel_summary}\n[추가 특성]: {json.dumps(self.recent_pixel, ensure_ascii=False)}"
            try:
                response = await self.gemini_model.generate_content_async(prompt)
                self.past_pixel_summary = response.text.strip()
                self.recent_pixel = []
            except Exception as e:
                print(f"⚠️ PIXEL 정보 요약 API 호출 실패 (기억 유지): {e}")

    # 병합 로직은 단순 메모리 연산이므로 동기 유지
    def get_combined_ml(self, base_ml: list) -> list:
        result = base_ml.copy() if isinstance(base_ml, list) else []
        if self.past_ml_summary:
            result.append({"[누적 과거 이력 요약]": self.past_ml_summary})
        result.extend(self.recent_ml)
        return result

    def get_combined_pixel(self, base_pixel: list) -> list:
        result = base_pixel.copy() if isinstance(base_pixel, list) else []
        if self.past_pixel_summary:
            result.append(f"[누적 과거 특성 요약] {self.past_pixel_summary}")
        result.extend(self.recent_pixel)
        return result
