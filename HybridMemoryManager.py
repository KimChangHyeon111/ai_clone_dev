from typing import List
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

# 대화 턴 관리를 위한 Pydantic 스키마
class Turn(BaseModel):
    interviewer: str
    customer: str

class HybridMemoryManager:
    def __init__(
        self,
        max_tokens: int = 2000,
        project_id: str = None,
        location: str = "global" # 참고: 서울 리전은 asia-northeast3
    ):
        self.max_tokens = max_tokens
        self.past_summary: str = ""
        self.recent_turns: List[Turn] = []

        # 전체 텍스트 재계산 방지를 위한 토큰 상태 추적 변수
        self._current_token_count: int = 0
        self._summary_token_count: int = 0

        if project_id:
            vertexai.init(project=project_id, location=location)
        else:
            vertexai.init(location=location)

        self.gemini_model = GenerativeModel("gemini-2.5-flash-lite")

    def _count_tokens(self, text: str) -> int:
        """단일 텍스트의 토큰 수 계산"""
        if not text:
            return 0
        response = self.gemini_model.count_tokens(text)
        return response.total_tokens

    def get_formatted_history(self) -> str:
        """FGI 프롬프트에 주입할 <CONVERSATION_HISTORY> 포맷 생성"""
        history_str = ""
        if self.past_summary:
            history_str += f"<PAST_SUMMARY>\n{self.past_summary}\n</PAST_SUMMARY>\n"

        if self.recent_turns:
            history_str += "<RECENT_TURNS>\n"
            for turn in self.recent_turns:
                history_str += f"Interviewer: {turn.interviewer}\nCustomer: {turn.customer}\n"
            history_str += "</RECENT_TURNS>"

        return history_str.strip()

    def add_interaction(self, interviewer_input: str, customer_response: str):
        """새로운 대화 턴을 추가하고 상태를 업데이트"""
        new_turn = Turn(interviewer=interviewer_input, customer=customer_response)
        self.recent_turns.append(new_turn)

        # 1. 새 턴에 대한 토큰 수만 별도로 계산하여 누적
        turn_text = f"Interviewer: {new_turn.interviewer}\nCustomer: {new_turn.customer}\n"
        self._current_token_count += self._count_tokens(turn_text)

        # 2. 임계치 초과 여부 확인 (while 루프 제거)
        # past_summary와 recent_turns의 합산 토큰 검사
        total_tokens = self._summary_token_count + self._current_token_count

        if total_tokens > self.max_tokens:
            self._manage_memory_window()

    def _manage_memory_window(self):
        """
        토큰 초과 시, 최근 대화의 절반을 한 번에 요약본으로 압축하여
        API 호출(while 반복)을 최소화하고 문맥을 보존합니다.
        """
        # 요약할 대상이 충분하지 않다면 안전장치로 리턴
        if len(self.recent_turns) <= 1:
            return

        # 청크 단위 분리: 전체 턴의 절반(또는 특정 비율)을 가져옴
        chunk_size = max(1, len(self.recent_turns) // 2)
        turns_to_summarize = self.recent_turns[:chunk_size]

        # recent_turns 업데이트 (남은 대화만 유지)
        self.recent_turns = self.recent_turns[chunk_size:]

        # 기존 요약 + 청크 대화를 병합하여 새로운 요약 생성
        self._update_summary_batch(turns_to_summarize)

        # 요약 및 리스트 갱신 완료 후, 전체 토큰 상태 재동기화 (이때 1회 전체 계산)
        self._resync_token_counts()

    def _update_summary_batch(self, turns: List[Turn]):
        """여러 턴을 한 번에 기존 요약본에 통합 (1회 호출)"""

        # 묶음 텍스트 생성
        new_dialogue_text = "\n".join(
            [f"Interviewer: {t.interviewer}\nCustomer: {t.customer}" for t in turns]
        )

        prompt = f"""
다음은 FGI 인터뷰어와 고객의 과거 대화 요약본과 추가로 진행된 대화 내용입니다.
이를 바탕으로 고객의 성향, 주요 불만, 관심사 등이 누락되지 않게 요약본을 갱신해주세요.
답변은 오직 요약된 텍스트만 출력하세요.

[기존 요약]
{self.past_summary if self.past_summary else "없음"}

[추가된 대화 묶음]
{new_dialogue_text}
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            self.past_summary = response.text.strip()
        except Exception as e:
            print(f"요약 모델 호출 중 에러 발생: {e}")
            # 에러 발생 시 fallback (본 코드에서는 임시 문자열 덧붙임 유지)
            fallback_text = " | [일부 요약 실패로 원문 보존됨]"
            self.past_summary += fallback_text

    def _resync_token_counts(self):
        """요약 완료 후 토큰 카운터 상태를 정확하게 재동기화"""
        self._summary_token_count = self._count_tokens(self.past_summary) if self.past_summary else 0

        # 남아있는 recent_turns의 총 토큰 수 재계산
        recent_text = "".join([f"Interviewer: {t.interviewer}\nCustomer: {t.customer}\n" for t in self.recent_turns])
        self._current_token_count = self._count_tokens(recent_text)
