import re
import json
from typing import List, Tuple
from google.genai import types
import numpy as np 

# 💡 깃허브 원본 클래스 임포트
from SimulationEngine import PersonaAgent
from ContextManager import FGIDataSchema

class _SharedMemoryRouter:
    """메인 실행 루프에서 .shared_memory.add_interaction 호출 시 호환성을 맞추기 위한 프록시"""
    def __init__(self, orchestrator):
        self.orch = orchestrator
    def add_interaction(self, speaker: str, text: str, target: str = "All"):
        self.orch.broadcast(speaker=speaker, text=text, target=target)

class MultiAgentOrchestrator:
    def __init__(
        self,
        embedder_model,
        srg_keys: List[str],
        fgi_profiles: List[dict],
        sys_tmpl_path: str,
        user_tmpl_path: str,
        free_talk_sys_tmpl_path: str,
        free_talk_user_tmpl_path: str,
        project_id: str,
        bucket_name: str, # 💡 에러 픽스: 필수 파라미터 추가
        context_manager = None
    ):
        self.names = []
        self.agents = {}
        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.free_talk_sys_tmpl_path = free_talk_sys_tmpl_path
        self.free_talk_user_tmpl_path = free_talk_user_tmpl_path
        self.ctx = context_manager

        for i, srg_key in enumerate(srg_keys):
            name = f"Customer_{chr(65+i)}"
            self.names.append(name)

            # 💡 에러 픽스: 깃허브 PersonaAgent의 파라미터 규격 완벽 매핑
            self.agents[name] = PersonaAgent(
                project_id=project_id,
                bucket_name=bucket_name,
                srg_key=srg_key,
                fgi_profile=fgi_profiles[i],
                sys_tmpl_path=sys_tmpl_path,
                user_tmpl_path=user_tmpl_path,
                context_manager=self.ctx
            )

        self.chat_mode = "SEQUENTIAL"
        self.targeted_agent = None
        self.last_agent = self.names[-1] if self.names else None
        self.last_moderator_msg = ""
        self.shared_memory = _SharedMemoryRouter(self)
        self.embedder_model = embedder_model

    def broadcast(self, speaker: str, text: str, target: str = "All", exclude: str = None):
        """방 안의 모든 에이전트 메모리에 대화를 전파합니다."""
        for agent_name, agent in self.agents.items():
            if agent_name != exclude:
                # 💡 원본 메모리 클래스와 업데이트된 메모리 클래스를 모두 호환하는 방어 코드
                try:
                    agent.memory.add_interaction(speaker=speaker, text=text, target=target)
                except TypeError:
                    agent.memory.add_interaction(interviewer_input=speaker, customer_response=text)

    def _get_next_agent(self) -> str:
        if not self.names: return ""
        idx = self.names.index(self.last_agent)
        return self.names[(idx + 1) % len(self.names)]

    def _retrieve_dynamic_context(self, query_vector: np.ndarray, embeddings: np.ndarray, raw_data: list, threshold: float = 0.7) -> list:
        """
        임계치(threshold) 이상의 데이터를 모두 가져오되,
        만약 조건을 만족하는 데이터가 하나도 없더라도 가장 유사한 Top 1은 무조건 반환합니다.
        """
        if not raw_data or embeddings is None or len(raw_data) == 0:
            return []

        # 1. 코사인 유사도 계산 (벡터가 정규화된 상태라고 가정)
        similarities = np.dot(embeddings, query_vector.T).flatten()

        # 2. 유사도 내림차순 정렬 인덱스
        sorted_indices = similarities.argsort()[::-1]

        # 3. 임계치(threshold) 이상인 인덱스만 필터링
        valid_indices = [idx for idx in sorted_indices if similarities[idx] >= threshold]

        # 4. 💡 [핵심] 만약 임계치를 넘은 게 하나도 없거나, Top 1이 누락되었다면 강제 추가
        top_1_idx = sorted_indices[0]
        if top_1_idx not in valid_indices:
            valid_indices.insert(0, top_1_idx) # 맨 앞에 1등 꽂아주기

        # 결과 리스트 반환 (점수가 높은 순으로)
        return [raw_data[i] for i in valid_indices]

    def process_turn(self, user_input: str, promo_info: str, debug = True) -> Tuple[str, str, str]:
        target = None
        display_msg = ""
        input_upper = user_input.upper()

        # =====================================================================
        # 1. 3-Way 라우팅 로직
        # =====================================================================
        target_candidate = f"Customer_{input_upper}"

        if target_candidate in self.names and len(user_input.strip()) == 1:
            self.chat_mode = "TARGETED"
            self.targeted_agent = target_candidate
            target = target_candidate

        elif user_input.strip() != '':
            if '@FREE_TALK' in input_upper:
                self.chat_mode = "FREE_TALK"
                self.targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@FREE_TALK', '[자유 토론]', user_input, flags=re.IGNORECASE).strip()
            elif '@ALL' in input_upper or '@모두' in input_upper:
                self.chat_mode = "SEQUENTIAL"
                self.targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@ALL|@모두', '', user_input, flags=re.IGNORECASE).strip()
            else:
                mentioned = False
                for name in self.names:
                    char = name.split('_')[1]
                    if re.search(fr'@{char}(?![a-zA-Z])', user_input, re.IGNORECASE):
                        self.chat_mode = "TARGETED"
                        self.targeted_agent = name
                        target = name
                        display_msg = re.sub(fr'@{char}(?![a-zA-Z])', f'@{name}', user_input, flags=re.IGNORECASE).strip()
                        mentioned = True
                        break
                if not mentioned:
                    if self.chat_mode == "TARGETED" and self.targeted_agent:
                        target = self.targeted_agent
                        display_msg = user_input
                    else:
                        target = self._get_next_agent()
                        display_msg = user_input
        else:
            if self.chat_mode == "TARGETED" and self.targeted_agent:
                target = self.targeted_agent
            else:
                target = self._get_next_agent()

        if display_msg:
            self.last_moderator_msg = display_msg

        # =====================================================================
        # 2. 동적 컨텍스트 세팅 및 질문 브로드캐스팅
        # =====================================================================
        current_name = target
        active_agent = self.agents[current_name] # 💡 타겟이 된 PersonaAgent 인스턴스
        is_free_talk = (self.chat_mode == "FREE_TALK")

        active_sys_path = self.free_talk_sys_tmpl_path if is_free_talk else self.sys_tmpl_path
        active_user_path = self.free_talk_user_tmpl_path if is_free_talk else self.user_tmpl_path

        # 💡 [강화 1] 동조 금지 및 구조 복제 원천 차단 태그
        anti_parroting_tag = "\n[SYSTEM RULE (CRITICAL): 다른 참여자의 발언 구조, 단어, 표현을 절대 복사하거나 동조하지 마세요. 의견이 비슷하다면 짧게 동의하고, 같은 단어를 반복하지 마세요.]"

        if display_msg:
            current_context_input = display_msg + anti_parroting_tag
            self.broadcast(speaker="Moderator", text=display_msg, exclude=current_name)
        else:
            if is_free_talk:
                current_context_input = "[자유 토론 진행 중. 앞 사람의 의견에 동의/반박/질문 하세요.]" + anti_parroting_tag
            else:
                # 💡 [강화 2] 묻지 않은 인사 및 자기소개 강제 생략 앵커링
                virtual_prompt = f"{current_name}님, 앞서 드린 질문('{self.last_moderator_msg}')에 대해 어떻게 생각하시나요? (요청하지 않은 인사, 자기소개 생략, 본론만 즉답 요망)"

                current_context_input = virtual_prompt + anti_parroting_tag
                self.broadcast(speaker="Moderator", text=virtual_prompt, exclude=current_name)
        # =====================================================================
        # 3. 모델 발화 및 결과 적재
        # =====================================================================
        # 대화 턴에서 기억 인출
        query_vector = self.embedder_model.encode([current_context_input], normalize_embeddings=True).astype(np.float32)

        # 2. 💡 방금 만든 검색 함수로 ML과 PIXEL 각각 추출 (임계치 0.7 설정)
        dynamic_ml = self._retrieve_dynamic_context(
            query_vector=query_vector,
            embeddings=active_agent.ml_embeddings,
            raw_data=active_agent.full_ml_history,
            threshold=0.7
        )

        dynamic_pixel = self._retrieve_dynamic_context(
            query_vector=query_vector,
            embeddings=active_agent.pixel_embeddings,
            raw_data=active_agent.full_pixel,
            threshold=0.7
        )


        try:
            history_str = active_agent.memory.get_formatted_history(current_name=current_name)
        except TypeError:
            history_str = active_agent.memory.get_formatted_history()

        # 3. 💡 추출된 동적 기억을 프롬프트 데이터로 전달
        valid_data = FGIDataSchema(
            ms=active_agent.profile["ms_core_mindset"],
            ml=dynamic_ml,             # 검색된 과거 구매 이력 주입!
            pixel=dynamic_pixel,       # 검색된 성향/관심사 주입!
            promo_info=promo_info,
            conversation_history=history_str,
            user_input=current_context_input
        )
        # 💡 PersonaAgent 내부의 ctx와 client를 그대로 활용하여 LLM 렌더링
        sys_inst, user_content = self.ctx.build_prompt(sys_path=active_sys_path, user_path=active_user_path, data=valid_data)

        # =====================================================================
        # 💡 [여기에 추가] 프롬프트 전체 출력 (디버깅용)
        # =====================================================================
        if DEBUG:
            print("\n" + "="*40 + f" [DEBUG: {current_name}에게 주입된 프롬프트] " + "="*40)
            print("[SYSTEM INSTRUCTION]")
            print(sys_inst)
            print("-" * 100)
            print("[USER CONTENT]")
            print(user_content)
            print("="*105 + "\n")
        # =====================================================================

        # =====================================================================
        response = active_agent.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.5,
                response_mime_type="application/json"
            )
        )

        try:
            res_data = json.loads(response.text)
            response_text = res_data.get("response", response.text).strip()
        except json.JSONDecodeError:
            response_text = response.text.strip()

        # 타겟 본인 메모리에 기록
        try:
            active_agent.memory.add_interaction(speaker=current_name, text=response_text, target="Group" if is_free_talk else "All")
        except TypeError:
            active_agent.memory.add_interaction(interviewer_input=current_name, customer_response=response_text)

        # 방 안의 다른 사람들에게 답변 내용 전파
        self.broadcast(speaker=current_name, text=response_text, exclude=current_name)

        self.last_agent = current_name
        return current_name, display_msg, response_text
