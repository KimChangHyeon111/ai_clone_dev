import re
import json
import asyncio
import numpy as np
from typing import List, Tuple, Optional
from google.genai import types

# 💡 앞서 완성한 비동기 컴포넌트들을 임포트한다고 가정
from async_core.PersonaAgentAsync import PersonaAgentAsync
from async_core.ContextManagerAsync import ContextManagerAsync
from common.schemas import FGIDataSchema
from common.utils import *

class _SharedMemoryRouterAsync:
    """비동기 환경에 맞춘 공유 메모리 라우터"""
    def __init__(self, orchestrator):
        self.orch = orchestrator
        
    # 💡 async def 로 변경
    async def add_interaction(self, speaker: str, text: str, target: str = "All"):
        await self.orch.broadcast(speaker=speaker, text=text, target=target)


class MultiAgentOrchestratorAsync:
    def __init__(
        self,
        agents,
        srg_keys: List[str],
        fgi_profiles: List[dict],
        sys_tmpl_path: str,
        user_tmpl_path: str,
        free_talk_sys_tmpl_path: str,
        free_talk_user_tmpl_path: str,
        embedder_model,            
        context_manager=None
    ):
        self.names = []
        self.agents = agents
        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.free_talk_sys_tmpl_path = free_talk_sys_tmpl_path
        self.free_talk_user_tmpl_path = free_talk_user_tmpl_path
        
        # 💡 비동기 컨텍스트 매니저 주입
        self.ctx = context_manager or ContextManagerAsync()
        self.embedder_model = embedder_model 

        for i, srg_key in enumerate(srg_keys):
            name = f"Customer_{chr(65+i)}"
            self.names.append(name)

            base_pixel = fgi_profiles[i].get("pixel", [])
            if isinstance(base_pixel, str):
                base_pixel = [base_pixel] 
            self.agents[name].profile["pixel"] = base_pixel

        self.chat_mode = "SEQUENTIAL"
        self.targeted_agent = None
        self.last_agent = self.names[-1] if self.names else None
        self.last_moderator_msg = ""
        
        # 💡 비동기 라우터 연결
        self.shared_memory = _SharedMemoryRouterAsync(self)

    # 💡 1. 브로드캐스트 비동기화 및 병렬(Concurrency) 처리
    async def broadcast(self, speaker: str, text: str, target: str = "All", exclude: str = None):
        """방 안의 모든 에이전트 메모리에 대화를 동시에 전파합니다."""
        tasks = []
        for agent_name, agent in self.agents.items():
            if agent_name != exclude:
                # Type 에러 방어(인터페이스 호환)
                try:
                    tasks.append(agent.memory.add_interaction(speaker=speaker, text=text, target=target))
                except TypeError:
                    tasks.append(agent.memory.add_interaction(interviewer_input=speaker, customer_response=text))
        
        # 모든 에이전트의 메모리를 병렬로 한꺼번에 업데이트!
        if tasks:
            await asyncio.gather(*tasks)

    def _get_next_agent(self) -> str:
        if not self.names: return ""
        idx = self.names.index(self.last_agent)
        return self.names[(idx + 1) % len(self.names)]
    
    def _parse_intent_and_target(self, user_input: str) -> Tuple[str, Optional[str], str, str]:
        # (기존 의도 분석 로직 완벽히 동일 - 단순 문자열 처리이므로 동기 유지)
        input_upper = user_input.upper().strip()
        target_candidate = f"Customer_{input_upper}"
        
        chat_mode = self.chat_mode
        targeted_agent = self.targeted_agent
        display_msg = ""
        target = None

        if target_candidate in self.names and len(user_input.strip()) == 1:
            chat_mode = "TARGETED"
            targeted_agent = target_candidate
            target = target_candidate
        elif user_input.strip() != '':
            if '@FREE_TALK' in input_upper:
                chat_mode = "FREE_TALK"
                targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@FREE_TALK', '[자유 토론]', user_input, flags=re.IGNORECASE).strip()
            elif '@ALL' in input_upper or '@모두' in input_upper:
                chat_mode = "SEQUENTIAL"
                targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@ALL|@모두', '', user_input, flags=re.IGNORECASE).strip()
            else:
                mentioned = False
                for name in self.names:
                    char = name.split('_')[1]
                    if re.search(fr'@{char}(?![a-zA-Z])', user_input, re.IGNORECASE):
                        chat_mode = "TARGETED"
                        targeted_agent = name
                        target = name
                        display_msg = re.sub(fr'@{char}(?![a-zA-Z])', f'@{name}', user_input, flags=re.IGNORECASE).strip()
                        mentioned = True
                        break
                if not mentioned:
                    if chat_mode == "TARGETED" and targeted_agent:
                        target = targeted_agent
                        display_msg = user_input
                    else:
                        target = self._get_next_agent()
                        display_msg = user_input
        else:
            if chat_mode == "TARGETED" and targeted_agent:
                target = targeted_agent
            else:
                target = self._get_next_agent()

        return chat_mode, targeted_agent, target, display_msg

    # (Numpy 연산은 매우 빠르므로 동기 유지)
    def _retrieve_dynamic_context(self, query_vector: np.ndarray, embeddings: np.ndarray, raw_data: list, threshold: float = 0.7) -> list:
        if not raw_data or embeddings is None or len(raw_data) == 0: return []
        similarities = np.dot(embeddings, query_vector.T).flatten()
        sorted_indices = similarities.argsort()[::-1]
        valid_indices = [idx for idx in sorted_indices if similarities[idx] >= threshold]

        top_1_idx = sorted_indices[0]
        if top_1_idx not in valid_indices:
            valid_indices.insert(0, top_1_idx) 

        return [raw_data[i] for i in valid_indices]

    # 💡 2. process_turn 비동기 파이프라인으로 개편
    async def process_turn(self, user_input: str, promo_info: str, DEBUG = False, temperature = 0.5) -> Tuple[str, str, str]:
        chat_mode, targeted_agent, target, display_msg = self._parse_intent_and_target(user_input)
        
        self.chat_mode = chat_mode
        self.targeted_agent = targeted_agent
        if display_msg:
            self.last_moderator_msg = display_msg

        current_name = target
        active_agent = self.agents[current_name]
        is_free_talk = (self.chat_mode == "FREE_TALK")

        active_sys_path = self.free_talk_sys_tmpl_path if is_free_talk else self.sys_tmpl_path
        active_user_path = self.free_talk_user_tmpl_path if is_free_talk else self.user_tmpl_path

        # 💡 브로드캐스트 비동기 대기
        if display_msg:
            await self.broadcast(speaker="Moderator", text=display_msg, exclude=current_name)
        elif not is_free_talk:
            virtual_prompt = f"{current_name}님, 앞서 드린 질문('{self.last_moderator_msg}')에 대해 응답해주세요."
            await self.broadcast(speaker="Moderator", text=virtual_prompt, exclude=current_name)

        embed_query = display_msg if display_msg else (self.last_moderator_msg if not is_free_talk else "[자유 토론]")
        
        # 임베딩 생성 워커 쓰레드 오프로딩 (CPU 방어)
        query_vector = await asyncio.to_thread(
            self.embedder_model.encode,
            [embed_query], 
            normalize_embeddings=True
        )
        query_vector = query_vector.astype(np.float32)

        dynamic_ml = self._retrieve_dynamic_context(
            query_vector=query_vector, embeddings=active_agent.ml_embeddings, raw_data=active_agent.full_ml_history, threshold=0.7
        )
        dynamic_pixel = self._retrieve_dynamic_context(
            query_vector=query_vector, embeddings=active_agent.pixel_embeddings, raw_data=active_agent.full_pixel, threshold=0.7
        )

        # 💡 컨텍스트 누적 비동기 대기
        await active_agent.memory.add_dynamic_context(dynamic_ml, dynamic_pixel)

        base_ml = active_agent.profile.get("ml_transaction_history", [])
        base_pixel = active_agent.profile.get("pixel", [])

        combined_ml = active_agent.memory.get_combined_ml(base_ml)
        combined_pixel = active_agent.memory.get_combined_pixel(base_pixel)

        try:
            history_str = active_agent.memory.get_formatted_history(current_name=current_name)
        except TypeError:
            history_str = active_agent.memory.get_formatted_history()

        other_names_str = ", ".join([n for n in self.names if n != current_name])

        valid_data = FGIDataSchema(
            ms=active_agent.profile.get("ms_core_mindset", ""),
            ml=combined_ml,
            pixel=combined_pixel,
            promo_info=promo_info,
            conversation_history=history_str,
            user_input=display_msg,               
            current_name=current_name,            
            other_names=other_names_str,          
            last_moderator_msg=self.last_moderator_msg 
        )

        # 💡 프롬프트 생성 비동기 대기
        sys_inst, user_content = await self.ctx.build_prompt(sys_path=active_sys_path, user_path=active_user_path, data=valid_data)

        if DEBUG:
            print(f"\n[DEBUG: {current_name} 프롬프트 주입 완료]")

        # 💡 3. 네트워크 통신: 클라이언트 API 비동기(.aio) 호출
        response = await active_agent.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=temperature,
                response_mime_type="application/json"
            )
        )

        try:
            clean_txt = clean_json_string(response.text)
            res_data = json.loads(clean_txt)
            response_text = res_data.get("response", response.text).strip()
        except json.JSONDecodeError:
            response_text = response.text.strip()

        # 💡 내 메모리와 다른 참여자 메모리 병렬 업데이트
        try:
            await active_agent.memory.add_interaction(speaker=current_name, text=response_text, target="Group" if is_free_talk else "All")
        except TypeError:
            await active_agent.memory.add_interaction(interviewer_input=current_name, customer_response=response_text)

        await self.broadcast(speaker=current_name, text=response_text, exclude=current_name)

        self.last_agent = current_name
        return current_name, display_msg, response_text
