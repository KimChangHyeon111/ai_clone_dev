import json
import asyncio
import numpy as np

# GCP SDK
from google import genai
from google.genai import types

# 💡 비동기로 개편된 ContextManagerAsync를 가져옵니다.
from async_core.ContextManagerAsync import ContextManagerAsync, FGIDataSchema
from async_core.HybridMemoryManagerAsync import HybridMemoryManagerAsync
from common.utils import clean_json_string

class PersonaAgentAsync:
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        srg_key: str,
        fgi_profile: dict,
        sys_tmpl_path: str,
        user_tmpl_path: str,
        location: str = "global",
        model_name: str = 'gemini-2.5-flash-lite', # 💡 누락되었던 쉼표 추가
        context_manager = None
    ):
        self.srg_key = srg_key

        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path

        # 💡 기본값을 ContextManagerAsync로 변경
        self.ctx = context_manager or ContextManagerAsync()
        self.model_name = model_name

        self.memory = HybridMemoryManagerAsync()

        self.client = genai.Client(vertexai=True, project=project_id, location=location)

        self.profile = {
            "ms_core_mindset": fgi_profile.get("ms", ""),
            "ml_transaction_history": fgi_profile.get("ml", [])
        }
        self.is_valid = bool(self.profile["ms_core_mindset"])

        self.full_ml_history = fgi_profile.get("full_ml", [])
        self.ml_embeddings = fgi_profile.get("ml_embeddings", np.array([]))

        self.full_pixel = fgi_profile.get("full_pixel", [])
        self.pixel_embeddings = fgi_profile.get("pixel_embeddings", np.array([]))

    async def process_interviewer_turn(self, interviewer_input: str, promo_info: str = "프로모션 정보 없음") -> str:
        if not self.is_valid:
            return "[시스템 알림] 데이터가 부족합니다."

        valid_data = FGIDataSchema(
            ms=self.profile["ms_core_mindset"],
            ml=self.profile["ml_transaction_history"],
            promo_info=promo_info,
            conversation_history=self.memory.get_formatted_history(),
            user_input=interviewer_input
        )

        # 💡 [핵심 변경점] asyncio.to_thread 래핑을 벗겨내고 네이티브 비동기 호출
        sys_inst, user_content = await self.ctx.build_prompt(
            sys_path=self.sys_tmpl_path,
            user_path=self.user_tmpl_path,
            data=valid_data
        )

        prompt = f"Interviewer: {user_content}\nCustomer: "

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.6
            )
        )

        try:
            clean_txt = clean_json_string(response.text)
            res_data = json.loads(clean_txt)
            response_text = res_data.get("response", response.text).strip()
        except json.JSONDecodeError:
            response_text = response.text.strip()

        await self.memory.add_interaction(interviewer_input, response_text)

        return response_text
