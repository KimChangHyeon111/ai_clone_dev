import io
import json
import uuid
import polars as pl
import numpy as np
import os
import json
from typing import List, Tuple

# GCP SDK

import vertexai
from google import genai
from google.genai import types
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, GenerationConfig

from DataLoader import DataLoader
from ContextManager import ContextManager, FGIDataSchema, SimulationDataSchema
from HybridMemoryManager import HybridMemoryManager


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
        project_id: str,
        bucket_name: str,
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
        
        # =====================================================================
        # 💡 [추가] 에이전트 장기 기억 장치 (Vector DB) 탑재
        # =====================================================================
        self.full_ml_history = fgi_profile.get("full_ml", [])
        self.ml_embeddings = fgi_profile.get("ml_embeddings", np.array([]))
        
        self.full_pixel = fgi_profile.get("full_pixel", [])
        self.pixel_embeddings = fgi_profile.get("pixel_embeddings", np.array([]))

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

class MultiPersonaAgent:
    def __init__(
        self,
        srg_keys: List[str],
        fgi_profiles: List[dict],
        sys_tmpl_path: str,
        user_tmpl_path: str,
        project_id: str,
        location: str = "global",
        context_manager = None
    ):
        self.srg_keys = srg_keys
        self.fgi_profiles = fgi_profiles
        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.ctx = context_manager or ContextManager()
        
        # 모델 클라이언트 초기화
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        
        # 참가자 상태 관리
        self.participants = {}
        self.names = []
        
        for i, srg_key in enumerate(srg_keys):
            name = f"Customer_{chr(65+i)}" # Customer_A, Customer_B...
            self.names.append(name)
            
            profile = fgi_profiles[i]
            self.participants[name] = {
                "srg_key": srg_key,
                "ms_core_mindset": profile.get("ms", ""),
                "ml_transaction_history": profile.get("ml", []),
                "is_valid": bool(profile.get("ms", ""))
            }
            
        # [핵심] FGI 룸에 참석한 전원의 대화를 기록하기 위해 메모리 매니저를 하나만 사용
        self.shared_memory = HybridMemoryManager()
        
        self.last_agent = self.names[-1] if self.names else None
        self.free_talk_mode = False

    def _get_next_agent(self) -> str:
        """다음 차례의 에이전트를 반환합니다."""
        if not self.names: 
            return ""
        idx = self.names.index(self.last_agent)
        next_idx = (idx + 1) % len(self.names)
        return self.names[next_idx]

    def process_turn(self, user_input: str, promo_info: str) -> Tuple[str, str, str]:
        """
        입력된 명령어에 따라 타겟 에이전트를 설정하고, 응답을 반환합니다.
        반환값: (발화하는 에이전트 이름, 화면에 출력할 Moderator 메시지, 생성된 응답)
        """
        target = None
        display_msg = ""
        
        # --- 1. 사용자 입력 라우팅 및 지목 로직 ---
        input_upper = user_input.upper()
        target_candidate = f"Customer_{input_upper}"
        
        # A, B 처럼 단일 알파벳만 입력한 경우 (자유 토론 해제 후 해당 유저 지목)
        if target_candidate in self.names and len(user_input) == 1:
            target = target_candidate
            self.free_talk_mode = False
            
        # 텍스트 내용이 있는 경우
        elif user_input.strip() != '':
            if re.search(r'@free_talk', user_input, re.IGNORECASE):
                self.free_talk_mode = True
                target = self._get_next_agent()
                display_msg = re.sub(r'@free_talk', '[자유 토론]', user_input, flags=re.IGNORECASE)
            else:
                mentioned = False
                # @A, @B 등을 통한 특정인 지목 감지
                for name in self.names:
                    char = name.split('_')[1]
                    if re.search(fr'@{char}', user_input, re.IGNORECASE):
                        self.free_talk_mode = False
                        target = name
                        display_msg = re.sub(fr'@{char}', f'@{name}님', user_input, flags=re.IGNORECASE)
                        mentioned = True
                        break
                        
                # 지목 태그 없이 질문만 들어온 경우
                if not mentioned:
                    self.free_talk_mode = False
                    target = self._get_next_agent()
                    display_msg = user_input
            
            # Moderator(사회자) 발화를 공유 메모리에 기록
            self.shared_memory.add_interaction("Moderator", display_msg)
            
        # 엔터만 친 경우 (기존 모드 유지하며 교대)
        else:
            target = self._get_next_agent()

        # --- 2. 타겟 에이전트 답변 생성 로직 ---
        current_name = target
        other_names = [n for n in self.names if n != current_name]
        other_name_str = ", ".join(other_names)

        # 시스템 프롬프트(지침) 동적 구성
        if self.free_talk_mode:
            direction = (
                f"[행동 지침: 당신의 이름은 {current_name}입니다. 현재는 다른 참여자({other_name_str})와 의견을 나누는 자유 토론입니다. "
                f"진행자(Moderator)의 개입이 없다면, 이전 대화에 있는 다른 사람의 의견에 동의/반박/질문하며 대화를 주도하세요. "
                f"본인 페르소나에 맞춰 반드시 2~3문장 이내 짧은 구어체로 작성하세요.]\n\n"
            )
        else:
            direction = (
                f"[행동 지침: 당신의 이름은 {current_name}입니다. "
                f"상대방의 말꼬리를 잡지 말고, 진행자(Moderator)의 질문에 집중하여 본인의 관점에서 대답하세요. "
                f"본인 페르소나에 맞춰 반드시 2~3문장 이내 짧은 구어체로 작성하세요.]\n\n"
            )

        target_profile = self.participants[current_name]
        if not target_profile["is_valid"]:
            return current_name, display_msg, "[시스템 알림] 대상 페르소나의 데이터가 부족합니다."

        # 전체 공유 히스토리 가져오기
        history_str = self.shared_memory.get_formatted_history()
        
        # 스키마 바인딩 (USER_INPUT 필드에 행동 지침과 현재 상황을 밀어넣어 프롬프트 완성)
        valid_data = FGIDataSchema(
            ms=target_profile["ms_core_mindset"],
            ml=target_profile["ml_transaction_history"],
            promo_info=promo_info,
            conversation_history=history_str,
            user_input=direction + (display_msg if display_msg else "[이전 대화에 이어 계속 발화하세요]")
        )

        sys_inst, user_content = self.ctx.build_prompt(
            sys_path=self.sys_tmpl_path,
            user_path=self.user_tmpl_path,
            data=valid_data
        )

        # 모델 호출
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"{user_content}\n{current_name}: ",
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.6,
                response_mime_type="application/json"
            )
        )
        
        # 결과 파싱 (fgi_user.md 포맷에서 response 키 추출)
        try:
            res_data = json.loads(response.text)
            response_text = res_data.get("response", response.text).strip()
        except json.JSONDecodeError:
            response_text = response.text.strip()
            
        # 에이전트의 답변을 공유 메모리에 기록
        self.shared_memory.add_interaction(current_name, response_text)
        
        self.last_agent = current_name
        return current_name, display_msg, response_text
