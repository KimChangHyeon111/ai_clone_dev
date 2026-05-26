# 필요 라이브러리: !pip install transformers torch redis sentence-transformers
import os
import uuid
import numpy as np
import torch
import hashlib
import asyncio
from typing import Tuple, List, Dict, Any

from transformers import pipeline
from sentence_transformers import SentenceTransformer

import redis
import redis.asyncio as aioredis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.query import Query

# Redis 버전 호환성 처리
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# =====================================================================
# 1. 하위 모듈 정의
# =====================================================================

class InputGuardrail:
    """[0-1] SLM 기반 프롬프트 인젝션 방어 (외부 모델 주입 방식)"""
    def __init__(self, classifier_pipeline):
        self.classifier = classifier_pipeline

    def is_safe(self, text: str) -> Tuple[bool, str]:
        # ⚠️ 동기 연산이므로 메인 비동기 루프에서 직접 호출 시 병목 발생 (to_thread로 우회 필요)
        result = self.classifier(text)[0]
        if result['label'] == 'INJECTION' and result['score'] > 0.8:
            return False, "PROMPT_INJECTION_DETECTED"
        return True, "SAFE"

class SemanticCacheRedisAsync:
    """[0-4] Redis 기반 비동기 시맨틱 캐시 (장애격리 완비)"""
    def __init__(self, embedder: SentenceTransformer, redis_host='localhost', redis_port=6379, threshold=0.8):
        self.threshold = threshold
        self.embedder = embedder
        self.dimension = self.embedder.get_embedding_dimension()

        self.redis_client = aioredis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=False, 
            socket_timeout=2.0
        )
        self.is_alive = True
        self.index_name = "fgi_cache_idx"

    async def initialize(self):
        try:
            await self._setup_index()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            print("⚠️ [경고] Redis 서버에 연결할 수 없습니다. 캐시 기능을 비활성화하고 LLM으로 우회합니다.")
            self.is_alive = False

    async def _setup_index(self):
        try:
            await self.redis_client.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            schema = (
                TagField("persona_id"), 
                TextField("query"), 
                VectorField("vector", "HNSW", {
                    "TYPE": "FLOAT32", "DIM": self.dimension, 
                    "DISTANCE_METRIC": "IP", "M": 16, "EF_CONSTRUCTION": 200
                })
            )
            definition = IndexDefinition(prefix=["idx:"], index_type=IndexType.HASH)
            await self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)

    async def _check_redis_policy(self):
        if not self.is_alive: return
        try:
            raw_policy = await self.redis_client.config_get('maxmemory-policy')
            maxmemory_policy = raw_policy.get(b'maxmemory-policy', b'').decode('utf-8')
            if maxmemory_policy == 'noeviction':
                print("⚠️ 경고: 현재 Redis 'maxmemory-policy'가 'noeviction'입니다.")
        except redis.exceptions.ResponseError:
            pass
    
    def _get_exact_key(self, persona_id: str, query: str) -> str:
        query_clean = query.strip()
        query_hash = hashlib.sha256(query_clean.encode('utf-8')).hexdigest()
        return f"exact_cache:{persona_id}:{query_hash}"

    async def check_cache(self, persona_id: str, query: str) -> Tuple[str, float]:
        if not getattr(self, 'is_alive', True): return None, 0.0 
        
        try:
            exact_key = self._get_exact_key(persona_id, query)
            exact_cached = await self.redis_client.get(exact_key)
            
            if exact_cached:
                return exact_cached.decode('utf-8'), 1.0 
            
            # 임베딩 생성 (CPU/GPU 연산이 무거울 경우 이 부분도 to_thread 고려 가능)
            query_vector = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32).tobytes()
            dist_threshold = 1.0 - self.threshold

            q = (Query(f"(@persona_id:{{{persona_id}}})=>[KNN 1 @vector $vec AS distance]")
                .return_fields("distance").sort_by("distance").dialect(2))
            
            results = await self.redis_client.ft(self.index_name).search(q, query_params={"vec": query_vector})
            similarity = 0.0

            if results.docs:
                doc = results.docs[0]
                distance = float(doc.distance)
                similarity = 1.0 - distance
                
                if distance <= dist_threshold:
                    idx_key = doc.id 
                    data_key = idx_key.replace("idx:", "data:")
                    
                    response_val = await self.redis_client.get(data_key)
                    if response_val:
                        return response_val.decode('utf-8'), similarity
                    
            return None, similarity
        
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            self.is_alive = False 
            return None, 0.0 

    async def add_to_cache(self, persona_id: str, query: str, response: str, ttl_seconds: int = 604800):
        if not getattr(self, 'is_alive', True): return
        try:
            exact_key = self._get_exact_key(persona_id, query)
            query_vector = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32).tobytes()
            unique_id = str(uuid.uuid4())
            idx_key = f"idx:{unique_id}"
            data_key = f"data:{unique_id}"
            
            pipe = self.redis_client.pipeline()
            pipe.setex(exact_key, ttl_seconds, response)
            pipe.setex(data_key, ttl_seconds, response)
            pipe.hset(idx_key, mapping={"persona_id": persona_id, "vector": query_vector})
            pipe.expire(idx_key, ttl_seconds)
            
            await pipe.execute()
            
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            self.is_alive = False

class ExecutionRouter:
    """[0-2] 의도 기반 작업 파이프라인 조립기"""
    def __init__(self):
        self.pipelines = {
            "MASS_SIMULATION": ["DataLoader.load_mass_data", "ContextManager.build_mass_context", "SimulationEngine.run_batch", "BatchResultValidator.validate"],
            "REALTIME_FGI": ["DataLoader.load_fgi_profile", "PersonaAgent.chat"],
            "EXTRACT_CAMPAIGN": ["DataLoader.load_text", "SimulationEngine.extract"]
        }

    def get_pipeline(self, intent: str) -> List[str]:
        if intent not in self.pipelines:
            raise ValueError(f"지원하지 않는 Intent 입니다: {intent}")
        return self.pipelines[intent]

# =====================================================================
# 2. 통합 모듈 (InputPreprocessorAsync)
# =====================================================================

class InputPreprocessorAsync:
    """[0] 입력 및 사전 처리 계층 (비동기 특화)"""

    def __init__(self, guardrail_classifier, embedder: SentenceTransformer, redis_host='localhost'):
        print("초기화 중: InputGuardrail...")
        self.guardrail = InputGuardrail(classifier_pipeline=guardrail_classifier)

        print("초기화 중: SemanticCacheRedisAsync...")
        self.cache = SemanticCacheRedisAsync(embedder=embedder, redis_host=redis_host, threshold=0.8)

        print("초기화 중: ExecutionRouter...")
        self.router = ExecutionRouter()

        print("✅ InputPreprocessorAsync 준비 완료!")
    
    async def initialize(self):
        await self.cache.initialize()

    async def process_request(self, user_id: str, intent: str, query: str, **kwargs) -> Dict[str, Any]:
        # 💡 [핵심 개선] 동기식 딥러닝 추론 함수가 이벤트 루프를 막지 않도록 별도 스레드로 분리 실행
        is_safe, reason = await asyncio.to_thread(self.guardrail.is_safe, query)
        if not is_safe:
            return {"status": "BLOCKED", "reason": reason, "message": "보안 정책 차단"}

        top_similarity = 0.0

        if intent == "REALTIME_FGI":
            cached_response, similarity = await self.cache.check_cache(user_id, query)
            top_similarity = similarity
            
            if cached_response:
                return {"status": "CACHE_HIT", "response": cached_response, "similarity_score": top_similarity}

        try:
            pipeline_steps = self.router.get_pipeline(intent)
        except ValueError as e:
            return {"status": "ERROR", "reason": "INVALID_INTENT", "message": str(e)}

        return {"status": "PROCEED", "intent": intent, "query": query, "pipeline": pipeline_steps, "top_similarity": top_similarity}    

    async def save_to_cache(self, user_id: str, intent: str, query: str, response: str) -> bool:
        """[Write] LLM이 생성한 유효한 답변을 캐시에 저장합니다."""
        if intent == "REALTIME_FGI" and response:
            await self.cache.add_to_cache(user_id, query, response)
            return True
        return False
