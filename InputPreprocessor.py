# 필요 라이브러리: !pip install transformers torch redis sentence-transformers
import os
import uuid
import numpy as np
import redis
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.query import Query
from typing import Tuple, List, Dict, Any

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
    
    # 💡 수정됨: 외부에서 로드한 classifier 파이프라인을 그대로 받습니다.
    def __init__(self, classifier_pipeline):
        self.classifier = classifier_pipeline

    def is_safe(self, text: str) -> Tuple[bool, str]:
        result = self.classifier(text)[0]
        if result['label'] == 'INJECTION' and result['score'] > 0.8:
            return False, "PROMPT_INJECTION_DETECTED"
        return True, "SAFE"

class SemanticCacheRedis:
    """[0-4] Redis 기반 시맨틱 캐시 (외부 모델 주입 방식)"""
    def __init__(self, embedder: SentenceTransformer, redis_host='localhost', redis_port=6379, threshold=0.9):
        self.threshold = threshold
        self.embedder = embedder
        # 버전 호환성을 위해 get_embedding_dimension() 사용
        self.dimension = self.embedder.get_embedding_dimension() 
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.index_name = "fgi_cache_idx"
        self._setup_index()

    def _setup_index(self):
        try:
            self.redis_client.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            schema = (
                TagField("persona_id"), TextField("query"), TextField("response"),
                VectorField("vector", "FLAT", {"TYPE": "FLOAT32", "DIM": self.dimension, "DISTANCE_METRIC": "COSINE"})
            )
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)

    def check_cache(self, persona_id: str, query: str) -> Tuple[str, float]:
        query_vector = self.embedder.encode([query]).astype(np.float32).tobytes()
        dist_threshold = 1.0 - self.threshold
        
        q = (Query(f"(@persona_id:{{{persona_id}}})=>[KNN 1 @vector $vec AS distance]")
             .return_fields("response", "distance").sort_by("distance").dialect(2))
        
        results = self.redis_client.ft(self.index_name).search(q, query_params={"vec": query_vector})
        if results.docs:
            doc = results.docs[0]
            distance = float(doc.distance)
            if distance <= dist_threshold:
                # [수정된 부분] 결과값이 bytes일 경우에만 디코딩 수행
                response_val = doc.response
                if isinstance(response_val, bytes):
                    response_val = response_val.decode('utf-8')
                    
                return response_val, (1.0 - distance)
        return None, 0.0
        
    def add_to_cache(self, persona_id: str, query: str, response: str, ttl_seconds: int = 604800):
        query_vector = self.embedder.encode([query]).astype(np.float32).tobytes()
        doc_id = f"cache:{uuid.uuid4()}"
        mapping = {
            "persona_id": persona_id,
            "query": query,
            "response": response,
            "vector": query_vector
        }
        self.redis_client.hset(doc_id, mapping=mapping)
        self.redis_client.expire(doc_id, ttl_seconds)

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
# 2. 통합 모듈 (InputPreprocessor)
# =====================================================================

class InputPreprocessor:
    """[0] 입력 및 사전 처리 계층"""
    
    # 💡 수정됨: guardrail_classifier와 embedder 두 모델을 모두 외부에서 받습니다.
    def __init__(self, guardrail_classifier, embedder: SentenceTransformer, redis_host='localhost'):
        print("초기화 중: InputGuardrail...")
        self.guardrail = InputGuardrail(classifier_pipeline=guardrail_classifier)
        
        print("초기화 중: SemanticCacheRedis...")
        self.cache = SemanticCacheRedis(embedder=embedder, redis_host=redis_host, threshold=0.9)
        
        print("초기화 중: ExecutionRouter...")
        self.router = ExecutionRouter()
        
        print("✅ InputPreprocessor 준비 완료!")

    def process_request(self, user_id: str, intent: str, query: str, **kwargs) -> Dict[str, Any]:
        """사용자 요청 사전 처리"""
        is_safe, reason = self.guardrail.is_safe(query)
        if not is_safe:
            return {"status": "BLOCKED", "reason": reason, "message": "보안 정책 차단"}

        if intent == "REALTIME_FGI":
            cached_response, similarity = self.cache.check_cache(user_id, query)
            if cached_response:
                return {"status": "CACHE_HIT", "response": cached_response, "similarity_score": similarity}

        try:
            pipeline_steps = self.router.get_pipeline(intent)
        except ValueError as e:
            return {"status": "ERROR", "reason": "INVALID_INTENT", "message": str(e)}

        return {"status": "PROCEED", "intent": intent, "query": query, "pipeline": pipeline_steps}

    # 💡 새로 추가된 메서드: 전처리기를 통한 안전한 캐시 적재 창구
    def save_to_cache(self, user_id: str, intent: str, query: str, response: str) -> bool:
        """[Write] LLM이 생성한 유효한 답변을 캐시에 저장합니다."""
        if intent == "REALTIME_FGI" and response:
            self.cache.add_to_cache(user_id, query, response)
            return True
        return False
