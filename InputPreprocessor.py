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
import hashlib


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

        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False, socket_timeout=2.0)
        self.is_alive = True
        self.index_name = "fgi_cache_idx"
        self._setup_index()

    def _setup_index(self):
            try:
                self.redis_client.ft(self.index_name).info()
            except redis.exceptions.ResponseError:
                # 💡 response 필드를 인덱스 스키마에서 완전히 제거합니다.
                schema = (
                    TagField("persona_id"), 
                    # query도 벡터 검색에 직접 안 쓰인다면 빼도 되지만, 디버깅을 위해 남겨둠
                    TextField("query"), 
                    VectorField("vector", "HNSW", {
                        "TYPE": "FLOAT32", "DIM": self.dimension, 
                        "DISTANCE_METRIC": "IP", "M": 16, "EF_CONSTRUCTION": 200
                    })
                )
                # 접두어를 idx: 로 변경하여 색인용 데이터만 인덱싱하게 만듭니다.
                definition = IndexDefinition(prefix=["idx:"], index_type=IndexType.HASH)
                self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)

    def _check_redis_policy(self):
        """실제 운영 환경에서 maxmemory-policy가 volatile-* 계열로 되어있는지 체크"""
        try:
            maxmemory_policy = self.redis_client.config_get('maxmemory-policy').get(b'maxmemory-policy', b'').decode('utf-8')
            if maxmemory_policy == 'noeviction':
                print("⚠️ 경고: 현재 Redis 'maxmemory-policy'가 'noeviction'입니다.")
                print("메모리가 가득 차면 쓰기 에러가 발생하므로, 'volatile-lru' 또는 'volatile-lfu'로 변경을 권장합니다.")
        except redis.exceptions.ResponseError:
            # GCP Memorystore 등 일부 클라우드 환경에서는 CONFIG GET 명령어가 차단되어 있을 수 있음
            pass
    
    def _get_exact_key(self, persona_id: str, query: str) -> str:
        query_clean = query.strip()
        query_hash = hashlib.sha256(query_clean.encode('utf-8')).hexdigest()
        return f"exact_cache:{persona_id}:{query_hash}"

    def check_cache(self, persona_id: str, query: str) -> Tuple[str, float]:
        if not getattr(self, 'is_alive', True): return None, 0.0 # 죽어있으면 바로 패스
        
        try:
            # [Step 1] Exact Match (텍스트 완전 일치 검사) - 고속 조회
            exact_key = self._get_exact_key(persona_id, query)
            exact_cached = self.redis_client.get(exact_key)
            
            if exact_cached:
                return exact_cached.decode('utf-8'), 1.0  # 완전 일치하므로 유사도 1.0 보장
            
            # [Step 2] Semantic Match (벡터 유사도 검색) - 캐시 미스 시 수행
            query_vector = self.embedder.encode([query], normalize_embeddings = True).astype(np.float32).tobytes()
            dist_threshold = 1.0 - self.threshold

            # 💡 return_fields에서 response를 빼고, 문서의 ID 자체만 가져옵니다.
            q = (Query(f"(@persona_id:{{{persona_id}}})=>[KNN 1 @vector $vec AS distance]")
                .return_fields("distance").sort_by("distance").dialect(2))
            
            results = self.redis_client.ft(self.index_name).search(q, query_params={"vec": query_vector})
            similarity = 0.0
            if results.docs:
                doc = results.docs[0]
                distance = float(doc.distance)
                similarity = 1.0 - distance
                if distance <= dist_threshold:
                    # doc.id는 "idx:1234-5678..." 형태입니다.
                    idx_key = doc.id 
                    # "idx:"를 "data:"로 바꿔서 실제 답변이 있는 Key를 만듭니다.
                    data_key = idx_key.replace("idx:", "data:")
                    
                    # 실제 긴 답변 텍스트를 별도로 가져옵니다. (O(1) 속도라 매우 빠름)
                    response_val = self.redis_client.get(data_key)
                    if response_val:
                        return response_val.decode('utf-8'), similarity
                    
            return None, similarity
        
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            self.is_alive = False # 단말기 고장 표시
            return None, 0.0 # 에러 내지 말고 LLM으로 우회

    def add_to_cache(self, persona_id: str, query: str, response: str, ttl_seconds: int = 604800):
        if not getattr(self, 'is_alive', True): return
        try:
            # 1. Exact Match용 단순 Key-Value 저장 (String 구조)
            exact_key = self._get_exact_key(persona_id, query)
            self.redis_client.setex(exact_key, ttl_seconds, response)

            # 2. Semantic Match용 Vector 저장 (Hash 구조)
            query_vector = self.embedder.encode([query], normalize_embeddings = True).astype(np.float32).tobytes()
            unique_id = str(uuid.uuid4())
            idx_key = f"idx:{unique_id}"
            data_key = f"data:{unique_id}"
            
            mapping = {
                "persona_id": persona_id,
                "query": query,
                "vector": query_vector
            }
            
            pipe = self.redis_client.pipeline()
            
            # 💡 분리 저장: 
            # 무거운 response는 단순 String으로 저장 (인덱스 엔진이 무시함)
            pipe.setex(data_key, ttl_seconds, response)
            
            # 가벼운 메타데이터와 벡터만 Hash로 저장 (인덱스 엔진이 추적함)
            pipe.hset(idx_key, mapping=mapping)
            pipe.expire(idx_key, ttl_seconds)
            
            pipe.execute()

        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            self.is_alive = False # 저장 실패해도 티 내지 말고 넘어가기

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
