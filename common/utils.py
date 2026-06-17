import re
import time
import inspect
import functools


def clean_json_string(text: str) -> str:
    """LLM 응답에서 Markdown JSON 포맷팅 찌꺼기를 안전하게 제거합니다."""
    return re.sub(r'^```json\n|^```|```$', '', text.strip(), flags=re.MULTILINE).strip()


def debug_node(func):
    """LangGraph 노드 진입/완료/에러를 로깅하는 데코레이터 (동기/비동기 모두 지원)."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(state, *args, **kwargs):
            node_name = func.__name__
            print(f"\n[DEBUG] 🚀 [진입] {node_name} 실행 시작...")
            start_time = time.time()
            try:
                result = await func(state, *args, **kwargs)
                elapsed = time.time() - start_time
                print(f"[DEBUG] ✅ [완료] {node_name} (소요시간: {elapsed:.2f}초)")
                return result
            except Exception as e:
                print(f"[DEBUG] ❌ [에러] {node_name} 실패: {e}")
                raise
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(state, *args, **kwargs):
            node_name = func.__name__
            print(f"\n[DEBUG] 🚀 [진입] {node_name} 실행 시작...")
            start_time = time.time()
            try:
                result = func(state, *args, **kwargs)
                elapsed = time.time() - start_time
                print(f"[DEBUG] ✅ [완료] {node_name} (소요시간: {elapsed:.2f}초)")
                return result
            except Exception as e:
                print(f"[DEBUG] ❌ [에러] {node_name} 실패: {e}")
                raise
        return sync_wrapper
