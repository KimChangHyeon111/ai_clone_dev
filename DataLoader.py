import polars as pl
import numpy as np
from typing import Union, Dict, List, Optional, Literal, Any
from kiwipiepy import Kiwi
import bm25s
import gc
import faiss

class DataLoader:
    """
    프로모션 데이터와 고객 데이터를 로드, 검색, 필터링하여
    FGI용 단건 프로필 또는 대량 시뮬레이션용 데이터셋을 생성하는 파이프라인 클래스.
    """
    def __init__(
        self,
        model: Any,
        ms_table_path: str,
        ml_table_path: str,
        promo_item: str,
        promo_info: str
    ):
        # 1. Models & Tools
        self.model = model
        self.kiwi = Kiwi()

        # 4. 일관되지 않은 상태 관리 개선 (지연 로딩을 위한 초기화)
        # 생성자에서 필수 재료(경로 및 정보)를 모두 주입받습니다.
        self.ms_table_path = ms_table_path
        self.ml_table_path = ml_table_path
        self.promo_item = promo_item
        self.promo_info = promo_info

        # 내부 캐싱용 프라이빗 변수
        self._ms_table: Optional[pl.DataFrame] = None
        self._ml_table: Optional[pl.DataFrame] = None
        self._retriever: Optional[bm25s.BM25] = None
        self._encoded_promo: Optional[np.ndarray] = None

        # Filtered Results (이들은 로직 실행 결과이므로 유지)
        self.filtered_ms: Optional[pl.DataFrame] = None
        self.filtered_ml: Optional[pl.DataFrame] = None

    # =====================================================================
    # 지연 로딩(Lazy Loading) 프로퍼티: 필요할 때만 메모리에 올립니다.
    # =====================================================================
    @property
    def ms_table(self) -> pl.DataFrame:
        if self._ms_table is None:
            print("MS 테이블 로드 중...")
            self._ms_table = pl.read_parquet(self.ms_table_path)
        return self._ms_table

    @property
    def ml_table(self) -> pl.DataFrame:
        if self._ml_table is None:
            print("ML 테이블 로드 중...")
            self._ml_table = pl.read_parquet(self.ml_table_path)
        return self._ml_table

    @property
    def retriever(self) -> bm25s.BM25:
        if self._retriever is None:
            print("BM25 인덱싱 수행 중...")
            self._retriever = bm25s.BM25()
            self._retriever.index(self.ml_table['tokenized_corpus'])
        return self._retriever

    @property
    def encoded_promo(self) -> np.ndarray:
        if self._encoded_promo is None:
            self._encoded_promo = self.model.encode(
                [self.promo_info], normalize_embeddings=True
            ).astype(np.float32)
        return self._encoded_promo

    # =====================================================================
    # 비즈니스 로직
    # =====================================================================
    def find_relevant_ms_faiss(self, top_n: Optional[int] = None, top_m_percent: Optional[float] = None) -> 'DataLoader':
        # 프로퍼티를 호출하는 순간 자동으로 데이터 로드 보장 (방어 코드 제거)
        query_vector = self.encoded_promo

        # 2. 메모리 비효율성 개선: np.vstack을 사용하여 불필요한 객체 생성 최소화 및 gc.collect() 제거
        encoded_ms_np = np.vstack(self.ms_table.get_column('encoded_sentences').to_numpy()).astype(np.float32)

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        total_rows, dimension = encoded_ms_np.shape

        k = total_rows
        if top_n is not None:
            k = min(top_n, total_rows)
        elif top_m_percent is not None:
            k = int(total_rows * top_m_percent)

        print(f"상위 {k}건 FAISS GPU 검색 시작...")
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dimension)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        gpu_index.add(encoded_ms_np)
        scores, indices = gpu_index.search(query_vector, k)

        self.filtered_ms = self.ms_table[indices[0]].with_columns(
            pl.Series("ms_similarity_score", scores[0])
        )
        return self

    def _yield_hybrid_tokenize(self, data_array: List[str], chunk_size: int = 10000, n_gram_sizes: List[int] = [2, 3]):
        for start_idx in range(0, len(data_array), chunk_size):
            chunk_data = data_array[start_idx : start_idx + chunk_size]
            text_chunk = pl.Series(chunk_data).fill_null("").cast(pl.String).to_list()
            analyzed_docs = self.kiwi.tokenize(text_chunk)

            for original_text, analyzed_tokens in zip(text_chunk, analyzed_docs):
                morphemes = [t.form for t in analyzed_tokens if t.tag.startswith('N') or t.tag == 'SL']
                text_no_space = original_text.replace(" ", "")
                n_grams = [text_no_space[i:i+n] for n in n_gram_sizes for i in range(len(text_no_space) - n + 1)]
                yield morphemes + n_grams
            gc.collect()

    def find_relevant_ml(self, top_n: Optional[int] = None, top_m_percent: Optional[float] = None) -> 'DataLoader':
        print("BM25 연관 상품 검색 시작...")
        tokenized_query = list(self._yield_hybrid_tokenize([self.promo_item]))
        score = self.retriever.get_scores(tokenized_query[0])

        filtered_df = self.ml_table.with_columns(
            pl.Series('ml_similarity_score', score)
        ).sort("ml_similarity_score", descending=True)

        if top_n is not None:
            filtered_df = filtered_df.head(top_n)
        if top_m_percent is not None:
            filtered_df = filtered_df.filter(
                pl.col('ml_similarity_score') > pl.col('ml_similarity_score').quantile(top_m_percent)
            )

        self.filtered_ml = filtered_df
        return self

    def get_fgi_profile(self, clone_srg_key: str, df_pd_de_tot: pl.DataFrame, df_pixel: pl.DataFrame = None) -> dict:
        if self.filtered_ms is None or self.filtered_ml is None:
             raise ValueError("필터링(find_relevant_ms_faiss / find_relevant_ml)을 먼저 실행해야 합니다.")

        # 1. MS (Core Mindset) 추출
        ms_row = self.filtered_ms.filter(pl.col('srg_key') == clone_srg_key)
        ms_data = ms_row['ms_full'][0] if not ms_row.is_empty() else ""

        # 2. [기존 로직] 이번 귀리 음료 프로모션과 '직접' 연관된 이력 추출
        ml_history_target = (
            df_pd_de_tot
            .filter(pl.col('srg_key') == clone_srg_key)
            .filter(pl.col('pd_nm_full').is_in(self.filtered_ml['pd_nm_full'].to_list()))
        )
        ml_data_target = ml_history_target.select([
            pl.col('dt').alias('구매일자'),
            pl.col('pd_nm_full').alias('구매상품'),
            pl.col('buy_am').alias('구매금액')
        ]).to_dicts()

        # =====================================================================
        # 💡 [새로 추가된 로직] 동적 검색(Dynamic Retrieval)을 위한 장기 기억 세팅
        # =====================================================================
        
        # 3. 해당 고객의 1년 치 '전체' 구매 이력(Full ML)과 임베딩 생성
        full_ml_df = df_pd_de_tot.filter(pl.col('srg_key') == clone_srg_key)
        full_ml_data = full_ml_df.select([
            pl.col('dt').alias('구매일자'),
            pl.col('pd_nm_full').alias('구매상품'),
            pl.col('buy_am').alias('구매금액')
        ]).to_dicts()
        
        # 텍스트로 변환하여 실시간 임베딩 (고객 1명이므로 매우 빠름)
        if full_ml_data:
            ml_texts = [f"{item['구매일자']}에 {item['구매상품']}을(를) {item['구매금액']}원에 구매" for item in full_ml_data]
            # DataLoader 생성 시 주입받은 self.model(Snowflake 등) 활용
            ml_embeddings = self.model.encode(ml_texts, normalize_embeddings=True).astype(np.float32)
        else:
            ml_embeddings = np.array([])

        # 4. 해당 고객의 전체 마이크로 어트리뷰트(PIXEL)와 임베딩 생성
        full_pixel_data = []
        pixel_embeddings = np.array([])
        
        if df_pixel is not None:
            pixel_df = df_pixel.filter(pl.col('srg_key') == clone_srg_key)
            if not pixel_df.is_empty():
                # PIXEL 텍스트 컬럼명이 'pixel_attr'이라고 가정 (실제 컬럼명에 맞게 수정 필요)
                if 'pixel_attr' in pixel_df.columns:
                    full_pixel_data = pixel_df['pixel_attr'].to_list()
                    pixel_embeddings = self.model.encode(full_pixel_data, normalize_embeddings=True).astype(np.float32)

        return {
            "ms": ms_data,
            "ml": ml_data_target,         # 프롬프트에 기본으로 박힐 타겟 이력
            "promo_info": self.promo_info,
            "full_ml": full_ml_data,      # 💡 에이전트 장기 기억용 (원시 데이터)
            "ml_embeddings": ml_embeddings, # 💡 에이전트 장기 기억용 (벡터)
            "full_pixel": full_pixel_data,
            "pixel_embeddings": pixel_embeddings
        }
    def get_mass_simulation_df(self, df_pd_de_tot: pl.DataFrame, join_type: Literal['inner', 'left'] = 'left') -> pl.DataFrame:
        if self.filtered_ms is None or self.filtered_ml is None:
            raise ValueError("필터링(find_relevant_ms_faiss / find_relevant_ml)을 먼저 실행해야 합니다.")

        ms_base = self.filtered_ms.select([
            pl.col('srg_key'),
            pl.col('ms_full').alias('ms_core_mindset')
        ])

        relevant_items = self.filtered_ml.select('pd_nm_full')

        ml_filtered = (
            df_pd_de_tot
            .join(relevant_items, on='pd_nm_full', how='inner')
            .with_columns(pl.col('dt').cast(pl.String))
        )

        # 1. map_elements(안티패턴) 제거: Polars 네이티브 JSON 인코딩 및 문자열 병합 사용
        # 파이썬 GIL을 거치지 않고 Rust 단에서 병렬 처리되므로 연산 속도가 극적으로 빨라집니다.
        ml_aggregated = ml_filtered.select([
            'srg_key', 'dt', 'pd_nm_full', 'buy_am'
        ]).group_by('srg_key').agg([
            pl.struct(['dt', 'pd_nm_full', 'buy_am']).alias('history')
        ]).with_columns(
            (
                pl.lit('[') +
                pl.col('history').list.eval(
                    pl.element().struct.json_encode()
                ).list.join(',') +
                pl.lit(']')
            ).alias('ml_transaction_history')
        ).select(['srg_key', 'ml_transaction_history'])

        target_df = ms_base.join(ml_aggregated, on='srg_key', how=join_type)

        print(f"최종 시뮬레이션 대상: {target_df.height}명 생성 완료.")
        return target_df
