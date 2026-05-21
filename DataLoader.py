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
