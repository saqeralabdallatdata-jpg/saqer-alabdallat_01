import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from config import settings

class EmbeddingService:
    def __init__(self):
        model_name = settings.EMBEDDING_MODEL_REGISTRY[settings.ACTIVE_MODEL_VERSION]
        # تحميل الموديل لمرة واحدة في الـ Shared Memory
        self.model = SentenceTransformer(model_name)
        # سيمافور لمنع اختناق الـ Memory عند معالجة طلبات ضخمة متزامنة
        self.semaphore = asyncio.Semaphore(2) 

    def get_model_name(self) -> str:
        return settings.EMBEDDING_MODEL_REGISTRY[settings.ACTIVE_MODEL_VERSION]

    async def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes a block of texts in a single batch operation using safe background threads"""
        async with self.semaphore:
            # تشغيل الـ Batch Inference الثقيل برمجياً في Thread معزول تماماً عن الـ Event Loop
            embeddings = await asyncio.to_thread(
                self.model.encode, 
                texts, 
                batch_size=32, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            return embeddings