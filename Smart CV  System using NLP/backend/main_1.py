import logging
import time
import asyncio
from typing import List
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from models.schemas import BatchATSResponse, MatchResult, CandidateProfile, SkillAnalytics, PerformanceMetrics
from services.embedding_service import EmbeddingService
from services.parser_service import ParserService
from services.nlp_extractor import NLPExtractor
from services.matcher_service import MatcherService
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnterpriseATSApp")

app = FastAPI(title=settings.PROJECT_NAME, version="2.3.0")

# Dependency Injection Singletons
embedding_service = EmbeddingService()
parser_service = ParserService()
nlp_extractor = NLPExtractor()


def CPU_bound_profile_processor(payload_name: str, cv_text: str, tfidf_score: float, embedding_score: float, jd_skills: List[str]) -> MatchResult:
    """Processes NLP extraction inside isolated threads (Can be scaled to ProcessPoolExecutor for massive loads)"""
    contacts = nlp_extractor.extract_contact_info(cv_text)
    years = nlp_extractor.extract_experience(cv_text)
    edu = nlp_extractor.extract_education(cv_text)
    cv_skills = nlp_extractor.extract_skills(cv_text)
    
    matched_skills = [s for s in cv_skills if s in jd_skills]
    missing_skills = [s for s in jd_skills if s not in cv_skills]
    match_rate = len(matched_skills) / len(jd_skills) if jd_skills else 0.0
    
    weight_embedding = settings.DEFAULT_WEIGHT_EMBEDDING
    weight_tfidf = round(1.0 - weight_embedding, 2)
    
    final_score = (tfidf_score * weight_tfidf) + (embedding_score * weight_embedding)
    final_score_pct = round(max(0.0, min(1.0, final_score)) * 100, 2)
    
    rec = "🟢 Highly Recommended" if final_score_pct >= 75 else "🟡 Consider for Review" if final_score_pct >= 45 else "🔴 Rejected"
    
    prof = CandidateProfile(
        candidate_id=payload_name,
        name=payload_name.split('.')[0].replace('_', ' ').title(),
        email=contacts["email"],
        phone=contacts["phone"],
        linkedin=contacts["linkedin"],
        github=contacts["github"],
        experience_years=years,
        education_tier=edu,
        extracted_skills=cv_skills
    )
    
    return MatchResult(
        profile=prof,
        tfidf_score=round(tfidf_score, 4),
        embedding_score=round(embedding_score, 4),
        final_hybrid_score=final_score_pct,
        explainability=SkillAnalytics(matched_skills=matched_skills, missing_skills=missing_skills, skill_match_rate=round(match_rate * 100, 2)),
        recommendation=rec
    )


@app.post(f"{settings.API_V1_STR}/match", response_model=BatchATSResponse)
async def process_ats_batch(
    job_description: str = Form(...),
    weight_embedding: float = Form(0.6),
    files: List[UploadFile] = File(...)
):
    start_time = time.time()
    logger.info(f"Initiating batch transaction for {len(files)} files via Vectorized Pipeline.")
    
    if abs((weight_embedding + round(1.0 - weight_embedding, 2)) - 1.0) > 0.001:
        raise HTTPException(status_code=400, detail="Configuration parameters imbalance.")
        
    # 🛡️ الـ Set هنا معزولة تماماً داخل نطاق الـ Request (Per-Request Scope) 
    # هي آمنة حالياً لأنها لا تتشارك مع أي Threads أخرى خارج هذا الـ Request.
    seen_hashes = set()
    unique_files = []
    duplicate_count = 0
    
    async def read_and_filter(file: UploadFile):
        nonlocal duplicate_count
        content = await file.read()
        f_hash = parser_service.calculate_file_hash(content)
        if f_hash in seen_hashes:
            duplicate_count += 1
            return None
        seen_hashes.add(f_hash)
        return {"name": file.filename, "bytes": content}

    file_tasks = [read_and_filter(f) for f in files]
    processed_payloads = await asyncio.gather(*file_tasks)
    unique_files = [p for p in processed_payloads if p is not None]

    if not unique_files:
        raise HTTPException(status_code=400, detail="No unique document artifacts remaining to process.")

    # 1. الـ Parsing واستخراج النصوص
    cleaned_jd = parser_service.clean_text(job_description)
    parsed_resumes = [str(parser_service.parse_file(f["name"], f["bytes"])) for f in unique_files]
    
    # 2. حساب مصفوفة الـ TF-IDF (Context Aware)
    try:
        tfidf_scores = MatcherService.calculate_tfidf_scores(cleaned_jd, parsed_resumes)
    except Exception as tfidf_err:
        logger.error(f"TF-IDF Error: {str(tfidf_err)}")
        tfidf_scores = [0.0] * len(unique_files)

    # 3. ⚡ الحل الأسطوري: حساب الـ Embeddings دفعة واحدة (Batch Encoding Pipeline)
    # نقوم بعمل الـ Fit للـ JD والـ CVs معاً في مصفوفة واحدة لتمريرها للموديل دفعة واحدة
    all_texts_to_encode = [cleaned_jd] + parsed_resumes
    try:
        all_embeddings = await embedding_service.encode_batch(all_texts_to_encode)
    except Exception as model_err:
        logger.error(f"Model Inference Failure: {str(model_err)}")
        raise HTTPException(status_code=500, detail="Semantic Engine Pipeline breakdown.")

    jd_embedding = all_embeddings[0]
    cv_embeddings = all_embeddings[1:]
    
    # حساب معايير الـ Vectors بشكل مسبق لحماية الـ Pipeline
    norm_jd = np.linalg.norm(jd_embedding)

    # 4. حساب الـ Cosine Similarity باستخدام الـ Vectorization السريع بـ NumPy
    embedding_scores = []
    for cv_emb in cv_embeddings:
        norm_cv = np.linalg.norm(cv_emb)
        # ✅ حل مشكلة الـ Math Stability والـ Division by Zero الصارمة
        if norm_jd == 0.0 or norm_cv == 0.0:
            embedding_scores.append(0.0)
        else:
            dot_prod = np.dot(jd_embedding, cv_emb)
            embedding_scores.append(float(dot_prod / (norm_jd * norm_cv)))

    # 5. توزيع المهام في الـ Background حماية للـ Event Loop
    jd_skills = nlp_extractor.extract_skills(cleaned_jd)
    
    processing_tasks = [
        asyncio.to_thread(
            CPU_bound_profile_processor, 
            unique_files[i]["name"], 
            parsed_resumes[i], 
            tfidf_scores[i], 
            embedding_scores[i], 
            jd_skills
        )
        for i in range(len(unique_files))
    ]
    
    leaderboard = await asyncio.gather(*processing_tasks)
    leaderboard.sort(key=lambda x: x.final_hybrid_score, reverse=True)
    
    processing_time = time.time() - start_time
    
    return BatchATSResponse(
        job_description_summary=job_description[:100] + "...",
        total_processed=len(unique_files),
        duplicates_detected=duplicate_count,
        engine_model_used=embedding_service.get_model_name(),
        leaderboard=leaderboard,
        performance_summary=PerformanceMetrics(processing_time_seconds=round(processing_time, 4))
    )