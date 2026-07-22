from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

class MatcherService:
    @staticmethod
    def calculate_tfidf_scores(jd_text: str, resumes_list: List[str]) -> List[float]:
        if not resumes_list:
            return []
        all_corpus = [jd_text] + resumes_list
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_corpus)
        
        jd_vector = tfidf_matrix[0:1]
        cv_vectors = tfidf_matrix[1:]
        
        scores = cosine_similarity(jd_vector, cv_vectors).flatten()
        return scores.tolist()