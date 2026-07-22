import re
from typing import List, Tuple

class NLPExtractor:
    def __init__(self):
        self.skills_lookup = {
            "python": ["python", "python3", "pythonic"],
            "fastapi": ["fastapi", "fast-api"],
            "docker": ["docker", "dockerized", "docker-compose"],
            "kubernetes": ["kubernetes", "k8s"],
            "xgboost": ["xgboost", "xgb"],
            "aws": ["aws", "amazon web services"],
            "sql": ["sql", "mysql", "postgresql"]
        }

    def extract_skills(self, cleaned_text: str) -> List[str] :
        found_skills = []
        for skill_key, variants in self.skills_lookup.items():
            for variant in variants:
                if re.search(r'\b' + re.escape(variant) + r'\b', cleaned_text):
                    found_skills.append(skill_key)
                    break
        return list(set(found_skills))

    def extract_contact_info(self, text: str) -> dict:
        email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        phone = re.search(r'\+?\d{10,14}', text)
        linkedin = re.search(r'linkedin\.com/in/[\w\-]+', text)
        github = re.search(r'github\.com/[\w\-]+', text)
        return {
            "email": email.group(0) if email else "Not Found",
            "phone": phone.group(0) if phone else "Not Found",
            "linkedin": linkedin.group(0) if linkedin else "Not Found",
            "github": github.group(0) if github else "Not Found"
        }

    def extract_experience(self, text: str) -> float:
        patterns = [
            r'(?:over|more than)?\s*(\d+)\s*(?:\+|–|-)?\s*(?:years|yrs)\s*(?:of\s*experience)?',
            r'experience\s*:\s*(\d+)\s*yrs'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return 0.0

    def extract_education(self, text: str) -> str:
        if re.search(r'\b(phd|ph\.d)\b', text):
            return "PhD"
        if re.search(r'\b(master|msc|m\.sc|ms|meng)\b', text):
            return "Master"
        if re.search(r'\b(bachelor|bsc|b\.sc|bs)\b', text):
            return "Bachelor"
        return "Not Specified"