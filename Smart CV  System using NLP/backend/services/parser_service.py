import io
import pypdf
import docx
import hashlib
import re

class ParserService:
    @staticmethod
    def calculate_file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+\s*', ' ', text)
        text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def parse_file(self, file_name: str, file_bytes: bytes) -> str:
        text = ""
        if file_name.endswith('.pdf'):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = " ".join([p.text for p in doc.paragraphs])
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
        return self.clean_text(text)