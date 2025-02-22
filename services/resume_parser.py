from pypdf import PdfReader
import io


class ResumeParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        """
        Parse a PDF file and extract its text content
        """
        pdf = PdfReader(io.BytesIO(file_bytes))

        # Extract text from all pages
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

        return text.strip()