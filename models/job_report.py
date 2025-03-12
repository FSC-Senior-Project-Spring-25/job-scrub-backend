import datetime
from typing import Optional

from pydantic import BaseModel


class JobReport(BaseModel):
    title: str
    company: str
    url: str
    description: str
    job_type: str
    location: str = "Remote"
    skills: list[str] = []
    benefits: Optional[list[str]] = []
    date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d")
    salary: Optional[str] = None
