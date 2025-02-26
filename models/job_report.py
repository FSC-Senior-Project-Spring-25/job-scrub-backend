import datetime
from typing import Optional

from pydantic import BaseModel


class JobReport(BaseModel):
    title: str
    company: str
    url: str
    description: str
    date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d")
    salary: Optional[str] = None
    skills: Optional[list[str]] = None
    benefits: Optional[list[str]] = None
    location: Optional[str] = None
    job_type: Optional[str] = None