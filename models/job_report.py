import datetime
from typing import Optional

from pydantic import BaseModel


class JobReport(BaseModel):
    title: str
    company: str
    url: str
    date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d")
    description: Optional[str] = None
    salary: Optional[str] = None
    skills: Optional[list[str]] = None
    location: Optional[str] = None
    job_type: Optional[str] = None