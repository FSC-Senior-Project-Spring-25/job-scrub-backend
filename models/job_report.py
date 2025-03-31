import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    address: str
    lat: float
    lon: float


class LocationType(Enum):
    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"

class JobType(Enum):
    FULL_TIME = "fulltime"
    PART_TIME = "parttime"
    INTERNSHIP = "internship"
    CONTRACT = "contract"
    VOLUNTEER = "volunteer"

class JobReport(BaseModel):
    title: str
    company: str
    url: str
    description: str
    job_type: JobType = Field(..., alias="jobType")
    skills: list[str] = []
    location: Location
    location_type: LocationType = Field(..., alias="locationType")
    benefits: Optional[list[str]] = []
    date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d")
    salary: Optional[str] = None


class AutoFillJobReport(BaseModel):
    content: str
