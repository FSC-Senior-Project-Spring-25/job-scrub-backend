"""
Remote-OK → Pinecone loader  |  existing index “job-postings”
----------------------------------------------------------------
activate venv ➜  cd job-scrub-backend ➜
    python -u -m services.job_fetch
"""

from __future__ import annotations
import asyncio, logging, os, sys
from datetime import datetime, timezone
from typing import List

import aiohttp, requests
from dotenv import load_dotenv
from pinecone import Pinecone

from models.job_report import JobReport, Location, JobType, LocationType
from services.jobs_posting import JobsPostingService
from services.text_embedder import TextEmbedder
from services.gemini import GeminiLLM

# ─── logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

# ─── feed & constants ───────────────────────────────────────
FEED_URL   = "https://remoteok.com/remote-dev-jobs.json"
HEADERS    = {"User-Agent": "Mozilla/5.0"}
INDEX_NAME = "job-postings"          # 1024-d, cosine, serverless us-east-1

# ─── Pinecone bootstrap ─────────────────────────────────────
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env     = os.getenv("PINECONE_ENV", "aws-us-east-1")

if not api_key:
    logging.error("Set PINECONE_API_KEY in .env"); sys.exit(1)

pc = Pinecone(api_key=api_key, environment=env)
existing_names = {idx["name"] for idx in pc.list_indexes()}

if INDEX_NAME not in existing_names:
    logging.error("Index '%s' not found in environment '%s'. "
                  "Current indexes: %s", INDEX_NAME, env, existing_names)
    sys.exit(1)

index = pc.Index(INDEX_NAME)
logging.info("Connected to Pinecone index '%s' (%s)", INDEX_NAME, env)

# ─── helpers ────────────────────────────────────────────────
def row_to_report(row: dict) -> JobReport:
    return JobReport(
        title=row["position"],
        company=row["company"],
        url=row["url"],
        description=row.get("description") or "",
        location=Location(lat=0, lon=0, address=row.get("location") or "Remote"),
        jobType=JobType.FULL_TIME,
        locationType=LocationType.REMOTE,
        posted_at=datetime.fromtimestamp(
            row.get("epoch", datetime.now().timestamp()), tz=timezone.utc
        ),
    )

# ─── main coroutine ─────────────────────────────────────────
async def ingest(batch: int = 50):
    logging.info("Downloading Remote-OK feed …")
    rows: List[dict] = requests.get(FEED_URL, headers=HEADERS, timeout=15).json()[1:batch+1]
    logging.info("Processing %s rows", len(rows))

    async with aiohttp.ClientSession() as session:
        poster = JobsPostingService(TextEmbedder(), index, GeminiLLM(), session)

        async def upsert_one(row: dict):
            jr  = row_to_report(row)
            vec = await poster.post_job(jr)
            logging.info("✔ upserted %-60s (id=%s)", jr.title[:60], vec)

        await asyncio.gather(*(upsert_one(r) for r in rows))

    logging.info("✅ Finished – vectors live in index '%s'.", INDEX_NAME)

# ─── CLI entry-point ────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(ingest())
