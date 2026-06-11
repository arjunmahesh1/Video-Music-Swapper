"""Sonic Segments site: intake, campaign status, and before/after preview.

Run with:  uvicorn sonic_segments.web.main:app --port 8000
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from ..intelligence import DemographicsKB
from ..intelligence.mood import MOODS
from ..pipeline.campaign import prepare_and_run
from ..pipeline.jobs import JobStore
from ..sources import sources_info

BASE = Path(__file__).resolve().parent
UPLOAD_STAGING = Path("output/uploads")

app = FastAPI(title="Sonic Segments")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


# ---------- pages -----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/campaign/{job_id}", response_class=HTMLResponse)
def campaign_page(request: Request, job_id: str):
    job = JobStore.get().get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown campaign.")
    return templates.TemplateResponse(request, "campaign.html", {"job_id": job_id})


@app.get("/preview/{job_id}", response_class=HTMLResponse)
def preview_page(request: Request, job_id: str):
    job = JobStore.get().get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown campaign.")
    manifest_path = job.dir / "manifest.json"
    if not manifest_path.exists():
        return RedirectResponse(f"/campaign/{job_id}")
    manifest = json.loads(manifest_path.read_text())
    return templates.TemplateResponse(
        request, "preview.html", {"m": manifest, "manifest_json": json.dumps(manifest)}
    )


# ---------- api -------------------------------------------------------------

@app.get("/api/meta")
def meta():
    kb = DemographicsKB.default()
    return {
        "segments": kb.list_segments(),
        "moods": [
            {"id": mood, "label": spec["label"], "adjectives": spec["adjectives"]}
            for mood, spec in MOODS.items()
        ],
        "sources": sources_info(),
        "max_cuts": 5,
    }


@app.post("/api/campaigns")
async def create_campaign(
    mode: str = Form("variants"),
    brand: str = Form(""),
    vibe: str = Form(""),
    url: str = Form(""),
    moods: str = Form(""),
    demographics: str = Form(""),
    sources: str = Form(""),
    reference_query: str = Form(""),
    video: UploadFile | None = File(None),
    own_track: UploadFile | None = File(None),
):
    mood_list = [m for m in moods.split(",") if m.strip()]
    demo_list = [d for d in demographics.split(",") if d.strip()]
    source_list = [s for s in sources.split(",") if s.strip()]

    if not mood_list:
        raise HTTPException(400, "Pick at least one mood.")
    if mode == "variants" and not demo_list:
        raise HTTPException(400, "Pick at least one target demographic.")
    if not source_list:
        raise HTTPException(400, "Pick at least one music source.")
    if video is None and not url.strip():
        raise HTTPException(400, "Upload the ad file or paste a link to it.")

    staged_video = await _stage_upload(video, VIDEO_SUFFIXES) if video else None
    staged_track = await _stage_upload(own_track, AUDIO_SUFFIXES) if own_track else None

    params = {
        "mode": mode,
        "brand": brand.strip(),
        "vibe": vibe.strip(),
        "url": url.strip() or None,
        "staged_file": str(staged_video) if staged_video else None,
        "moods": mood_list,
        "demographics": demo_list,
        "sources": source_list,
        "reference_query": reference_query.strip() or None,
        "uploaded_tracks": [str(staged_track)] if staged_track else [],
    }
    job = JobStore.get().submit("campaign", params, prepare_and_run)
    return {"job_id": job.id, "status_url": f"/campaign/{job.id}"}


@app.get("/api/campaigns/{job_id}")
def campaign_status(job_id: str):
    job = JobStore.get().get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown campaign.")
    return job.as_dict()


@app.get("/media/{job_id}/{file_path:path}")
def media(job_id: str, file_path: str):
    job = JobStore.get().get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown campaign.")
    target = (job.dir / file_path).resolve()
    if not str(target).startswith(str(job.dir.resolve())) or not target.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(target)


# ---------- spotify ---------------------------------------------------------

@app.get("/api/spotify/status")
def spotify_status():
    try:
        from spotify_helper import SpotifyManager

        manager = SpotifyManager()
        if manager.authenticate():
            user = manager.sp.current_user()
            return {"connected": True, "user": user.get("display_name") or user.get("id")}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
    return {"connected": False}


@app.get("/spotify/login")
def spotify_login():
    from spotify_helper import SpotifyManager

    try:
        return RedirectResponse(SpotifyManager().get_auth_url())
    except Exception as exc:
        raise HTTPException(500, f"Spotify auth not configured: {exc}")


@app.get("/callback")
def spotify_callback(code: str = "", error: str = ""):
    from spotify_helper import SpotifyManager

    if error or not code:
        return RedirectResponse(f"/?spotify=denied#intake")
    try:
        ok = SpotifyManager().handle_redirect_code(code)
    except Exception:
        ok = False
    return RedirectResponse(f"/?spotify={'connected' if ok else 'failed'}#intake")


# ---------- helpers ---------------------------------------------------------

async def _stage_upload(upload: UploadFile, allowed: set[str]) -> Path:
    suffix = Path(upload.filename or "file").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}.")
    UPLOAD_STAGING.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_STAGING / f"{uuid.uuid4().hex[:10]}{suffix}"
    with open(target, "wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return target
