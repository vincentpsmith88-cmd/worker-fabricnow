"""Fabric Now API. Run: uvicorn main:app --host 0.0.0.0 --port 8000"""
import base64, io, json, logging, os, re, shutil, threading, time, uuid, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

import pipeline
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # local dev; in Docker/hosting, real env vars are used

log = logging.getLogger("fabricnow")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

OPENAI_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-image-1.5")
IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1536x1024")
IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "high")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-5")
AI_VISION_PROVIDER = os.getenv("AI_VISION_PROVIDER", "auto")
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "2"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
RATE_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
RETENTION_H = int(os.getenv("RETENTION_HOURS", "72"))
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
INTERNAL_SECRET = os.getenv("FABRIC_NOW_INTERNAL_SECRET", "")
ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",") if o.strip()]
JOBS = Path(os.getenv("DATA_DIR", "./data")) / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

_client = None


def get_client():
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("The server has no OPENAI_API_KEY configured.")
        _client = openai.OpenAI(timeout=300, max_retries=2)
    return _client


if not os.getenv("OPENAI_API_KEY"):
    log.warning("OPENAI_API_KEY is not set. Add it to backend/.env or the environment. Generation will fail until then.")
pool = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_CONCURRENT_JOBS", "3")))
lock, hits = threading.Lock(), {}

app = FastAPI(title="Fabric Now API", docs_url=None if os.getenv("ENV") == "production" else "/docs")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


def _meta(jid):
    try: return json.loads((JOBS / jid / "job.json").read_text())
    except Exception: raise HTTPException(404, "Project not found")

def _save(jid, **kw):
    with lock:
        p = JOBS / jid / "job.json"
        m = json.loads(p.read_text()); m.update(kw)
        p.write_text(json.dumps(m))

def _valid(jid):
    try: return str(uuid.UUID(jid)) == jid
    except ValueError: return False

def _internal_auth(request: Request):
    """Only Style Backend may call this worker in production."""
    if INTERNAL_SECRET and request.headers.get("x-fabric-internal-secret") != INTERNAL_SECRET:
        raise HTTPException(401, "Worker authentication required.")

def _user_id(request: Request):
    uid = request.headers.get("x-workspace-user-id", "").strip()
    if not uid:
        raise HTTPException(401, "Workspace user required.")
    return uid

def _purge():
    cut = time.time() - RETENTION_H * 3600
    for d in JOBS.iterdir():
        if d.is_dir() and d.stat().st_mtime < cut: shutil.rmtree(d, ignore_errors=True)

def _rate(request: Request):
    # The Node workspace already authenticates the user. Rate limiting must therefore
    # be keyed to the workspace user, not the Node server's shared IP.
    uid = _user_id(request)
    key = f"user:{uid}"
    now = time.time()
    with lock:
        h = [t for t in hits.get(key, []) if now - t < 3600]
        if len(h) >= RATE_PER_HOUR:
            raise HTTPException(429, "Hourly generation limit reached. Try again later.")
        hits[key] = h + [now]



def _claude_quality_check(sheet: Image.Image, plan):
    """Optional second-opinion vision gate. It never replaces the deterministic CV gate."""
    key=os.getenv("ANTHROPIC_API_KEY")
    if not key or not plan: return {"enabled":False,"ok":True,"issues":[]}
    b=io.BytesIO(); sheet.save(b,"PNG")
    prompt=("Review this generated sewing-pattern sheet against the planned pieces below. "
            "Return JSON only: {\"ok\":true,\"issues\":[],\"missing\":[],\"extra\":[],\"labels_ok\":true}. "
            "Be conservative: flag obvious missing/extra pieces, clipped pieces, unreadable labels, "
            "solid-black fills, or a sheet that is clearly a fashion illustration instead of flat pieces.\n"
            + json.dumps(plan["pieces"],ensure_ascii=False))
    payload={"model":os.getenv("ANTHROPIC_VISION_MODEL","claude-sonnet-4-5"),"max_tokens":1500,
             "system":"You are a strict garment-pattern QA reviewer. JSON only.",
             "messages":[{"role":"user","content":[
                 {"type":"text","text":prompt},
                 {"type":"image","source":{"type":"base64","media_type":"image/png","data":base64.b64encode(b.getvalue()).decode()}}
             ]}]}
    try:
        req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=json.dumps(payload).encode(),
            headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=90) as res: d=json.loads(res.read())
        txt="\n".join(x.get("text","") for x in d.get("content",[]) if x.get("type")=="text")
        txt=re.sub(r"^```json\s*|\s*```$","",txt.strip(),flags=re.I)
        q=json.loads(txt); q["enabled"]=True; return q
    except Exception as e:
        log.warning("Claude pattern QA skipped: %s",e)
        return {"enabled":True,"ok":True,"issues":["QA unavailable"],"missing":[],"extra":[]}

def _run(jid):
    d = JOBS / jid
    try:
        m = _meta(jid)
        src = (d / "input.png").read_bytes()
        _save(jid, status="processing", step=0, stage="Reading the garment and fabric")
        plan = None
        try:
            provider = AI_VISION_PROVIDER.lower()
            vision_client = get_client() if provider == "openai" or (provider == "auto" and not os.getenv("ANTHROPIC_API_KEY")) else None
            plan = pipeline.analyze(vision_client, VISION_MODEL, src, m["garment"], m["lining"])
        except (openai.AuthenticationError, openai.RateLimitError):
            raise
        except Exception:
            log.exception("garment analysis failed; continuing with a generic prompt")
        swatch = (d / "swatch.png").read_bytes() if (d / "swatch.png").exists() else (
            pipeline.crop_swatch(src, plan.get("swatch")) if plan else None)
        _save(jid, step=1, stage="AI is determining garment type and mathematical pattern geometry", plan={"garment": plan["garment"], "region_family": plan.get("region_family"), "style_family": plan.get("style_family"), "scale_basis": plan.get("scale_basis"), "geometry_confidence": plan.get("geometry_confidence"), "pieces": plan["pieces"]})
        _save(jid, step=2, stage="Rendering exact numeric geometry and fabric pieces")
        pieces = pipeline.export_all(None, d / "export", src, plan=plan, swatch_bytes=swatch)
        qa = {"enabled": False, "mode": "deterministic-geometry", "ok": True}
        _save(jid, qa=qa, output_format="fabricnow-basic-pattern-v3-mathematical-geometry")
        if not pieces:
            raise ValueError("No separate pieces were detected. Try a clearer, full-length photo.")
        _save(jid, status="done", step=3, stage="Ready", pieces=pieces)
    except openai.RateLimitError:
        _save(jid, status="failed", error="The image service is busy or out of quota. Try again shortly.")
    except openai.AuthenticationError:
        log.error("OPENAI_API_KEY rejected"); _save(jid, status="failed", error="Server is not configured correctly.")
    except openai.BadRequestError as e:
        log.warning("bad request: %s", e)
        _save(jid, status="failed", error="The image service rejected this photo. Try a different one.")
    except ValueError as e:
        _save(jid, status="failed", error=str(e))
    except Exception:
        log.exception("job %s failed", jid)
        _save(jid, status="failed", error="Something went wrong while generating. Please try again.")



AFRICAN_GARMENTS = [
    "Agbada","Boubou / Bubu","Dashiki","Kaba and Slit","Buba and Iro","Aso-Ebi Set",
    "Aso-Oke Set","Kitenge Two-Piece","Shweshwe Dress","Ankara Maxi Dress","Mermaid Gown",
    "Corset Gown","Peplum Dress","Senator Suit","Kaftan","Djellaba","Kanzu","Jumpsuit",
    "Wrap Dress","Tunic","Kimono","Skirt Set","Blazer Set","Children's Wear","Gele","Headwrap"
]

@app.get("/api/ai/catalog")
def ai_catalog(request: Request):
    _internal_auth(request); _user_id(request)
    return {
        "garments": AFRICAN_GARMENTS,
        "generators":[
            {"id":"african-garment","label":"African Garment Generator"},
            {"id":"fabric-print","label":"Fabric / Print Generator"},
            {"id":"colorways","label":"Colorway Generator"},
            {"id":"listing","label":"Product Listing Writer"}
        ],
        "vision_provider": AI_VISION_PROVIDER,
        "claude_enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai_enabled": bool(os.getenv("OPENAI_API_KEY"))
    }

def _image_data_url(raw):
    return "data:image/png;base64,"+base64.b64encode(raw).decode()

@app.post("/api/ai/generate-image")
async def ai_generate_image(request: Request, file: UploadFile | None = File(None),
                            prompt: str = Form(""), mode: str = Form("african-garment")):
    _internal_auth(request); _user_id(request); _rate(request)
    if not prompt.strip(): raise HTTPException(400,"A generation prompt is required.")
    client=get_client()
    images=[]
    if file:
        raw=await file.read(MAX_UPLOAD_MB*1024*1024+1)
        if len(raw)>MAX_UPLOAD_MB*1024*1024: raise HTTPException(413,"Reference image is too large.")
        img=_prep(raw); b=io.BytesIO(); img.save(b,"PNG"); images=[("reference.png",b.getvalue(),"image/png")]
    mode_prompts={
      "african-garment":"Create a clean studio product render of the requested African garment. Preserve the requested textile motif and garment construction. Full garment visible, elegant fashion presentation, no text, no watermark.",
      "fabric-print":"Create a seamless, tileable textile print swatch. Flat fabric surface, repeat-ready edges, consistent motif scale, no garment, no model, no text.",
      "colorways":"Create a clean presentation of the same garment/fabric design in multiple distinct colourways. Preserve silhouette and motif geometry; no text.",
      "mockup":"Apply the supplied fabric/design faithfully to the requested garment on a clean fashion model or mannequin. Preserve motif scale and placement.",
      "technical-flat":"Create a clean fashion technical flat of the requested African garment, front and back views, black linework on white, construction details visible, no model, no decorative rendering, no text.",
      "lookbook":"Create a polished African fashion lookbook scene featuring the requested garment and textile. Keep the garment design faithful, editorial styling, no brand logos, no text.",
      "aso-ebi":"Create coordinated Aso-Ebi outfit variations for a group using the same supplied fabric identity. Show distinct silhouettes while preserving the exact textile motif and colour family.",
      "cutting-layout":"Create a clear fabric cutting-layout concept for the listed pattern pieces on the requested fabric width. Label pieces and grain direction; do not invent dimensions."
    }
    full=(mode_prompts.get(mode,mode_prompts["african-garment"])+"\nUser brief: "+prompt[:1500])
    kwargs=dict(model=OPENAI_MODEL,image=images or None,prompt=full,size=IMAGE_SIZE,quality=IMAGE_QUALITY,n=1)
    if images:
        res=client.images.edit(**kwargs)
    else:
        # Images.generate is used when there is no reference image.
        kwargs.pop("image",None)
        res=client.images.generate(model=OPENAI_MODEL,prompt=full,size=IMAGE_SIZE,quality=IMAGE_QUALITY,n=1)
    return {"mode":mode,"image_base64":res.data[0].b64_json}

@app.post("/api/ai/listing")
async def ai_listing(request: Request, garment: str = Form(""), fabric: str = Form(""),
                     style: str = Form(""), notes: str = Form("")):
    _internal_auth(request); _user_id(request)
    prompt=("Write a FabricNow digital textile/fashion product listing. Return JSON only with "
            "title, short_description, description, category, style, fabric, tags, seo_title, seo_description. "
            "Do not invent certifications or measurements. "
            f"Garment: {garment}; Fabric: {fabric}; Style: {style}; Notes: {notes[:1000]}")
    key=os.getenv("ANTHROPIC_API_KEY")
    if key:
        payload={"model":os.getenv("ANTHROPIC_TEXT_MODEL","claude-sonnet-4-5"),"max_tokens":1800,
                 "messages":[{"role":"user","content":prompt}]}
        req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=json.dumps(payload).encode(),
             headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},method="POST")
        try:
            with urllib.request.urlopen(req,timeout=90) as res: d=json.loads(res.read())
            txt="\n".join(x.get("text","") for x in d.get("content",[]) if x.get("type")=="text")
            txt=re.sub(r"^```json\s*|\s*```$","",txt.strip(),flags=re.I)
            return json.loads(txt)
        except Exception as e: log.warning("Claude listing failed: %s",e)
    client=get_client()
    r=client.chat.completions.create(model=VISION_MODEL,response_format={"type":"json_object"},
        messages=[{"role":"user","content":prompt}])
    return json.loads(r.choices[0].message.content)


@app.get("/api/health")
def health():
    return {"ok": True, "auth_required": bool(ACCESS_TOKEN)}


def _prep(raw: bytes) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA"); bg = Image.new("RGB", img.size, "white"); bg.paste(img, mask=img.split()[-1]); img = bg
    img = img.convert("RGB"); img.thumbnail((2048, 2048))
    return img


@app.post("/api/jobs", status_code=202)
async def create_job(request: Request, file: UploadFile = File(...), swatch: UploadFile | None = File(None),
                     garment: str = Form("Auto-detect"), notes: str = Form(""), lining: str = Form("none")):
    _internal_auth(request)
    user_id = _user_id(request)
    _rate(request)
    limit = MAX_UPLOAD_MB * 1024 * 1024
    raw = await file.read(limit + 1)
    sraw = await swatch.read(limit + 1) if swatch else b""
    if len(raw) > limit or len(sraw) > limit:
        raise HTTPException(413, f"Photos must be smaller than {MAX_UPLOAD_MB} MB.")
    try:
        img = _prep(raw)
        simg = _prep(sraw) if sraw else None
    except Exception:
        raise HTTPException(400, "Upload a JPG, PNG or WebP photo.")
    _purge()
    jid = str(uuid.uuid4()); d = JOBS / jid; d.mkdir(parents=True)
    img.save(d / "input.png", "PNG")
    if simg: simg.save(d / "swatch.png", "PNG")
    (d / "job.json").write_text(json.dumps(dict(
        id=jid, user_id=user_id, status="queued", step=0, stage="Reading the garment and fabric", created=int(time.time()),
        garment="Auto-detect", notes=notes[:500], lining=str(lining).lower()[:20], pieces=[], plan=None, error=None)))
    pool.submit(_run, jid)
    return _meta(jid)


@app.get("/api/jobs")
def list_jobs(request: Request, ids: str = ""):
    _internal_auth(request)
    user_id = _user_id(request)
    out = []
    requested = [i for i in ids.split(",") if _valid(i)][:50] if ids else []
    if requested:
        candidates = requested
    else:
        candidates = [d.name for d in JOBS.iterdir() if d.is_dir()]
        candidates.sort(key=lambda jid: (JOBS / jid / "job.json").stat().st_mtime if (JOBS / jid / "job.json").exists() else 0, reverse=True)
        candidates = candidates[:100]
    for jid in candidates:
        try:
            meta = json.loads((JOBS / jid / "job.json").read_text())
            if meta.get("user_id") == user_id:
                out.append(meta)
        except Exception:
            pass
    return out


@app.get("/api/jobs/{jid}")
def get_job(request: Request, jid: str):
    _internal_auth(request)
    user_id = _user_id(request)
    if not _valid(jid): raise HTTPException(404, "Project not found")
    meta = _meta(jid)
    if meta.get("user_id") != user_id: raise HTTPException(404, "Project not found")
    return meta


@app.get("/api/jobs/{jid}/files/{path:path}")
def get_file(request: Request, jid: str, path: str):
    _internal_auth(request)
    user_id = _user_id(request)
    if not _valid(jid): raise HTTPException(404)
    if _meta(jid).get("user_id") != user_id: raise HTTPException(404)
    base = (JOBS / jid / "export").resolve()
    f = (base / path).resolve()
    if not f.is_relative_to(base) or not f.is_file(): raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{jid}/export.zip")
def export_zip(request: Request, jid: str):
    _internal_auth(request)
    user_id = _user_id(request)
    if not _valid(jid): raise HTTPException(404)
    if _meta(jid).get("user_id") != user_id: raise HTTPException(404)
    if _meta(jid)["status"] != "done": raise HTTPException(409, "Project is not ready yet.")
    root, z = f"fabric Now {jid[:6]}", JOBS / jid / "export.zip"
    if not z.exists():
        tmp = z.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted((JOBS / jid / "export").rglob("*")):
                if f.is_file(): zf.write(f, f"{root}/{f.relative_to(JOBS / jid / 'export')}")
        tmp.replace(z)
    return FileResponse(z, media_type="application/zip", filename=f"{root}.zip")


@app.delete("/api/jobs/{jid}", status_code=204)
def delete_job(request: Request, jid: str):
    _internal_auth(request)
    user_id = _user_id(request)
    if _valid(jid):
        try:
            if _meta(jid).get("user_id") != user_id: raise HTTPException(404, "Project not found")
            shutil.rmtree(JOBS / jid, ignore_errors=True)
        except HTTPException: raise
