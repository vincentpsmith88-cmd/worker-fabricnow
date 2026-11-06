"""Fabric Now AI pattern pipeline.

Photo -> vision analysis -> numeric pattern geometry -> deterministic SVG/PNG rendering -> ZIP.
Vision analysis can use OpenAI or Anthropic Claude. Image-generation models are not used to draw production pattern boundaries.
"""
import base64, io, json, logging, os, re, urllib.request, urllib.error
from pathlib import Path

import cv2
import openai
import numpy as np
from PIL import Image, ImageDraw, ImageOps

log = logging.getLogger("fabricnow")

ANALYZE = """You are an expert garment pattern engineer and African fashion construction analyst.
Analyze THIS photograph, not a generic garment. Automatically determine the garment type and regional/style family from what is visibly supported. Do not require the user to select a garment type.[[HINT]]

Return ONLY valid JSON with this exact structure:
{
  "garment":"exact inferred garment type",
  "region_family":"African regional/style family when supported, otherwise null",
  "style_family":"specific style",
  "scale_basis":"photo_inferred_standard_body|photo_inferred_garment|unknown",
  "geometry_confidence":0.0,
  "pieces":[
    {"name":"Front Bodice","cut":"Cut 1","notes":"...","width_mm":420,"height_mm":520,"points":[{"x":0,"y":0},{"x":420,"y":0},{"x":400,"y":520},{"x":0,"y":520}],"holes":[],"grainline":{"x1":210,"y1":40,"x2":210,"y2":480}}
  ],
  "swatch":{"x":0.0,"y":0.0,"w":0.2,"h":0.2}
}

GEOMETRY RULES:
- Identify all separate fabric components required to construct the photographed garment, including mirrored pairs, facings, waistbands, cuffs, ruffles, collars and sleeve pieces where visible/necessary.
- Treat this as a pattern-drafting task, never an illustration task. Return 2D cutting-piece outlines as numeric coordinates in millimetres.
- Draft each piece as its own complete closed contour, with a local origin at its top-left and points ordered clockwise.
- Curves are essential: use 24-80 points for armholes, necklines, sleeve caps, crotch curves, curved hems and shaped panels. Never simplify a curved piece to a rectangle, capsule or generic blob.
- Reproduce garment-specific proportions: shoulder slope, armhole depth, waist shaping, dart intake, sleeve cap, flare, gathers and hem shape.
- Coordinates must remain within width_mm and height_mm; those dimensions must tightly bound the contour.
- Avoid self-intersections, duplicate points, zero-area shapes and disconnected contours. Each piece must be a single manufacturable 2D cut shape.
- Use holes only for real cut-outs. Include grainline when construction supports it.
- Identify the full set of pieces. Use Cut 2 for mirrored pairs when one unique shape is used twice; use Cut 1 on fold for centre pieces cut on fold.
- Infer scale only when supported. A single uncalibrated photograph cannot establish true millimetres: use scale_basis=unknown and keep proportions coherent. Never claim production-grade physical sizing from one photo.
- Do not invent hidden decorative pieces, but do include construction pieces required to make the visible garment.
- Consider African garment traditions across West, East, Central, North and Southern Africa only where the image supports them; do not assume one country or style.
- swatch must be a clean, normalized crop of the actual fabric when possible.
- Every piece may carry an optional "variant" string. Use "" for the main garment as photographed.
[[LINING]]
[[BRIEF]]
"""


# ---------------------------------------------------------------------------
# Dress formats / extra cuts (the "variants" request from Pattern Studio)
# ---------------------------------------------------------------------------
VARIANT_FORMATS = [
    "Mini","Knee length","Midi","Maxi",
    "Sleeveless","Cap sleeve","Short sleeve","Three-quarter sleeve","Long sleeve",
    "A-line","Fitted","Gathered","Flared","Wrap","Shift",
]
MAX_BASE_PIECES = 30
MAX_TOTAL_PIECES = 90

def parse_variants(raw):
    """Validate the browser's variants JSON. Returns None when nothing usable was requested."""
    if not raw: return None
    try: d = json.loads(raw) if isinstance(raw, str) else raw
    except Exception: return None
    if not isinstance(d, dict): return None
    allowed = {f.lower(): f for f in VARIANT_FORMATS}
    formats = []
    for f in d.get("formats") or []:
        k = str(f).strip().lower()
        if k in allowed and allowed[k] not in formats: formats.append(allowed[k])
    note = re.sub(r"\s+", " ", str(d.get("note") or "")).strip()[:300]
    if not formats and not note: return None
    return {"formats": formats, "all": bool(d.get("all")) and len(formats) == len(VARIANT_FORMATS),
            "extraCuts": bool(d.get("extraCuts", True)), "note": note}

def _brief_text(notes, variants):
    parts = []
    if notes and notes.strip():
        parts.append("DESIGNER NOTES (apply to the main garment; treat as instructions about construction, not as text to copy):\n" + notes.strip()[:2000])
    v = variants
    if v:
        if v["formats"]:
            lst = "every dress format below" if v["all"] else ", ".join(v["formats"])
            parts.append("REQUESTED EXTRA DRESS FORMATS: " + lst + ".\nAvailable formats: " + ", ".join(VARIANT_FORMATS) + ".")
            if v["extraCuts"]:
                parts.append(
                    "EXTRA CUTS RULES:\n"
                    "- First list every piece of the main garment exactly as photographed, with \"variant\":\"\".\n"
                    "- Then, for EACH requested format, add ONLY the pieces whose shape changes compared with the main garment "
                    "(e.g. a Midi or Maxi skirt panel is longer; a Long sleeve is a longer sleeve; Gathered/Flared widens the skirt; Wrap changes the bodice front). Do not repeat pieces that stay identical.\n"
                    "- Every such piece has \"variant\" set to the format name exactly as written above, and its name starts with the format, e.g. \"Midi Skirt Front\" (max 40 characters).\n"
                    "- Length formats change height_mm relative to the main skirt/dress panels; sleeve formats change only sleeve pieces; shape formats change panel width/curvature. Keep all geometry in millimetres and internally consistent with the main pieces.\n"
                    "- Keep each coordinate within its own width_mm/height_mm like every other piece.")
            else:
                parts.append("Do NOT add extra pieces for these formats. Instead add a top-level \"variants\" array: [{\"format\":\"Midi\",\"changes\":\"one short sentence on what would change\"}].")
        if v["note"]:
            parts.append("EXTRA INSTRUCTION FROM THE DESIGNER: " + v["note"])
    return "\n\n".join(parts)

def _analyze_text(garment, lining, notes="", variants=None):
    hint = f" (the requested type is {garment.lower()})" if garment and garment != "Auto-detect" else ""
    lining_rule = "- Include lining pieces when applicable.\n" if lining not in (False, "none") else "- Do not list lining pieces.\n"
    return (ANALYZE.replace("[[HINT]]", hint).replace("[[LINING]]", lining_rule)
                   .replace("[[BRIEF]]", _brief_text(notes, variants)))

PROMPT = """You are a professional pattern drafter and textile designer.
Image 1 is the garment photo{garment}.
{swatch}
Create a clean 2D technical pattern sheet with exactly the supplied sewing pieces, each shown as a separate, complete cut shape:
{pieces}

Drafting quality:
- Follow the garment's actual silhouette, construction, panel divisions, pleats/gathers, sleeve shape, neckline, collar, cuffs, waistband, ruffles and closures.
- Every piece must be a complete, recognizable flat pattern shape with smooth accurate curves, not a rectangle, rough crop or fashion illustration.
- Keep every piece fully visible, separated, upright and straight-on. No overlap, clipping, rotation or perspective.
- Arrange in a balanced grid with consistent spacing and generous margins.
- Use a plain white background. No checkerboard, mannequin, person, hands, shadows, decorative props or fake transparency.

Fabric:
- Preserve the actual garment print, colour palette and motif character.
- If Image 2 exists, use it as the exact fabric reference. Keep the same motif scale and repeat density across all pattern pieces; do not independently stretch the print to fit each shape.
- Never substitute solid black or unrelated fabric.

Labels:
- Place the exact supplied label under each piece, readable and separate from the shape.
- Do not add title blocks, rulers, dimensions, extra pieces or decorative text.

{lining_rule}
{notes}"""

def build_prompt(plan, garment: str, notes: str, lining, has_swatch: bool) -> str:
    g = f" ({plan['garment']})" if plan and plan.get("garment") else (
        f" (a {garment.lower()})" if garment and garment != "Auto-detect" else "")
    pieces = "\n".join(
        f'{i}. "{p["name"]} ({p["cut"]})"' for i, p in enumerate(plan["pieces"], 1)
    ) if plan else 'Every piece needed to construct the exact garment.'
    swatch = ("Image 2 is a close-up of the exact fabric. Reproduce THIS fabric faithfully.\n"
              if has_swatch else "")
    lining_rule = ("Include lining pieces only when they are explicitly in the plan."
                   if lining in (True, "partial", "full") else
                   "Do not invent lining pieces.")
    n = f"Designer notes: {notes.strip()[:500]}" if notes.strip() else ""
    return PROMPT.format(garment=g, swatch=swatch, pieces=pieces,
                         lining_rule=lining_rule, notes=n)

def _openai_analyze(client, model, original_png, garment, lining, notes="", variants=None):
    text = _analyze_text(garment, lining, notes, variants)
    url = "data:image/png;base64," + base64.b64encode(original_png).decode()
    r = client.chat.completions.create(
        model=model, response_format={"type":"json_object"},
        messages=[{"role":"user","content":[
            {"type":"text","text":text},
            {"type":"image_url","image_url":{"url":url,"detail":"high"}}
        ]}]
    )
    return json.loads(r.choices[0].message.content)

def _claude_analyze(original_png, garment, lining, notes="", variants=None):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not configured.")
    text = _analyze_text(garment, lining, notes, variants)
    payload = {
        "model": os.getenv("ANTHROPIC_VISION_MODEL","claude-sonnet-4-5"),
        "max_tokens": 16000 if (variants and variants.get("formats") and variants.get("extraCuts")) else 5000,
        "system": "Return valid JSON only. You are an expert garment construction and African fashion pattern analyst.",
        "messages":[{"role":"user","content":[
            {"type":"text","text":text},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":base64.b64encode(original_png).decode()}}
        ]}]
    }
    req=urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as res:
        d=json.loads(res.read())
    txt="\n".join(x.get("text","") for x in d.get("content",[]) if x.get("type")=="text")
    txt=re.sub(r"^```json\s*|\s*```$","",txt.strip(),flags=re.I)
    return json.loads(txt)

def _clean_points(points, width, height):
    out=[]
    for pt in points or []:
        try:
            x=float(pt.get("x")); y=float(pt.get("y"))
        except Exception:
            continue
        if 0 <= x <= width and 0 <= y <= height:
            out.append({"x":round(x,2),"y":round(y,2)})
    return out

def normalize_plan(d, variants=None):
    if not isinstance(d,dict): raise ValueError("AI returned an invalid garment analysis.")
    pieces=[]
    for p in d.get("pieces",[]) if isinstance(d.get("pieces"),list) else []:
        if not p.get("name"): continue
        try:
            w=float(p.get("width_mm",0)); h=float(p.get("height_mm",0))
        except Exception: w=h=0
        pts=_clean_points(p.get("points"),w,h) if w>0 and h>0 else []
        if len(pts)<3:
            continue
        holes=[]
        for hole in p.get("holes",[]) or []:
            hp=_clean_points(hole,w,h)
            if len(hp)>=3: holes.append(hp)
        grain=p.get("grainline") or None
        if grain:
            try:
                grain={k:round(float(grain[k]),2) for k in ("x1","y1","x2","y2")}
            except Exception: grain=None
        variant=str(p.get("variant") or "").strip()[:40]
        if variant and variants and variants.get("formats"):
            low={f.lower():f for f in VARIANT_FORMATS}
            variant=low.get(variant.lower(),"")  # unknown format names are treated as main pieces
        elif not (variants and variants.get("formats")): variant=""
        pieces.append({
            "variant":variant,
            "name":str(p["name"])[:40], "cut":str(p.get("cut","Cut 1"))[:40],
            "notes":str(p.get("notes",""))[:160], "width_mm":round(w,2), "height_mm":round(h,2),
            "points":pts, "holes":holes, "grainline":grain
        })
    main=[x for x in pieces if not x["variant"]][:MAX_BASE_PIECES]
    extra=[x for x in pieces if x["variant"]][:MAX_TOTAL_PIECES-len(main)]
    pieces=main+extra
    if not pieces: raise ValueError("AI could not identify mathematically usable garment geometry.")
    return {
        "garment":str(d.get("garment","Unknown garment"))[:120],
        "region_family":str(d.get("region_family") or "")[:100],
        "style_family":str(d.get("style_family") or "")[:100],
        "scale_basis":str(d.get("scale_basis") or "unknown")[:80],
        "geometry_confidence":max(0.0,min(1.0,float(d.get("geometry_confidence",0.0) or 0.0))),
        "pieces":pieces, "swatch":d.get("swatch"),
        "variants_requested":variants,
        "variant_notes":[{"format":str(v.get("format",""))[:40],"changes":str(v.get("changes",""))[:200]}
                         for v in (d.get("variants") or []) if isinstance(v,dict)][:20],
    }

def analyze(client, model, original_png: bytes, garment, lining, notes="", variants=None):
    provider=os.getenv("AI_VISION_PROVIDER","auto").lower()
    if provider=="claude" or (provider=="auto" and os.getenv("ANTHROPIC_API_KEY")):
        try:
            return normalize_plan(_claude_analyze(original_png,garment,lining,notes,variants),variants)
        except Exception as e:
            if provider=="claude": raise
            log.warning("Claude geometry analysis unavailable, falling back to OpenAI: %s",e)
    if client is None:
        raise ValueError("OpenAI vision fallback is unavailable because OPENAI_API_KEY is not configured.")
    return normalize_plan(_openai_analyze(client, model, original_png, garment, lining, notes, variants),variants)

def variant_report(plan):
    """What the engine actually delivered versus what was requested. Never claims more than exists."""
    v=plan.get("variants_requested") or {}
    want=v.get("formats") or []
    got=sorted({p["variant"] for p in plan["pieces"] if p.get("variant")}, key=lambda f: VARIANT_FORMATS.index(f))
    missing=[f for f in want if f not in got]
    return {"requested":want,"delivered":got,"missing":missing,
            "extra_cut_pieces":sum(1 for p in plan["pieces"] if p.get("variant")),
            "extra_cuts":bool(v.get("extraCuts",True))}
def crop_swatch(original_png: bytes, box):
    try:
        im=Image.open(io.BytesIO(original_png)).convert("RGB"); W,H=im.size
        x,y,w,h=[float(box[k]) for k in "xywh"]
        x0,y0,x1,y1=int(max(x,0)*W),int(max(y,0)*H),int(min(x+w,1)*W),int(min(y+h,1)*H)
    except (TypeError,KeyError,ValueError): return None
    if x1-x0<96 or y1-y0<96: return None
    b=io.BytesIO(); im.crop((x0,y0,x1,y1)).save(b,"PNG"); return b.getvalue()

def _alpha_ok(model): return model.startswith("gpt-image-1")

def generate_sheet(client, model, size, quality, images, prompt, fallback=""):
    files=[(f"image{i}.png",b,"image/png") for i,b in enumerate(images,1)]
    def call(m):
        kw=dict(model=m,image=files,prompt=prompt,size=size,quality=quality,n=1)
        if _alpha_ok(m): kw.update(background="transparent",output_format="png")
        else: kw["prompt"]=prompt.replace("Transparent background.","Plain pure white background (#FFFFFF), no texture, gradient or shadow.")
        return client.images.edit(**kw)
    try: res=call(model)
    except (openai.NotFoundError,openai.PermissionDeniedError):
        if not fallback or fallback==model: raise
        res=call(fallback)
    return ensure_transparent(Image.open(io.BytesIO(base64.b64decode(res.data[0].b64_json))).convert("RGBA"))

def check(sheet, planned):
    a=np.array(sheet)[...,3]>24
    clipped=bool(a[:3].any() or a[-3:].any() or a[:,:3].any() or a[:,-3:].any())
    H,W=a.shape
    _,_,st,_=cv2.connectedComponentsWithStats(a.astype(np.uint8),8)
    count=int((st[1:,4]>=max(500,int(.0005*H*W))).sum())
    return (not clipped and (planned==0 or count>=.75*planned)),count

def ensure_transparent(img):
    a=np.array(img)
    if a[...,3].min()<250: return img
    white=(a[...,:3].min(axis=2)>=238).astype(np.uint8)
    n,lab=cv2.connectedComponents(white,4)
    border=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    a[np.isin(lab,list(border)),3]=0
    return Image.fromarray(a)

def _png(arr):
    b=io.BytesIO(); Image.fromarray(arr).save(b,"PNG",optimize=True); return b.getvalue()

def _reading_order(items,height):
    items.sort(key=lambda p:p["cy"]); rows=[]
    for p in items:
        if rows and abs(p["cy"]-np.mean([q["cy"] for q in rows[-1]]))<height*.15: rows[-1].append(p)
        else: rows.append([p])
    return [p for r in rows for p in sorted(r,key=lambda p:p["cx"])]

def _svg_path(points, ox=0, oy=0, scale=1.0):
    if not points: return ""
    return "M " + " ".join(f"{(p['x']+ox)*scale:.3f},{(p['y']+oy)*scale:.3f}" for p in points) + " Z"

def _texture_plane(texture, size, repeat_px=260):
    """Build one consistent fabric-repeat plane so every piece shares the same motif scale."""
    if texture is None:
        return None
    tw = th = max(96, int(repeat_px))
    tile = ImageOps.fit(texture.convert("RGB"), (tw, th), method=Image.Resampling.LANCZOS)
    # Mirror alternate tiles to reduce obvious seams when the supplied swatch is not tileable.
    mirror_x = ImageOps.mirror(tile)
    mirror_y = ImageOps.flip(tile)
    mirror_xy = ImageOps.flip(mirror_x)
    variants = ((tile, mirror_x), (mirror_y, mirror_xy))
    plane = Image.new("RGB", size, (255, 255, 255))
    for row, y in enumerate(range(0, size[1], th)):
        for col, x in enumerate(range(0, size[0], tw)):
            plane.paste(variants[row % 2][col % 2], (x, y))
    return plane

def _pattern_layout(plan):
    S = float(os.getenv("PATTERN_PX_PER_MM", "3"))
    margin_mm, gap_mm, label_mm = 30, 35, 24
    max_width_mm = float(os.getenv("PATTERN_SHEET_WIDTH_MM", "900"))
    widest_piece = max((p["width_mm"] for p in plan["pieces"]), default=0)
    max_width_mm = max(max_width_mm, widest_piece + margin_mm * 2)
    x = y = margin_mm
    row_h = 0
    max_right = margin_mm
    for p in plan["pieces"]:
        w, h = p["width_mm"], p["height_mm"]
        if x > margin_mm and x + w > max_width_mm - margin_mm:
            x = margin_mm
            y += row_h
            row_h = 0
        p["sheet_x_mm"], p["sheet_y_mm"] = x, y
        x += w + gap_mm
        row_h = max(row_h, h + label_mm + gap_mm)
        max_right = max(max_right, x)
    sheet_w_mm = max(300, min(max_width_mm, max_right + margin_mm))
    sheet_h_mm = max(300, y + row_h + margin_mm)
    return S, (int(sheet_w_mm * S), int(sheet_h_mm * S))

def _render_geometry(plan, swatch_bytes=None):
    """Render model-proposed numeric geometry deterministically; AI never paints the cut boundaries."""
    S, sheet_size = _pattern_layout(plan)
    # Transparent master sheet; the human-facing technical sheet is composited onto white below.
    img = Image.new("RGBA", sheet_size, (255, 255, 255, 0))
    texture = None
    if swatch_bytes:
        try:
            texture = Image.open(io.BytesIO(swatch_bytes)).convert("RGB")
        except Exception:
            texture = None
    repeat_px = int(os.getenv("FABRIC_REPEAT_PX", "260"))
    fabric_plane = _texture_plane(texture, sheet_size, repeat_px) if texture else None
    for idx, p in enumerate(plan["pieces"], 1):
        ox, oy = p["sheet_x_mm"], p["sheet_y_mm"]
        x0, y0 = int(ox * S), int(oy * S)
        W, H = max(1, int(p["width_mm"] * S)), max(1, int(p["height_mm"] * S))
        poly = [(int((q["x"] + ox) * S), int((q["y"] + oy) * S)) for q in p["points"]]
        mask = Image.new("L", sheet_size, 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        if fabric_plane:
            layer = Image.new("RGBA", sheet_size, (255, 255, 255, 0))
            layer.paste(fabric_plane.crop((x0, y0, x0 + W, y0 + H)).convert("RGBA"), (x0, y0))
            img = Image.composite(layer, img, mask)
        else:
            ImageDraw.Draw(img).polygon(poly, fill=(238, 238, 238, 255))
        dr = ImageDraw.Draw(img)
        dr.line(poly + [poly[0]], fill=(24, 24, 24, 255), width=max(2, int(S)))
        if p.get("grainline"):
            g = p["grainline"]
            dr.line([(int((g["x1"] + ox) * S), int((g["y1"] + oy) * S)),
                     (int((g["x2"] + ox) * S), int((g["y2"] + oy) * S))],
                    fill=(40, 40, 40, 255), width=max(1, int(S * .7)))
        dr.text((x0, int((oy + p["height_mm"] + 7) * S)), f"{idx:02d}  {p['name']}", fill=(20, 20, 20, 255))
    return img, S

def _write_piece_assets(out, plan, swatch_bytes=None):
    S = float(os.getenv("PATTERN_PX_PER_MM", "3"))
    (out / "svg_outline").mkdir(parents=True, exist_ok=True)
    (out / "svg_full_print").mkdir(parents=True, exist_ok=True)
    texture = None
    if swatch_bytes:
        try:
            texture = Image.open(io.BytesIO(swatch_bytes)).convert("RGB")
        except Exception:
            texture = None
    repeat_px = int(os.getenv("FABRIC_REPEAT_PX", "260"))
    pieces = []
    for k, p in enumerate(plan["pieces"], 1):
        w, h = p["width_mm"], p["height_mm"]
        W, H = max(1, int(w * S)), max(1, int(h * S))
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon([(int(q["x"] * S), int(q["y"] * S)) for q in p["points"]], fill=255)
        # Same fixed repeat size and origin for every asset: motifs no longer balloon or shrink per piece.
        fabric_plane = _texture_plane(texture, (W, H), repeat_px) if texture else None
        tile = fabric_plane if fabric_plane else Image.new("RGB", (W, H), (238, 238, 238))
        crop = tile.convert("RGBA")
        crop.putalpha(mask)
        name = f"piece_{k:02d}"
        crop.save(out / f"{name}.png", optimize=True)
        path = _svg_path(p["points"], scale=S)
        hdr = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><path d="{path}" fill="none" fill-rule="evenodd" stroke="black" stroke-width="2"/>'
        if p.get("grainline"):
            g = p["grainline"]
            hdr += f'<line x1="{g["x1"]*S:.3f}" y1="{g["y1"]*S:.3f}" x2="{g["x2"]*S:.3f}" y2="{g["y2"]*S:.3f}" stroke="black" stroke-width="1" stroke-dasharray="8 6"/>'
        (out / "svg_outline" / f"{name}_outline.svg").write_text(hdr + "</svg>")
        if texture:
            b = io.BytesIO(); crop.save(b, "PNG")
            href = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
            (out / "svg_full_print" / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><image width="{W}" height="{H}" href="{href}"/><path d="{path}" fill="none" stroke="black" stroke-width="1"/></svg>')
        else:
            (out / "svg_full_print" / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><path d="{path}" fill="#eeeeee" stroke="black" stroke-width="1"/></svg>')
        pieces.append({"id": name, "file": f"{name}.png", "width_mm": w, "height_mm": h,
                       "name": p["name"], "variant": p.get("variant", ""), "cut": p["cut"],
                       "notes": p["notes"], "geometry_points": p["points"], "grainline": p.get("grainline")})
    return pieces

def export_all(sheet, out: Path, original_png: bytes, plan=None, original_name="ORIGINAL PICTURE.png", swatch_bytes=None):
    if not plan or not plan.get("pieces"): raise ValueError("No AI geometry plan was produced.")
    out.mkdir(parents=True,exist_ok=True); (out/"ORIGINAL PICTURE.png").write_bytes(original_png)
    rendered,S=_render_geometry(plan,swatch_bytes)
    rgba=np.array(rendered); _png(rgba).startswith(b"\x89PNG")
    rendered.save(out/"00_full_transparent.png", "PNG")
    white_sheet = Image.new("RGBA", rendered.size, (255, 255, 255, 255))
    white_sheet.alpha_composite(rendered)
    white_sheet.convert("RGB").save(out/"Fabric Now 1.png", "PNG", optimize=True)
    pieces=_write_piece_assets(out,plan,swatch_bytes)
    manifest={
      "format":"fabricnow-basic-pattern-v3-mathematical-geometry",
      "source_image":"ORIGINAL PICTURE.png",
      "technical_sheet":"Fabric Now 1.png",
      "transparent_sheet":"00_full_transparent.png",
      "garment":plan.get("garment"),"region_family":plan.get("region_family"),"style_family":plan.get("style_family"),
      "scale_basis":plan.get("scale_basis"),"geometry_confidence":plan.get("geometry_confidence"),
      "pattern_px_per_mm":S,"pieces":pieces,
      "variants":variant_report(plan) if plan.get("variants_requested") else None,
      "variant_notes":plan.get("variant_notes") or [],
      "svg":{"outline":"svg_outline","full_print":"svg_full_print"},
      "accuracy_note":"Geometry is AI-inferred from the supplied photograph and rendered deterministically from numeric coordinates. A single photo cannot establish true physical scale without a measurement or scale reference; verify physical sizing before production."
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"README.txt").write_text("Fabric Now Pattern export.\n\nClaude/OpenAI vision determines the garment type and numeric 2D geometry. The worker renders those coordinates deterministically into SVG and PNG; the final boundaries are not drawn by an image-generation model.\n\nPhysical dimensions inferred from one photograph are estimates unless a scale/measurement reference is supplied. Verify fit and production measurements before cutting.\n")
    return pieces
