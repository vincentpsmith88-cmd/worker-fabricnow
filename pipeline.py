"""Fabric Now AI pattern pipeline.

Photo -> vision analysis -> AI technical pattern sheet -> validation -> piece extraction -> ZIP.
OpenAI is used for image generation. Vision analysis can use OpenAI or Anthropic Claude.
"""
import base64, io, json, logging, os, re, urllib.request, urllib.error
from pathlib import Path

import cv2
import openai
import numpy as np
from PIL import Image, ImageDraw, ImageOps

log = logging.getLogger("fabricnow")

ANALYZE = """You are an expert garment pattern engineer and African fashion construction analyst.
Analyze THIS photograph, not a generic garment. Automatically determine the garment type and regional/style family from what is visibly supported. Do not require the user to select a garment type.

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
- Identify EVERY visible/necessary fabric piece for THIS garment: bodices, panels, sleeves, collars, cuffs, pockets, waistbands, belts, ties, facings, yokes, peplums, skirt/trouser panels and other construction pieces.
- The geometry is NOT an illustration. It is a 2D cutting-pattern approximation expressed in millimetres.
- Each piece has a local origin at its top-left. points must be ordered clockwise around the outer cut boundary.
- Use enough points to describe curves accurately (typically 16-60 points for curved pieces). Do not use a crude rectangle for a curved garment piece.
- Keep every coordinate between 0 and width_mm/height_mm.
- width_mm and height_mm must tightly bound the points.
- holes contains closed point lists only when the photo supports an actual cut-out/hole.
- grainline is optional; use it when the garment construction supports a clear grain direction.
- Estimate geometry from visible proportions and recognized garment construction. If physical scale cannot be established from the image, say so with scale_basis=unknown and keep geometry internally consistent. Never claim production-grade physical sizing from a single photo.
- Do not invent hidden pieces unless required by the visible construction.
- Mirrored pairs use one piece with Cut 2 when appropriate. Centre pieces can use Cut 1 on fold.
- African scope is continental: consider West, East, Central, North and Southern African garment traditions when visibly supported. Do not default to Ghanaian.
- swatch must be a normalized clean-fabric crop when possible.
{lining}
"""

PROMPT = """You are a professional pattern drafter and textile designer.
Image 1 is the garment photo{garment}.
{swatch}
Create ONE flat 2D technical pattern sheet containing exactly these sewing pieces, one isolated shape per line:
{pieces}

Construction:
- Match the photographed garment silhouette, proportions, panel divisions, pleats/gathers, sleeves, neckline, collars, belts and closures.
- Use realistic flat pattern-piece silhouettes, not a fashion illustration.
- Keep each piece completely separated from every other piece.
- Pieces are upright and straight-on; do not rotate them.
- Use a tidy grid with a 4% clear margin.
- No mannequin, person, body, hands, shadows or seam markings.

Fabric:
- Preserve the actual garment's visible print, colours and motif character.
- If Image 2 exists, Image 2 is the exact fabric reference. Reproduce its motif family, colours and approximate scale consistently across every piece.
- Never use solid black as a substitute for fabric.

Labels:
- Under each piece write ONLY its exact label from the supplied list.
- Labels must be readable and must not touch the pattern shape.
- Do not add title blocks, rulers, dimensions or extra pieces.

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

def _openai_analyze(client, model, original_png, garment, lining):
    hint = f" (the requested type is {garment.lower()})" if garment and garment != "Auto-detect" else ""
    text = ANALYZE.format(
        hint=hint,
        lining="- Include lining pieces when applicable.\n" if lining not in (False, "none") else "- Do not list lining pieces.\n"
    )
    url = "data:image/png;base64," + base64.b64encode(original_png).decode()
    r = client.chat.completions.create(
        model=model, response_format={"type":"json_object"},
        messages=[{"role":"user","content":[
            {"type":"text","text":text},
            {"type":"image_url","image_url":{"url":url,"detail":"high"}}
        ]}]
    )
    return json.loads(r.choices[0].message.content)

def _claude_analyze(original_png, garment, lining):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not configured.")
    hint = f" (the requested type is {garment.lower()})" if garment and garment != "Auto-detect" else ""
    text = ANALYZE.format(
        hint=hint,
        lining="- Include lining pieces when applicable.\n" if lining not in (False, "none") else "- Do not list lining pieces.\n"
    )
    payload = {
        "model": os.getenv("ANTHROPIC_VISION_MODEL","claude-sonnet-4-5"),
        "max_tokens": 5000,
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

def normalize_plan(d):
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
        pieces.append({
            "name":str(p["name"])[:40], "cut":str(p.get("cut","Cut 1"))[:40],
            "notes":str(p.get("notes",""))[:160], "width_mm":round(w,2), "height_mm":round(h,2),
            "points":pts, "holes":holes, "grainline":grain
        })
    pieces=pieces[:30]
    if not pieces: raise ValueError("AI could not identify mathematically usable garment geometry.")
    return {
        "garment":str(d.get("garment","Unknown garment"))[:120],
        "region_family":str(d.get("region_family") or "")[:100],
        "style_family":str(d.get("style_family") or "")[:100],
        "scale_basis":str(d.get("scale_basis") or "unknown")[:80],
        "geometry_confidence":max(0.0,min(1.0,float(d.get("geometry_confidence",0.0) or 0.0))),
        "pieces":pieces, "swatch":d.get("swatch")
    }

def analyze(client, model, original_png: bytes, garment, lining):
    provider=os.getenv("AI_VISION_PROVIDER","auto").lower()
    if provider=="claude" or (provider=="auto" and os.getenv("ANTHROPIC_API_KEY")):
        try:
            return normalize_plan(_claude_analyze(original_png,garment,lining))
        except Exception as e:
            if provider=="claude": raise
            log.warning("Claude geometry analysis unavailable, falling back to OpenAI: %s",e)
    if client is None:
        raise ValueError("OpenAI vision fallback is unavailable because OPENAI_API_KEY is not configured.")
    return normalize_plan(_openai_analyze(client, model, original_png, garment, lining))

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

def _render_geometry(plan, swatch_bytes=None):
    """Deterministically render Claude's measured geometry. No image model invents the final boundaries."""
    S=float(os.getenv("PATTERN_PX_PER_MM","2"))
    margin_mm=30; gap_mm=35; label_mm=18
    cols=[]; x=margin_mm; y=margin_mm; row_h=0; max_w=0
    for p in plan["pieces"]:
        w=p["width_mm"]+gap_mm; h=p["height_mm"]+label_mm+gap_mm
        if x>margin_mm and x+w>900: x=margin_mm; y+=row_h; row_h=0
        p["sheet_x_mm"]=x; p["sheet_y_mm"]=y; x+=w; row_h=max(row_h,h); max_w=max(max_w,x)
    sheet_w=max(1000,max_w+margin_mm); sheet_h=max(800,y+row_h+margin_mm)
    img=Image.new("RGBA",(int(sheet_w*S),int(sheet_h*S)),(255,255,255,255))
    texture=None
    if swatch_bytes:
        try: texture=Image.open(io.BytesIO(swatch_bytes)).convert("RGB")
        except Exception: texture=None
    for idx,p in enumerate(plan["pieces"],1):
        ox,oy=p["sheet_x_mm"],p["sheet_y_mm"]
        poly=[(int((q["x"]+ox)*S),int((q["y"]+oy)*S)) for q in p["points"]]
        mask=Image.new("L",img.size,0); ImageDraw.Draw(mask).polygon(poly,fill=255)
        if texture:
            tw=max(1,int(p["width_mm"]*S)); th=max(1,int(p["height_mm"]*S)); tile=ImageOps.fit(texture,(tw,th),method=Image.Resampling.LANCZOS)
            layer=Image.new("RGBA",img.size,(255,255,255,0)); layer.paste(tile,(int(ox*S),int(oy*S)),None); img=Image.composite(layer,img,mask)
        else:
            ImageDraw.Draw(img).polygon(poly,fill=(238,238,238,255))
        dr=ImageDraw.Draw(img); dr.line(poly+[poly[0]],fill=(20,20,20,255),width=max(2,int(S)))
        if p.get("grainline"):
            g=p["grainline"]; dr.line([(int((g["x1"]+ox)*S),int((g["y1"]+oy)*S)),(int((g["x2"]+ox)*S),int((g["y2"]+oy)*S))],fill=(40,40,40,255),width=max(1,int(S*.7)))
        dr.text((int(ox*S),int((oy+p["height_mm"]+7)*S)),f"{idx:02d} {p['name']}",fill=(0,0,0,255))
    return img, S

def _write_piece_assets(out, plan, swatch_bytes=None):
    S=float(os.getenv("PATTERN_PX_PER_MM","2")); (out/"svg_outline").mkdir(parents=True,exist_ok=True); (out/"svg_full_print").mkdir(parents=True,exist_ok=True)
    texture=None
    if swatch_bytes:
        try: texture=Image.open(io.BytesIO(swatch_bytes)).convert("RGB")
        except Exception: texture=None
    pieces=[]
    for k,p in enumerate(plan["pieces"],1):
        w,h=p["width_mm"],p["height_mm"]; W=max(1,int(w*S)); H=max(1,int(h*S));
        mask=Image.new("L",(W,H),0); ImageDraw.Draw(mask).polygon([(int(q["x"]*S),int(q["y"]*S)) for q in p["points"]],fill=255)
        if texture: tile=ImageOps.fit(texture,(W,H),method=Image.Resampling.LANCZOS)
        else: tile=Image.new("RGB",(W,H),(238,238,238))
        crop=tile.convert("RGBA"); crop.putalpha(mask)
        name=f"piece_{k:02d}"; crop.save(out/f"{name}.png")
        path=_svg_path(p["points"],scale=S)
        hdr=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><path d="{path}" fill="none" fill-rule="evenodd" stroke="black" stroke-width="2"/>'
        if p.get("grainline"):
            g=p["grainline"]; hdr+=f'<line x1="{g["x1"]*S:.3f}" y1="{g["y1"]*S:.3f}" x2="{g["x2"]*S:.3f}" y2="{g["y2"]*S:.3f}" stroke="black" stroke-width="1" stroke-dasharray="8 6"/>'
        (out/"svg_outline"/f"{name}_outline.svg").write_text(hdr+"</svg>")
        if texture:
            b=io.BytesIO(); crop.save(b,"PNG"); href="data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
            (out/"svg_full_print"/f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><image width="{W}" height="{H}" href="{href}"/><path d="{path}" fill="none" stroke="black" stroke-width="1"/></svg>')
        else:
            (out/"svg_full_print"/f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><path d="{path}" fill="#eeeeee" stroke="black" stroke-width="1"/></svg>')
        pieces.append({"id":name,"file":f"{name}.png","width_mm":w,"height_mm":h,"name":p["name"],"cut":p["cut"],"notes":p["notes"],"geometry_points":p["points"],"grainline":p.get("grainline")})
    return pieces

def export_all(sheet, out: Path, original_png: bytes, plan=None, original_name="ORIGINAL PICTURE.png", swatch_bytes=None):
    if not plan or not plan.get("pieces"): raise ValueError("No AI geometry plan was produced.")
    out.mkdir(parents=True,exist_ok=True); (out/"ORIGINAL PICTURE.png").write_bytes(original_png)
    rendered,S=_render_geometry(plan,swatch_bytes)
    rgba=np.array(rendered); _png(rgba).startswith(b"\x89PNG")
    rendered.save(out/"00_full_transparent.png","PNG")
    rendered.convert("RGB").save(out/"Fabric Now 1.png")
    pieces=_write_piece_assets(out,plan,swatch_bytes)
    manifest={
      "format":"fabricnow-basic-pattern-v3-mathematical-geometry",
      "source_image":"ORIGINAL PICTURE.png",
      "technical_sheet":"Fabric Now 1.png",
      "transparent_sheet":"00_full_transparent.png",
      "garment":plan.get("garment"),"region_family":plan.get("region_family"),"style_family":plan.get("style_family"),
      "scale_basis":plan.get("scale_basis"),"geometry_confidence":plan.get("geometry_confidence"),
      "pattern_px_per_mm":S,"pieces":pieces,
      "svg":{"outline":"svg_outline","full_print":"svg_full_print"},
      "accuracy_note":"Geometry is AI-inferred from the supplied photograph and rendered deterministically from numeric coordinates. A single photo cannot establish true physical scale without a measurement or scale reference; verify physical sizing before production."
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"README.txt").write_text("Fabric Now Pattern export.\n\nClaude/OpenAI vision determines the garment type and numeric 2D geometry. The worker renders those coordinates deterministically into SVG and PNG; the final boundaries are not drawn by an image-generation model.\n\nPhysical dimensions inferred from one photograph are estimates unless a scale/measurement reference is supplied. Verify fit and production measurements before cutting.\n")
    return pieces
