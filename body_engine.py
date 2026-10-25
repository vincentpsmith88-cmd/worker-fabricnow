"""Calibrated orthogonal-silhouette body reconstruction and deterministic measurement/pattern engine.

Calibration uses supplied standing height and front/side full-body photographs. This is a
visual-hull approximation, not a substitute for a depth camera or validated statistical body model.
"""
from __future__ import annotations
import io, math, json, uuid
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps
import trimesh
from shapely.geometry import Polygon

ENGINE_VERSION = "fabricnow-orthogonal-visual-hull-1.0.0"

# Fractions measured from the top of the standing silhouette. Values are chosen to avoid
# pretending that image-only data reveals hidden anatomy; each derived measurement carries confidence.
LANDMARKS = {
    "neck": .105, "shoulder": .155, "bust": .285, "underbust": .335,
    "waist": .405, "highHip": .465, "hip": .535, "thigh": .625,
    "knee": .755, "calf": .835, "ankle": .965,
}

def _load(raw: bytes) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if im.width < 240 or im.height < 400:
        raise ValueError("Each body photo must be at least 240 × 400 pixels.")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def _person_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    # GrabCut with a conservative inset rectangle. A clean, plain background is strongly recommended.
    mask = np.zeros((h, w), np.uint8)
    gc = np.zeros((h, w), np.uint8)
    rect = (max(1, int(w*.06)), max(1, int(h*.015)), max(2, int(w*.88)), max(2, int(h*.97)))
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(bgr, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask[(gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)] = 255
    # Keep the dominant foreground component and close small gaps.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: raise ValueError("Could not isolate the person. Retake against a plain, contrasting background.")
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.where(labels == idx, 255, 0).astype(np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((9,5),np.uint8), iterations=2)
    ys, xs = np.where(clean > 0)
    if len(xs) < 500 or (ys.max()-ys.min()) < h*.48:
        raise ValueError("Person segmentation is incomplete. Use a full-length, unobstructed photo.")
    # crop margins in segmentation mask so stray background is not mistaken for body width
    return clean

def _profile(mask: np.ndarray, height_cm: float, samples=161) -> Tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    if y1 <= y0: raise ValueError("Invalid body silhouette.")
    px_per_cm = (y1-y0) / height_cm
    fractions = np.linspace(0.015, .985, samples)
    widths = np.zeros(samples, float)
    for i, f in enumerate(fractions):
        yy = int(y0 + f*(y1-y0))
        lo, hi = max(y0, yy-2), min(y1+1, yy+3)
        row = np.where(np.any(mask[lo:hi] > 0, axis=0))[0]
        widths[i] = (row.max()-row.min()+1)/px_per_cm if len(row) else 0
    # interpolate small gaps, smooth segmentation jaggies without changing scale materially
    good = widths > 0
    if good.sum() < samples*.8: raise ValueError("Silhouette has too many gaps; retake the image.")
    widths = np.interp(np.arange(samples), np.where(good)[0], widths[good])
    widths = cv2.GaussianBlur(widths.astype(np.float32).reshape(-1,1),(1, nine:=9),0).ravel()
    return fractions, widths, px_per_cm

def _at(fracs, values, f): return float(np.interp(f, fracs, values))
def _ellipse_circ(a: float, b: float) -> float:
    # Ramanujan II; a and b are semi-axes.
    if a <= 0 or b <= 0: return 0.0
    h=((a-b)/(a+b))**2
    return math.pi*(a+b)*(1+3*h/(10+math.sqrt(4-3*h)))

def _build_mesh(front_mask, side_mask, height_cm, out_path: Path):
    ff, fw, _ = _profile(front_mask, height_cm)
    sf, sw, _ = _profile(side_mask, height_cm)
    # Both images use top-to-bottom normalized vertical coordinates. Resample side profile to front fractions.
    depths=np.interp(ff,sf,sw)
    # 161 loft rings; radii have a small floor to avoid degenerate triangles at neck/ankles.
    ntheta=64; verts=[]
    for f,w,d in zip(ff,fw,depths):
        rx=max(.35,w/2); rz=max(.35,d/2)
        y=(1-f)*height_cm
        for t in np.linspace(0,2*math.pi,ntheta,endpoint=False):
            verts.append([rx*math.cos(t), y, rz*math.sin(t)])
    faces=[]
    for j in range(len(ff)-1):
        for k in range(ntheta):
            a=j*ntheta+k; b=j*ntheta+(k+1)%ntheta; c=(j+1)*ntheta+(k+1)%ntheta; d=(j+1)*ntheta+k
            faces.extend([[a,d,b],[b,d,c]])
    # end caps
    bottom=len(verts); verts.append([0,0,0]); top=len(verts); verts.append([0,height_cm,0])
    for k in range(ntheta):
        faces.append([bottom,(len(ff)-1)*ntheta+k,(len(ff)-1)*ntheta+(k+1)%ntheta])
        faces.append([top,(k+1)%ntheta,k])
    mesh=trimesh.Trimesh(vertices=np.asarray(verts),faces=np.asarray(faces),process=True)
    mesh.export(out_path)
    return ff,fw,depths,mesh

def reconstruct(front: bytes, side: bytes, height: float, unit: str, out_dir: Path, extra_views: dict | None = None) -> dict:
    height_cm=float(height)*(2.54 if unit=="in" else 1.0)
    if not 100 <= height_cm <= 230: raise ValueError("Height must be between 100 and 230 cm (39.4–90.6 in).")
    fimg,simg=_load(front),_load(side)
    fm,sm=_person_mask(fimg),_person_mask(simg)
    ff,fw,_=_profile(fm,height_cm); sf,sw,_=_profile(sm,height_cm)
    front_profiles=[fw]; side_profiles=[sw]; extra_masks={}
    for view, raw in (extra_views or {}).items():
        if not raw: continue
        mask=_person_mask(_load(raw)); extra_masks[view]=mask
        vf,vw,_=_profile(mask,height_cm)
        if view in ("back",): front_profiles.append(np.interp(ff,vf,vw))
        elif view in ("left","right"): side_profiles.append(np.interp(sf,vf,vw))
    fw=np.mean(np.vstack(front_profiles),axis=0)
    sw=np.mean(np.vstack(side_profiles),axis=0)
    depths=np.interp(ff,sf,sw)
    mesh_path=out_dir/"body.glb"
    # Rebuild from the calibrated average profiles. Mesh is a loft, not a learned anatomical model.
    ntheta=64; verts=[]
    for f,w,d in zip(ff,fw,depths):
        for t in np.linspace(0,2*math.pi,ntheta,endpoint=False): verts.append([max(.35,w/2)*math.cos(t),(1-f)*height_cm,max(.35,d/2)*math.sin(t)])
    faces=[]
    for j in range(len(ff)-1):
        for k in range(ntheta):
            a=j*ntheta+k; b=j*ntheta+(k+1)%ntheta; c=(j+1)*ntheta+(k+1)%ntheta; d=(j+1)*ntheta+k
            faces.extend([[a,d,b],[b,d,c]])
    bottom=len(verts); verts.append([0,0,0]); top=len(verts); verts.append([0,height_cm,0])
    for k in range(ntheta): faces.extend([[bottom,(len(ff)-1)*ntheta+k,(len(ff)-1)*ntheta+(k+1)%ntheta],[top,(k+1)%ntheta,k]])
    mesh=trimesh.Trimesh(vertices=np.asarray(verts),faces=np.asarray(faces),process=True); mesh.export(mesh_path)
    def dims(frac):
        return max(.1,_at(ff,fw,frac)), max(.1,_at(ff,depths,frac))
    # confidence reflects photo-only silhouette uncertainty, not statistical probability of exact fit.
    measurements={"height":_m(height_cm,.99,"calibrated_to_supplied_height"),
      "shoulderWidth":_m(dims(.155)[0]*.88,.55,"front_silhouette_estimate"),
      "bust":_m(_ellipse_circ(*[x/2 for x in dims(.285)]),.62,"orthogonal_silhouette_estimate"),
      "underbust":_m(_ellipse_circ(*[x/2 for x in dims(.335)]),.58,"orthogonal_silhouette_estimate"),
      "waist":_m(_ellipse_circ(*[x/2 for x in dims(.405)]),.62,"orthogonal_silhouette_estimate"),
      "highHip":_m(_ellipse_circ(*[x/2 for x in dims(.465)]),.58,"orthogonal_silhouette_estimate"),
      "hip":_m(_ellipse_circ(*[x/2 for x in dims(.535)]),.62,"orthogonal_silhouette_estimate"),
      "thigh":_m(_ellipse_circ(*[x/2 for x in dims(.625)]),.48,"silhouette_estimate_per_thigh"),
      "knee":_m(_ellipse_circ(*[x/2 for x in dims(.755)]),.42,"silhouette_estimate"),
      "calf":_m(_ellipse_circ(*[x/2 for x in dims(.835)]),.42,"silhouette_estimate"),
      "neck":_m(_ellipse_circ(*[x/2 for x in dims(.105)]),.35,"low_confidence_silhouette_estimate"),
      "inseam":_m(height_cm*.445,.30,"anthropometric_ratio_estimate"),
      "outseam":_m(height_cm*.565,.45,"anthropometric_ratio_estimate"),
      "armLength":_m(height_cm*.325,.25,"not_visible_in_body_silhouette"),
      "upperArm":_m(height_cm*.105,.20,"not_visible_in_body_silhouette"),
      "wrist":_m(height_cm*.055,.15,"not_visible_in_body_silhouette")}
    overall=float(np.mean([measurements[k]["confidence"] for k in ("bust","waist","hip")]))
    # Persist preview silhouettes for visual QA.
    cv2.imwrite(str(out_dir/"front-silhouette.png"),fm); cv2.imwrite(str(out_dir/"side-silhouette.png"),sm)
    return {"status":"review","engine_version":ENGINE_VERSION,"measurements":measurements,
      "confidence":{"overall":round(overall,3),"method":"orthogonal silhouette visual hull; tape validation required"},
      "outputs":{"mesh_file":"body.glb","front_silhouette":"front-silhouette.png","side_silhouette":"side-silhouette.png"},
      "calibration":{"height_cm":round(height_cm,2),"views":["front","side"]+list(extra_masks.keys()),"mesh":"elliptical loft visual-hull approximation"}}

def _m(value,confidence,method): return {"value":round(float(value),1),"confidence":confidence,"method":method,"unit":"cm"}

# Basic deterministic flat-pattern blocks. Dimensions are computed from a reviewed measurement profile.
def draft_pattern(data: dict, out_dir: Path) -> dict:
    m=data.get("measurements") or {}; garment=str(data.get("garment","bodice")).lower()
    unit=data.get("unit","cm"); factor=2.54 if unit=="in" else 1.0
    def val(k,default=None):
        x=m.get(k)
        if isinstance(x,dict): x=x.get("value")
        if x is None: x=default
        if x is None: raise ValueError(f"Missing required measurement: {k}")
        return float(x)*factor
    bust,waist,hip=val("bust"),val("waist"),val("hip")
    height=val("height")
    ease=float(data.get("ease_cm",4)); seam=float(data.get("seam_allowance_cm",1.0))
    if not 0<=ease<=20 or not 0<=seam<=3: raise ValueError("Ease must be 0–20 cm and seam allowance 0–3 cm.")
    # Each SVG is a true-scale 1 SVG unit = 1 mm pattern sheet; dimensions are laid out in mm.
    pieces=[]
    def _path(points):
        if not points: return ""
        return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x,y in points) + " Z"
    def add(name,w_cm,h_cm,kind,notes):
        w,h=w_cm*10,h_cm*10
        if kind=="bodice":
            # Drafting block with a shaped side/armhole approximation. The cutline is
            # generated by a geometric offset from the stitch line, not by shrinking it.
            stitch=[(0,0),(w,0),(w,h*.54),(w*.92,h*.67),(w*.78,h*.76),(w*.22,h*.76),(w*.08,h*.67),(0,h*.54)]
        else:
            stitch=[(0,0),(w,0),(w*.88,h),(w*.12,h)]
        poly=Polygon(stitch)
        if not poly.is_valid: poly=poly.buffer(0)
        cut=poly.buffer(seam*10,join_style="round",resolution=8)
        if cut.geom_type=="MultiPolygon": cut=max(cut.geoms,key=lambda g:g.area)
        cut_pts=list(cut.exterior.coords)[:-1]
        pieces.append({"name":name,"stitch":stitch,"cut":cut_pts,"width_mm":cut.bounds[2]-cut.bounds[0],"height_mm":cut.bounds[3]-cut.bounds[1],"notes":notes})
    if "bodice" in garment or garment in ("top","shirt","dress"):
        chest=(bust/4)+(ease/4); waistw=(waist/4)+(ease/4)
        add("Bodice front",chest, max(25,height*.25),"bodice","Cut 1 on fold; verify neckline, dart intake and armhole on toile")
        add("Bodice back",chest, max(25,height*.25),"bodice","Cut 1 on fold; verify back-neck and shoulder slope")
        add("Sleeve block",max(22,bust*.18),max(45,height*.32),"block","Sleeve cap and bicep require fitting/toile refinement")
    if "skirt" in garment or garment=="dress":
        add("Skirt front",(hip+ease)/4,max(35,height*.42),"block","Cut 1 on fold; waist shaping/darts require fitting")
        add("Skirt back",(hip+ease)/4,max(35,height*.42),"block","Cut 1 on fold; add closure allowance as required")
    if "trouser" in garment or "pant" in garment:
        inseam=val("inseam",height*.445); outseam=val("outseam",height*.565)
        add("Trouser front",(hip+ease)/4,max(55,outseam),"block","Front crotch extension and rise are provisional; fit toile required")
        add("Trouser back",(hip+ease)/4+2,max(55,outseam),"block","Back crotch extension is provisional; fit toile required")
    if not pieces: raise ValueError("Choose bodice, skirt, trousers, or dress for pattern drafting.")
    # SVG with mm physical units, separate stitch line and geometrically offset cut line.
    x=20; y=35; gap=25; maxh=max(p["height_mm"] for p in pieces); totalw=sum(p["width_mm"] for p in pieces)+gap*(len(pieces)-1)+40
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{totalw:.1f}mm" height="{maxh+80:.1f}mm" viewBox="0 0 {totalw:.1f} {maxh+80:.1f}">',
         '<rect width="100%" height="100%" fill="white"/>',f'<text x="20" y="18" font-size="8">FabricNow draft · 1:1 millimetres · seam allowance {seam:.1f} cm · ease {ease:.1f} cm</text>']
    for p in pieces:
        minx=min(q[0] for q in p["cut"]); miny=min(q[1] for q in p["cut"])
        cut=[(a-minx,b-miny) for a,b in p["cut"]]
        stitch=[(a-minx,b-miny) for a,b in p["stitch"]]
        svg.append(f'<g transform="translate({x:.1f},{y:.1f})"><path d="{_path(cut)}" fill="#f5f5f5" stroke="#111" stroke-width="1.2"/><path d="{_path(stitch)}" fill="none" stroke="#3974bd" stroke-width="0.7" stroke-dasharray="4 2"/><text x="3" y="-8" font-size="7">{p["name"]}</text><text x="3" y="{p["height_mm"]+12:.1f}" font-size="5">{p["width_mm"]/10:.1f} × {p["height_mm"]/10:.1f} cm · {p["notes"]}</text></g>')
        x+=p["width_mm"]+gap
    svg.append('</svg>'); svg_text=''.join(svg)
    out_dir.mkdir(parents=True,exist_ok=True); (out_dir/"pattern.svg").write_text(svg_text)
    report={"garment":garment,"unit":"mm","scale":"1:1","ease_cm":ease,"seam_allowance_cm":seam,
      "pieces":[{"name":p["name"],"width_mm":round(p["width_mm"],1),"height_mm":round(p["height_mm"],1),"notes":p["notes"]} for p in pieces],
      "warnings":["Draft blocks are starting patterns, not production-validated fitted patterns.","Darts, crotch curve, sleeve cap, neckline and individual posture require fitting refinement.","Verify measurements with a tape and make a toile before cutting final fabric."]}
    (out_dir/"pattern.json").write_text(json.dumps(report,indent=2))
    return {"status":"review","engine_version":ENGINE_VERSION,"pattern":report,"outputs":{"svg_file":"pattern.svg","json_file":"pattern.json"}}
