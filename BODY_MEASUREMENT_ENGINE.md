# FabricNow Body Measurement Engine

## Processing flow

1. The existing FabricNow Style Backend authenticates the tailor and checks API Growth entitlement.
2. It sends the required front and side photos (plus optional back/left/right) to this worker over a private internal-secret header.
3. The worker applies EXIF orientation, image-size validation, GrabCut foreground segmentation and largest-person-component cleanup.
4. Supplied standing height calibrates pixels to centimetres independently in each view.
5. Front/back silhouette widths and side/left/right silhouette depths are fused into vertical profiles.
6. The worker creates a 64-vertex elliptical loft mesh and exports GLB. Circumferences use an ellipse-perimeter approximation (Ramanujan II) at bust, underbust, waist, high hip, hip, thigh, knee and calf levels.
7. Confidence and method metadata are returned per measurement. Proportional estimates for limbs are deliberately marked low confidence.
8. The tailor can correct measurements manually. Manual values are marked with confidence 1 and a manual-correction method tag.
9. The pattern engine drafts measurement-driven bodice, skirt, trouser and dress starting blocks. Shapely buffers the stitch-line geometry to create a separate seam-allowance cutline. SVG is exported in millimetres at 1:1 scale with JSON metadata.

## API endpoints

- `POST /body-measurements/process` — multipart form: `front`, `side`, optional `back`, `left`, `right`, `height`, `unit`, `scan_id`.
- `POST /body-measurements/patterns` — JSON: `scan_id`, `measurements`, `unit`, `garment`, `ease_cm`, `seam_allowance_cm`.
- `GET /body-measurements/files/{scan_id}/{filename}` — private GLB/silhouette files.
- `GET /body-measurements/files/{scan_id}/patterns/{pattern_id}/{filename}` — private pattern files.

## Required configuration

- `FABRIC_NOW_INTERNAL_SECRET`: required and must match the Style Backend value. Body endpoints fail closed if absent.
- `BODY_MAX_UPLOAD_MB=12`
- `DATA_DIR=/data` (persistent volume recommended; generated GLB/SVG files are stored under `body-scans`).
- Style Backend: `BODY_MEASUREMENT_WORKER_URL=https://...` and matching `FABRIC_NOW_INTERNAL_SECRET`.

## Engineering limitations

This implementation is an orthogonal silhouette visual-hull approximation. It is not SMPL/SOMA or another trained parametric anatomical model. It does not recover hidden geometry or exact tissue shape from photographs. Clothing, camera perspective, pose, shadows, segmentation errors and arm placement affect dimensions. The torso circumference values are estimates, not tape-equivalent measurements. Limb circumference/length estimates with low confidence should be measured manually. The generated blocks are starting patterns and do not yet include individual dart solving, posture compensation, crotch curve fitting, sleeve-cap balancing, fabric stretch modelling or a validated grading system. A tape check and toile fitting remain mandatory before production cutting.
