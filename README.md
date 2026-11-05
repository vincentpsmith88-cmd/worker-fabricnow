# Pattern worker

This is the image/pattern-processing worker used by the main Style/FabricNow
company workspace.

The Node/Express Style Backend remains the public backend of record. It handles
authentication, Google sign-in, workspace authorization, API keys, Stripe
billing and usage. It forwards authenticated pattern jobs to this worker.

Set:
- `FABRIC_NOW_SERVICE_URL` on the Node backend
- the same random `FABRIC_NOW_INTERNAL_SECRET` on Node and the worker
- `OPENAI_API_KEY` and the worker's image-model settings on the worker

The worker is not intended to be exposed directly to the browser.

## Body measurement and pattern engine

The worker also exposes private backend-to-backend endpoints:

- `POST /body-measurements/process`: front + side images, optional back/left/right, supplied standing height and unit. It runs GrabCut silhouette extraction, height calibration, orthogonal front/depth profile fusion, a 3D elliptical-loft mesh export (GLB), and derives circumference estimates using an ellipse perimeter approximation.
- `POST /body-measurements/patterns`: drafts starting bodice, skirt, trouser or dress blocks from a reviewed measurement profile and returns SVG (1:1 mm) plus JSON metadata.
- `GET /body-measurements/files/...`: private generated mesh, silhouette and pattern files.

Configure `FABRIC_NOW_INTERNAL_SECRET` on both worker and Style Backend (same strong secret), and `BODY_MEASUREMENT_WORKER_URL` on Style Backend. The body-measurement endpoints fail closed if the worker secret is absent.

**Accuracy boundary:** this is a calibrated two/ multi-view silhouette visual-hull approximation, not a learned anatomical reconstruction or depth-camera scan. It cannot infer hidden body surfaces, posture, tissue compression, exact bust projection, arm lengths or crotch shape from two photos. Low-confidence proportional estimates are explicitly labelled. The generated SVGs are measurement-driven starter blocks and are not production-certified fitted patterns; validate tape measurements and toile fit before cutting. Do not represent these outputs as guaranteed perfect fit.

## Dress formats and extra cuts

`POST /api/jobs` accepts an optional `variants` form field (JSON, sent by Pattern Studio via Style Backend):
`{"formats":["Midi","Maxi","Long sleeve"],"all":false,"extraCuts":true,"note":"..."}`.
Formats are validated against a fixed list (mini to maxi, sleeveless to long sleeve, A-line, fitted, gathered, flared, wrap, shift). The designer notes and the variants are now passed to the vision model.
With `extraCuts` on, the model adds only the pieces that change per format; each carries a `variant` field (also written to `manifest.json`). Pieces are limited to 30 main plus extras, 90 in total.
The finished job reports `variants_result` (requested / delivered / missing) and a `warnings` entry for any format the model did not return, so the app never claims cuts that were not produced.
