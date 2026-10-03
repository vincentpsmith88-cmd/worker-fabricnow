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
