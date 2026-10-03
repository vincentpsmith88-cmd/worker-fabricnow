# FabricNow Accurate Pattern Flow

The Basic Pattern workflow is now AI geometry analysis + deterministic rendering. The AI automatically identifies the garment type and African regional/style family from the photo. No garment-type selection is required in the Workspace.

Claude is preferred when `ANTHROPIC_API_KEY` is configured (`AI_VISION_PROVIDER=auto`). OpenAI vision is the fallback. The worker does not use an image-generation model to invent final pattern boundaries. Claude/OpenAI returns numeric 2D piece geometry, and the worker renders that geometry into SVG/PNG and maps the supplied fabric texture into each piece.

A single photograph cannot establish true physical scale without a measurement or scale reference. The manifest therefore records `scale_basis` and `geometry_confidence`.
