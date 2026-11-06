# FabricNow Accurate Pattern Flow

## Recommended model setup
- **Claude Sonnet 4.5 vision** (`ANTHROPIC_VISION_MODEL=claude-sonnet-4-5`) is the preferred garment-photo analyst when the Anthropic key is configured.
- **GPT-4.1 vision** (`OPENAI_VISION_MODEL=gpt-4.1`) is the OpenAI fallback for structured garment and pattern-geometry analysis.
- **GPT Image 2.5 Sunburst** (`OPENAI_IMAGE_MODEL=gpt-image-2.5-sunburst`) is for fashion concepts, lookbooks, fabric-print exploration and mockups only. It must not draw production pattern boundaries.

## Pattern-generation architecture
The AI interprets the garment photograph and returns numeric 2D piece geometry. The worker validates and renders those coordinates deterministically into SVG/PNG, then maps one shared fabric-repeat plane across the full set of pieces. The shared repeat keeps motif scale consistent instead of stretching the swatch independently to each piece.

The Basic Plan sample images are the visual benchmark: recognizable garment-specific silhouettes, complete separated pieces, consistent textile scale, clean white technical sheets and high-resolution individual PNG assets.

## Accuracy boundary
A single uncalibrated photograph cannot establish true physical scale, seam allowance, dart intake or hidden construction with certainty. Outputs inferred from photos are drafting starting points, not production-certified fitted patterns. Validate body/garment measurements and make a toile before cutting final fabric.
