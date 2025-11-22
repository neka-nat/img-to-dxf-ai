# img-to-dxf-ai

This project converts an image of a mechanical drawing into a DXF file.

## Installation

```bash
git clone https://github.com/your-username/img-to-dxf-ai.git
cd img-to-dxf-ai
uv sync
cp .env.example .env
# edit .env
```

## Usage

This script can generate a DXF file or an SVG file from an image.

```bash
uv run python scripts/make_outline.py <input_image_path>
```

## Dependencies

- Object detection
  - Gemini 2.5 pro
- Image Segmentation (Using either of the following models)
  - Nanobanana Pro
  - GPT Image
  - Reve

## Architecture

```mermaid
graph TD
    A[Input Image] --> B[Detect each projected view]
    B --> C[Colorize each view]
    C --> D[Separate each view]
    D --> E[Detect circles]
    E --> F[Build graph]
    F --> G[Export DXF/SVG]
```

## Demo
### Input
![input](./assets/b9-1.png)

### Detect each projected view
![detect](./assets/b9-1_1.png)

### Colorize each drawing element
* Red: Dimension lines (including extension lines, arrows, and dimension text)
* Black: Auxiliary lines such as dashed, dotted, chain, or double-chain lines
* Blue: Straight outer contour lines, excluding dimension and auxiliary lines
* Green: Circular or arc-shaped outer contour lines, excluding dimension and auxiliary lines

![colorize](./assets/colorized_0.png)

### Export outline as SVG
![svg](./assets/b9-1.svg)
