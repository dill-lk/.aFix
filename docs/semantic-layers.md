# Semantic Layers — Developer Guide

Every `.aFix` file contains a built-in **Semantic Object Manifest** (`OBJM` chunk). This makes the image a programmable component, not a static asset.

## The Object Manifest

At encode time, a segmentation model analyses the image and tags every significant region. The result is stored as a BSON document inside the `OBJM` chunk.

```json
{
  "version": "1.0",
  "objects": [
    { "id": "sky",    "label": "sky",        "category": "background" },
    { "id": "face_0", "label": "human_face", "category": "subject"    },
    { "id": "text_0", "label": "text",       "category": "overlay"    }
  ]
}
```

## Using the `afix-view` Web Component

### Installation

```bash
npm install afix-view
```

### Basic Usage

```html
<script type="module" src="node_modules/afix-view/afix-view.js"></script>

<afix-img id="hero" src="hero.afix"></afix-img>
```

### JavaScript Layer API

```js
const img = document.getElementById('hero');

// Wait until fully decoded
img.addEventListener('afix:ready', () => {

  // Style a layer
  img.layers['sky'].style.filter = 'hue-rotate(180deg)';

  // Hide a layer
  img.layers['text_0'].style.display = 'none';

  // Get layer bounding box
  const bbox = img.layers['face_0'].getBoundingBox();
  console.log(bbox); // { x, y, width, height }

  // List all available layers
  console.log(Object.keys(img.layers));
});
```

### CSS Layer Targeting

```css
/* Target a semantic layer with ::layer() pseudo-element */
#hero::layer(sky) {
  filter: blur(8px);
  opacity: 0.7;
}

#hero::layer(face_0) {
  outline: 3px solid hotpink;
}
```

## Programmatic Encoding (libafix CLI)

When converting from JPEG/PNG, the encoder will auto-detect semantic regions. You can also provide manual annotations:

```bash
afix-convert input.jpg output.afix \
  --semantic-model auto \
  --tag sky:background \
  --tag person:subject
```

## Layer Categories

| Category | Description | Typical IDs |
|----------|-------------|-------------|
| `background` | Non-subject regions | `sky`, `ground`, `wall` |
| `subject` | Primary focus | `face_0`, `person_0`, `product` |
| `overlay` | Text, logos, UI | `text_0`, `logo`, `watermark` |
| `depth` | Depth layers | `foreground`, `midground`, `background` |
