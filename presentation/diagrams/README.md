# MolE Presentation Diagrams

This directory contains individual Mermaid diagram files that can be used to generate visual elements for the MolE presentation.

## Diagram Files

### 01_high_level_architecture.mmd
**Purpose:** Overview of the complete MolE pipeline
**Usage:** Title slide or architecture overview
**Shows:** Molecular Graph → Tokenization → Embedding → Transformer → Prediction

### 02_disentangled_attention.mmd
**Purpose:** Detailed view of the four attention types
**Usage:** Core attention mechanism explanation
**Shows:** c2c, c2p, p2c, p2p attention interactions

### 03_embedding_layer.mmd
**Purpose:** How molecular tokens become embeddings
**Usage:** Embedding layer explanation
**Shows:** Word + Position + Type embeddings → Final embeddings

### 04_transformer_layer.mmd
**Purpose:** Internal structure of a transformer layer
**Usage:** Detailed layer architecture
**Shows:** Attention → FFN with residual connections and normalization

### 05_graph_to_sequence.mmd
**Purpose:** Molecular graph to sequence conversion
**Usage:** Input processing explanation
**Shows:** Graph components → Tokenization → Position encoding

### 06_multitask_prediction.mmd
**Purpose:** Multi-task prediction architecture
**Usage:** Output/prediction head explanation
**Shows:** Shared processing → Task-specific heads → Various outputs

### 07_attention_detail.mmd
**Purpose:** Comprehensive attention mechanism breakdown
**Usage:** Deep dive into attention computation
**Shows:** All projections and attention types in detail

### 08_training_pipeline.mmd
**Purpose:** Complete training process
**Usage:** Training methodology explanation
**Shows:** Data processing → Training → Optimization → Evaluation

### 09_model_scalability.mmd
**Purpose:** Model size and performance trade-offs
**Usage:** Scalability discussion
**Shows:** Different model sizes with performance metrics

## How to Generate Images

### Method 1: Online Mermaid Editor
1. Go to [Mermaid Live Editor](https://mermaid.live/)
2. Copy and paste the content from any `.mmd` file
3. Export as PNG or SVG

### Method 2: Command Line (Recommended)
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG (high resolution)
mmdc -i 01_high_level_architecture.mmd -o 01_high_level_architecture.png -w 1920 -H 1080

# Generate SVG (scalable)
mmdc -i 01_high_level_architecture.mmd -o 01_high_level_architecture.svg

# Batch generate all diagrams
for file in *.mmd; do
    mmdc -i "$file" -o "${file%.mmd}.png" -w 1920 -H 1080
done
```

### Method 3: VS Code Extension
1. Install "Mermaid Markdown Syntax Highlighting" extension
2. Open `.mmd` files in VS Code
3. Use preview or export features

## Image Specifications

### For Presentations
- **Format:** PNG or SVG
- **Resolution:** 1920x1080 minimum (for projection)
- **DPI:** 300 for print, 150 for screen
- **Background:** Transparent or white

### Quality Settings
```bash
# High quality for presentations
mmdc -i diagram.mmd -o diagram.png -w 1920 -H 1080 -s 2

# Medium quality for documents
mmdc -i diagram.mmd -o diagram.png -w 1280 -H 720 -s 1.5

# Web quality
mmdc -i diagram.mmd -o diagram.png -w 800 -H 600 -s 1
```

## Customization Tips

### Colors
- Modify `fill:#color` in the diagram files
- Use consistent colors across all diagrams
- Consider colorblind-friendly palettes

### Text
- Keep text concise and readable
- Use consistent font sizes
- Avoid overlapping text

### Layout
- Ensure proper spacing between elements
- Use subgraphs for logical grouping
- Maintain consistent direction (top-to-bottom, left-to-right)

## Integration with Presentation Software

### PowerPoint
1. Generate PNG files with transparent backgrounds
2. Insert → Pictures → From file
3. Use "Remove background" if needed
4. Resize while maintaining aspect ratio

### Google Slides
1. Generate PNG or SVG files
2. Insert → Image → Upload from computer
3. Crop and resize as needed

### LaTeX/Beamer
1. Generate PDF or SVG files
2. Use `\includegraphics` command
3. Specify width/height parameters

## Troubleshooting

### Common Issues
1. **Text too small:** Increase `-w` and `-H` parameters
2. **Blurry images:** Use higher `-s` scale factor
3. **Colors not showing:** Check color syntax in `.mmd` files
4. **Layout broken:** Verify Mermaid syntax

### Quality Check
- Test on actual presentation display
- Check readability from audience distance
- Ensure colors work on both light and dark backgrounds

## Advanced Usage

### Batch Processing
```bash
#!/bin/bash
# Generate all diagrams in multiple formats
for file in *.mmd; do
    base="${file%.mmd}"
    echo "Processing $file..."
    mmdc -i "$file" -o "${base}.png" -w 1920 -H 1080 -s 2
    mmdc -i "$file" -o "${base}.svg"
    mmdc -i "$file" -o "${base}.pdf"
done
```

### Configuration File
Create `mermaid.config.json`:
```json
{
  "theme": "default",
  "themeVariables": {
    "primaryColor": "#1565C0",
    "primaryTextColor": "#212121",
    "primaryBorderColor": "#1565C0",
    "lineColor": "#757575"
  }
}
```

Use with: `mmdc -i diagram.mmd -o diagram.png -c mermaid.config.json`

These diagrams provide comprehensive visual support for explaining the MolE architecture in technical presentations.