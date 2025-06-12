# Atom Environment Visual Atlas Generator

This tool creates a comprehensive visual atlas of atom environments found in the GuacaMol SMILES dataset. For each atom environment (radius 0, structural), it finds an example molecule, creates a visualization with the atom highlighted, and combines all visualizations into a multi-page PDF.

## Features

- 🔍 **Automated Discovery**: Scans GuacaMol SMILES to find example molecules for each atom environment
- 🎨 **Visual Highlighting**: Creates molecule images with the specific atom environment highlighted in red
- 📖 **Multi-page PDF**: Organizes visualizations in a 4×5 grid (20 molecules per page)
- 🧪 **Element-based Sorting**: Groups atom environments by element type (C, N, O, S, etc.) for better organization
- 📊 **Summary Statistics**: Provides coverage statistics, element distribution, and example listings
- ⚙️ **Configurable**: Adjustable scanning limits, image sizes, and output options

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_atlas.txt
```

The main dependencies are:
- `rdkit` - For molecular processing and visualization
- `matplotlib` - For PDF generation
- `pillow` - For image processing
- `tqdm` - For progress bars

## Usage

### Basic Usage

```bash
cd data/
python create_atomenv_atlas.py
```

This will:
1. Load the radius 0 structural vocabulary from `../mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl`
2. Scan **all molecules** in `guacamol_v1_all.smiles` until examples for all atom environments are found
3. Create `atomenv_atlas_radius0_structural.pdf`

### Advanced Usage

```bash
python create_atomenv_atlas.py \
    --smiles path/to/your/smiles_file.smiles \
    --vocab path/to/vocabulary.pkl \
    --output custom_atlas.pdf \
    --max-molecules 50000 \
    --mol-size 400 400

# Or scan all molecules (unlimited - this is now the default):
python create_atomenv_atlas.py \
    --smiles guacamol_v1_all.smiles \
    --output complete_atlas.pdf
```

### Command Line Options

- `--smiles`: Path to SMILES file (default: `guacamol_v1_all.smiles`)
- `--vocab`: Path to vocabulary pickle file (default: `../mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl`)
- `--output`: Output PDF filename (default: `atomenv_atlas_radius0_structural.pdf`)
- `--max-molecules`: Maximum molecules to scan (default: unlimited - scans until all environments found)
- `--mol-size`: Molecule image size as width height (default: 300 300)

### Testing

Run the test script to verify functionality with a smaller dataset:

```bash
python test_atlas.py
```

## Output Format

The generated PDF contains:

- **Title**: "Atom Environment Atlas (Radius 0, Structural) - Page X - Sorted by Element"
- **Grid Layout**: 4 columns × 5 rows per page (20 molecules)
- **Element-based Organization**: Environments grouped by element (C, N, O, S, P, F, Cl, Br, I, others)
- **Each Cell Contains**:
  - Molecule structure with highlighted atom (red)
  - Environment index number and element symbol
  - Structural environment hash value (radius 0, no features)
  - Functional environment hash value (radius 0, with features)
  - Atom index within the molecule

## Algorithm Overview

1. **Vocabulary Loading**: Loads the pre-computed atom environment vocabulary
2. **Molecule Scanning**: Iterates through SMILES file to find example molecules
3. **Environment Matching**: Uses Morgan fingerprints to identify atom environments
4. **Image Generation**: Creates molecule visualizations with RDKit
5. **PDF Assembly**: Combines images into a multi-page PDF with matplotlib

## File Structure

```
data/
├── create_atomenv_atlas.py     # Main atlas generator script
├── test_atlas.py               # Test script
├── requirements_atlas.txt      # Python dependencies
├── README_atlas.md            # This documentation
├── guacamol_v1_all.smiles     # GuacaMol SMILES data
└── atomenv_atlas_radius0_structural.pdf  # Output atlas (generated)
```

## Example Output

Each page of the PDF shows 20 molecules arranged in a 4×5 grid. The molecules are organized by element type (carbon environments first, then nitrogen, oxygen, etc.). Each molecule:
- Has one atom highlighted in red (the atom with the specific environment)
- Shows the environment index number, element symbol, structural hash, and functional hash
- Displays both structural (radius 0, no features) and functional (radius 0, with features) environment hashes
- Displays the atom index for reference
- Is grouped with other environments of the same element type

## Performance Notes

- **Scanning Speed**: Processes ~1000-5000 molecules per second
- **Default Behavior**: Scans all molecules until 100% coverage is achieved
- **Memory Usage**: Minimal - processes molecules one at a time
- **Output Size**: PDF size depends on number of environments (~1-10 MB typical)
- **Coverage**: Aims for 100% coverage by default (scans entire dataset if needed)
- **Runtime**: May take 10-30 minutes for complete GuacaMol scan depending on vocabulary size

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure SMILES and vocabulary files exist
2. **ImportError**: Install dependencies with `pip install -r requirements_atlas.txt`
3. **Long runtime**: For testing, set `--max-molecules 10000` to limit scanning
4. **Memory Issues**: Set `--max-molecules` to a smaller number
5. **Empty PDF**: Check vocabulary file format or increase `--max-molecules` if using a limit

### Debug Tips

- Run `test_atlas.py` first to verify setup
- Check vocabulary loading with the test script
- Verify SMILES file format (should be one SMILES per line)

## Customization

### Different Vocabularies

To use different atom environment vocabularies:

```bash
python create_atomenv_atlas.py \
    --vocab ../mole/data/vocabularies/vocabulary_radius1_structural_guacamol_v1.pkl \
    --output radius1_atlas.pdf
```

### Custom Grid Layout

Modify the `grid_cols` and `grid_rows` attributes in the `AtomEnvAtlasGenerator` class for different page layouts.

### Highlighting Colors

Change the highlighting color by modifying the `highlight_colors` dictionary in the `create_molecule_image` method.

## Contributing

Feel free to extend this tool for:
- Different radius environments
- Alternative visualization styles
- Additional molecular properties
- Interactive HTML output
- Batch processing multiple vocabularies 