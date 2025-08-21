# ChEMBL Multi-Target Dataset Extraction

This directory contains scripts for extracting single-target and multi-target datasets from the complete ChEMBL dataset. The scripts properly handle three-state classification: 1 (active), -1 (inactive), 0 (missing/not measured).

## 📁 Files

- **`extract_single_target.py`** - Main extraction script supporting both single and multi-target extraction
- **`extract_assays_from_metadata.py`** - Metadata-driven extraction using analysis results
- **`README.md`** - This documentation file

## 🚀 Quick Start

### Single Target Extraction
```bash
python extract_single_target.py \
    --target_assay_id CHEMBL1794580 \
    --csv ../chembl_complete_dataset_full.csv \
    --output single_target_dataset.csv
```

### Multi-Target Extraction
```bash
python extract_single_target.py \
    --target_assay_ids "['CHEMBL1614161', 'CHEMBL1614441', 'CHEMBL1614458', 'CHEMBL1614530', 'CHEMBL1738442', 'CHEMBL1794345', 'CHEMBL1794580']" \
    --csv ../chembl_complete_dataset_full.csv \
    --output multi_target_dataset.csv
```

### Metadata-Driven Extraction
```bash
python extract_assays_from_metadata.py \
    --csv ../chembl_complete_dataset_full.csv \
    --metadata ../chembl_target_metadata.csv \
    --output metadata_based_dataset.csv
```

## 📊 Script Details

### `extract_single_target.py`

**Purpose**: Extract compounds measured in one or more target assays from the complete ChEMBL dataset.

**Features**:
- ✅ Single target extraction (backward compatible)
- ✅ Multi-target extraction
- ✅ Memory-efficient chunked processing
- ✅ Three-state classification handling
- ✅ Flexible input formats
- ✅ Progress tracking and statistics

**Usage**:
```bash
# Single target
python extract_single_target.py \
    --target_assay_id CHEMBL1794580 \
    --csv input_dataset.csv \
    --output output_dataset.csv

# Multiple targets (comma-separated)
python extract_single_target.py \
    --target_assay_ids CHEMBL1794580,CHEMBL1614161,CHEMBL1614441 \
    --csv input_dataset.csv \
    --output output_dataset.csv

# Multiple targets (list format)
python extract_single_target.py \
    --target_assay_ids "['CHEMBL1794580', 'CHEMBL1614161', 'CHEMBL1614441']" \
    --csv input_dataset.csv \
    --output output_dataset.csv

# Custom chunk size for memory management
python extract_single_target.py \
    --target_assay_ids CHEMBL1794580,CHEMBL1614161 \
    --csv input_dataset.csv \
    --output output_dataset.csv \
    --chunk_size 500
```

**Arguments**:
- `--target_assay_id`: Single CHEMBL assay ID (e.g., CHEMBL1794580)
- `--target_assay_ids`: Multiple CHEMBL assay IDs (comma-separated or list format)
- `--csv`: Path to input chembl_complete_dataset.csv
- `--output`: Path to save the filtered dataset
- `--chunk_size`: Number of rows to process at a time (default: 1000)

### `extract_assays_from_metadata.py`

**Purpose**: Automatically extract assays from metadata analysis based on quality criteria.

**Features**:
- ✅ Metadata-driven assay selection
- ✅ Configurable quality thresholds
- ✅ Automatic filtering based on active/inactive counts
- ✅ Detailed assay statistics
- ✅ Integration with metadata analysis results

**Usage**:
```bash
# Default extraction (>10,000 active and inactive compounds)
python extract_assays_from_metadata.py \
    --csv ../chembl_complete_dataset_full.csv \
    --metadata ../chembl_target_metadata.csv \
    --output high_quality_dataset.csv

# Custom thresholds
python extract_assays_from_metadata.py \
    --csv ../chembl_complete_dataset_full.csv \
    --metadata ../chembl_target_metadata.csv \
    --output custom_dataset.csv \
    --min_active 5000 \
    --min_inactive 5000

# Memory-optimized processing
python extract_assays_from_metadata.py \
    --csv ../chembl_complete_dataset_full.csv \
    --metadata ../chembl_target_metadata.csv \
    --output dataset.csv \
    --chunk_size 500
```

**Arguments**:
- `--csv`: Path to input chembl_complete_dataset.csv
- `--metadata`: Path to chembl_target_metadata.csv
- `--output`: Path to save the filtered dataset
- `--min_active`: Minimum active compounds required (default: 10000)
- `--min_inactive`: Minimum inactive compounds required (default: 10000)
- `--chunk_size`: Number of rows to process at a time (default: 1000)

## 📈 Output Format

### Single Target Dataset
```
compound_id,smiles,compound_name,fold,activity
7,N=C(N)NC(=O)c1nc(Cl)c(N2CCCCCC2)nc1N,CHEMBL501701,fold_1,1
16,O=C(c1ccccc1)c1nccc2ccccc12,CHEMBL553669,fold_0,-1
17,Br.CCCCCCCn1c2c(c(=N)c3c1CCC3)CCCC2,CHEMBL536166,fold_0,1
```

### Multi-Target Dataset
```
compound_id,smiles,compound_name,fold,activity_CHEMBL1614161,activity_CHEMBL1614441,activity_CHEMBL1614458,activity_CHEMBL1614530,activity_CHEMBL1738442,activity_CHEMBL1794345,activity_CHEMBL1794580
7,N=C(N)NC(=O)c1nc(Cl)c(N2CCCCCC2)nc1N,CHEMBL501701,fold_1,0.0,0.0,-1.0,-1.0,1.0,1.0,1.0
16,O=C(c1ccccc1)c1nccc2ccccc12,CHEMBL553669,fold_0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0
17,Br.CCCCCCCn1c2c(c(=N)c3c1CCC3)CCCC2,CHEMBL536166,fold_0,0.0,1.0,0.0,0.0,1.0,-1.0,1.0
```

### Activity Values
- **1**: Active compound
- **-1**: Inactive compound  
- **0**: Not measured in this assay

## 🎯 Example Results

### Multi-Target Dataset Statistics
Based on the 7 high-quality assays from metadata analysis:

| Assay | Active | Inactive | Total Measured | Active % |
|-------|--------|----------|----------------|----------|
| CHEMBL1614161 | 11,702 | 13,734 | 25,436 | 46.0% |
| CHEMBL1614441 | 12,445 | 10,375 | 22,820 | 54.5% |
| CHEMBL1614458 | 17,193 | 33,805 | 50,998 | 33.7% |
| CHEMBL1614530 | 17,545 | 28,206 | 45,751 | 38.3% |
| CHEMBL1738442 | 30,791 | 11,330 | 42,121 | 73.1% |
| CHEMBL1794345 | 22,240 | 60,997 | 83,237 | 26.7% |
| CHEMBL1794580 | 17,962 | 45,954 | 63,916 | 28.1% |

**Total compounds with any measurement**: 212,637

## 🔧 Technical Details

### Memory Management
- **Chunked processing**: Processes data in configurable chunks (default: 1000 rows)
- **Streaming I/O**: Writes output incrementally to avoid memory issues
- **Column filtering**: Only loads required columns from input CSV

### Data Quality
- **Three-state classification**: Properly handles active (1), inactive (-1), and missing (0) values
- **Validation**: Checks for required columns and valid assay IDs
- **Statistics**: Provides detailed counts and percentages for each assay

### Error Handling
- **File validation**: Checks input files exist before processing
- **Column validation**: Verifies required columns are present
- **Assay validation**: Confirms target assay columns exist in dataset
- **Graceful failures**: Provides clear error messages and exits cleanly

## 🚨 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'pandas'"**
```bash
conda activate mole-py10
```

**"Input CSV file not found"**
- Check the path to your chembl_complete_dataset.csv file
- Ensure you're in the correct directory

**"Column 'assay_CHEMBLxxxxx' not found"**
- Verify the assay ID exists in your dataset
- Check the column naming convention (should be 'assay_CHEMBLxxxxx')

**Memory issues with large datasets**
- Reduce chunk size: `--chunk_size 500`
- Ensure sufficient RAM is available
- Consider processing smaller subsets first

### Performance Tips

1. **Use appropriate chunk sizes**: Larger chunks (2000-5000) for more RAM, smaller (500-1000) for less RAM
2. **Filter assays early**: Use metadata analysis to select only high-quality assays
3. **Monitor progress**: The script shows progress every 10,000 rows processed
4. **Check output size**: Multi-target datasets can be large; ensure sufficient disk space

## 📝 Examples

### Extract Top 7 Assays from Metadata Analysis
```bash
# This extracts the 7 assays with >10,000 active and inactive compounds
python extract_assays_from_metadata.py \
    --csv ../chembl_complete_dataset_full.csv \
    --metadata ../chembl_target_metadata.csv \
    --output top_7_assays_dataset.csv
```

### Extract Custom Assay Set
```bash
# Extract specific assays of interest
python extract_single_target.py \
    --target_assay_ids "['CHEMBL1794580', 'CHEMBL1614161', 'CHEMBL1614441']" \
    --csv ../chembl_complete_dataset_full.csv \
    --output custom_assays_dataset.csv
```

### Extract Single Target for Training
```bash
# Extract single target for binary classification training
python extract_single_target.py \
    --target_assay_id CHEMBL1794580 \
    --csv ../chembl_complete_dataset_full.csv \
    --output CHEMBL1794580_training_dataset.csv
```

## 🔗 Related Files

- `../chembl_complete_dataset_full.csv` - Complete ChEMBL dataset
- `../chembl_target_metadata.csv` - Target metadata analysis results
- `../chembl_target_metadata_analysis.ipynb` - Jupyter notebook for metadata analysis

## 📊 Dataset Quality

The scripts ensure high-quality datasets by:
- **Filtering missing data**: Only includes compounds with actual measurements
- **Validating activity values**: Ensures proper 1/-1/0 classification
- **Providing statistics**: Detailed counts and percentages for verification
- **Memory efficiency**: Handles large datasets without memory issues

This makes the extracted datasets ready for machine learning model training with proper binary classification targets.
