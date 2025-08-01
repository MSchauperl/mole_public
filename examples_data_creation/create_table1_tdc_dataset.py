"""
Script to create a filtered TDC dataset containing only the datasets mentioned in Table 1 of the mole paper.
"""
import os
import pandas as pd

# Define the datasets from Table 1 of the mole paper
table1_datasets = [
    'Caco2',           # ADME_Caco2_Wang
    'HIA',             # ADME_HIA_Hou  
    'Pgp',             # ADME_Pgp_Broccatelli
    'Bioavailability', # ADME_Bioavailability_Ma
    'Lipophilicity',   # ADME_Lipophilicity_AstraZeneca
    'Solubility',      # ADME_Solubility_AqSolDB
    'BBB',             # ADME_BBB_Martins
    'PPBR',            # ADME_PPBR_AZ
    'VDss',            # ADME_VDss_Lombardo
    'CYP2D6_inhibition',    # ADME_CYP2D6_Veith
    'CYP3A4_inhibition',    # ADME_CYP3A4_Veith
    'CYP2C9_inhibition',    # ADME_CYP2C9_Veith
    'CYP2D6_substrate',     # ADME_CYP2D6_Substrate_CarbonMangels
    'CYP3A4_substrate',     # ADME_CYP3A4_Substrate_CarbonMangels
    'CYP2C9_substrate',     # ADME_CYP2C9_Substrate_CarbonMangels
    'Half_life',            # ADME_Half_Life_Obach
    'Clearance_microsome',  # ADME_Clearance_Microsome_AZ
    'Clearance_hepatocyte', # ADME_Clearance_Hepatocyte_AZ
    'hERG',                 # Tox_hERG
    'Ames',                 # Tox_AMES
    'DILI',                 # Tox_DILI
    'LD50'                  # Tox_LD50_Zhu
]

# Mapping from Table 1 dataset names to actual column names in the merged TDC file
dataset_mapping = {
    'Caco2': 'ADME_Caco2_Wang',
    'HIA': 'ADME_HIA_Hou',
    'Pgp': 'ADME_Pgp_Broccatelli',
    'Bioavailability': 'ADME_Bioavailability_Ma',
    'Lipophilicity': 'ADME_Lipophilicity_AstraZeneca',
    'Solubility': 'ADME_Solubility_AqSolDB',
    'BBB': 'ADME_BBB_Martins',
    'PPBR': 'ADME_PPBR_AZ',
    'VDss': 'ADME_VDss_Lombardo',
    'CYP2D6_inhibition': 'ADME_CYP2D6_Veith',
    'CYP3A4_inhibition': 'ADME_CYP3A4_Veith',
    'CYP2C9_inhibition': 'ADME_CYP2C9_Veith',
    'CYP2D6_substrate': 'ADME_CYP2D6_Substrate_CarbonMangels',
    'CYP3A4_substrate': 'ADME_CYP3A4_Substrate_CarbonMangels',
    'CYP2C9_substrate': 'ADME_CYP2C9_Substrate_CarbonMangels',
    'Half_life': 'ADME_Half_Life_Obach',
    'Clearance_microsome': 'ADME_Clearance_Microsome_AZ',
    'Clearance_hepatocyte': 'ADME_Clearance_Hepatocyte_AZ',
    'hERG': 'Tox_hERG',
    'Ames': 'Tox_AMES',
    'DILI': 'Tox_DILI',
    'LD50': 'Tox_LD50_Zhu'
}

def main():
    # Load the full merged TDC dataset
    input_file = 'data/tdc/tdc_adme_tox_merged.csv'
    output_file = 'data/tdc/tdc_table1_datasets.csv'
    
    print(f"Loading full TDC dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Get the columns we want to keep (SMILES + Table 1 datasets)
    columns_to_keep = ['SMILES']
    available_columns = []
    missing_columns = []
    
    for table1_dataset, column_name in dataset_mapping.items():
        if column_name in df.columns:
            columns_to_keep.append(column_name)
            available_columns.append(table1_dataset)
        else:
            missing_columns.append(table1_dataset)
    
    print(f"Available Table 1 datasets: {available_columns}")
    print(f"Missing Table 1 datasets: {missing_columns}")
    
    # Filter the dataset
    df_filtered = df[columns_to_keep]
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Save the filtered dataset
    df_filtered.to_csv(output_file, index=False)
    print(f"Saved filtered dataset to {output_file}")
    
    # Print summary statistics
    print("\nSummary of available data:")
    for col in columns_to_keep[1:]:  # Skip SMILES
        non_null_count = df_filtered[col].notnull().sum()
        print(f"{col}: {non_null_count} non-null values")

if __name__ == "__main__":
    main()