"""
Script to create a unified TDC ADME+Tox dataset.
- Merges all ADME and Tox datasets from TDC on the SMILES column.
- Output: mole_public/data/tdc_adme_tox_merged.csv

Requirements:
    pip install tdc pandas
"""
import os
import logging
import pandas as pd
from tdc.single_pred import ADME, Tox
from tdc.utils import retrieve_benchmark_names
from functools import reduce

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'tdc_adme_tox_merged.csv')


def load_and_prepare_dataset(cls, name, prefix):
    """
    Loads a TDC dataset and returns a DataFrame with SMILES and labeled column.
    """
    try:
        df = cls(name=name).get_data()[['Drug', 'Y']].rename(
            columns={'Drug': 'SMILES', 'Y': f'{prefix}_{name}'}
        )
        logging.info(f"Loaded {prefix} dataset: {name} ({len(df)} rows)")
        return df
    except Exception as e:
        logging.error(f"Failed to load {prefix} dataset {name}: {e}")
        return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Expanded dataset list from TDC website (ADME)
    adme_dataset_names = [
        # Absorption
        'Caco2_Wang', 'PAMPA_NCATS', 'Approved_PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'HydrationFreeEnergy_FreeSolv',
        # Distribution
        'BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo', 'B3DB_Classification', 'B3DB_Regression',
        # Metabolism
        'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels',
        # Excretion
        'Half_Life_Obach', 'Clearance_Microsome_AZ', 'Clearance_Hepatocyte_AZ', 'HLM', 'RLM',
    ]

    from tdc.utils import retrieve_label_name_list

    # List of single-label Tox datasets (from TDC website)
    single_label_tox = [
        'LD50_Zhu', 'hERG', 'hERG_Karim', 'AMES', 'DILI', 'Skin Reaction', 'Carcinogens_Lagunin', 'ClinTox'
    ]
    # Multi-label datasets
    multi_label_tox = [
        ('herg_central', 'hERG_Central'),
        ('Tox21', 'Tox21'),
        ('ToxCast', 'ToxCast'),
    ]

    dfs = []
    # Load ADME datasets
    for name in adme_dataset_names:
        df = None
        try:
            df = load_and_prepare_dataset(ADME, name, 'ADME')
        except Exception:
            pass
        if df is not None:
            dfs.append(df)
        else:
            logging.warning(f"ADME Dataset {name} could not be loaded.")

    # Load single-label Tox datasets
    for name in single_label_tox:
        df = None
        try:
            df = load_and_prepare_dataset(Tox, name, 'Tox')
        except Exception:
            pass
        if df is not None:
            dfs.append(df)
        else:
            logging.warning(f"Tox Dataset {name} could not be loaded.")

    # Load multi-label Tox datasets
    for dataset_name, prefix in multi_label_tox:
        try:
            label_list = retrieve_label_name_list(dataset_name)
        except Exception as e:
            logging.warning(f"Could not retrieve label list for {dataset_name}: {e}")
            continue
        for label in label_list:
            try:
                from tdc.single_pred import Tox
                data = Tox(name=dataset_name, label_name=label).get_data()[['Drug', 'Y']].rename(
                    columns={'Drug': 'SMILES', 'Y': f'Tox_{prefix}_{label}'}
                )
                logging.info(f"Loaded Tox multi-label dataset: {prefix} label: {label} ({len(data)} rows)")
                dfs.append(data)
            except Exception as e:
                logging.warning(f"Failed to load {prefix} label {label}: {e}")

    if not dfs:
        logging.error("No datasets loaded. Exiting.")
        return

    logging.info(f"Merging {len(dfs)} datasets on SMILES column.")
    merged = reduce(lambda left, right: pd.merge(left, right, on='SMILES', how='outer'), dfs)
    logging.info(f"Merged dataset shape: {merged.shape}")

    merged.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved merged dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
