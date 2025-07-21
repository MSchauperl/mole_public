"""
Task configuration for Table 1 datasets from the MolE paper.
Maps datasets to their prediction head type, evaluation metrics, and TDC column names.
"""

from typing import Dict, List, Tuple
import pandas as pd

# Task configuration mapping Table 1 datasets to their properties
TASK_CONFIG = {
    'regression': {
        'tasks': [
            'Caco2', 'Lipophilicity', 'Solubility', 'PPBR', 'VDss',
            'Half_life', 'Clearance_microsome', 'Clearance_hepatocyte'
        ],
        'metrics': [
            'MAE', 'MAE', 'MAE', 'MAE', 'Spearman',
            'Spearman', 'Spearman', 'Spearman'
        ],
        'tdc_columns': [
            'ADME_Caco2_Wang', 'ADME_Lipophilicity_AstraZeneca', 'ADME_Solubility_AqSolDB',
            'ADME_PPBR_AZ', 'ADME_VDss_Lombardo', 'ADME_Half_Life_Obach',
            'ADME_Clearance_Microsome_AZ', 'ADME_Clearance_Hepatocyte_AZ'
        ]
    },
    'classification': {
        'tasks': [
            'HIA', 'Pgp', 'Bioavailability', 'BBB', 'CYP3A4_substrate',
            'CYP2D6_inhibition', 'CYP3A4_inhibition', 'CYP2C9_inhibition',
            'CYP2D6_substrate', 'CYP2C9_substrate'
        ],
        'metrics': [
            'AUROC', 'AUROC', 'AUROC', 'AUROC', 'AUROC',
            'AUPRC', 'AUPRC', 'AUPRC', 'AUPRC', 'AUPRC'
        ],
        'tdc_columns': [
            'ADME_HIA_Hou', 'ADME_Pgp_Broccatelli', 'ADME_Bioavailability_Ma',
            'ADME_BBB_Martins', 'ADME_CYP3A4_Substrate_CarbonMangels',
            'ADME_CYP2D6_Veith', 'ADME_CYP3A4_Veith', 'ADME_CYP2C9_Veith',
            'ADME_CYP2D6_Substrate_CarbonMangels', 'ADME_CYP2C9_Substrate_CarbonMangels'
        ]
    }
}

# MolE paper results for comparison
MOLE_PAPER_RESULTS = {
    'Caco2': {'metric': 'MAE', 'result': 0.329, 'std': 0.008, 'status': 'Not Best'},
    'HIA': {'metric': 'AUROC', 'result': 0.984, 'std': 0.005, 'status': 'Not Best'},
    'Pgp': {'metric': 'AUROC', 'result': 0.93, 'std': 0.005, 'status': 'Not Best'},
    'Bioavailability': {'metric': 'AUROC', 'result': 0.64, 'std': 0.046, 'status': 'Not Best'},
    'Lipophilicity': {'metric': 'MAE', 'result': 0.406, 'std': 0.009, 'status': 'Best'},
    'Solubility': {'metric': 'MAE', 'result': 0.776, 'std': 0.019, 'status': 'Not Best'},
    'BBB': {'metric': 'AUROC', 'result': 0.903, 'std': 0.003, 'status': 'Not Best'},
    'PPBR': {'metric': 'MAE', 'result': 7.229, 'std': 0.168, 'status': 'Best'},
    'VDss': {'metric': 'Spearman', 'result': 0.644, 'std': 0.013, 'status': 'Not Best'},
    'CYP2D6_inhibition': {'metric': 'AUPRC', 'result': 0.679, 'std': 0.006, 'status': 'Not Best'},
    'CYP3A4_inhibition': {'metric': 'AUPRC', 'result': 0.876, 'std': 0.002, 'status': 'Not Best'},
    'CYP2C9_inhibition': {'metric': 'AUPRC', 'result': 0.782, 'std': 0.001, 'status': 'Not Best'},
    'CYP2D6_substrate': {'metric': 'AUPRC', 'result': 0.692, 'std': 0.017, 'status': 'Not Best'},
    'CYP3A4_substrate': {'metric': 'AUROC', 'result': 0.692, 'std': 0.019, 'status': 'Best'},
    'CYP2C9_substrate': {'metric': 'AUPRC', 'result': 0.409, 'std': 0.014, 'status': 'Not Best'},
    'Half_life': {'metric': 'Spearman', 'result': 0.578, 'std': 0.032, 'status': 'Best'},
    'Clearance_microsome': {'metric': 'Spearman', 'result': 0.632, 'std': 0.008, 'status': 'Best'},
    'Clearance_hepatocyte': {'metric': 'Spearman', 'result': 0.456, 'std': 0.027, 'status': 'Not Best'}
}

def get_task_info() -> Dict:
    """
    Get comprehensive task information including task types, metrics, and TDC columns.
    
    Returns:
        Dict containing task information organized by prediction head type
    """
    return TASK_CONFIG

def get_regression_tasks() -> List[str]:
    """Get list of regression task names."""
    return TASK_CONFIG['regression']['tasks']

def get_classification_tasks() -> List[str]:
    """Get list of classification task names."""
    return TASK_CONFIG['classification']['tasks']

def get_all_tasks() -> List[str]:
    """Get list of all task names."""
    return TASK_CONFIG['regression']['tasks'] + TASK_CONFIG['classification']['tasks']

def get_tdc_column_mapping() -> Dict[str, str]:
    """
    Get mapping from task names to TDC column names.
    
    Returns:
        Dict mapping task name to TDC column name
    """
    mapping = {}
    for task_type in ['regression', 'classification']:
        for task, column in zip(TASK_CONFIG[task_type]['tasks'], TASK_CONFIG[task_type]['tdc_columns']):
            mapping[task] = column
    return mapping

def get_metric_mapping() -> Dict[str, str]:
    """
    Get mapping from task names to evaluation metrics.
    
    Returns:
        Dict mapping task name to metric name
    """
    mapping = {}
    for task_type in ['regression', 'classification']:
        for task, metric in zip(TASK_CONFIG[task_type]['tasks'], TASK_CONFIG[task_type]['metrics']):
            mapping[task] = metric
    return mapping

def get_task_type_mapping() -> Dict[str, str]:
    """
    Get mapping from task names to task type (regression/classification).
    
    Returns:
        Dict mapping task name to task type
    """
    mapping = {}
    for task_type in ['regression', 'classification']:
        for task in TASK_CONFIG[task_type]['tasks']:
            mapping[task] = task_type
    return mapping

def validate_dataset_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Validate that all required TDC columns are present in the dataset.
    
    Args:
        df: DataFrame containing TDC data
        
    Returns:
        Tuple of (available_columns, missing_columns)
    """
    all_required_columns = []
    for task_type in ['regression', 'classification']:
        all_required_columns.extend(TASK_CONFIG[task_type]['tdc_columns'])
    
    available_columns = [col for col in all_required_columns if col in df.columns]
    missing_columns = [col for col in all_required_columns if col not in df.columns]
    
    return available_columns, missing_columns

def print_task_summary():
    """Print a summary of all tasks and their configurations."""
    print("=== Table 1 Task Configuration Summary ===")
    
    for task_type in ['regression', 'classification']:
        print(f"\n{task_type.upper()} TASKS ({len(TASK_CONFIG[task_type]['tasks'])} tasks):")
        for i, (task, metric, column) in enumerate(zip(
            TASK_CONFIG[task_type]['tasks'],
            TASK_CONFIG[task_type]['metrics'],
            TASK_CONFIG[task_type]['tdc_columns']
        )):
            mole_result = MOLE_PAPER_RESULTS.get(task, {})
            status = mole_result.get('status', 'Unknown')
            print(f"  {i+1:2d}. {task:20s} | {metric:8s} | {column:35s} | {status}")
    
    print(f"\nTotal tasks: {len(get_all_tasks())}")
    print(f"Regression tasks: {len(get_regression_tasks())}")
    print(f"Classification tasks: {len(get_classification_tasks())}")

if __name__ == "__main__":
    print_task_summary()