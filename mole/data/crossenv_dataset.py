"""
Cross-Environment Dataset for MolE Pretraining

This dataset creates training data where:
- Input tokens: Radius 0 structural atom environments
- Target tokens: Radius 1 functional atom environments

This enables the model to learn rich chemical representations by predicting
functional atom environments from structural ones.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem


from mole.data.vocabulary import open_dictionary


class CrossEnvMolDataset(Dataset):
    """Dataset for cross-environment masked language modeling pretraining"""

    def __init__(
        self,
        smiles: pd.Series,
        input_vocab_path: str,
        target_vocab_path: str,
        input_radius: int = 0,
        target_radius: int = 1,
        input_use_features: bool = False,
        target_use_features: bool = True,
        mask_prob: float = 0.15,
        replace_prob: float = 0.8,
        random_prob: float = 0.1,
        max_length: Optional[int] = None,
        cls_token: bool = True,
        enable_masking: bool = True,
    ):
        """
        Initialize cross-environment dataset

        Args:
            smiles: Series of SMILES strings
            input_vocab_path: Path to input vocabulary (radius 0 structural)
            target_vocab_path: Path to target vocabulary (radius 1 functional)
            input_radius: Radius for input atom environments
            target_radius: Radius for target atom environments
            input_use_features: Whether to use features for input environments
            target_use_features: Whether to use features for target environments
            mask_prob: Probability of masking a token (default 15%)
            replace_prob: Probability of replacing with [MASK] (default 80%)
            random_prob: Probability of replacing with random token (default 10%)
            max_length: Maximum sequence length (None for no limit)
            cls_token: Whether to add CLS token at beginning
            enable_masking: Whether to apply masking (False for classification-only mode)
        """
        self.smiles = smiles
        self.input_radius = input_radius
        self.target_radius = target_radius
        self.input_use_features = input_use_features
        self.target_use_features = target_use_features
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.max_length = max_length
        self.cls_token = cls_token
        self.enable_masking = enable_masking

        # Load vocabularies
        self.input_vocab = open_dictionary(input_vocab_path)
        self.target_vocab = open_dictionary(target_vocab_path)

        # Get special token IDs
        self.input_pad_id = self.input_vocab.get("PAD", 0)
        self.input_mask_id = self.input_vocab.get("MASK", 1)
        self.input_unk_id = self.input_vocab.get("UNK", 2)
        self.input_cls_id = self.input_vocab.get("CLS", 3)

        self.target_pad_id = self.target_vocab.get("PAD", 0)
        self.target_unk_id = self.target_vocab.get("UNK", 2)

        # Vocabulary sizes
        self.input_vocab_size = len(self.input_vocab)
        self.target_vocab_size = len(self.target_vocab)

        print(f"CrossEnvMolDataset initialized:")
        print(
            f"  Input vocab: {self.input_vocab_size} tokens (radius={input_radius}, features={input_use_features})"
        )
        print(
            f"  Target vocab: {self.target_vocab_size} tokens (radius={target_radius}, features={target_use_features})"
        )
        print(
            f"  Masking: {mask_prob:.1%} prob, {replace_prob:.1%} [MASK], {random_prob:.1%} random"
        )

    def get_atom_environments(
        self, mol: Chem.Mol, radius: int, use_features: bool
    ) -> List[int]:
        """Extract atom environments for a molecule"""
        if mol is None:
            return []

        info = {}
        _ = AllChem.GetMorganFingerprint(
            mol,
            radius=radius,
            bitInfo=info,
            useFeatures=use_features,
            includeRedundantEnvironments=True,
        )

        atom_envs = [None] * mol.GetNumAtoms()
        for atom_idx in range(mol.GetNumAtoms()):
            for bit, atom_list in info.items():
                for atom_info_tuple in atom_list:
                    if atom_info_tuple[0] == atom_idx and atom_info_tuple[1] == radius:
                        atom_envs[atom_idx] = bit
                        break
                if atom_envs[atom_idx] is not None:
                    break

        return atom_envs

    def encode_environments(
        self, environments: List[int], vocab: Dict[str, int], unk_id: int
    ) -> List[int]:
        """Encode atom environments using vocabulary"""
        tokens = []
        found_count = 0
        for env in environments:
            if env is None:
                tokens.append(unk_id)
            else:
                # Fix: Use integer key directly instead of converting to string
                token = vocab.get(env, unk_id)
                if token != unk_id:
                    found_count += 1
                tokens.append(token)
        
# Vocabulary coverage debugging removed - issue was fixed (int vs string key mismatch)
        
        return tokens

    def create_mlm_sample(
        self, input_tokens: List[int], target_tokens: List[int]
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """Create masked language modeling sample"""
        input_tokens = np.array(input_tokens, dtype=np.int64)
        target_tokens = np.array(target_tokens, dtype=np.int64)

        # Create copy for masking
        masked_input = input_tokens.copy()

        # Initialize labels (all -100 means ignore in loss)
        labels = np.full_like(target_tokens, -100, dtype=np.int64)

        # Decide which positions to mask (avoid CLS token if present)
        start_idx = 1 if self.cls_token else 0
        mask_decisions = np.random.random(len(input_tokens)) < self.mask_prob
        if self.cls_token:
            mask_decisions[0] = False  # Never mask CLS token

        masked_positions = np.where(mask_decisions)[0]

        # Ensure at least one token is masked (excluding CLS)
        if len(masked_positions) == 0 and len(input_tokens) > start_idx:
            pos = np.random.randint(start_idx, len(input_tokens))
            masked_positions = [pos]
            mask_decisions[pos] = True

        # Apply masking strategy
        for pos in masked_positions:
            rand = np.random.random()

            if rand < self.replace_prob:
                # Replace with [MASK] token
                masked_input[pos] = self.input_mask_id
            elif rand < self.replace_prob + self.random_prob:
                # Replace with random token (avoid special tokens)
                random_token = np.random.randint(4, min(self.input_vocab_size, 1000))
                masked_input[pos] = random_token
            # else: keep original token

            # Set target label for this position
            labels[pos] = target_tokens[pos]

        return masked_input.tolist(), labels.tolist(), mask_decisions

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Data:
        """Get a single training sample"""
        smiles_str = self.smiles.iloc[idx]

        try:
            # Parse molecule
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles_str}")

            # Extract input and target atom environments
            input_envs = self.get_atom_environments(
                mol, self.input_radius, self.input_use_features
            )
            target_envs = self.get_atom_environments(
                mol, self.target_radius, self.target_use_features
            )

            if len(input_envs) == 0 or len(target_envs) == 0:
                raise ValueError(f"No atom environments found for: {smiles_str}")

            if len(input_envs) != len(target_envs):
                raise ValueError(
                    f"Mismatch in environment lengths: {len(input_envs)} vs {len(target_envs)}"
                )

            # Encode with vocabularies
            input_tokens = self.encode_environments(
                input_envs, self.input_vocab, self.input_unk_id
            )
            target_tokens = self.encode_environments(
                target_envs, self.target_vocab, self.target_unk_id
            )

            # Add CLS token if required
            if self.cls_token:
                input_tokens = [self.input_cls_id] + input_tokens
                target_tokens = [
                    self.target_unk_id
                ] + target_tokens  # CLS target is ignored

            # Apply length limit if specified
            if self.max_length is not None:
                input_tokens = input_tokens[: self.max_length]
                target_tokens = target_tokens[: self.max_length]

            # Create MLM sample (conditionally apply masking)
            if self.enable_masking:
                masked_input, labels, mask = self.create_mlm_sample(
                    input_tokens, target_tokens
                )
            else:
                # For classification-only mode: use original tokens, no masking
                masked_input = input_tokens
                labels = [-100] * len(target_tokens)  # Ignore all labels for MLM loss
                mask = [False] * len(input_tokens)  # No tokens are masked

            # Create distance matrix for molecular graph
            dist_mat = Chem.GetDistanceMatrix(mol)
            dist_mat[dist_mat == 1.0e08] = -1

            # Convert to sparse format
            from scipy import sparse

            dist_mat_sparse = sparse.coo_matrix(dist_mat + 1)

            # Adjust for CLS token
            if self.cls_token:
                # Add row and column for CLS token
                dist_mat_sparse = sparse.vstack(
                    [np.zeros(dist_mat_sparse.shape[0])[None, :], dist_mat_sparse]
                )
                dist_mat_sparse = sparse.hstack(
                    [np.zeros(dist_mat_sparse.shape[0])[:, None], dist_mat_sparse]
                )

            # Create PyTorch tensors
            data_dict = {
                "x": torch.tensor(masked_input, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "original_input": torch.tensor(input_tokens, dtype=torch.long),
                "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
                "edge_index": torch.tensor(
                    np.array([dist_mat_sparse.row, dist_mat_sparse.col]),
                    dtype=torch.long,
                ),
                "edge_attr": torch.tensor(dist_mat_sparse.data, dtype=torch.long),
                "smiles": smiles_str,
            }

            return Data.from_dict(data_dict)

        except Exception as e:
            # Return a dummy sample for failed molecules
            print(f"Warning: Failed to process molecule {idx}: {e}")
            return self.__handle_fail_case__(idx, smiles_str)

    def __handle_fail_case__(self, idx: int, smiles_str: str) -> Data:
        """Creates a dummy sample for a molecule that failed to process"""
        # Create minimal dummy sample
        dummy_input = (
            [self.input_cls_id, self.input_unk_id]
            if self.cls_token
            else [self.input_unk_id]
        )
        dummy_target = (
            [self.target_unk_id, self.target_unk_id]
            if self.cls_token
            else [self.target_unk_id]
        )
        dummy_labels = [-100, -100] if self.cls_token else [-100]

        data_dict = {
            "x": torch.tensor(dummy_input, dtype=torch.long),
            "labels": torch.tensor(dummy_labels, dtype=torch.long),
            "original_input": torch.tensor(dummy_input, dtype=torch.long),
            "target_tokens": torch.tensor(dummy_target, dtype=torch.long),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_attr": torch.tensor([1, 1], dtype=torch.long),
            "smiles": smiles_str,
            "dummy": torch.tensor(True, dtype=torch.bool),
        }

        return Data.from_dict(data_dict)
