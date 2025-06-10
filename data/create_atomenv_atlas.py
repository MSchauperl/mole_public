#!/usr/bin/env python3
"""
Atom Environment Atlas Generator

This script creates a visual atlas of all atom environments (radius 0, structural)
found in the GuacaMol dataset. For each atom environment, it finds an example molecule,
visualizes it with the atom highlighted, and combines all visualizations into a
multi-page PDF with 4 columns × 5 rows per page.

Author: AI Assistant
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm

# For PDF generation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import io


class AtomEnvAtlasGenerator:
    """Generate a visual atlas of atom environments from GuacaMol SMILES."""

    def __init__(
        self,
        smiles_file: str = "guacamol_v1_all.smiles",
        vocab_file: str = "../mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        output_pdf: str = "atomenv_atlas_radius0_structural.pdf",
        max_molecules_to_scan: int = None,
        molecules_per_page: int = 20,
        mol_size: Tuple[int, int] = (300, 300),
    ):
        """
        Initialize the atlas generator.

        Args:
            smiles_file: Path to GuacaMol SMILES file
            vocab_file: Path to vocabulary pickle file
            output_pdf: Output PDF filename
            max_molecules_to_scan: Maximum molecules to scan (None = scan all until complete)
            molecules_per_page: Number of molecules per page (4x5 grid = 20)
            mol_size: Size of each molecule image
        """
        self.smiles_file = Path(smiles_file)
        self.vocab_file = Path(vocab_file)
        self.output_pdf = Path(output_pdf)
        self.max_molecules_to_scan = max_molecules_to_scan
        self.molecules_per_page = molecules_per_page
        self.mol_size = mol_size

        # Grid layout (4 columns × 5 rows)
        self.grid_cols = 4
        self.grid_rows = 5

        # Data storage
        self.vocabulary = {}
        self.atom_env_examples = {}  # atom_env_hash -> (smiles, atom_idx, element)

    def load_vocabulary(self) -> Dict:
        """Load the atom environment vocabulary."""
        print(f"📚 Loading vocabulary from {self.vocab_file}")

        if not self.vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}")

        with open(self.vocab_file, "rb") as f:
            self.vocabulary = pickle.load(f)

        # Filter out special tokens
        atom_envs = {
            k: v
            for k, v in self.vocabulary.items()
            if isinstance(k, int) and k not in ["PAD", "MASK", "UNK", "CLS"]
        }

        print(f"✅ Loaded {len(atom_envs)} atom environments")
        print(
            f"   Special tokens: {[k for k in self.vocabulary.keys() if isinstance(k, str)]}"
        )

        return atom_envs

    def get_atom_environments(
        self, mol: Chem.Mol, radius: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Get atom environments for a molecule.

        Args:
            mol: RDKit molecule
            radius: Environment radius

        Returns:
            List of (atom_idx, env_hash) tuples
        """
        if mol is None:
            return []

        try:
            # Generate Morgan fingerprints for each atom
            fp_info = {}
            AllChem.GetMorganFingerprint(mol, radius, bitInfo=fp_info)

            atom_envs = []
            for env_hash, atom_lists in fp_info.items():
                for atom_idx, _ in atom_lists:
                    atom_envs.append((atom_idx, env_hash))

            return atom_envs
        except Exception as e:
            print(f"⚠️  Error getting atom environments: {e}")
            return []

    def scan_for_examples(self):
        """Scan SMILES file to find example molecules for each atom environment."""
        print(f"🔍 Scanning {self.smiles_file} for atom environment examples")

        if not self.smiles_file.exists():
            raise FileNotFoundError(f"SMILES file not found: {self.smiles_file}")

        atom_envs = self.load_vocabulary()
        target_envs = set(atom_envs.keys())
        found_envs = set()

        print(f"🎯 Looking for {len(target_envs)} atom environments")
        if self.max_molecules_to_scan is None:
            print("📝 Scanning all molecules until all environments are found")
        else:
            print(f"📝 Scanning up to {self.max_molecules_to_scan:,} molecules")

        with open(self.smiles_file, "r") as f:
            for i, line in enumerate(tqdm(f, desc="Scanning molecules")):
                if (
                    self.max_molecules_to_scan is not None
                    and i >= self.max_molecules_to_scan
                ):
                    break

                smiles = line.strip()
                if not smiles or smiles.startswith("#"):
                    continue

                # Handle different file formats (take first column)
                if "\t" in smiles:
                    smiles = smiles.split("\t")[0]
                elif "," in smiles:
                    smiles = smiles.split(",")[0]

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Get atom environments for this molecule
                atom_envs_in_mol = self.get_atom_environments(mol, radius=0)

                for atom_idx, env_hash in atom_envs_in_mol:
                    if env_hash in target_envs and env_hash not in found_envs:
                        # Get the element of the highlighted atom
                        atom = mol.GetAtomWithIdx(atom_idx)
                        element = atom.GetSymbol()
                        self.atom_env_examples[env_hash] = (smiles, atom_idx, element)
                        found_envs.add(env_hash)

                        # Stop early if we found all environments
                        if len(found_envs) == len(target_envs):
                            print(
                                f"🎉 Found all {len(target_envs)} atom environments after {i+1:,} molecules!"
                            )
                            break

                # Progress update
                if i % 50000 == 0 and found_envs:
                    print(
                        f"   Found {len(found_envs)}/{len(target_envs)} environments after {i:,} molecules"
                    )

        print(
            f"✅ Found examples for {len(found_envs)}/{len(target_envs)} atom environments"
        )

        if len(found_envs) < len(target_envs):
            missing = target_envs - found_envs
            print(
                f"⚠️  Missing examples for {len(missing)} environments: {list(missing)[:10]}..."
            )
            if self.max_molecules_to_scan is not None:
                print(
                    f"💡 Consider increasing --max-molecules (currently {self.max_molecules_to_scan:,}) or remove limit"
                )

    def create_molecule_image(self, smiles: str, highlight_atom: int) -> Image.Image:
        """
        Create a molecule image with highlighted atom.

        Args:
            smiles: SMILES string
            highlight_atom: Atom index to highlight

        Returns:
            PIL Image of the molecule
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Create error image
            img = Image.new("RGB", self.mol_size, color="white")
            return img

        try:
            # Create drawer
            drawer = rdMolDraw2D.MolDraw2DCairo(*self.mol_size)

            # Set up highlighting
            highlight_atoms = [highlight_atom]
            highlight_colors = {highlight_atom: (1.0, 0.0, 0.0)}  # Red

            # Draw molecule with highlighting
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_colors,
            )
            drawer.FinishDrawing()

            # Convert to PIL Image
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))

            return img

        except Exception as e:
            print(f"⚠️  Error creating image for {smiles}: {e}")
            # Create error image
            img = Image.new("RGB", self.mol_size, color="lightgray")
            return img

    def create_atlas_pdf(self):
        """Create the multi-page PDF atlas."""
        print(f"📖 Creating PDF atlas: {self.output_pdf}")

        if not self.atom_env_examples:
            print(
                "❌ No atom environment examples found. Run scan_for_examples() first."
            )
            return

        # Sort environments by element first, then by vocabulary index for consistent ordering
        def sort_key(item):
            env_hash, (smiles, atom_idx, element) = item
            vocab_idx = self.vocabulary.get(env_hash, float("inf"))
            # Define element priority (common elements first)
            element_priority = {
                "C": 0,
                "N": 1,
                "O": 2,
                "S": 3,
                "P": 4,
                "F": 5,
                "Cl": 6,
                "Br": 7,
                "I": 8,
            }
            priority = element_priority.get(element, 999)  # Unknown elements last
            return (priority, element, vocab_idx)

        sorted_envs = sorted(self.atom_env_examples.items(), key=sort_key)

        with PdfPages(self.output_pdf) as pdf:
            total_envs = len(sorted_envs)
            pages_needed = (
                total_envs + self.molecules_per_page - 1
            ) // self.molecules_per_page

            print(f"📄 Creating {pages_needed} pages for {total_envs} environments")

            for page_idx in range(pages_needed):
                print(f"   Creating page {page_idx + 1}/{pages_needed}")

                # Calculate figure size based on grid
                fig_width = self.grid_cols * 4  # 4 inches per molecule
                fig_height = self.grid_rows * 4  # 4 inches per molecule

                fig, axes = plt.subplots(
                    self.grid_rows, self.grid_cols, figsize=(fig_width, fig_height)
                )
                fig.suptitle(
                    f"Atom Environment Atlas (Radius 0, Structural) - Page {page_idx + 1}\nSorted by Element",
                    fontsize=16,
                    fontweight="bold",
                )

                # Flatten axes for easier indexing
                if self.grid_rows == 1:
                    axes = [axes]
                elif self.grid_cols == 1:
                    axes = [[ax] for ax in axes]
                axes_flat = [ax for row in axes for ax in row]

                # Fill grid with molecules
                start_idx = page_idx * self.molecules_per_page
                end_idx = min(start_idx + self.molecules_per_page, total_envs)

                for i, ax in enumerate(axes_flat):
                    mol_idx = start_idx + i

                    if mol_idx < end_idx:
                        # Get environment data
                        env_hash, (smiles, atom_idx, element) = sorted_envs[mol_idx]
                        vocab_idx = self.vocabulary.get(env_hash, "?")

                        # Create molecule image
                        img = self.create_molecule_image(smiles, atom_idx)

                        # Display image
                        ax.imshow(img)
                        ax.set_title(
                            f"Env #{vocab_idx} ({element})\nHash: {env_hash}\nAtom: {atom_idx}",
                            fontsize=8,
                            pad=5,
                        )
                        ax.axis("off")
                    else:
                        # Empty cell
                        ax.axis("off")

                plt.tight_layout()
                pdf.savefig(fig, dpi=150, bbox_inches="tight")
                plt.close(fig)

        print(f"✅ Atlas saved to {self.output_pdf}")

    def generate_atlas(self):
        """Main method to generate the complete atlas."""
        print("🚀 Starting Atom Environment Atlas Generation")
        print("=" * 50)

        try:
            # Step 1: Scan for examples
            self.scan_for_examples()

            # Step 2: Create PDF
            self.create_atlas_pdf()

            print("\n🎉 Atlas generation complete!")
            print(f"📁 Output: {self.output_pdf.absolute()}")

        except Exception as e:
            print(f"❌ Error during atlas generation: {e}")
            raise

    def create_summary_stats(self):
        """Create summary statistics about the atom environments."""
        print("\n📊 Atom Environment Summary")
        print("-" * 30)

        atom_envs = {k: v for k, v in self.vocabulary.items() if isinstance(k, int)}

        print(f"Total atom environments: {len(atom_envs)}")
        print(f"Examples found: {len(self.atom_env_examples)}")
        print(f"Coverage: {len(self.atom_env_examples)/len(atom_envs)*100:.1f}%")

        # Element statistics
        if self.atom_env_examples:
            from collections import Counter

            elements = [data[2] for data in self.atom_env_examples.values()]
            element_counts = Counter(elements)

            print("\n🧪 Element Distribution:")
            for element, count in sorted(element_counts.items()):
                percentage = count / len(self.atom_env_examples) * 100
                print(f"  {element}: {count} environments ({percentage:.1f}%)")

            print("\nExample environments (first 10):")
            for i, (env_hash, (smiles, atom_idx, element)) in enumerate(
                list(self.atom_env_examples.items())[:10]
            ):
                vocab_idx = self.vocabulary.get(env_hash, "?")
                print(
                    f"  {env_hash} (#{vocab_idx}, {element}): {smiles[:30]}... atom {atom_idx}"
                )


def main():
    """Main function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Atom Environment Visual Atlas"
    )
    parser.add_argument(
        "--smiles", default="guacamol_v1_all.smiles", help="Path to SMILES file"
    )
    parser.add_argument(
        "--vocab",
        default="../mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--output",
        default="atomenv_atlas_radius0_structural.pdf",
        help="Output PDF filename",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Maximum molecules to scan (default: unlimited)",
    )
    parser.add_argument(
        "--mol-size",
        type=int,
        nargs=2,
        default=[300, 300],
        help="Molecule image size (width height)",
    )

    args = parser.parse_args()

    # Create generator
    generator = AtomEnvAtlasGenerator(
        smiles_file=args.smiles,
        vocab_file=args.vocab,
        output_pdf=args.output,
        max_molecules_to_scan=args.max_molecules,
        mol_size=tuple(args.mol_size),
    )

    # Generate atlas
    generator.generate_atlas()
    generator.create_summary_stats()


if __name__ == "__main__":
    main()
