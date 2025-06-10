#!/usr/bin/env python3
"""
Test script for the Atom Environment Atlas Generator.
This creates a smaller test version to verify functionality.
"""

from create_atomenv_atlas import AtomEnvAtlasGenerator


def test_atlas_generation():
    """Test the atlas generation with a smaller dataset."""
    print("🧪 Testing Atom Environment Atlas Generator")
    print("=" * 50)

    # Create test generator with smaller limits
    generator = AtomEnvAtlasGenerator(
        smiles_file="guacamol_v1_all.smiles",
        vocab_file="../mole/data/vocabularies/vocabulary_radius0_structural_guacamol_v1.pkl",
        output_pdf="test_atomenv_atlas.pdf",
        max_molecules_to_scan=10000,  # Smaller limit for testing (default is None = unlimited)
        mol_size=(200, 200),  # Smaller images for testing
    )

    try:
        # Test vocabulary loading
        print("\n1. Testing vocabulary loading...")
        atom_envs = generator.load_vocabulary()
        print(f"   ✅ Loaded {len(atom_envs)} atom environments")

        # Test molecule scanning (limited)
        print("\n2. Testing molecule scanning...")
        generator.scan_for_examples()

        if generator.atom_env_examples:
            print(
                f"   ✅ Found examples for {len(generator.atom_env_examples)} environments"
            )

            # Test PDF creation
            print("\n3. Testing PDF creation...")
            generator.create_atlas_pdf()
            print(f"   ✅ Created PDF: {generator.output_pdf}")
        else:
            print("   ⚠️  No examples found, skipping PDF creation")

        # Show summary
        generator.create_summary_stats()

        print("\n🎉 Test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_atlas_generation()
