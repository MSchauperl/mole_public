#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute pairwise overlap (intersection) counts of compounds across given assay columns "
			"from a wide chembl_complete_dataset.csv without loading the full file into memory."
		)
	)
	parser.add_argument(
		"--csv",
		type=Path,
		required=True,
		help="Path to chembl_complete_dataset.csv (wide format, columns like assay_CHEMBLXXXXX)",
	)
	parser.add_argument(
		"--assays",
		nargs="+",
		required=True,
		help="List of assay IDs (e.g. CHEMBL1614161 CHEMBL1614441 ...)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("assay_overlap_matrix.csv"),
		help="Path to write the overlap matrix CSV",
	)
	parser.add_argument(
		"--compound-col",
		default="compound_id",
		help="Name of compound identifier column",
	)
	parser.add_argument(
		"--present-threshold",
		type=float,
		default=0.5,
		help=(
			"Value threshold to treat an assay cell as present/True. "
			"Cells >= threshold are counted as present (default 0.5)."
		),
	)
	return parser.parse_args()


def stream_overlap_counts(
	csv_path: Path,
	assays: List[str],
	compound_col: str,
	present_threshold: float,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], int]:
	"""
	Stream the CSV and compute:
	- per_assay_count: present count per assay
	- pair_counts: pairwise intersection counts
	- all_count: count of compounds present in all assays
	"""
	assay_cols = [f"assay_{aid}" for aid in assays]

	per_assay_count: Dict[str, int] = defaultdict(int)
	pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
	all_count = 0

	with csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		missing = [c for c in assay_cols if c not in reader.fieldnames]
		if missing:
			raise ValueError(
				f"Missing expected columns in CSV: {missing}. Available: {reader.fieldnames[:10]} ..."
			)

		for row in reader:
			present = []
			for aid, col in zip(assays, assay_cols):
				val = row[col]
				if val == "" or val is None:
					is_present = False
				else:
					try:
						is_present = float(val) >= present_threshold
					except ValueError:
						# Non-numeric; treat non-empty as present
						is_present = True
				if is_present:
					per_assay_count[aid] += 1
					present.append(aid)

			# pairwise
			for a, b in combinations(sorted(present), 2):
				pair_counts[(a, b)] += 1

			# all assays
			if len(present) == len(assays):
				all_count += 1

	return per_assay_count, pair_counts, all_count


def write_matrix(
	assays: List[str],
	per_assay_count: Dict[str, int],
	pair_counts: Dict[Tuple[str, str], int],
	all_count: int,
	output_path: Path,
) -> None:
	# Build square matrix with headers
	header = ["assay"] + assays
	rows: List[List[str]] = []
	for a in assays:
		row: List[str] = [a]
		for b in assays:
			if a == b:
				row.append(str(per_assay_count.get(a, 0)))
			else:
				key = (a, b) if a < b else (b, a)
				row.append(str(pair_counts.get(key, 0)))
		rows.append(row)

	with output_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["all_assays_intersection", all_count])
		writer.writerow(header)
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	per, pairs, all_count = stream_overlap_counts(
		csv_path=args.csv,
		assays=args.assays,
		compound_col=args.compound_col,
		present_threshold=args.present_threshold,
	)
	write_matrix(args.assays, per, pairs, all_count, args.output)
	print(f"Wrote overlap matrix to {args.output} (all-assays intersection: {all_count})")


if __name__ == "__main__":
	main()
