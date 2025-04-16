import argparse
import logging
import os
from typing import List
import warnings
import sys

import numpy as np
import onnxruntime
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch

from mole.data.dataloaders import MolDataModule
from mole.models.mole import Supervised
from mole.models.encoders import Encoder

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

DEFAULT_VOCAB_NAME = "vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"


def find_vocab_path(name=DEFAULT_VOCAB_NAME):
    script_dir = os.path.dirname(__file__)
    potential_paths = [
        os.path.join(script_dir, "..", "data", "vocabularies", name),
        os.path.join(script_dir, "..", "..", "data", "vocabularies", name),
    ]
    for path in potential_paths:
        if os.path.exists(path):
            return path
    logger.warning(
        f"Vocabulary file {name} not found in standard locations. Using default name."
    )
    return name


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, help="smiles used as input for the model")
    parser.add_argument(
        "--task", default="regression", help="classification or regression"
    )
    parser.add_argument("--num_tasks", default=1, type=int, help="number of outputs")
    parser.add_argument(
        "--num_classes",
        default=1,
        type=int,
        help="number of classes in classification task",
    )
    parser.add_argument(
        "--pretrained_model",
        required=True,
        help="path to pretrained model (.ckpt or .onnx)",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size use for loader"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of CPU workers"
    )
    parser.add_argument(
        "--accelerator", default="auto", type=str, help="Choose accelerator to use"
    )
    parser.add_argument(
        "--vocabulary_path",
        type=str,
        default=None,
        help=f"Path to vocabulary pkl file. Defaults to finding {DEFAULT_VOCAB_NAME}.",
    )
    args = parser.parse_args(argv)
    return args


def encode(
    smiles: List[str],
    pretrained_model: str,
    vocabulary_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> np.ndarray:
    datamodule_args = {
        "data": smiles,
        "vocabulary_inp": vocabulary_path,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    datamodule = MolDataModule(**datamodule_args)

    try:
        model = Encoder.load_from_checkpoint(pretrained_model, strict=False)
    except Exception as e:
        logger.error(
            f"Failed to load Encoder model from checkpoint {pretrained_model}: {e}"
        )
        raise

    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, enable_progress_bar=False, logger=False
    )

    model.eval()
    with torch.no_grad():
        outputs = trainer.predict(model, datamodule=datamodule)

    if outputs is None or not outputs:
        logger.error("Trainer prediction returned None or empty list.")
        return np.array([])

    try:
        embeddings = torch.cat(outputs).cpu().numpy()
    except Exception as e:
        logger.error(
            f"Failed to process model outputs for embedding concatenation: {e}"
        )
        if outputs:
            logger.error(
                f"First output element type: {type(outputs[0])}, content: {outputs[0]}"
            )
        return np.array([])

    return embeddings


def predict_onnx(
    smiles: List[str],
    pretrained_model: str,
    vocabulary_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    so = onnxruntime.SessionOptions()
    so.inter_op_num_threads = num_workers
    so.intra_op_num_threads = 2

    try:
        ort_session = onnxruntime.InferenceSession(pretrained_model, sess_options=so)
    except Exception as e:
        logger.error(f"Failed to load ONNX model {pretrained_model}: {e}")
        raise

    datamodule_args = {
        "data": smiles,
        "vocabulary_inp": vocabulary_path,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    datamodule = MolDataModule(**datamodule_args)
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    outputs = []
    try:
        for batch in dataloader:
            input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
            relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
            ort_inputs = {
                "input_ids": input_ids.cpu().numpy(),
                "input_mask": input_mask.cpu().numpy(),
                "relative_pos": relative_pos.cpu().numpy(),
            }
            ort_outs = ort_session.run(None, ort_inputs)
            outputs.append(ort_outs[0])
    except Exception as e:
        logger.error(f"Error during ONNX prediction loop: {e}")
        raise

    if not outputs:
        logger.warning("ONNX prediction loop resulted in empty output list.")
        return np.array([])

    predictions = np.concatenate(outputs)

    return predictions


def predict_ckpt(
    smiles: List[str],
    task: str,
    num_tasks: int,
    num_classes: int,
    pretrained_model: str,
    vocabulary_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> np.ndarray:
    datamodule_args = {
        "data": smiles,
        "vocabulary_inp": vocabulary_path,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    datamodule = MolDataModule(**datamodule_args)

    try:
        model = Supervised.load_from_checkpoint(
            pretrained_model,
            strict=False,
        )
    except Exception as e:
        logger.error(
            f"Failed to load Supervised model from checkpoint {pretrained_model}: {e}"
        )
        raise

    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, enable_progress_bar=False, logger=False
    )

    model.eval()
    with torch.no_grad():
        outputs = trainer.predict(model, datamodule=datamodule)

    if outputs is None or not outputs:
        logger.error("Trainer prediction returned None or empty list.")
        return np.array([])

    try:
        predictions = torch.cat([output["logits"] for output in outputs]).cpu().numpy()
    except (KeyError, TypeError) as e:
        logger.error(f"Failed to extract 'logits' from model outputs: {e}")
        if outputs:
            logger.error(
                f"First output element type: {type(outputs[0])}, content: {outputs[0]}"
            )
        return np.array([])
    except Exception as e:
        logger.error(
            f"Failed to process model outputs for prediction concatenation: {e}"
        )
        return np.array([])

    return predictions


def predict(
    smiles: List[str],
    task: str = "regression",
    num_tasks: int = 1,
    num_classes: int = 1,
    pretrained_model: str = "null",
    vocabulary_path: str = "null",
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> np.ndarray:
    if vocabulary_path == "null" or vocabulary_path is None:
        resolved_vocab_path = find_vocab_path()
        if resolved_vocab_path == DEFAULT_VOCAB_NAME:
            raise FileNotFoundError(
                f"Vocabulary '{DEFAULT_VOCAB_NAME}' not found automatically. "
                f"Use --vocabulary_path."
            )
    else:
        resolved_vocab_path = vocabulary_path
    if not os.path.exists(resolved_vocab_path):
        raise FileNotFoundError(
            f"Specified vocabulary file not found: {resolved_vocab_path}"
        )

    if pretrained_model == "null" or not pretrained_model:
        raise ValueError(
            "A pretrained model path (--pretrained_model) must be provided."
        )

    if ".onnx" in pretrained_model:
        predictions = predict_onnx(
            smiles,
            pretrained_model,
            vocabulary_path=resolved_vocab_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        predictions = predict_ckpt(
            smiles=smiles,
            task=task,
            num_tasks=num_tasks,
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            vocabulary_path=resolved_vocab_path,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator=accelerator,
        )

    return predictions


def main(argv=None):
    args = parse_args(argv)
    if args.vocabulary_path is None:
        args.vocabulary_path = find_vocab_path()
        if args.vocabulary_path == DEFAULT_VOCAB_NAME:
            print(
                f"Error: Vocabulary file '{DEFAULT_VOCAB_NAME}' not found. Use --vocabulary_path.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif not os.path.exists(args.vocabulary_path):
        print(
            f"Error: Specified vocabulary file not found: {args.vocabulary_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    filename = None
    if os.path.isfile(args.smiles):
        filename = args.smiles
        file_ext = filename.split(".")[-1].lower()
        if file_ext == "csv":
            loader = pd.read_csv
        elif file_ext == "parquet":
            loader = pd.read_parquet
        else:
            print(
                f"Error: Unsupported input file type '.{file_ext}'. Use CSV or Parquet.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            df = loader(filename)
            if "smiles" not in df.columns:
                print(
                    f"Error: Input file '{filename}' must contain a 'smiles' column.",
                    file=sys.stderr,
                )
                sys.exit(1)
            smiles_list = df.smiles.to_list()
        except Exception as e:
            print(f"Error reading input file {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        smiles_list = args.smiles.split()
        if not smiles_list:
            print("Error: No SMILES provided.", file=sys.stderr)
            sys.exit(1)

    try:
        output = predict(
            smiles=smiles_list,
            task=args.task,
            num_tasks=args.num_tasks,
            num_classes=args.num_classes,
            pretrained_model=args.pretrained_model,
            vocabulary_path=args.vocabulary_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
        )
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

    if filename:
        output_df = pd.DataFrame(output)
        output_df.columns = [f"prediction_{i}" for i in range(output_df.shape[1])]
        result = pd.concat([df, output_df], axis=1)
        try:
            result.to_csv("predictions.csv", index=False)
            print("Predictions saved to predictions.csv")
        except Exception as e:
            print(f"Error saving predictions to predictions.csv: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Predictions:")
        print(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
