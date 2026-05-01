"""DeBERTa fine-tuning pipeline for six-target essay scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    TRANSFORMER_DEPS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on local CUDA/Python stack
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    AutoModel = None
    AutoTokenizer = None
    get_cosine_schedule_with_warmup = None
    TRANSFORMER_DEPS_AVAILABLE = False

from feedback_ell.baseline import save_submission
from feedback_ell.constants import TARGET_COLUMNS, TEXT_COLUMN
from feedback_ell.data import load_sample_submission, load_test, load_train, make_folds
from feedback_ell.metrics import columnwise_rmse, mcrmse
from feedback_ell.utils import ensure_dir, set_seed, write_json


class EssayDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int,
        targets: np.ndarray | None = None,
    ) -> None:
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class EssayRegressor(nn.Module if nn is not None else object):
    def __init__(self, model_name: str, dropout: float = 0.1, pooling: str = "mean") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, len(TARGET_COLUMNS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.head(self.dropout(pooled))


@dataclass
class FoldResult:
    fold: int
    best_mcrmse: float
    column_rmse: dict[str, float]
    model_path: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels", None)
            batch = _batch_to_device(batch, device)
            outputs = model(**batch)
            preds.append(outputs.detach().cpu().numpy())
            if labels is not None:
                batch["labels"] = labels
    return np.concatenate(preds, axis=0)


def train_one_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold: int,
    config: dict[str, Any],
    tokenizer,
    model_dir: Path,
) -> tuple[FoldResult, np.ndarray, np.ndarray]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    device = _device()
    trn = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val = train_df[train_df["fold"] == fold].reset_index(drop=True)

    train_ds = EssayDataset(
        trn[TEXT_COLUMN].tolist(),
        tokenizer,
        int(model_cfg["max_length"]),
        trn[TARGET_COLUMNS].to_numpy(dtype=float),
    )
    valid_ds = EssayDataset(
        val[TEXT_COLUMN].tolist(),
        tokenizer,
        int(model_cfg["max_length"]),
        val[TARGET_COLUMNS].to_numpy(dtype=float),
    )
    test_ds = EssayDataset(test_df[TEXT_COLUMN].tolist(), tokenizer, int(model_cfg["max_length"]))
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["train_batch_size"]), shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=int(train_cfg["eval_batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg["eval_batch_size"]), shuffle=False)

    model = EssayRegressor(model_cfg["name"], float(model_cfg["dropout"]), model_cfg["pooling"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_steps = max(1, len(train_loader) * int(train_cfg["epochs"]))
    warmup_steps = int(total_steps * float(train_cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg["fp16"]) and device.type == "cuda")
    criterion = nn.SmoothL1Loss()
    best_score = float("inf")
    best_pred = None
    patience = 0
    model_path = model_dir / f"deberta_fold{fold}.pt"

    for _epoch in range(int(train_cfg["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"fold {fold}", leave=False)
        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device)
            batch = _batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg["fp16"]) and device.type == "cuda"):
                outputs = model(**batch)
                loss = criterion(outputs, labels) / int(train_cfg["gradient_accumulation_steps"])
            scaler.scale(loss).backward()
            if step % int(train_cfg["gradient_accumulation_steps"]) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_pred = predict(model, valid_loader, device)
        score = mcrmse(val[TARGET_COLUMNS].to_numpy(dtype=float), val_pred)
        if score < best_score:
            best_score = score
            best_pred = val_pred
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= int(train_cfg["early_stopping_patience"]):
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_pred = predict(model, test_loader, device)
    assert best_pred is not None
    fold_result = FoldResult(
        fold=fold,
        best_mcrmse=float(best_score),
        column_rmse=columnwise_rmse(val[TARGET_COLUMNS].to_numpy(dtype=float), best_pred),
        model_path=str(model_path),
    )
    return fold_result, best_pred, test_pred


def run_transformer(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    artifact_dir = ensure_dir(config["output"]["artifacts_dir"])
    model_dir = ensure_dir(config["output"]["models_dir"])
    submission_dir = ensure_dir(config["output"]["submissions_dir"])
    if not TRANSFORMER_DEPS_AVAILABLE:
        summary = {
            "name": "deberta_v3_base",
            "status": "skipped",
            "reason": "PyTorch/Transformers are not installed for the active Python environment.",
            "cv_mcrmse": None,
            "column_rmse": {},
            "fold_scores": [],
            "submission_path": None,
            "trained_folds": [],
        }
        write_json(summary, artifact_dir / "transformer_metrics.json")
        return summary

    train = load_train(config["data"]["train_path"])
    test = load_test(config["data"]["test_path"])
    sample = load_sample_submission(config["data"]["sample_submission_path"])
    if config.get("debug_rows"):
        train = train.head(int(config["debug_rows"])).copy()
        test = test.head(min(len(test), 16)).copy()
        sample = sample.head(len(test)).copy()
    train = make_folds(train, int(config["n_splits"]), int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    oof = np.zeros_like(y)
    test_pred = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
    fold_results = []
    folds = config.get("folds_to_train") or sorted(train["fold"].unique().tolist())

    for fold in folds:
        result, val_pred, fold_test_pred = train_one_fold(train, test, int(fold), config, tokenizer, model_dir)
        val_idx = train.index[train["fold"] == int(fold)].to_numpy()
        oof[val_idx] = val_pred
        test_pred += fold_test_pred / len(folds)
        fold_results.append(result)

    trained_mask = np.any(oof != 0, axis=1)
    if trained_mask.any():
        cv_score = mcrmse(y[trained_mask], oof[trained_mask])
        col_score = columnwise_rmse(y[trained_mask], oof[trained_mask])
    else:
        cv_score = float("nan")
        col_score = {}

    submission_path = submission_dir / "submission_deberta_v3_base.csv"
    save_submission(sample, test_pred, submission_path)
    summary = {
        "name": "deberta_v3_base",
        "cv_mcrmse": cv_score,
        "column_rmse": col_score,
        "fold_scores": [item.__dict__ for item in fold_results],
        "submission_path": str(submission_path),
        "trained_folds": folds,
    }
    write_json(summary, artifact_dir / "transformer_metrics.json")
    np.save(artifact_dir / "deberta_oof.npy", oof)
    np.save(artifact_dir / "deberta_test.npy", test_pred)
    return summary
