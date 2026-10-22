# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock experiment tracker and dataset registry for the applied_science_research_team MAS."""

import json

_EXPERIMENTS = {
    "EXP-201": {"metric_name": "accuracy", "metric_value": 0.873, "runtime_minutes": 142, "status": "completed", "hyperparams": {"lr": 0.001, "batch_size": 256}, "owner": "Mei_Wong"},
    "EXP-202": {"metric_name": "f1", "metric_value": 0.791, "runtime_minutes": 88, "status": "completed", "hyperparams": {"lr": 0.0005, "batch_size": 128}, "owner": "Hassan_Ali"},
    "EXP-203": {"metric_name": "rmse", "metric_value": 4.21, "runtime_minutes": 312, "status": "running", "hyperparams": {"lr": 0.0001, "batch_size": 64}, "owner": "Olga_Petrov"},
    "EXP-204": {"metric_name": "accuracy", "metric_value": 0.928, "runtime_minutes": 51, "status": "completed", "hyperparams": {"lr": 0.002, "batch_size": 512}, "owner": "Mei_Wong"},
    "EXP-205": {"metric_name": "auc", "metric_value": 0.842, "runtime_minutes": 215, "status": "failed", "hyperparams": {"lr": 0.01, "batch_size": 256}, "owner": "Hassan_Ali"},
}

_DATASETS = {
    "DS-IMG-001": {"name": "Cells_HE_2024", "modality": "image", "license": "CC-BY-4.0", "rows": 48720, "columns": ["image_path", "label", "patient_id"], "owner": "Mei_Wong", "version": "1.3.0"},
    "DS-TXT-002": {"name": "ClinicalNotes_v2", "modality": "text", "license": "MIT", "rows": 211440, "columns": ["note_id", "text", "icd10"], "owner": "Hassan_Ali", "version": "2.0.1"},
    "DS-TAB-003": {"name": "Sensor_Telemetry_Q3", "modality": "tabular", "license": "Apache-2.0", "rows": 1820000, "columns": ["timestamp", "sensor_id", "value", "unit"], "owner": "Olga_Petrov", "version": "0.9.4"},
}


def get_experiment_metrics(exp_id: str) -> str:
    """Return the headline metric for an experiment."""
    e = _EXPERIMENTS.get(exp_id)
    if not e:
        return json.dumps({"error": f"unknown exp_id {exp_id}"})
    return json.dumps({"exp_id": exp_id, "metric_name": e["metric_name"], "metric_value": e["metric_value"]})


def get_experiment_runtime(exp_id: str) -> str:
    """Return the wall-clock training time in minutes."""
    e = _EXPERIMENTS.get(exp_id)
    if not e:
        return json.dumps({"error": f"unknown exp_id {exp_id}"})
    return json.dumps({"exp_id": exp_id, "runtime_minutes": e["runtime_minutes"]})


def get_experiment_status(exp_id: str) -> str:
    """Return the experiment status."""
    e = _EXPERIMENTS.get(exp_id)
    if not e:
        return json.dumps({"error": f"unknown exp_id {exp_id}"})
    return json.dumps({"exp_id": exp_id, "status": e["status"]})


def get_experiment_hyperparams(exp_id: str) -> str:
    """Return the hyperparameters of an experiment."""
    e = _EXPERIMENTS.get(exp_id)
    if not e:
        return json.dumps({"error": f"unknown exp_id {exp_id}"})
    return json.dumps({"exp_id": exp_id, "hyperparams": e["hyperparams"]})


def get_experiment_owner(exp_id: str) -> str:
    """Return the owner of an experiment."""
    e = _EXPERIMENTS.get(exp_id)
    if not e:
        return json.dumps({"error": f"unknown exp_id {exp_id}"})
    return json.dumps({"exp_id": exp_id, "owner": e["owner"]})


def list_experiments() -> str:
    """List all experiment IDs."""
    return json.dumps({"exp_ids": list(_EXPERIMENTS.keys())})


def get_dataset_info(dataset_id: str) -> str:
    """Return name, modality, and license for a dataset."""
    d = _DATASETS.get(dataset_id)
    if not d:
        return json.dumps({"error": f"unknown dataset_id {dataset_id}"})
    return json.dumps({"dataset_id": dataset_id, "name": d["name"], "modality": d["modality"], "license": d["license"]})


def get_dataset_size(dataset_id: str) -> str:
    """Return the row count of a dataset."""
    d = _DATASETS.get(dataset_id)
    if not d:
        return json.dumps({"error": f"unknown dataset_id {dataset_id}"})
    return json.dumps({"dataset_id": dataset_id, "rows": d["rows"]})


def get_dataset_columns(dataset_id: str) -> str:
    """Return the column list of a dataset."""
    d = _DATASETS.get(dataset_id)
    if not d:
        return json.dumps({"error": f"unknown dataset_id {dataset_id}"})
    return json.dumps({"dataset_id": dataset_id, "columns": d["columns"]})


def get_dataset_owner(dataset_id: str) -> str:
    """Return the owner of a dataset."""
    d = _DATASETS.get(dataset_id)
    if not d:
        return json.dumps({"error": f"unknown dataset_id {dataset_id}"})
    return json.dumps({"dataset_id": dataset_id, "owner": d["owner"]})


def get_dataset_version(dataset_id: str) -> str:
    """Return the current version string of a dataset."""
    d = _DATASETS.get(dataset_id)
    if not d:
        return json.dumps({"error": f"unknown dataset_id {dataset_id}"})
    return json.dumps({"dataset_id": dataset_id, "version": d["version"]})


def list_datasets() -> str:
    """List all dataset IDs."""
    return json.dumps({"dataset_ids": list(_DATASETS.keys())})
