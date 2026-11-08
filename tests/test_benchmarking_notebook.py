"""Offline tests for the self-contained timing helper in the Colab notebook."""

import ast
import json
from pathlib import Path

import pytest
import torch


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "role3_validation_subset_benchmark.ipynb"
)


def load_benchmark_callable():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # The notebook's first code cell contains a Colab %pip magic.
            continue
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "benchmark_callable"
            for node in tree.body
        ):
            namespace = {"torch": torch}
            exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
            return namespace["benchmark_callable"]
    raise AssertionError("benchmark_callable was not found in the notebook")


def test_benchmark_callable_runs_warmups_and_measurements_on_cpu():
    benchmark_callable = load_benchmark_callable()
    calls = []

    result = benchmark_callable(
        lambda: calls.append(len(calls)) or torch.ones(1),
        device="cpu",
        warmup_runs=3,
        measured_runs=10,
    )

    assert len(calls) == 13
    assert result["warmup_runs"] == 3
    assert result["measured_runs"] == 10
    assert len(result["latencies_seconds"]) == 10
    assert result["mean_seconds"] >= 0
    assert result["median_seconds"] >= 0
    assert result["standard_deviation_seconds"] >= 0
    assert result["standard_deviation_kind"] == "sample"
    assert result["peak_gpu_memory_mb"] is None


def test_benchmark_callable_rejects_too_few_measured_runs():
    benchmark_callable = load_benchmark_callable()
    with pytest.raises(ValueError, match="at least 2"):
        benchmark_callable(lambda: None, device="cpu", measured_runs=1)
