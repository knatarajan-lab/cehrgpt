"""
End-to-end smoke test for the Qwen2 backbone against real CoMET-tokenized data.

Runs, for each training YAML given:
  1. an SDPA backend probe on the local GPU in the run's dtype, which fails the test if
     PyTorch would fall back to the math backend (that materialises a
     `(heads, L, L)` score matrix and defeats the point of sdpa);
  2. a handful of real optimizer steps through `hf_cehrgpt_pretrain_runner`, using the
     real tokenizer, dataset mapping, collator and Trainer;
  3. assertions on the saved checkpoint: backbone, geometry and reloadability.

Everything except step 2's data comes from the YAML, so this exercises the same code path
as a full run. Intended to be run on the training box before committing to a full run.

Usage:
    python -m cehrgpt.tools.smoke_test_comet_qwen2 \
        --yaml_file /path/to/comet_s_qwen2.yaml \
                    /path/to/comet_xgpt_matched_qwen2.yaml \
        --max_steps 20
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

# Keys overridden so the run is short and writes nowhere permanent.
def build_overrides(output_dir: str, prepared_dir: str, max_steps: int) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "model_name_or_path": output_dir,
        "dataset_prepared_path": prepared_dir,
        "max_steps": max_steps,
        "num_train_epochs": 1,
        "save_strategy": "steps",
        "save_steps": max_steps,
        "evaluation_strategy": "no",
        "use_early_stopping": False,
        "load_best_model_at_end": False,
        "overwrite_output_dir": True,
        "report_to": "none",
        "logging_steps": 1,
        "save_total_limit": 1,
        "streaming": False,
        "preprocessing_num_workers": 4,
        # Single-process loading keeps the smoke test quick to start and easy to debug.
        # prefetch_factor must be cleared alongside it: TrainingArguments rejects a
        # prefetch_factor with num_workers=0, and both CoMET YAMLs legitimately set
        # prefetch_factor: 8 for real multi-worker runs.
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
    }


def probe_sdpa_backend(dtype_name: str, packed_len: int) -> bool:
    """Check that a fused SDPA kernel accepts the real mask shape in the run's dtype."""
    import torch

    from cehrgpt.models.hf_cehrgpt import create_sample_packing_attention_mask
    from cehrgpt.models.qwen2 import build_sdpa_params

    if not torch.cuda.is_available():
        print("  no CUDA device; skipping the SDPA backend probe")
        return True

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        dtype_name
    ]
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"  device={properties.name} sm_{properties.major}{properties.minor} "
        f"dtype={dtype_name} packed_len={packed_len}"
    )

    attention_mask = torch.ones((1, packed_len), dtype=torch.long, device=device)
    for boundary in range(2048, packed_len, 2048):
        attention_mask[0, boundary] = 0
    packed = create_sample_packing_attention_mask(attention_mask)[:, None, :, :]
    additive = packed.to(dtype=dtype)
    additive = (1.0 - additive) * torch.finfo(dtype).min
    causal = torch.tril(
        torch.ones((packed_len, packed_len), dtype=torch.bool, device=device)
    ).view(1, 1, packed_len, packed_len)
    mask = additive.masked_fill(~causal, torch.finfo(dtype).min)

    query = torch.randn((1, 12, packed_len, 64), device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    params = build_sdpa_params(query, key, value, mask)
    efficient = torch.backends.cuda.can_use_efficient_attention(params, False)
    flash = torch.backends.cuda.can_use_flash_attention(params, False)
    print(f"  can_use_efficient_attention={efficient} can_use_flash_attention={flash}")

    if not (efficient or flash):
        print(
            "  FAIL: SDPA would fall back to the math backend. Note bfloat16 needs "
            "sm_80+ for mem-efficient attention; try fp16, or lower "
            "max_tokens_per_batch."
        )
        return False

    # Confirm the memory profile, not just eligibility.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
    torch.cuda.synchronize()
    extra = (torch.cuda.max_memory_allocated() - baseline) / 2**30
    materialised = 12 * packed_len * packed_len * mask.element_size() / 2**30
    print(
        f"  fused kernel extra memory {extra:.2f} GiB "
        f"(a materialised score matrix would be {materialised:.2f} GiB)"
    )
    del out, query, key, value, mask
    torch.cuda.empty_cache()
    return extra < materialised / 4


def run_one(yaml_file: Path, max_steps: int, keep: bool) -> bool:
    print("=" * 78)
    print(yaml_file.name)
    print("=" * 78)

    with open(yaml_file) as stream:
        config = yaml.safe_load(stream)

    dtype_name = "bf16" if config.get("bf16") else ("fp16" if config.get("fp16") else "fp32")
    packed_len = (
        config.get("max_tokens_per_batch", config["max_position_embeddings"])
        if config.get("sample_packing")
        else config["max_position_embeddings"]
    )

    print("[1/3] SDPA backend probe")
    if not probe_sdpa_backend(dtype_name, packed_len):
        return False

    temp_dir = tempfile.mkdtemp(prefix="comet_smoke_")
    output_dir = os.path.join(temp_dir, "model")
    prepared_dir = os.path.join(temp_dir, "prepared")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    smoke_config = copy.deepcopy(config)
    smoke_config.update(build_overrides(output_dir, prepared_dir, max_steps))
    smoke_yaml = os.path.join(temp_dir, "smoke.yaml")
    with open(smoke_yaml, "w") as stream:
        yaml.safe_dump(smoke_config, stream)

    print(f"\n[2/3] {max_steps} training steps via hf_cehrgpt_pretrain_runner")
    print(f"  data      : {smoke_config['data_folder']}")
    print(f"  tokenizer : {smoke_config['tokenizer_name_or_path']}")
    print(f"  scratch   : {temp_dir}")
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "cehrgpt.runners.hf_cehrgpt_pretrain_runner",
            smoke_yaml,
        ],
        check=False,
    )
    if completed.returncode != 0:
        print(f"  FAIL: runner exited {completed.returncode}")
        return False

    print("\n[3/3] checkpoint assertions")
    ok = True
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"  FAIL: no config.json under {output_dir}")
        return False
    with open(config_path) as stream:
        saved = json.load(stream)

    expected = {
        "backbone": "qwen2",
        "n_layer": config["num_hidden_layers"],
        "n_embd": config["hidden_size"],
        "n_head": config["n_head"],
        "num_key_value_heads": config["num_key_value_heads"],
        "n_inner": config["inner_dim"],
        "decoder_mlp": config["decoder_mlp"],
        "activation_function": config["activation_function"],
        "resid_pdrop": config["resid_pdrop"],
        "n_positions": config["max_position_embeddings"],
    }
    for key, want in expected.items():
        got = saved.get(key)
        status = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {status} {key:22s} expected={want!r} saved={got!r}")

    from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
    from cehrgpt.runners.gpt_runner_util import (
        read_backbone,
        resolve_attn_implementation,
    )

    backbone = read_backbone(output_dir)
    model = CEHRGPT2LMHeadModel.from_pretrained(
        output_dir, attn_implementation=resolve_attn_implementation(backbone)
    )
    print(
        f"  ok  reloaded: block={type(model.cehrgpt.h[0]).__name__} "
        f"impl={model.config._attn_implementation} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    if type(model.cehrgpt.h[0]).__name__ != "Qwen2Block":
        ok = False

    if keep:
        print(f"  scratch kept at {temp_dir}")
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_file", required=True, nargs="+")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument(
        "--keep_scratch",
        action="store_true",
        help="Do not delete the temporary output/prepared-dataset directories.",
    )
    args = parser.parse_args()

    results = {}
    for yaml_file in args.yaml_file:
        results[yaml_file] = run_one(Path(yaml_file), args.max_steps, args.keep_scratch)
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for yaml_file, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {yaml_file}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
