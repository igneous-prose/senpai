# Training Code Style Guide

This guide distills the style lessons from DrivAerML's
[`train_drivaerml.py`][drivaerml-train] and applies them to this repository's
speculative-decoding work. The goal is simple training code that is rigorous
enough for expensive ML runs: every major knob has a clear owner, every
artifact path is explicit, and invalid configurations fail before they waste GPU
time.

The relevant TorchSpec references for the first Qwen3-8B script are its
[SGLang Qwen3-8B train-with-decode config][torchspec-qwen3-decode-config]
and [`torchspec/train_entry.py`][torchspec-train-entry].

## Core Lessons

### Make The Entrypoint A Contract

A training script is not only a convenience wrapper. It is the executable
contract for a run: data source, target model, draft model shape, optimizer
schedule, seeds, logging, checkpoints, and hardware placement.

Do:

```python
@dataclass(frozen=True)
class RunArgs:
    target_model: str
    dataset: str
    train_gpus: int
    inference_gpus: int
    output_dir: Path
```

Do not:

```python
args = parse_args()
config = {}
config.update(vars(args))
```

The first version makes the run boundary visible. The second version lets
unreviewed keys drift into the trainer.

### Use Strong Defaults, Then Validate Them

Good ML defaults are specific, not vague. DrivAerML names its default manifest,
normalizers, point counts, scheduler, precision, and save cadence. TorchSpec's
Qwen3 train-with-decode config names the target model, chat template, draft
length, hidden-state layers, and GPU split.

Do:

```python
if args.inference_gpus < 1:
    raise ValueError("inference_gpus must be at least 1")
if args.speculative_num_draft_tokens < args.speculative_num_steps + 1:
    raise ValueError("draft tokens must cover all speculative steps plus one verifier token")
```

Do not:

```python
if bad_config:
    print("warning: weird config, continuing")
```

Warnings are fine for non-load-bearing hints. For shape, hardware, loss, data,
and checkpoint identity, fail early.

### Keep The Main Path Readable

The DrivAerML script has one clear sequence:

1. Parse typed arguments.
2. Configure distributed state and seed.
3. Build dataloaders.
4. Build model.
5. Build optimizer, schedule, loss, and metrics.
6. Print a concise run summary.
7. Train.
8. Tear down distributed state.

For TorchSpec, the equivalent main path should be:

1. Parse the launch contract.
2. Validate GPU and data assumptions.
3. Write the resolved TorchSpec config.
4. Print the target, dataset, hardware split, and command.
5. Run `python -m torchspec.train_entry`.

Do not hide that flow behind generic "manager" or "runner" objects unless the
repo already has a real abstraction that needs them.

### Put Effects Near Their Purpose

Filesystem writes, environment mutation, subprocess launch, W&B setup, and
distributed initialization should be easy to find.

Do:

```python
config_path = write_torchspec_config(args)
env = build_child_environment(args)
subprocess.run(build_command(args, config_path), env=env, check=True)
```

Do not:

```python
prepare_everything(args)
```

A broad helper hides the expensive side effects a reviewer most needs to audit.

### Treat Seeds As Artifact Identity

DrivAerML seeds each rank through the data-parallel rank and uses deterministic
validation sampling. Speculative-decoding runs need the same discipline because
dataset ordering, generated continuations, and draft-model initialization can
all move the result.

Do:

```yaml
training:
  seed: 42
dataset:
  shuffle_dataset: true
```

Record the seed with the resolved config and output directory. Do not rely on a
shell history entry as the only record of the run.

### Make Hardware Placement Explicit

Speculative training has two different GPU consumers: inference engines and
training workers. The Qwen3 TorchSpec decode recipe uses one SGLang inference
GPU and two training GPUs by default.

Do:

```text
CUDA_VISIBLE_DEVICES=0,1,2
training GPUs: 2
inference GPUs: 1
SGLang TP size: 1
```

Do not:

```python
num_gpus = torch.cuda.device_count()
```

Total GPU count is not enough. The script must say which side gets the GPUs
and how tensor parallelism is configured.

### Keep Logging Rank-Aware

DrivAerML initializes W&B only on rank 0 and disables it on the other ranks.
TorchSpec handles distributed logging internally, but the launcher should still
avoid noisy duplicate setup. Prefer one concise preflight summary before the
trainer takes over.

Do:

```text
target_model: Qwen/Qwen3-8B
dataset: lightseekorg/kimi-mtp-dataset
output_dir: /runs/qwen3-8b-eagle3
train_with_decode: true
```

Do not print hundreds of derived fields in the launcher. TorchSpec already
saves a full resolved config snapshot.

### Preserve Domain Metrics

Training loss is not the success metric for speculative decoding. A good
training script can launch training, but the run is only useful if later evals
measure acceptance and speed.

Track, at minimum:

- training loss and learning rate;
- accepted length per verifier pass;
- acceptance rate by draft position;
- target/draft latency split;
- generated tokens per second;
- source-family held-out metrics;
- target checkpoint, tokenizer revision, and chat template.

Do not report "the model trained" as a research result. Report what the
trained speculator buys under the verifier.

## Good And Bad Patterns

### Config Boundary

Good:

```python
def build_torchspec_config(args: LaunchArgs) -> dict:
    return {
        "model": {"target_model_path": args.target_model},
        "dataset": {"train_data_path": args.dataset, "prompt_key": "conversations"},
        "training": {"train_with_decode": True, "seed": args.seed},
    }
```

Bad:

```python
cfg = load_yaml("some_config.yaml")
cfg["training"]["train_with_decode"] = True
cfg["training"]["whatever"] = args.whatever
```

The good version makes the supported surface obvious. The bad version mutates
an inherited blob and makes it hard to know which knobs are intentional.

### Data Paths

Good:

```python
if looks_local(args.dataset) and not Path(args.dataset).exists():
    raise FileNotFoundError(args.dataset)
```

Bad:

```python
dataset = args.dataset or "sample.jsonl"
```

Silent sample-data fallbacks are dangerous. A smoke-test mode can exist, but it
must be explicit.

### Expensive Defaults

Good:

```text
Default dataset: lightseekorg/kimi-mtp-dataset
Default command supports --dry-run and --num-train-steps 10 for smoke tests.
```

Bad:

```python
if args.debug:
    dataset = "examples/data/sample_conversations.jsonl"
else:
    dataset = "lightseekorg/kimi-mtp-dataset"
```

Debug flags often get copied into serious runs. Use explicit data paths and
explicit step limits.

### Subprocess Launch

Good:

```python
cmd = [python, "-m", "torchspec.train_entry", "--config", str(config_path)]
cmd.extend(args.torchspec_overrides)
subprocess.run(cmd, env=env, check=True)
```

Bad:

```python
os.system(f"python -m torchspec.train_entry --config {config_path} {extra}")
```

Structured command lists avoid quoting bugs and preserve clear failure through
`check=True`.

### Optional Integrations

Good:

```yaml
logging:
  report_to: none
```

Bad:

```python
try:
    import wandb
except Exception:
    pass
```

Optional integrations should be explicit run choices. Broad imports and silent
fallbacks hide broken logging.

## Checklist For New Training Scripts

- The script names the exact target model and tokenizer family.
- It writes a resolved config to the run output directory.
- It validates GPU split, draft length, sequence length, and local data paths.
- It exposes only the knobs needed for the first credible run.
- It passes additional expert overrides through to the underlying trainer.
- It prints a compact preflight summary before launching.
- It has a dry-run path that does not touch GPUs.
- It does not implement a parallel trainer when a maintained framework already
  owns that behavior.
- It leaves acceptance, throughput, and exactness metrics as first-class next
  steps rather than pretending loss is enough.

[drivaerml-train]: https://github.com/wandb/drivaerml/blob/main/scripts/train_drivaerml.py
[torchspec-qwen3-decode-config]: https://github.com/lightseekorg/TorchSpec/blob/main/configs/train_with_decode/sglang_qwen3_8b.yaml
[torchspec-train-entry]: https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/train_entry.py
