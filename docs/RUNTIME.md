# hy3d_runtime — operational guide

`hy3d_runtime/` owns the cross-cutting runtime concerns: device & dtype
resolution, weight residency, memory monitoring, request lifecycle, recovery
state machines, and telemetry. Entry points (`api_server.py`, `gradio_app.py`,
`demo.py`, `model_worker.py`) are thin shells over this package.

## Quick map

| Module | Responsibility |
|---|---|
| `device.py`         | `pick_device` / `pick_dtype` / `gpu_inventory` / `cuda_available`. CPU always returns fp32 (see Hard Rules). |
| `config.py`         | `RuntimeConfig` dataclass merging env + argparse. Graded `--low_vram_mode`. |
| `budgets.py`        | `InputLimits` (user-input caps) + `MeshBudget` (output knobs, cluster-aware). |
| `weight_manager.py` | Process-local registry with `ref_count` / `active_stage` / `last_access` / `mutable_state` / RLock. |
| `memory_monitor.py` | VRAM + reserved/allocated/fragmentation + psutil CPU RAM + swap + pinned-buffer accounting. |
| `orchestrator.py`   | Synchronous stage→device routing. Fallback ladder: GPU → CPU pinned → CPU paged → MeshBudget downgrade → `OOMError`. Disk offload is **opt-in only**. |
| `request_context.py`| Per-request UUID, deadline, cancel, artifact tracking, cleanup callbacks. |
| `recovery.py`       | Formalized `ShapeOOMRecovery` and `PaintOOMRecovery` state machines with hard retry cap (`MAX_RETRIES_PER_STAGE=3`). |
| `telemetry.py`      | JSON-line events with severity levels (INFO/WARNING/ERROR/CRITICAL); optional Prometheus exporter. |
| `offload.py`        | `attach_cpu_offload` (sequential `cpu_offload_with_hook`) + `attach_disk_offload` (gated by `RuntimeConfig.enable_disk_offload`). |
| `safetensors_mmap.py` | `load_safetensors_mmap(path)` via `safetensors.safe_open` — avoids bulk-copy at load time. |

## Hard rules

1. **CPU runs fp32 only.** `pick_dtype(cpu, want=anything)` returns
   `torch.float32` and logs a one-shot warning if a caller asks for half
   precision on CPU. PyTorch CPU fp16/bf16 is dramatically slower than fp32,
   unstable, and vectorization-poor.
2. **VRAM is NOT pooled across GPUs.** The orchestrator distributes pipeline
   stages across devices, but a stage that doesn't fit in one GPU's VRAM (after
   CPU offload + MeshBudget downgrade) does not fit, period. A 3 × 15 GB
   system gives you 15 GB of usable per-stage VRAM, not 45 GB. If the smallest
   profile still doesn't fit, the API returns HTTP 507 (`StageDoesNotFit`)
   with a structured payload that includes the "VRAM is not pooled"
   explanation.
3. **Disk offload is opt-in only.** Automatic GPU → CPU → disk fallbacks
   risk minutes-long stalls and swap thrashing. The automatic ladder STOPS
   at CPU paged. Disk offload requires `--enable-disk-offload`
   (or `HY3D_ENABLE_DISK_OFFLOAD=1`), prints a one-time warning at startup,
   and emits a `disk_offload_demote` WARNING telemetry alert on every move.
4. **CPU mode is SHAPE_GEN only.** The custom rasterizer requires CUDA.
   `--device cpu` produces a textureless (shape-only) GLB; the paint pipeline
   raises `RasterizerNotAvailable` with a documented message instead of
   crashing deep in a kernel import. See `docs/CPU_MODE.md`.
5. **One process per GPU.** `WeightManager` is process-local and not safe for
   multi-worker pools across CUDA contexts. `api_server.py` refuses to launch
   when `WEB_CONCURRENCY > 1`. For horizontal scaling, run one process per GPU
   behind a load balancer (nginx, k8s service).
6. **All operations execute on each device's default CUDA stream.** No
   `Stream`-factory usage in Phase 1–6 (verified by CI lint). Async preload
   that requires non-default streams is deferred to a future Phase 7
   workstream and does not appear in the public API of the current modules.

## Memory tiers and the fallback ladder

| Tier | When | Cost | How to invoke |
|---|---|---|---|
| GPU resident | Stage active                  | 0           | Default. |
| CPU paged    | Between stages (automatic)    | ~0.5-2 s    | `--low_vram_mode conservative` (graded). |
| CPU pinned   | Between stages (opt-in)       | ~0.2-1 s    | `WeightManager.demote(target='cpu_pinned')`. Pinned cap: `min(4 GB, total_ram*0.10)`. |
| Disk offload | Operator opt-in only          | ~1-5 s NVMe / 10-30 s HDD per layer swap | `--enable-disk-offload` and optionally `--prefer-disk-offload`. |
| Profile downgrade | Automatic on OOM         | Smaller output | `MeshBudget.demote_one_step()` walks `ultra → high → standard → draft`. |
| `StageDoesNotFit` | All exhausted at draft   | Permanent failure | HTTP 507 with structured payload. |

## MeshBudget tuning (cluster-aware)

The `auto` profile selector consults the full GPU inventory plus current load:

| Hardware | `auto` selects | Why |
|---|---|---|
| CPU only         | `draft`    | CPU-bound; profile is moot. |
| 1 × 8 GB         | `draft`    | Standard's UNet (4 GB) + DiT (3 GB) + overhead won't fit. |
| 1 × 15 GB        | `standard` | All stages fit one-at-a-time; CPU offload between stages keeps headroom. |
| 1 × 24 GB        | `high`     | Single GPU holds DiT + multiview UNet concurrently. |
| 1 × 48 GB+       | `ultra`    | Single GPU has comfortable headroom. |
| 2 × 15 GB        | `high`     | SHAPE_GEN on GPU-A, PAINT on GPU-B; no swap between stages. |
| 2 × 24 GB        | `ultra`    | Same distribution, comfortable per-stage headroom. |
| 3 × 15 GB        | `high`     | Per-stage VRAM still 15 GB (no pooling); third GPU enables concurrent request pipelining only. |
| 3 × 24 GB        | `ultra`    | Each device has ultra headroom. |
| 8 × 80 GB (H100) | `ultra`    | Trivially fits. |

Pressure-driven adjustments after the base pick:
- `+1 rung` if any GPU is at high pressure (free < 10%).
- `+1 rung` if `>1` request is in flight.
- `-1 rung` if `--low_vram_mode aggressive` is active (offload frees ~30-50% effective VRAM).

## Recovery state machines

Both shape and paint recovery enforce **`MAX_RETRIES_PER_STAGE = 3`**. No
state retries the same configuration twice — every retry mutates a parameter
(texture_size, profile, low_vram_mode). Terminal states:

- `ShapeOOMRecovery`: `SUCCESS` | `FAIL` (raises `OOMError` or `StageDoesNotFit`).
- `PaintOOMRecovery`: `SUCCESS` | `PARTIAL_SUCCESS` (smaller texture) | `DEGRADED_SUCCESS` (shape-only fallback). Never returns a partial/corrupt paint artifact.

Each transition emits `telemetry.event("shape_recovery_step" | "paint_recovery_step", level=...)`.

## Telemetry severity

| Level | Examples |
|---|---|
| INFO     | `stage_start`, `stage_end`, `mesh_budget_resolved`, `routing_table`. |
| WARNING  | `paint_recovery_step` (non-terminal), `shape_recovery_step` (non-terminal), `pressure_event` (high), `disk_offload_demote`, `fragmentation_detected`, `request_cancelled`, `request_timeout`. |
| ERROR    | Recovery terminal `FAIL`, `request_ended` (outcome=error). |
| CRITICAL | `swap_panic_mode` (planned), `cuda_context_corrupt` (planned), `StageDoesNotFit`. |

The default sink is stderr + a JSON-line log file (configurable). When
`prometheus_client` is installed, counters are exported automatically.

## CUDA stream invariant (CI lint)

```bash
grep -r 'torch\.cuda\.Stream' hy3d_runtime/ hy3dshape/hy3dshape/ hy3dpaint/
```
must return zero matches. Phase 7 (async preload) will introduce stream
semantics but is out of scope for the current refactor.

## Risk acknowledgement

- **PCIe / NVLink / NUMA topology awareness** is a future workstream. The
  current orchestrator routes by VRAM size and compute capability only.
  Same-NUMA-node GPUs are cheaper to hand off between than cross-socket GPUs;
  NVLink is ~10× faster than PCIe for tensor moves. Not blocking for
  Phase 1–6 since stage hand-offs are infrequent and the artifact (coarse
  mesh) is small.
- **Allocator fragmentation** beyond `(reserved - allocated) / reserved`
  heuristic is not automatically remediated. If long-running servers show
  persistent fragmentation events, set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.1+) and
  restart.
- **Locking** uses `threading.RLock` per `WeightEntry`. Phase 7 async preload
  will require lock-order hierarchy, promote/demote queues, and
  cancellation-safe locks; deliberately deferred.
