# Execution Logs Index

> **Reproducibility Documentation**: Complete index of all execution logs generated during thesis experiments.

This document serves as a central index for all log files in the repository. Logs are kept in their original locations alongside the code that generated them to preserve context and demonstrate authenticity.

---

## 📋 Log Inventory

### 1. Brain Surgery Logs (Methods A & B)

| Log File | Location | Description |
|----------|----------|-------------|
| `dequant_FP8.log` | `brain_surgery/dequantize_FP8/` | FP8 → BF16 dequantization for Method B bridge model |
| `raw_dequant_12041842.log` | `brain_surgery/dequantize_INT4/` | INT4 dequantization attempt (exploratory) |
| `raw_dequant_12041842.err` | `brain_surgery/dequantize_INT4/` | INT4 dequantization error trace |

**What These Logs Prove:**
- Successful dequantization of FP8 weights to BF16
- Block-128 scale expansion algorithm execution
- Weight autopsy verification (Mean ≈ 0, Std ≈ 0.02)

---

### 2. Quantization Logs (Method C)

| Log File | Location | Description |
|----------|----------|-------------|
| `quantize_FP8.log` | `quantization/scripts/FP8_quantization/` | Main FP8 quantization of R0-32B |
| `quantize_FP8.err` | `quantization/scripts/FP8_quantization/` | FP8 quantization warnings/errors |
| `quantize_AWQ_INT4_12001782.log` | `quantization/scripts/INT4_quantization/` | AWQ INT4 quantization attempt |
| `quantize_AWQ_INT4_12001782.err` | `quantization/scripts/INT4_quantization/` | INT4 quantization errors |
| `quantize_INT4.log` | `quantization/scripts/INT4_quantization/` | Additional INT4 logs |

**What These Logs Prove:**
- Calibration data loading (123 instances, 3.1M tokens)
- Per-layer quantization progress
- FP8 scale factor computation
- Model save verification

---

### 3. Server Logs (Inference & Evaluation)

| Log File | Location | Description |
|----------|----------|-------------|
| `vllm_gpu0.log` | `server/` | vLLM server on GPU 0 |
| `vllm_gpu1.log` | `server/` | vLLM server on GPU 1 |
| `vllm_gpu2.log` | `server/` | vLLM server on GPU 2 |
| `vllm_gpu3.log` | `server/` | vLLM server on GPU 3 |
| `load_balancer.log` | `server/` | Multi-GPU request distribution |

**What These Logs Prove:**
- Model loading success on each GPU
- Inference request handling
- Response generation times
- Memory usage during evaluation

---

## 📊 Log Analysis Scripts

### View Quantization Progress

```bash
# Check FP8 quantization completion
grep "Saving" quantization/scripts/FP8_quantization/quantize_FP8.log

# Count processed layers
grep -c "Quantizing layer" quantization/scripts/FP8_quantization/quantize_FP8.log
```

### Check Server Health

```bash
# Check if all GPUs loaded model
for i in 0 1 2 3; do
  echo "=== GPU $i ==="
  grep "Model loaded" server/vllm_gpu$i.log | tail -1
done
```

### Verify Dequantization Quality

```bash
# Extract autopsy results from dequantization log
grep -E "mean|std|WARNING" brain_surgery/dequantize_FP8/dequant_FP8.log
```

---

## 📋 Log Timestamps

| Experiment | SLURM Job ID | Date | Duration |
|------------|--------------|------|----------|
| FP8 Quantization | - | - | ~8 hours |
| INT4 Quantization | 12001782 | - | ~6 hours |
| INT4 Dequantization | 12041842 | - | ~4 hours |
| FP8 Dequantization | - | - | ~2 hours |

---

## 🔐 Log Integrity

These logs are preserved in their original state to demonstrate:

1. **Reproducibility**: Exact commands and parameters used
2. **Authenticity**: Timestamps and job IDs match HPC records
3. **Completeness**: Full execution traces, not cherry-picked excerpts
4. **Transparency**: Errors and warnings are included, not hidden

---

## 📁 Log File Structure

Logs follow the standard output convention:

- `.log` files: Standard output (stdout) - progress, results, info
- `.err` files: Standard error (stderr) - warnings, errors, debug info

For SLURM jobs, filenames include the job ID (e.g., `*_12001782.log`) for HPC record matching.

---

## 🗄️ Archival Note

All logs are included in version control to ensure:

1. Exact experimental conditions are documented
2. Results can be verified against execution records
3. Future researchers can understand the experimental process
4. Thesis examiners can verify claims

**Note**: Large binary outputs (model checkpoints) are excluded via `.gitignore`, but text logs are preserved.
