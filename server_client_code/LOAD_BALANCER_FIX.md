# Load Balancer Connection Error Fix

## Problem Identified

The evaluation runs were failing with generic "Connection error" messages, and the load balancer logs showed:

```
PROXY ERROR to http://localhost:8000: 
PROXY ERROR to http://localhost:8001: 
PROXY ERROR to http://localhost:8002: 
PROXY ERROR to http://localhost:8003:
```

**Key Issues:**
1. **No error details**: The logs only showed "PROXY ERROR" without the actual error message
2. **vLLM logs empty**: All `vllm_gpu*.log` files were empty (0 or 1 byte)
3. **No health monitoring**: No way to know which backends were actually running

## Root Cause Analysis

### Issue 1: vLLM Servers Not Running
The empty log files suggest the vLLM servers either:
- Never started successfully
- Crashed immediately after startup
- Logs aren't being written due to buffering or permission issues

**Most likely**: The servers crashed during initialization, possibly due to:
- Model loading failures
- GPU memory issues
- Port conflicts
- Missing dependencies

### Issue 2: Poor Error Logging
The original load balancer code had minimal error logging:

```python
except Exception as e:
    print(f"PROXY ERROR to {backend_url}: {e}", file=sys.stderr, flush=True)
```

But the error message `{e}` was appearing as empty, suggesting:
- The exception had no string representation
- The error was being truncated
- Connection errors were generic with no details

### Issue 3: No Health Monitoring
There was no mechanism to:
- Check if backends are actually running
- Track backend health over time
- Alert when backends become unavailable
- Provide diagnostics for troubleshooting

## Solutions Implemented

### 1. Enhanced Error Logging

**Added detailed error categorization:**

```python
except aiohttp.ClientConnectorError as e:
    error_type = type(e).__name__
    error_msg = f"Connection error: {error_type} - {str(e)}"
    print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}")
    print(f"  Backend health: {backend_health[backend_url]['status']}")
```

**New error details include:**
- Timestamp of error
- Error type (ClientConnectorError, TimeoutError, etc.)
- Full error message and details
- Current backend health status
- Full traceback for unexpected errors

**Example improved output:**
```
[2025-11-17T14:23:45] PROXY ERROR to http://localhost:8000: Connection error: ClientConnectorError - Cannot connect to host localhost:8000 ssl:default [Connect call failed ('127.0.0.1', 8000)]
  Backend health: connection_refused (ClientConnectorError)
```

### 2. Backend Health Monitoring

**Added continuous health checks:**

```python
backend_health = {
    backend: {
        "status": "unknown",
        "last_check": None,
        "error_count": 0
    } for backend in BACKENDS
}

async def check_backend_health(session, backend_url):
    """Check if a backend is responsive."""
    try:
        async with session.get(f"{backend_url}/health", timeout=5) as resp:
            if resp.status == 200:
                backend_health[backend_url]["status"] = "healthy"
                backend_health[backend_url]["error_count"] = 0
                return True
    except aiohttp.ClientConnectorError:
        backend_health[backend_url]["status"] = "connection_refused"
        return False
```

**Features:**
- Initial health check on startup
- Periodic health checks every 30 seconds
- Tracks error counts per backend
- Reports health status in logs

**Example health check output:**
```
[2025-11-17T14:24:00] Health Check:
  ✓ http://localhost:8000: healthy (errors: 0)
  ✗ http://localhost:8001: connection_refused (errors: 15)
  ✓ http://localhost:8002: healthy (errors: 0)
  ✗ http://localhost:8003: timeout (errors: 3)
```

### 3. Diagnostic Script

Created `diagnose_servers.sh` to help troubleshoot server issues:

```bash
./diagnose_servers.sh
```

**The script checks:**
1. **Port status**: Which ports have processes listening
2. **vLLM logs**: Last 20 lines of each vLLM server log
3. **Backend connectivity**: HTTP health check for each backend
4. **GPU status**: Current GPU utilization and memory
5. **Process list**: All running vLLM processes
6. **Network listeners**: Detailed port information

## How to Use

### 1. Check Current Status

```bash
cd /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/compare
./diagnose_servers.sh
```

This will show you:
- Which vLLM servers are running
- Why servers failed (from logs)
- Which ports are actually listening
- GPU status

### 2. Read Enhanced Load Balancer Logs

The new load balancer provides much better logging:

```bash
# Check load balancer log
tail -f load_balancer.log

# Look for specific errors
grep "PROXY ERROR" load_balancer.log | head -20

# Check health status
grep "Health Check" load_balancer.log | tail -5
```

### 3. Restart Services

If servers are down, restart them:

```bash
# Kill existing processes
for port in 8000 8001 8002 8003 8080; do
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

# Restart the multi-server setup
sbatch serve_r0_model_multi.sh
```

### 4. Monitor Startup

Watch the health checks during startup:

```bash
tail -f load_balancer.log | grep -E "(Health Check|PROXY ERROR|Backend health)"
```

## Common Issues and Solutions

### Issue: "connection_refused"
**Cause**: vLLM server not running on that port

**Solutions:**
1. Check vLLM logs: `tail -100 vllm_gpu0.log`
2. Look for errors during model loading
3. Check GPU memory: `nvidia-smi`
4. Verify model path exists
5. Check for port conflicts: `lsof -i:8000`

### Issue: "timeout"
**Cause**: Server running but not responding (overloaded or hung)

**Solutions:**
1. Check GPU utilization: `nvidia-smi`
2. Review vLLM logs for warnings
3. Reduce batch size in server config
4. Reduce concurrent requests
5. Restart the hung server

### Issue: Empty vLLM logs
**Cause**: Server crashed during startup or logs not being flushed

**Solutions:**
1. Check SLURM job logs: `tail serve_r0_model_multi*.log`
2. Look for Python errors during startup
3. Run vLLM server manually to see errors:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
       --model /path/to/model \
       --port 8000
   ```
4. Check model files and permissions
5. Verify conda environment is correct

### Issue: All servers failing
**Cause**: Systematic issue (model path, permissions, GPU access)

**Solutions:**
1. Check model path: `ls -la /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview`
2. Verify GPU access: `nvidia-smi`
3. Check conda environment: `conda list | grep vllm`
4. Review SLURM job status: `squeue -u $USER`
5. Check node resources: `sinfo -N -l`

## Expected Behavior

### Healthy System

```
[2025-11-17T14:24:00] Health Check:
  ✓ http://localhost:8000: healthy (errors: 0)
  ✓ http://localhost:8001: healthy (errors: 0)
  ✓ http://localhost:8002: healthy (errors: 0)
  ✓ http://localhost:8003: healthy (errors: 0)
```

### System with Issues

```
[2025-11-17T14:24:00] Health Check:
  ✗ http://localhost:8000: connection_refused (ClientConnectorError) (errors: 45)
  ✓ http://localhost:8001: healthy (errors: 0)
  ✗ http://localhost:8002: timeout (errors: 3)
  ✓ http://localhost:8003: healthy (errors: 0)
```

**Action**: Check GPU 0 and GPU 2 servers - they're down/hung

## Next Steps

1. **Run diagnostics**: `./diagnose_servers.sh > server_diagnostic.log 2>&1`
2. **Share diagnostic output** to identify the root cause
3. **Check vLLM logs** for specific error messages
4. **Verify model and GPU availability**
5. **Restart services** once issues are identified

The enhanced logging will now provide **much more useful information** for troubleshooting connection issues!










