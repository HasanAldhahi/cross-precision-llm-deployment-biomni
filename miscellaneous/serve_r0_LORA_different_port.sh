#!/bin/bash
#SBATCH --job-name=serve_r0_model
#SBATCH --output=serve_MAX_r0_model_%j.log
#SBATCH --error=serve_MAX_r0_model_%j.err
#SBATCH --time=24:00:00
#SBATCH -p kisski-h100
#SBATCH -G H100:4
#SBATCH --nodes=1
#SBATCH -C inet


echo "==============================================="
echo "Starting Biomni-R0 Multi-Server Setup (4× vLLM instances)"
echo "Node: $(hostname)"
echo "==============================================="

# --- START OF CORRECTIONS ---

# Step 1: Load the correct environment modules for your HPC system.
echo "Loading environment modules for CUDA and GCC..."
module load gcc/13.2.0
module load gcc/13.2.0-nvptx
module load cuda/12.6.2

echo "✓ Modules loaded successfully."

# Step 2: Ensure nvcc (NVIDIA's CUDA Compiler) is in the PATH.
# This is critical for vLLM's on-the-fly kernel compilation.
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc could not be found in PATH even after loading the CUDA module. Please check the module's configuration. Exiting."
    exit 1
fi
echo "✓ nvcc found at: $(which nvcc)"

# --- END OF CORRECTIONS ---

# Setup environment
echo "Initializing Conda..."
module load miniforge3/24.3.0-0

echo "Activating Conda environment..."
# Note: Using 'source activate' can be more robust in some SLURM environments
source activate /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check vLLM version
if python -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
    echo "vLLM version: $VLLM_VERSION"
else
    echo "Installing vLLM..."
    pip install -U vllm
fi

# Proxy bypass
export NO_PROXY="localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.usr.hpc.gwdg.de"
export no_proxy="$NO_PROXY"

MODEL_PATH="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B-FP8"
LORA_PATH="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/lora_extraction_results/lora_basic_rank_256"  
LORA_NAME="biomni-reasoning"

LOAD_BALANCER_PORT=9200

# Kill any existing processes on these ports
echo "Cleaning up any old processes on ports 9100-9103 and 9200..."
for port in 9100 9101 9102 9103 9200; do
    lsof -ti:$port | xargs --no-run-if-empty kill -9
done

echo ""
echo "==============================================="
echo "Launching 4 vLLM servers (1 per GPU)..."
echo "==============================================="
# --enable-lora \
# --max-lora-rank 128 \
# --lora-modules $LORA_NAME=$LORA_PATH \

# Launch server on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 9100 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --trust-remote-code \
    --enable-lora \
    --max-lora-rank 256 \
    --lora-modules $LORA_NAME=$LORA_PATH \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    > vllm_different_port_gpu0.log 2>&1 &
echo "✓ Server 1 starting on GPU 0, port 9100 (PID: $!)"

# Launch server on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 9101 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --trust-remote-code \
    --enable-lora \
    --max-lora-rank 256 \
    --lora-modules $LORA_NAME=$LORA_PATH \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    > vllm_different_port_gpu1.log 2>&1 &
echo "✓ Server 2 starting on GPU 1, port 9101 (PID: $!)"

# Launch server on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 9102 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --trust-remote-code \
    --enable-lora \
    --max-lora-rank 256 \
    --lora-modules $LORA_NAME=$LORA_PATH \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    > vllm_different_port_gpu2.log 2>&1 &
echo "✓ Server 3 starting on GPU 2, port 9102 (PID: $!)"

# Launch server on GPU 3
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 9103 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --trust-remote-code \
    --enable-lora \
    --max-lora-rank 256 \
    --lora-modules $LORA_NAME=$LORA_PATH \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    > vllm_different_port_gpu3.log 2>&1 &
echo "✓ Server 4 starting on GPU 3, port 9103 (PID: $!)"

echo ""
echo "Waiting 60 seconds for servers to initialize..."
sleep 60

# Create Python load balancer script
cat > load_balancer.py << 'EOF'
import asyncio
import aiohttp
from aiohttp import web
import itertools
import sys
import traceback
from datetime import datetime

BACKEND_PORTS = [9100, 9101, 9102, 9103]
# Use localhost since vLLM servers are running on the same node
BACKENDS = [f"http://localhost:{port}" for port in BACKEND_PORTS]
backend_cycle = itertools.cycle(BACKENDS)

# Track backend health status
backend_health = {backend: {"status": "unknown", "last_check": None, "error_count": 0} for backend in BACKENDS}

async def check_backend_health(session, backend_url):
    """Check if a backend is responsive."""
    try:
        async with session.get(f"{backend_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                backend_health[backend_url]["status"] = "healthy"
                backend_health[backend_url]["error_count"] = 0
                return True
            else:
                backend_health[backend_url]["status"] = f"unhealthy (HTTP {resp.status})"
                return False
    except aiohttp.ClientConnectorError as e:
        backend_health[backend_url]["status"] = f"connection_refused ({type(e).__name__})"
        backend_health[backend_url]["error_count"] += 1
        return False
    except asyncio.TimeoutError:
        backend_health[backend_url]["status"] = "timeout"
        backend_health[backend_url]["error_count"] += 1
        return False
    except Exception as e:
        backend_health[backend_url]["status"] = f"error ({type(e).__name__}: {str(e)})"
        backend_health[backend_url]["error_count"] += 1
        return False
    finally:
        backend_health[backend_url]["last_check"] = datetime.now().isoformat()

async def health_monitor(app):
    """Periodically check backend health."""
    session = app['client_session']
    while True:
        print(f"\n[{datetime.now().isoformat()}] Health Check:", file=sys.stderr, flush=True)
        for backend in BACKENDS:
            is_healthy = await check_backend_health(session, backend)
            status_symbol = "✓" if is_healthy else "✗"
            error_count = backend_health[backend]["error_count"]
            status_info = backend_health[backend]["status"]
            print(f"  {status_symbol} {backend}: {status_info} (errors: {error_count})", file=sys.stderr, flush=True)
        await asyncio.sleep(30)

async def proxy_handler(request):
    """Round-robin proxy to backends using a shared client session."""
    backend_url = next(backend_cycle)
    # Get the single, persistent session from the application context
    session = request.app['client_session']
    
    url = f"{backend_url}{request.path_qs}"

    try:
        # Use the shared session to make the request
        async with session.request(
            method=request.method,
            url=url,
            headers=request.headers,
            data=await request.read(),
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            body = await resp.read()
            return web.Response(
                body=body,
                status=resp.status,
                headers=resp.headers
            )
    except aiohttp.ClientConnectorError as e:
        error_type = type(e).__name__
        error_msg = f"Connection error: {error_type} - {str(e)}"
        print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}", file=sys.stderr, flush=True)
        print(f"  Backend health: {backend_health[backend_url]['status']}", file=sys.stderr, flush=True)
        backend_health[backend_url]["error_count"] += 1
        return web.Response(text=f"Proxy error to {backend_url}: {error_msg}", status=502)
    except asyncio.TimeoutError as e:
        error_msg = f"Timeout after 600s"
        print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}", file=sys.stderr, flush=True)
        backend_health[backend_url]["error_count"] += 1
        return web.Response(text=f"Proxy error to {backend_url}: {error_msg}", status=504)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"{error_type}: {str(e)}"
        print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}", file=sys.stderr, flush=True)
        print(f"  Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        backend_health[backend_url]["error_count"] += 1
        return web.Response(text=f"Proxy error to {backend_url}: {error_msg}", status=502)

async def startup_tasks(app):
    """Create a single, shared ClientSession on application startup."""
    print("Initializing shared ClientSession for the application...", file=sys.stderr, flush=True)
    connector = aiohttp.TCPConnector(limit_per_host=100, limit=400)
    # Store the session in the app object so handlers can access it
    app['client_session'] = aiohttp.ClientSession(connector=connector)
    
    # Start health monitoring
    print("Starting backend health monitor...", file=sys.stderr, flush=True)
    app['health_monitor_task'] = asyncio.create_task(health_monitor(app))
    
    # Do an initial health check
    print("\nInitial backend health check:", file=sys.stderr, flush=True)
    for backend in BACKENDS:
        is_healthy = await check_backend_health(app['client_session'], backend)
        status_symbol = "✓" if is_healthy else "✗"
        status_info = backend_health[backend]["status"]
        print(f"  {status_symbol} {backend}: {status_info}", file=sys.stderr, flush=True)

async def cleanup_tasks(app):
    """Close the shared ClientSession on application cleanup."""
    print("Shutting down health monitor...", file=sys.stderr, flush=True)
    if 'health_monitor_task' in app:
        app['health_monitor_task'].cancel()
        try:
            await app['health_monitor_task']
        except asyncio.CancelledError:
            pass
    
    print("Closing shared ClientSession...", file=sys.stderr, flush=True)
    await app['client_session'].close()

if __name__ == '__main__':
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    
    # Register the startup and cleanup tasks
    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    
    port = 9200
    print(f"Load balancer starting on port {port}")
    print(f"Backends: {BACKENDS}")
    web.run_app(app, host='0.0.0.0', port=port)

EOF

echo ""
echo "==============================================="
echo "Starting load balancer on port $LOAD_BALANCER_PORT..."
echo "==============================================="
python load_balancer.py > load_balancer.log 2>&1 &
LB_PID=$!
echo "✓ Load balancer started (PID: $LB_PID)"

echo ""
echo "==============================================="
echo "ALL SERVERS READY"
echo "==============================================="
echo "Connect to: http://$(hostname):9200/v1"
echo "Or IP: http://$(hostname -I | awk '{print $1}'):9200/v1"
echo "==============================================="

# Wait indefinitely until the job is cancelled

# Sleep 7 minutes to let vLLM servers fully initialize
echo ""
echo "Sleeping 7 minutes (420s) for server warm-up..."
sleep 1200
echo "✓ Warm-up complete!"

# Run the evaluation script
echo ""
echo "==============================================="
echo "STARTING EVALUATION (A1 + QWEN 3 32B-FP8-LORA-256)"
echo "==============================================="
nohup bash run_r0_slurm_MAX_B1_different_port.sh > r0_final_LORA_different_port_corected_RANK_256.log 2>&1 &
EVAL_PID=$!
echo "✓ Evaluation started (PID: $EVAL_PID)"
echo "  Log: r0_final_LORA_different_port_corected_RANK_256.log"

# Wait for evaluation to complete
echo "Waiting for evaluation to finish..."
wait $EVAL_PID
EVAL_EXIT_CODE=$?
echo "✓ Evaluation finished with exit code: $EVAL_EXIT_CODE"

# Cleanup: kill load balancer and any vLLM processes
echo ""
echo "==============================================="
echo "CLEANUP: Stopping all servers..."
echo "==============================================="
kill $LB_PID 2>/dev/null
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
echo "✓ All servers stopped. Script complete."

exit $EVAL_EXIT_CODE