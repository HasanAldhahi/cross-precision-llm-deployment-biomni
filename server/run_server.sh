#!/bin/bash
#SBATCH --job-name=serve_r0_model
#SBATCH --output=serve_MAX_r0_model_%j.log
#SBATCH --error=serve_MAX_r0_model_%j.err
#SBATCH --time=24:00:00
#SBATCH -p kisski-h100
#SBATCH -G H100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
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
LORA_PATH="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/lora_extraction_results/lora_basic_original_base_rank_256"  
LORA_NAME="biomni-reasoning"

LOAD_BALANCER_PORT=8080

# Kill any existing processes on these ports
echo "Cleaning up any old processes on ports 8000-8003 and 8080..."
for port in 8000 8001 8002 8003 8080; do
    lsof -ti:$port | xargs --no-run-if-empty kill -9
done

echo ""
echo "==============================================="
echo "Launching 4 vLLM servers (1 per GPU)..."
echo "==============================================="

# Launch server on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 8000 \
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
    > vllm_gpu0.log 2>&1 &
echo "✓ Server 1 starting on GPU 0, port 8000 (PID: $!)"

# Launch server on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 8001 \
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
    > vllm_gpu1.log 2>&1 &
echo "✓ Server 2 starting on GPU 1, port 8001 (PID: $!)"

# Launch server on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 8002 \
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
    > vllm_gpu2.log 2>&1 &
echo "✓ Server 3 starting on GPU 2, port 8002 (PID: $!)"

# Launch server on GPU 3
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port 8003 \
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
    > vllm_gpu3.log 2>&1 &
echo "✓ Server 4 starting on GPU 3, port 8003 (PID: $!)"

echo ""
echo "Waiting 60 seconds for servers to initialize..."
sleep 60

# Create Python load balancer script with SMART queue-aware distribution
cat > load_balancer.py << 'EOF'
"""
Smart Load Balancer with Queue-Aware Distribution
- Tracks in-flight requests per backend
- Routes to backend with LEAST pending requests
- Avoids queue saturation on any single GPU
"""
import asyncio
import aiohttp
from aiohttp import web
import sys
import traceback
from datetime import datetime
import threading

BACKEND_PORTS = [8000, 8001, 8002, 8003]
BACKENDS = [f"http://localhost:{port}" for port in BACKEND_PORTS]

# Track backend state: health + in-flight request count
backend_state = {
    backend: {
        "status": "unknown",
        "last_check": None,
        "error_count": 0,
        "in_flight": 0,  # Current pending requests
        "total_requests": 0,  # Total requests sent
        "total_completed": 0,  # Total completed (success + error)
    } for backend in BACKENDS
}

# Lock for thread-safe counter updates
state_lock = threading.Lock()

def get_least_loaded_backend():
    """
    Select backend with least in-flight requests.
    Only considers healthy backends. Falls back to least loaded if all unhealthy.
    """
    with state_lock:
        # Filter healthy backends
        healthy_backends = [
            b for b in BACKENDS 
            if backend_state[b]["status"] == "healthy"
        ]
        
        # If no healthy backends, use all backends
        candidates = healthy_backends if healthy_backends else BACKENDS
        
        # Find backend with minimum in-flight requests
        min_backend = min(candidates, key=lambda b: backend_state[b]["in_flight"])
        
        # Increment in-flight counter
        backend_state[min_backend]["in_flight"] += 1
        backend_state[min_backend]["total_requests"] += 1
        
        return min_backend

def release_backend(backend_url, success=True):
    """Decrement in-flight counter when request completes."""
    with state_lock:
        backend_state[backend_url]["in_flight"] = max(0, backend_state[backend_url]["in_flight"] - 1)
        backend_state[backend_url]["total_completed"] += 1

async def check_backend_health(session, backend_url):
    """Check if a backend is responsive."""
    try:
        async with session.get(f"{backend_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                backend_state[backend_url]["status"] = "healthy"
                backend_state[backend_url]["error_count"] = 0
                return True
            else:
                backend_state[backend_url]["status"] = f"unhealthy (HTTP {resp.status})"
                return False
    except aiohttp.ClientConnectorError as e:
        backend_state[backend_url]["status"] = f"connection_refused"
        backend_state[backend_url]["error_count"] += 1
        return False
    except asyncio.TimeoutError:
        backend_state[backend_url]["status"] = "timeout"
        backend_state[backend_url]["error_count"] += 1
        return False
    except Exception as e:
        backend_state[backend_url]["status"] = f"error ({type(e).__name__})"
        backend_state[backend_url]["error_count"] += 1
        return False
    finally:
        backend_state[backend_url]["last_check"] = datetime.now().isoformat()

async def health_monitor(app):
    """Periodically check backend health and log queue status."""
    session = app['client_session']
    while True:
        print(f"\n[{datetime.now().isoformat()}] === SMART LOAD BALANCER STATUS ===", file=sys.stderr, flush=True)
        total_in_flight = 0
        for backend in BACKENDS:
            is_healthy = await check_backend_health(session, backend)
            status_symbol = "✓" if is_healthy else "✗"
            state = backend_state[backend]
            in_flight = state["in_flight"]
            total_in_flight += in_flight
            total_req = state["total_requests"]
            completed = state["total_completed"]
            print(f"  {status_symbol} {backend}: {state['status']} | In-flight: {in_flight:3d} | Total: {total_req} | Completed: {completed}", file=sys.stderr, flush=True)
        print(f"  📊 Total in-flight across all backends: {total_in_flight}", file=sys.stderr, flush=True)
        await asyncio.sleep(30)

async def proxy_handler(request):
    """Smart proxy: routes to backend with least pending requests."""
    # Select least loaded backend
    backend_url = get_least_loaded_backend()
    session = request.app['client_session']
    url = f"{backend_url}{request.path_qs}"
    
    # Log routing decision for debugging
    with state_lock:
        queue_status = {b.split(':')[-1]: backend_state[b]["in_flight"] for b in BACKENDS}
    print(f"[{datetime.now().isoformat()}] Routing to {backend_url} | Queue depths: {queue_status}", file=sys.stderr, flush=True)

    try:
        async with session.request(
            method=request.method,
            url=url,
            headers=request.headers,
            data=await request.read(),
            timeout=aiohttp.ClientTimeout(total=1800)  # 30 min timeout for long D1 agent runs
        ) as resp:
            body = await resp.read()
            release_backend(backend_url, success=True)
            return web.Response(
                body=body,
                status=resp.status,
                headers=resp.headers
            )
    except aiohttp.ClientConnectorError as e:
        release_backend(backend_url, success=False)
        error_msg = f"Connection error: {type(e).__name__}"
        print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}", file=sys.stderr, flush=True)
        backend_state[backend_url]["error_count"] += 1
        return web.Response(text=f"Proxy error to {backend_url}: {error_msg}", status=502)
    except asyncio.TimeoutError:
        release_backend(backend_url, success=False)
        error_msg = "Timeout after 1800s"
        print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}", file=sys.stderr, flush=True)
        backend_state[backend_url]["error_count"] += 1
        return web.Response(text=f"Proxy error to {backend_url}: {error_msg}", status=504)
    except Exception as e:
        release_backend(backend_url, success=False)
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[{datetime.now().isoformat()}] PROXY ERROR to {backend_url}: {error_msg}", file=sys.stderr, flush=True)
        print(f"  Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        backend_state[backend_url]["error_count"] += 1
        return web.Response(text=f"Proxy error to {backend_url}: {error_msg}", status=502)

async def status_endpoint(request):
    """Endpoint to check load balancer status: GET /lb_status"""
    with state_lock:
        status = {
            "timestamp": datetime.now().isoformat(),
            "backends": {
                backend: {
                    "status": backend_state[backend]["status"],
                    "in_flight": backend_state[backend]["in_flight"],
                    "total_requests": backend_state[backend]["total_requests"],
                    "total_completed": backend_state[backend]["total_completed"],
                    "error_count": backend_state[backend]["error_count"],
                }
                for backend in BACKENDS
            },
            "total_in_flight": sum(backend_state[b]["in_flight"] for b in BACKENDS)
        }
    return web.json_response(status)

async def startup_tasks(app):
    """Initialize session and start health monitoring."""
    print("🚀 Initializing SMART Load Balancer with queue-aware routing...", file=sys.stderr, flush=True)
    connector = aiohttp.TCPConnector(limit_per_host=100, limit=400)
    app['client_session'] = aiohttp.ClientSession(connector=connector)
    
    print("Starting backend health monitor...", file=sys.stderr, flush=True)
    app['health_monitor_task'] = asyncio.create_task(health_monitor(app))
    
    print("\nInitial backend health check:", file=sys.stderr, flush=True)
    for backend in BACKENDS:
        is_healthy = await check_backend_health(app['client_session'], backend)
        status_symbol = "✓" if is_healthy else "✗"
        print(f"  {status_symbol} {backend}: {backend_state[backend]['status']}", file=sys.stderr, flush=True)

async def cleanup_tasks(app):
    """Cleanup on shutdown."""
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
    # Status endpoint for monitoring
    app.router.add_get('/lb_status', status_endpoint)
    # Proxy all other requests
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    
    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    
    port = 8080
    print(f"🚀 SMART Load Balancer starting on port {port}")
    print(f"📊 Queue-aware routing to backends: {BACKENDS}")
    print(f"📍 Status endpoint: http://localhost:{port}/lb_status")
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
echo "Waiting for all vLLM servers to be ready..."
echo "==============================================="

MAX_WAIT_TIME=3600  # Maximum 1 hour wait
CHECK_INTERVAL=30   # Check every 30 seconds
ELAPSED=0

all_servers_ready() {
    local all_ready=true
    for port in 8000 8001 8002 8003; do
        if ! curl -s -f -m 5 "http://localhost:${port}/health" > /dev/null 2>&1; then
            all_ready=false
            echo "  ✗ Port ${port}: Not ready"
        else
            echo "  ✓ Port ${port}: Ready"
        fi
    done
    $all_ready
}

while [ $ELAPSED -lt $MAX_WAIT_TIME ]; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Health check (${ELAPSED}s elapsed):"
    
    if all_servers_ready; then
        echo ""
        echo "==============================================="
        echo "✓ ALL SERVERS ARE READY!"
        echo "==============================================="
        break
    fi
    
    echo "  Waiting ${CHECK_INTERVAL}s before next check..."
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT_TIME ]; then
    echo ""
    echo "==============================================="
    echo "ERROR: Servers did not become ready within ${MAX_WAIT_TIME}s"
    echo "==============================================="
    echo "Check individual server logs:"
    echo "  - vllm_gpu0.log"
    echo "  - vllm_gpu1.log"
    echo "  - vllm_gpu2.log"
    echo "  - vllm_gpu3.log"
    exit 1
fi

# Additional 60s buffer after all servers report healthy
echo ""
echo "Waiting additional 60s buffer for full initialization..."
sleep 60
echo "✓ Ready to start evaluation!"

echo ""
echo "==============================================="
echo "ALL SERVERS READY"
echo "==============================================="
echo "Connect to: http://$(hostname):8080/v1"
echo "Or IP: http://$(hostname -I | awk '{print $1}'):8080/v1"
echo "==============================================="

# Run the evaluation script
echo ""
echo "==============================================="
echo "STARTING EVALUATION (A1 + R0-32B)"
echo "==============================================="
nohup bash run_r0_slurm_MAX_B1.sh > r0_final_A1_R0_32B.log 2>&1 &
EVAL_PID=$!
echo "✓ Evaluation started (PID: $EVAL_PID)"
echo "  Log: r0_final_A1_R0_32B.log"

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

