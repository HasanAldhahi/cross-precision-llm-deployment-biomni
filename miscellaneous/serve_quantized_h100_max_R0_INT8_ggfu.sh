#!/bin/bash
#SBATCH --job-name=serve_r0_model
#SBATCH --output=serve_MAX_r0_model_%j.log
#SBATCH --error=serve_MAX_r0_model_%j.err
#SBATCH --time=48:00:00
#SBATCH -p kisski-h100
#SBATCH -G H100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
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

MODEL_PATH="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/models/Biomni-R0-32B-Preview-GGUF/Biomni-R0-32B-Preview.Q8_0.gguf"
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
    --served-model-name "biomni/Biomni-R0-32B-Preview" \
    --tensor-parallel-size 1 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    --enforce-eager \
    > vllm_gpu0.log 2>&1 &
echo "✓ Server 1 starting on GPU 0, port 8000 (PID: $!)"

# Launch server on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "biomni/Biomni-R0-32B-Preview" \
    --tensor-parallel-size 1 \
    --port 8001 \
    --host 0.0.0.0 \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    --enforce-eager \
    > vllm_gpu1.log 2>&1 &
echo "✓ Server 2 starting on GPU 1, port 8001 (PID: $!)"

# Launch server on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "biomni/Biomni-R0-32B-Preview" \
    --tensor-parallel-size 1 \
    --port 8002 \
    --host 0.0.0.0 \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    --enforce-eager \
    > vllm_gpu2.log 2>&1 &
echo "✓ Server 3 starting on GPU 2, port 8002 (PID: $!)"

# Launch server on GPU 3
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "biomni/Biomni-R0-32B-Preview" \
    --tensor-parallel-size 1 \
    --port 8003 \
    --host 0.0.0.0 \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --disable-log-requests \
    --enforce-eager \
    > vllm_gpu3.log 2>&1 &
echo "✓ Server 4 starting on GPU 3, port 8003 (PID: $!)"

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

BACKEND_PORTS = [8000, 8001, 8002, 8003]
BACKENDS = [f"http://localhost:{port}" for port in BACKEND_PORTS]
backend_cycle = itertools.cycle(BACKENDS)

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
    except Exception as e:
        print(f"PROXY ERROR to {backend_url}: {e}", file=sys.stderr, flush=True)
        return web.Response(text=f"Proxy error to {backend_url}: {e}", status=502)

async def startup_tasks(app):
    """Create a single, shared ClientSession on application startup."""
    print("Initializing shared ClientSession for the application...")
    connector = aiohttp.TCPConnector(limit_per_host=100, limit=400)
    # Store the session in the app object so handlers can access it
    app['client_session'] = aiohttp.ClientSession(connector=connector)

async def cleanup_tasks(app):
    """Close the shared ClientSession on application cleanup."""
    print("Closing shared ClientSession...")
    await app['client_session'].close()

if __name__ == '__main__':
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    
    # Register the startup and cleanup tasks
    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    
    port = 8080
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
echo "Connect to: http://$(hostname):8080/v1"
echo "Or IP: http://$(hostname -I | awk '{print $1}'):8080/v1"
echo "==============================================="

# Wait indefinitely until the job is cancelled
wait