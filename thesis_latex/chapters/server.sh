
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

as well as load balancer 
pick the most important functions here and describe it 

""Smart Load Balancer with Queue-Aware Distribution
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
    web.run_app(app, host='0.0.0.0', port=port)"
