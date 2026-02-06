#!/usr/bin/env python3
"""
Biomni-R0-32B-Preview Evaluation Script
Run the Biomni R0 model on GPU for evaluation tasks
"""

import sys
import os
from pathlib import Path
import argparse
import subprocess
import time
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from biomni.agent import A1

# Configuration for Biomni-R0-32B-Preview
MODEL_NAME = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"
DEFAULT_PORT = 30000
DEFAULT_HOST = "0.0.0.0"
ENVIRONMENT_PATH = '/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env'

def check_server_running(base_url):
    """Check if SGLang server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_sglang_server(port=DEFAULT_PORT, host=DEFAULT_HOST, tp=2, rope_factor=1.0):
    """
    Start SGLang server for Biomni-R0-32B-Preview
    
    Args:
        port: Server port (default: 30000)
        host: Server host (default: 0.0.0.0)
        tp: Tensor parallelism - number of GPUs (default: 2 for 2x80GB, use 4 for 4x40GB)
        rope_factor: RoPE scaling factor (1.0-4.0, higher for longer contexts)
    """
    print(f"🚀 Starting SGLang server for {MODEL_NAME}...")
    print(f"   Port: {port}")
    print(f"   Host: {host}")
    print(f"   Tensor Parallelism (GPUs): {tp}")
    print(f"   RoPE Scaling Factor: {rope_factor}")
    print()
    
    json_override = (
        '{"rope_scaling":{"rope_type":"yarn","factor":' + str(rope_factor) + 
        ',"original_max_position_embeddings":32768}, "max_position_embeddings": 131072}'
    )
    
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        "--port", str(port),
        "--host", host,
        "--mem-fraction-static", "0.8",
        "--tp", str(tp),
        "--trust-remote-code",
        "--json-model-override-args", json_override
    ]
    
    print("Command:", " ".join(cmd))
    print("\n" + "="*80)
    print("⚠️  Server will run in background. Starting server...")
    print("="*80 + "\n")
    
    # Start server in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    base_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    max_wait = 300  # 5 minutes
    wait_interval = 5
    
    print("Waiting for server to start...", end="", flush=True)
    for i in range(0, max_wait, wait_interval):
        if check_server_running(base_url):
            print(" ✅ Server is ready!")
            return process, base_url
        print(".", end="", flush=True)
        time.sleep(wait_interval)
    
    print(" ❌ Failed to start")
    print("\nServer stdout:", process.stdout.read() if process.stdout else "")
    print("\nServer stderr:", process.stderr.read() if process.stderr else "")
    raise RuntimeError("Failed to start SGLang server within timeout")

def run_evaluation(base_url, task_description=None, timeout_seconds=1000):
    """
    Run evaluation with A1 agent using Biomni-R0-32B-Preview
    
    Args:
        base_url: Base URL of the SGLang server
        task_description: Task description for the agent
        timeout_seconds: Timeout for agent execution
    """
    print(f"\n{'='*80}")
    print(f"🧬 BIOMNI-R0-32B-PREVIEW EVALUATION")
    print(f"{'='*80}")
    print(f"Timestamp: {os.popen('date').read().strip()}")
    print(f"Base URL: {base_url}")
    print(f"Environment Path: {ENVIRONMENT_PATH}")
    print(f"{'='*80}\n")
    
    # Default task if none provided
    if task_description is None:
        task_description = """Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
            measured by the change in T cell receptor (TCR) signaling between acute
            (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
            Generate 32 genes that maximize the perturbation effect."""
    
    try:
        # Create A1 agent with R0 model
        agent = A1(
            path=ENVIRONMENT_PATH,
            llm=MODEL_NAME,
            source='custom',
            use_tool_retriever=True,
            timeout_seconds=timeout_seconds,
            base_url=f"{base_url}/v1",
            api_key="EMPTY",  # SGLang doesn't require API key
        )
        
        print(f"✅ A1 agent created successfully with model: {MODEL_NAME}")
        print(f"\n{'='*80}")
        print("📋 TASK:")
        print(f"{'='*80}")
        print(task_description)
        print(f"{'='*80}\n")
        
        # Run the task
        print("🔄 Running task...\n")
        log = agent.go(task_description)
        
        print(f"\n{'='*80}")
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model: {MODEL_NAME}")
        print(f"Agent: A1")
        print(f"End time: {os.popen('date').read().strip()}")
        print(f"{'='*80}\n")
        
        return log
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR: Evaluation failed")
        print(f"{'='*80}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run Biomni-R0-32B-Preview model for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server and run evaluation
  python run_R0.py --start-server
  
  # Use existing server
  python run_R0.py --base-url http://localhost:30000
  
  # Start server with 4 GPUs (40GB each)
  python run_R0.py --start-server --tp 4
  
  # Custom task
  python run_R0.py --base-url http://localhost:30000 --task "Your task here"
        """
    )
    
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start SGLang server before running evaluation"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of existing SGLang server (e.g., http://localhost:30000)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for SGLang server (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host for SGLang server (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallelism (number of GPUs): 2 for 2x80GB, 4 for 4x40GB (default: 2)"
    )
    parser.add_argument(
        "--rope-factor",
        type=float,
        default=1.0,
        help="RoPE scaling factor (1.0-4.0, higher for longer contexts, default: 1.0)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Custom task description for evaluation"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="Timeout in seconds for agent execution (default: 1000)"
    )
    
    args = parser.parse_args()
    
    server_process = None
    
    try:
        # Determine base URL
        if args.start_server:
            server_process, base_url = start_sglang_server(
                port=args.port,
                host=args.host,
                tp=args.tp,
                rope_factor=args.rope_factor
            )
        elif args.base_url:
            base_url = args.base_url
            if not check_server_running(base_url):
                print(f"❌ Error: Server at {base_url} is not running")
                print("   Start the server first or use --start-server")
                sys.exit(1)
        else:
            print("❌ Error: Must specify either --start-server or --base-url")
            print("   Run with --help for usage information")
            sys.exit(1)
        
        # Run evaluation
        run_evaluation(
            base_url=base_url,
            task_description=args.task,
            timeout_seconds=args.timeout
        )
        
    finally:
        # Clean up server if we started it
        if server_process:
            print("\n🛑 Shutting down SGLang server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("✅ Server stopped")

if __name__ == "__main__":
    main()
