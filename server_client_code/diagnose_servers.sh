#!/bin/bash

echo "==============================================="
echo "vLLM Server Diagnostics"
echo "==============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

echo "1. Checking if processes are running on ports 8000-8003:"
for port in 8000 8001 8002 8003 8080; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  ✓ Port $port: Active (PID: $pid)"
        ps aux | grep $pid | grep -v grep | head -1
    else
        echo "  ✗ Port $port: No process listening"
    fi
done

echo ""
echo "2. Checking vLLM log files (last 20 lines each):"
for i in 0 1 2 3; do
    log_file="vllm_gpu$i.log"
    if [ -f "$log_file" ]; then
        size=$(wc -l < "$log_file")
        echo ""
        echo "--- $log_file ($size lines) ---"
        if [ $size -gt 0 ]; then
            tail -20 "$log_file"
        else
            echo "  (empty file)"
        fi
    else
        echo ""
        echo "--- $log_file ---"
        echo "  (file not found)"
    fi
done

echo ""
echo "3. Testing backend connectivity:"
for port in 8000 8001 8002 8003; do
    echo -n "  Port $port: "
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$port/health 2>/dev/null)
    if [ "$response" = "200" ]; then
        echo "✓ Healthy (HTTP 200)"
    elif [ "$response" = "000" ]; then
        echo "✗ Connection refused"
    else
        echo "✗ HTTP $response"
    fi
done

echo ""
echo "4. GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "5. Python processes (vLLM):"
ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep

echo ""
echo "6. Network listeners:"
netstat -tlnp 2>/dev/null | grep -E ":(8000|8001|8002|8003|8080)" || ss -tlnp | grep -E ":(8000|8001|8002|8003|8080)"

echo ""
echo "==============================================="
echo "Diagnostics complete"
echo "==============================================="










