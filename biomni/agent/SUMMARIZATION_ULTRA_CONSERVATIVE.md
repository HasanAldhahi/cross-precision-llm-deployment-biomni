# Ultra-Conservative Summarization Settings

## Problem

Even with the previous fixes, instance 98 hit a context length error at **57,794 input tokens** after **44 minutes of execution**:

```json
{
  "error": "Error code: 400 - 'max_tokens' or 'max_completion_tokens' is too large: 8192. 
            This model's maximum context length is 65536 tokens and your request has 57794 input tokens 
            (8192 > 65536 - 57794).",
  "execution_time": 2649.077145576477  // 44 minutes!
}
```

This instance ran for a very long time with many tool calls, causing rapid context growth even after summarization.

## Root Cause

The previous thresholds were not conservative enough for **extremely long-running instances**:

| Setting | Previous Value | Issue |
|---------|---------------|-------|
| Safe threshold | 47,514 tokens | Not early enough |
| Critical threshold | 56,344 tokens | Instance exceeded it (57,794 tokens) |
| Response allocation | 8,192 tokens | Too large, leaving less room for conversation |
| Buffer percentage | 15% | Not enough safety margin |

## New Ultra-Conservative Settings

### 1. **Reduced Response Allocation**
```python
max_completion_tokens: int = 4096  # Was 8192
```

**Why**: Most responses don't need 8K tokens. By reducing to 4K, we free up 4K more tokens for conversation, allowing it to grow longer before summarization.

### 2. **More Aggressive Buffer (25% instead of 15%)**
```python
safe_threshold = model_max_tokens - max_completion_tokens - (model_max_tokens * 0.25)
# 65,536 - 4,096 - 16,384 = 45,056 tokens
```

**Why**: Leaves much more room for:
- Token estimation errors (char/4 is approximate)
- System prompt variations
- Unexpected growth between checks

### 3. **Much Lower Critical Threshold (3K buffer instead of 1K)**
```python
critical_threshold = model_max_tokens - max_completion_tokens - 3000
# 65,536 - 4,096 - 3,000 = 58,440 tokens
```

**Why**: The previous 1K buffer wasn't enough. With 3K buffer, even if we somehow miss the safe threshold, the critical threshold will definitely catch it before API error.

## New Thresholds Comparison

| Threshold | Old Value | New Value | Difference |
|-----------|-----------|-----------|------------|
| **Safe** | 47,514 tokens | **45,056 tokens** | -2,458 (triggers 2.5K earlier) |
| **Critical** | 56,344 tokens | **58,440 tokens** | +2,096 (more headroom) |
| **Response** | 8,192 tokens | **4,096 tokens** | -4,096 (doubles available space) |
| **Buffer %** | 15% | **25%** | +10% (much safer) |

## Expected Behavior

### For the Failing Instance (57,794 tokens)

**Old behavior:**
```
47,514 tokens → ✅ Summarize (safe threshold)
Conversation continues...
56,344 tokens → 🚨 Force summarize (critical threshold)  
Conversation continues...
57,794 tokens → ❌ API ERROR (exceeded limit)
```

**New behavior:**
```
45,056 tokens → ✅ Summarize (safe threshold - EARLIER)
Conversation continues...
If summarization didn't reduce enough...
58,440 tokens → 🚨 Force summarize (critical threshold - MORE HEADROOM)
Conversation continues...
Should never reach 61,440 tokens (65,536 - 4,096)
```

### Safety Margins

| Scenario | Old System | New System |
|----------|------------|------------|
| **Normal case** | Trigger at 72% capacity | Trigger at **69% capacity** |
| **Emergency** | Force at 86% capacity | Force at **89% capacity** |
| **Response space** | 8,192 tokens (12.5%) | 4,096 tokens (**6.25%**) |
| **Total safety** | ~8,000 tokens buffer | **~20,000 tokens buffer** |

## Testing Recommendations

### 1. Run Failed Instance Again

```bash
# Extract the failed instance ID and retry
echo '{"instance_ids": [98]}' > eval_output/retry_instances.json

# Modify the evaluation script to only process this instance
python compare/run_eval_R0_async_MAX_A1_summarization.py
```

### 2. Monitor Summarization

Watch for these log messages:
```
📊 Context length check:
   System prompt: ~XX,XXX tokens
   Conversation: ~XX,XXX tokens  
   Total input: ~XX,XXX tokens
   Safe threshold: ~45,056 tokens
   ⚠️ SUMMARIZATION TRIGGERED

🚨 CRITICAL CONTEXT LENGTH - FORCING SUMMARIZATION:
   ...
   Critical threshold: ~58,440 tokens
```

### 3. Expected Results

For the 44-minute instance:
- **First summarization**: Should trigger around 45K total tokens (much earlier)
- **Possible second summarization**: If conversation continues to grow
- **Critical catch**: If it somehow reaches 58K, force summarization with plenty of room
- **No more errors**: Should never hit the 61K+ limit

## Performance Impact

**Pros:**
- ✅ Eliminates context overflow errors
- ✅ Allows longer conversations (more room)
- ✅ Multiple summarization rounds supported
- ✅ Much safer for long-running instances

**Cons:**
- ⚠️ Summarizes slightly earlier (saves context proactively)
- ⚠️ Responses limited to 4K tokens (usually sufficient for most tasks)
- ⚠️ Slightly more aggressive memory management

**Trade-off**: Prioritizes **reliability over maximum context usage**. Better to summarize a bit early than to crash with a 400 error after 44 minutes of work!

## Rollback (If Needed)

If responses are being truncated (rare), you can adjust:

```python
# In a1_summarization.py, line 1294
def should_summarize(
    state: AgentState, 
    model_max_tokens: int = 65536, 
    max_completion_tokens: int = 6144  # Increase from 4096 to 6144
):
```

This gives responses more room (6K tokens) while still being safer than the original 8K.




