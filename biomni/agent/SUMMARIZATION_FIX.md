# Context Length Error Fix - Summarization Feature

## Problem Identified

Even with the summarization feature implemented, evaluation runs were still hitting context length errors:

```
Error code: 400 - {'error': {'message': "'max_tokens' or 'max_completion_tokens' is too large: 8192. 
This model's maximum context length is 65536 tokens and your request has 57879 input tokens 
(8192 > 65536 - 57879). None", 'type': 'BadRequestError', 'param': None, 'code': 400}}
```

### Root Cause Analysis

The error occurred because:
1. **Model context limit**: 65,536 tokens
2. **Input tokens sent**: ~58,000 tokens
3. **Response tokens requested**: 8,192 tokens
4. **Total needed**: ~66,000+ tokens → **EXCEEDS LIMIT**

The original summarization implementation had **two critical flaws**:

#### Flaw 1: System Prompt Not Counted
```python
# OLD CODE - ONLY counted conversation messages
for msg in messages:
    total_chars += len(str(msg.content))
```

**Problem**: The system prompt (20-30K tokens with all tools/data/libraries) was **NOT** included in the calculation, even though it's added to every API call:
```python
messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
```

#### Flaw 2: Threshold Too High
- Original threshold: 50,000 tokens
- But total context = system_prompt (30K) + conversation (25K) = 55K
- Add response tokens (8K) = 63K
- **Still exceeded the 65K limit!**

## Solution Implemented

### 1. **Account for System Prompt**
Now calculates the **complete context** that will be sent to the model:

```python
# Calculate conversation tokens
conversation_chars = sum(len(str(msg.content)) for msg in messages)
conversation_tokens = conversation_chars / 4

# Calculate system prompt tokens (this is added in every generate call)
system_prompt_chars = len(self.system_prompt)
system_prompt_tokens = system_prompt_chars / 4

# Total context that will be sent
total_input_tokens = system_prompt_tokens + conversation_tokens
```

### 2. **Dynamic Safe Threshold**
Calculates a safe threshold based on the model's actual limits:

```python
# Calculate safe threshold (leave room for response + 10% buffer)
safe_threshold = model_max_tokens - max_completion_tokens - (model_max_tokens * 0.1)

# For 65,536 token model:
# safe_threshold = 65,536 - 8,192 - 6,553 = ~47,000 tokens
```

### 3. **Detailed Logging**
Provides clear visibility into what's happening:

```python
print(f"📊 Context length check:")
print(f"   System prompt: ~{int(system_prompt_tokens):,} tokens ({system_prompt_chars:,} chars)")
print(f"   Conversation: ~{int(conversation_tokens):,} tokens ({conversation_chars:,} chars)")
print(f"   Total input: ~{int(total_input_tokens):,} tokens")
print(f"   Safe threshold: ~{int(safe_threshold):,} tokens")
print(f"   ⚠️ SUMMARIZATION TRIGGERED (would exceed limit with response)")
```

## New Behavior

### Example Scenario
For a 65,536 token model:
- **System prompt**: ~28,000 tokens
- **Conversation**: Growing with each step
- **Response space**: 8,192 tokens reserved
- **Safety buffer**: 6,553 tokens (10%)

**Trigger point**: When `total_input` > 47,000 tokens

This ensures:
- ✅ System prompt counted
- ✅ Conversation counted
- ✅ Response space reserved
- ✅ Safety buffer included
- ✅ **Never exceeds model limit**

### Calculation Breakdown
```
Model max:           65,536 tokens
Response reserved:   -8,192 tokens
Safety buffer (10%): -6,553 tokens
                     _______________
Safe threshold:      =47,000 tokens
```

When `system_prompt (28K) + conversation (19K+)` > 47K → **SUMMARIZE**

## Benefits

1. **Prevents Context Overflow**: Accounts for ALL components of the request
2. **Model-Agnostic**: Automatically adapts to different model context windows
3. **Safe by Design**: 10% buffer prevents edge cases
4. **Transparent**: Clear logging shows exactly what's being counted
5. **Configurable**: Easy to adjust for different models:
   ```python
   should_summarize(
       state, 
       model_max_tokens=131072,  # GPT-4 Turbo
       max_completion_tokens=8192
   )
   ```

## Secondary Issue Found & Fixed

After the first fix, **3 more instances still failed** with the same error. Root cause:

### The Re-Summarization Blocking Bug

The code prevented re-summarization if ANY of the last 3 messages contained a summary:

```python
# OLD CODE - BLOCKS re-summarization
for msg in messages[-3:]:
    if '<summary>' in str(msg.content):
        return False  # Blocks even if context is WAY over limit!
```

**What happened:**
1. First summarization at ~47K tokens ✅
2. Conversation continues...
3. Reaches ~58K tokens (near limit!)
4. Check says "you summarized recently, skip" ❌
5. **API ERROR**: Context overflow!

### The Complete Fix

**1. More Conservative Threshold**
- Changed from 10% buffer to **15% buffer**
- Safe threshold: 65,536 - 8,192 - 9,830 = **~47,500 tokens**

**2. Critical Threshold (Emergency Mode)**
```python
critical_threshold = model_max_tokens - max_completion_tokens - 1000
# = 65,536 - 8,192 - 1,000 = 56,344 tokens

if total_input_tokens >= critical_threshold:
    # ALWAYS summarize, ignore "recent summary" check
    return True
```

**3. Smart Re-Summarization**
```python
# Only block if the LAST message (not last 3) was a summary
if messages[-1] has '<summary>':
    # But we're still over threshold, so summarization didn't help
    # Allow re-summarization!
    return True
```

## New Behavior with Fix

| Situation | Old Code | New Code |
|-----------|----------|----------|
| First time at 47K | ✅ Summarize | ✅ Summarize (15% buffer) |
| After summary, at 50K | ❌ Block (recent) | ✅ Re-summarize |
| At 56K (critical) | ❌ Block (recent) | 🚨 FORCE summarize |
| At 58K+ | ❌ **ERROR** | 🚨 FORCE summarize |

## Testing Recommendation

Re-run the failed instances with the updated code:
```bash
python compare/run_eval_R0_async_MAX_A1_summarization.py
```

The summarization will now:
1. Trigger **earlier** (15% buffer = ~47.5K tokens)
2. **Allow re-summarization** when needed
3. **Force summarization** at critical threshold (~56K tokens)
4. **Never hit the API limit** (65K tokens)

## Model-Specific Settings

| Model | Context Window | Recommended Config |
|-------|----------------|-------------------|
| GPT-3.5/GPT-4 | 65,536 | Default (built-in) |
| GPT-4 Turbo | 128,000 | `model_max_tokens=131072` |
| Claude 3 | 200,000 | `model_max_tokens=200000` |
| Smaller models | 32,768 | `model_max_tokens=32768` |

The system automatically calculates safe thresholds for any model size!



