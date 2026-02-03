"""
B1 Agent: A1 with Dynamic Tool-Building Capability (STELLA)

This agent extends A1 with the ability to dynamically create new tools when needed.
It inherits ALL functionality from A1 and adds the "maqbosa" protocol for on-demand tool building.
"""

import os
import re
import inspect
import importlib
from typing import TypedDict, List, Optional, Any
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field

# Import the base A1 agent
from biomni.agent.a1 import A1, AgentState
from biomni.utils import function_to_api_schema as original_function_to_api_schema


class B1(A1):
    """
    B1 Agent: Inherits ALL A1 capabilities + Dynamic Tool Building
    
    This agent has everything A1 has:
    - All tools, data, software management
    - MCP integration
    - Conversation history & PDF export
    - Streaming interface
    - Result formatting
    
    PLUS the ability to dynamically create new tools via the "maqbosa" protocol.
    """
    
    # =========================================================================
    #  INTERNAL HELPER CLASS: FunctionGenerator (STELLA)
    # =========================================================================
    class _FunctionGenerator:
        """Specialized agent that generates executable Python code scripts."""
        
        def __init__(self, llm_instance: BaseChatModel, temperature: float = 0.7):
            self.llm = llm_instance
            self._configure()

        def _configure(self):
            self.system_prompt = """You are a senior Python engineer. Your sole task is to generate robust, idiomatic Python code that solves the user's task.
            
Follow these requirements STRICTLY:
1. Output ONLY Python code inside a single triple-backtick block (```python ... ```).
2. Do NOT include any explanations, prose, or text before or after the code block.
3. Include a concise docstring explaining the function's purpose, arguments, and what it returns.
4. Add all necessary library imports at the top of the script.
5. If external dependencies are used, add a comment showing the pip install command.

Generate Python code for the following task:
{task}
"""

        def go(self, task_description: str) -> str:
            """Main entry point for the generator."""
            prompt = self.system_prompt.format(task=task_description)
            message = self.llm.invoke(prompt)
            return self._extract_code_block(message.content)

        def _extract_code_block(self, s: str) -> str:
            """Extracts the Python code from the LLM's response."""
            if not s:
                return None
            m = re.search(r"```(?:python)?\s*(.+?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else None

    # =========================================================================
    #  B1 INITIALIZATION (Extends A1)
    # =========================================================================
    def __init__(self, *args, **kwargs):
        """Initialize B1 with all A1 capabilities + STELLA."""
        # Call parent (A1) constructor first
        super().__init__(*args, **kwargs)
        
        # Initialize STELLA components
        self.code_generator = self._FunctionGenerator(llm_instance=self.llm)
        self.stella_mode_activated = False
        
        print("\n🌟 B1 Agent initialized with A1 capabilities + STELLA tool-building")
        print("   Use 'maqbosa' in your thinking to activate dynamic tool creation\n")

    # =========================================================================
    #  STELLA INTEGRATION: Tools & Helpers
    # =========================================================================
    def _update_system_prompt_for_stella(self):
        """Injects the 'maqbosa' trigger instructions into the prompt."""
        stella_instructions = """

# ADVANCED MODE: Dynamic Tool Creation ("maqbosa")

**STRATEGY:**
1. First, try to solve with existing tools and `run_python_repl`
2. If you completely lack a necessary tool for a complex task, you can BUILD it
3. To activate tool-building mode, output the magic word **"maqbosa"** in your <think> tags

**EXAMPLE:**
<think>
I need to perform a complex restriction digest analysis. I don't have a tool for this, and writing it as a raw script is error-prone. I need to build a robust tool. Activating protocol: **maqbosa**.
</think>

Once activated, you will gain:
- `create_tool_from_description(task_description)`: To build the tool you need
- `request_critical_feedback()`: To get expert review if stuck
"""
        if "maqbosa" not in self.system_prompt:
            self.system_prompt += stella_instructions

    def _safe_function_to_api_schema(self, function_code: str, llm: BaseChatModel) -> dict:
        """Robust wrapper for schema generation with fallback."""
        try:
            schema = original_function_to_api_schema(function_code, llm)
            if isinstance(schema, dict) and schema.get("name"):
                return schema
        except Exception:
            pass

        # Fallback using Pydantic V1
        class Parameter(BaseModel):
            name: str = Field(...)
            type: str = Field(...)
            description: str = Field(...)
            default: Optional[Any] = Field(None)

        class APISchema(BaseModel):
            name: str = Field(...)
            description: str = Field(...)
            required_parameters: List[Parameter] = Field(...)
            optional_parameters: List[Parameter] = Field(...)
            
        try:
            json_llm = llm.with_structured_output(APISchema)
            prompt = ChatPromptTemplate.from_template(
                "Analyze this Python function code and output ONLY a JSON schema describing it.\nCode:\n```{code}```"
            )
            chain = prompt | json_llm
            api_model = chain.invoke({"code": function_code})
            return api_model.dict()
        except Exception as e:
            print(f"⚠️ Schema generation fallback failed: {e}")
            return {}

    def _register_stella_tools(self):
        """Activates the advanced tools mid-run."""
        print("\n⚡ MAQBOSA DETECTED: Activating STELLA Tool Builder & Critic ⚡")
        self.add_tool(self.create_tool_from_description)
        self.add_tool(self.request_critical_feedback)
        print("✅ Advanced tools registered.\n")

    def create_tool_from_description(self, task_description: str) -> str:
        """STELLA TOOL: Generates and registers a new Python tool on the fly."""
        print(f"\n--- 🔨 Building Tool: {task_description[:50]}... ---")
        
        # 1. Generate Code
        tool_code = self.code_generator.go(task_description)
        if not tool_code:
            return "Error: Failed to generate valid Python code."
        
        try:
            # 2. Setup Directory
            dynamic_tools_dir = os.path.join(self.path, "dynamic_tools")
            os.makedirs(dynamic_tools_dir, exist_ok=True)
            init_path = os.path.join(dynamic_tools_dir, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    pass

            # 3. Extract Name & Save
            match = re.search(r"def\s+(\w+)\s*\(", tool_code)
            if not match:
                return "Error: No function definition found in generated code."
            tool_name = match.group(1)
            
            tool_file_path = os.path.join(dynamic_tools_dir, f"{tool_name}.py")
            with open(tool_file_path, "w") as f:
                f.write(tool_code)

            # 4. Import & Register
            module_name = f"dynamic_tools.{tool_name}"
            spec = importlib.util.spec_from_file_location(module_name, tool_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            new_tool_function = getattr(module, tool_name)
            self.add_tool(new_tool_function)  # Uses A1's robust add_tool method
            
            return f"✅ Tool '{tool_name}' created and registered successfully. You can now use it."
        except Exception as e:
            return f"❌ Error creating tool: {e}"

    def request_critical_feedback(self) -> str:
        """STELLA TOOL: Placeholder for the critic logic handled in _execute."""
        return "Critical feedback requested."

    def _run_critic(self, state: AgentState) -> str:
        """STELLA LOGIC: Runs the critique."""
        print("\n--- 🧐 Expert Critic Reviewing... ---")
        feedback_prompt = f"You are an expert Critic. Review the agent's progress on: '{self.user_task}'. Identify any logical flaws or missing steps. Be concise and actionable."
        feedback = self.llm.invoke(state["messages"] + [HumanMessage(content=feedback_prompt)])
        return f"CRITIC FEEDBACK: {feedback.content}"

    # =========================================================================
    #  OVERRIDE: add_tool (Use safe schema generation for B1's dynamic tools)
    # =========================================================================
    def add_tool(self, api):
        """
        Override A1's add_tool to use B1's safe schema generator for dynamic tools.
        This ensures dynamically created tools work reliably.
        """
        try:
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)

            # Check if this is a dynamically created tool (from dynamic_tools module)
            if "dynamic_tools" in module_name:
                # Use B1's safe schema generator
                schema = self._safe_function_to_api_schema(function_code, self.llm)
                if not schema:
                    print(f"⚠️ Skipping registration for '{function_name}': Invalid schema.")
                    return None
            else:
                # Use A1's standard schema generation for regular tools
                from biomni.utils import function_to_api_schema
                schema = function_to_api_schema(function_code, self.llm)

            # Ensure metadata exists
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Tool: {function_name}"
            schema["module"] = module_name

            # Use A1's registration logic (tool registry, module2api, etc.)
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")

            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                existing_tool.update(schema)
            else:
                self.module2api[module_name].append(schema)

            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            import builtins
            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions[schema["name"]] = api

            print(f"Tool '{schema['name']}' successfully added.")
            self.configure()  # Reconfigure to update system prompt
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback
            traceback.print_exc()
            return None

    # =========================================================================
    #  OVERRIDE: _generate (Add loop detection + maqbosa activation)
    # =========================================================================
    def _generate(self, state: AgentState) -> AgentState:
        """Override A1's generate to add loop detection and maqbosa activation."""
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        response = self.llm.invoke(messages)
        msg = str(response.content)

        # Fix common tag errors
        if "<execute>" in msg and "</execute>" not in msg:
            msg += "</execute>"
        if "<solution>" in msg and "</solution>" not in msg:
            msg += "</solution>"
        if "<think>" in msg and "</think>" not in msg:
            msg += "</think>"

        state["messages"].append(AIMessage(content=msg.strip()))

        # === MAQBOSA DETECTION ===
        if not self.stella_mode_activated and "maqbosa" in msg.lower():
            self._register_stella_tools()
            self.stella_mode_activated = True

        # === LOOP DETECTION ===
        if not hasattr(self, '_action_history'):
            self._action_history = []
        
        if "<execute>" in msg:
            exec_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            if exec_match:
                action_sig = exec_match.group(1).strip()[:200]
                self._action_history.append(action_sig)
                
                # Check for loops: if last 3 actions are very similar
                if len(self._action_history) >= 3:
                    recent_actions = self._action_history[-3:]
                    similarity_count = 0
                    for i in range(len(recent_actions)-1):
                        for j in range(i+1, len(recent_actions)):
                            if self._strings_similar(recent_actions[i], recent_actions[j], threshold=0.7):
                                similarity_count += 1
                    
                    if similarity_count >= 2:
                        print("\n⚠️ LOOP DETECTED: Agent is repeating similar actions")
                        if not self.stella_mode_activated:
                            state["messages"].append(HumanMessage(content="""
You appear to be stuck in a loop, repeating similar actions without making progress.

You have TWO options:
1. If you need a specialized tool to solve this problem, activate advanced mode by including the word "maqbosa" in your thinking.
2. If you have enough information to provide a reasonable answer (even if not perfect), provide your best answer in <solution> tags NOW.

Do NOT continue with the same approach. Make a decision immediately."""))
                            state["next_step"] = "generate"
                            return state
                        else:
                            print("⚠️ Still looping even in STELLA mode. Forcing termination.")
                            state["next_step"] = "end"
                            state["messages"].append(AIMessage(content="<solution>Unable to find definitive answer after exhaustive search. Based on available information, I cannot conclusively determine the answer.</solution>"))
                            return state

        # Check conversation length
        if len(state["messages"]) > 50:
            print(f"\n⚠️ CONVERSATION TOO LONG ({len(state['messages'])} messages). Forcing termination.")
            state["next_step"] = "end"
            state["messages"].append(AIMessage(content="<solution>Unable to complete analysis within reasonable limits.</solution>"))
            return state

        # Route based on tags
        think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL)
        execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
        answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

        if answer_match:
            state["next_step"] = "end"
        elif execute_match:
            state["next_step"] = "execute"
        elif think_match:
            state["next_step"] = "generate"
        else:
            error_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage) and "no valid tags" in m.content.lower())
            if error_count >= 2:
                print("Detected repeated parsing errors, ending conversation")
                state["next_step"] = "end"
                state["messages"].append(AIMessage(content="<solution>Execution terminated due to repeated parsing errors.</solution>"))
            else:
                state["messages"].append(HumanMessage(content="Each response must include thinking process followed by either <execute> or <solution> tag. Please follow the instruction and regenerate."))
                state["next_step"] = "generate"

        return state
    
    def _strings_similar(self, s1: str, s2: str, threshold: float = 0.7) -> bool:
        """Check if two strings are similar using word overlap."""
        if not s1 or not s2:
            return False
        s1_set = set(s1.lower().split())
        s2_set = set(s2.lower().split())
        if not s1_set or not s2_set:
            return False
        intersection = len(s1_set & s2_set)
        union = len(s1_set | s2_set)
        return (intersection / union) > threshold if union > 0 else False

    # =========================================================================
    #  OVERRIDE: configure (Add STELLA instructions and handle critic calls)
    # =========================================================================
    def configure(self, self_critic=False, test_time_scale_round=0):
        """Override A1's configure to add STELLA instructions."""
        # Call parent configure first to set up everything
        super().configure(self_critic=self_critic, test_time_scale_round=test_time_scale_round)
        
        # Inject STELLA instructions
        self._update_system_prompt_for_stella()
        
        # Rebuild graph with STELLA-aware nodes
        def generate_with_stella(state: AgentState) -> AgentState:
            return self._generate(state)

        def execute_with_critic(state: AgentState) -> AgentState:
            # Check for critic call if STELLA is active
            if self.stella_mode_activated:
                last_msg = state["messages"][-1].content if state["messages"] else ""
                if "<execute>" in last_msg and "request_critical_feedback()" in last_msg:
                    feedback = self._run_critic(state)
                    state["messages"].append(AIMessage(content=f"\n<observation>{feedback}</observation>"))
                    return state
            
            # Otherwise, use A1's execute logic
            return self._execute_original(state)

        # Store original execute for delegation
        self._execute_original = self._get_execute_function_from_graph()

        # Rebuild workflow with B1's enhanced nodes
        from typing import Literal
        
        def routing_function(state: AgentState) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        workflow = StateGraph(AgentState)
        workflow.add_node("generate", generate_with_stella)
        workflow.add_node("execute", execute_with_critic)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("execute", "generate")
        workflow.add_conditional_edges(
            "generate",
            routing_function,
            path_map={"execute": "execute", "generate": "generate", "end": END}
        )

        self.app = workflow.compile()
        self.app.checkpointer = self.checkpointer
        
        print("✅ B1 agent configured with A1 capabilities + STELLA")

    def _get_execute_function_from_graph(self):
        """Extract the execute function from the compiled graph."""
        # This is a workaround to access the parent's execute logic
        # We'll use A1's logic by reading the messages and executing code
        def execute_from_a1(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)

                # Import A1's execution utilities
                from biomni.utils import run_with_timeout, run_r_code, run_bash_script
                from biomni.tool.support_tools import run_python_repl, get_captured_plots
                
                timeout = self.timeout_seconds
                
                # Check language and execute
                if code.strip().startswith("#!R"):
                    r_code = re.sub(r"^#!R", "", code, count=1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                elif code.strip().startswith("#!BASH"):
                    bash_script = re.sub(r"^#!BASH", "", code, count=1).strip()
                    result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                else:
                    self._clear_execution_plots()
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                if len(result) > 10000:
                    result = "Output truncated. First 10K chars:\n" + result[:10000]

                # Store execution results
                if not hasattr(self, "_execution_results"):
                    self._execution_results = []
                
                try:
                    plots = get_captured_plots().copy()
                except:
                    plots = []
                
                self._execution_results.append({
                    "triggering_message": last_message,
                    "images": plots,
                    "timestamp": datetime.now().isoformat()
                })

                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state
        
        return execute_from_a1

    # =========================================================================
    #  OVERRIDE: go (Reset STELLA state per run + lower recursion limit)
    # =========================================================================
    def go(self, prompt):
        """Override A1's go to reset STELLA state and add safety limits."""
        self.critic_count = 0
        self.user_task = prompt
        self.stella_mode_activated = False
        self._action_history = []

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(prompt)
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        # Reduced from 500 to 100 to prevent infinite loops
        config = {"recursion_limit": 100, "configurable": {"thread_id": datetime.now().isoformat()}}
        self.log = []

        final_state = None
        try:
            for s in self.app.stream(inputs, stream_mode="values", config=config):
                message = s["messages"][-1]
                from biomni.utils import pretty_print
                out = pretty_print(message)
                self.log.append(out)
                final_state = s
        except Exception as e:
            print(f"\n⚠️ Exception during execution: {e}")
            error_msg = f"<solution>Execution error: {str(e)[:200]}. Unable to complete task.</solution>"
            self.log.append(error_msg)
            if final_state is None:
                final_state = inputs
            return self.log, error_msg

        self._conversation_state = final_state
        return self.log, message.content
