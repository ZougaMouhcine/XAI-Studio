"""
XAI Studio — Agent Service (Orchestrator)
==========================================
Manages the agentic conversation loop with Ask / Agent / Plan modes.
Dispatches tool calls to PipelineService via the ToolExecutor.
"""

import json
import threading
from services.llm_client import create_all_providers, LLMError, RateLimitError
from services.agent_tools import TOOL_SCHEMAS, ToolExecutor
from services.pipeline_service import PipelineService
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum tool-call iterations per turn to prevent infinite loops.
MAX_TOOL_ITERATIONS = 8


# ── System Prompts ───────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = (
    "You are the XAI Studio AI Assistant — an expert in machine learning, "
    "data science, and explainable AI (XAI). You are embedded inside a desktop "
    "application called XAI Studio that helps users load data, preprocess it, "
    "train ML models, evaluate them, and explain predictions.\n\n"
    "IMPORTANT: Always respond in the same language the user writes in. "
    "If the user writes in French, respond in French. If in English, respond in English.\n\n"
    "Be concise, helpful, and professional. Use markdown formatting when appropriate "
    "(bullet points, bold, code blocks). When discussing models or metrics, "
    "be specific and precise."
)

SYSTEM_PROMPTS = {
    "ask": (
        SYSTEM_PROMPT_BASE + "\n\n"
        "MODE: ASK\n"
        "You are in Ask mode. Answer the user's questions about machine learning, "
        "data science, statistics, XAI techniques, and how to use the application. "
        "Do NOT call any tools. Provide informative, educational answers."
    ),
    "agent": (
        SYSTEM_PROMPT_BASE + "\n\n"
        "MODE: AGENT\n"
        "You are in Agent mode. You can interact with the application by calling tools. "
        "Use tools to help the user accomplish their goals: load data, set targets, "
        "preprocess, train models, evaluate, navigate between views, etc.\n\n"
        "Guidelines:\n"
        "- First check the pipeline status to understand the current state before taking action.\n"
        "- Explain what you're doing before and after calling tools.\n"
        "- If a step fails, explain the error and suggest a fix.\n"
        "- You can chain multiple tool calls to accomplish complex tasks.\n"
        "- After completing actions, summarize what was done."
    ),
    "plan": (
        SYSTEM_PROMPT_BASE + "\n\n"
        "MODE: PLAN\n"
        "You are in Plan mode. Create a detailed, step-by-step plan for the user's "
        "ML workflow. Do NOT call any tools — just plan.\n\n"
        "Structure your plan with:\n"
        "1. **Objective** — What the user wants to achieve\n"
        "2. **Steps** — Numbered steps with explanations\n"
        "3. **Recommendations** — Best practices and tips\n"
        "4. **Expected Outcome** — What the user should expect\n\n"
        "Tailor the plan based on what you know about the application's capabilities."
    ),
}


# ── Agent Service ────────────────────────────────────────────────────

class AgentService:
    """Orchestrates the AI agent conversation with tool-calling support."""

    def __init__(self):
        self.provider = None
        self._providers: dict = {}  # {"Groq": GroqProvider, "Gemini": GeminiProvider}
        self.pipeline = PipelineService()
        self.tool_executor: ToolExecutor | None = None
        self.conversation: list[dict] = []
        self._mode = "ask"
        self._is_busy = False

    # ── configuration ────────────────────────────────────────────────

    def configure(self, groq_key: str = "", gemini_key: str = "",
                  groq_model: str = "llama-3.3-70b-versatile",
                  gemini_model: str = "gemini-2.5-flash",
                  navigate_fn=None,
                  preferred_provider: str = ""):
        """Configure all available LLM providers and select the active one."""
        self._providers = create_all_providers(groq_key, gemini_key, groq_model, gemini_model)
        self.tool_executor = ToolExecutor(self.pipeline, navigate_fn)

        # Pick active provider: honor preference, else Groq > Gemini
        if preferred_provider and preferred_provider in self._providers:
            self.provider = self._providers[preferred_provider]
        elif self._providers:
            self.provider = next(iter(self._providers.values()))
        else:
            self.provider = None

        if self.provider:
            logger.info("Agent configured with %s provider", self.provider.name)
        else:
            logger.warning("No LLM API key configured — agent disabled")

    @property
    def is_configured(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        return self.provider.name if self.provider else "None"

    @property
    def available_providers(self) -> list[str]:
        """Return names of all configured providers."""
        return list(self._providers.keys())

    @property
    def has_multiple_providers(self) -> bool:
        return len(self._providers) > 1

    def set_provider(self, name: str) -> bool:
        """Switch the active provider by name. Returns True if successful."""
        if name in self._providers:
            self.provider = self._providers[name]
            logger.info("Switched active provider to: %s", name)
            return True
        return False

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value in SYSTEM_PROMPTS:
            self._mode = value
            logger.info("Agent mode set to: %s", value)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    # ── conversation management ──────────────────────────────────────

    def clear_conversation(self):
        """Reset the conversation history."""
        self.conversation = []
        logger.info("Agent conversation cleared")

    # ── main interface ───────────────────────────────────────────────

    def send_message(self, user_message: str,
                     on_token=None,
                     on_tool_call=None,
                     on_done=None,
                     on_error=None):
        """Send a message asynchronously (runs in a background thread).

        Callbacks (all called on the calling thread context):
            on_token(text: str)          — partial/full text from LLM
            on_tool_call(name, result)   — when a tool is called
            on_done(full_response: str)  — when the turn is complete
            on_error(error_msg: str)     — on any error
        """
        if not self.is_configured:
            if on_error:
                on_error("No API key configured. Open settings (⚙) to add your Groq or Gemini key.")
            return

        if self._is_busy:
            if on_error:
                on_error("Agent is busy. Please wait for the current request to finish.")
            return

        thread = threading.Thread(
            target=self._run_turn,
            args=(user_message, on_token, on_tool_call, on_done, on_error),
            daemon=True,
        )
        thread.start()

    def _run_turn(self, user_message, on_token, on_tool_call, on_done, on_error):
        """Execute a complete agent turn (potentially with multiple tool calls)."""
        self._is_busy = True
        try:
            # Add user message to conversation
            self.conversation.append({"role": "user", "content": user_message})

            # Build messages with system prompt
            messages = [{"role": "system", "content": SYSTEM_PROMPTS[self._mode]}]
            messages.extend(self.conversation)

            # Determine if tools are available in this mode
            use_tools = self._mode == "agent"
            tools = TOOL_SCHEMAS if use_tools else None
            tool_choice = "auto" if use_tools else "none"

            iterations = 0
            while iterations < MAX_TOOL_ITERATIONS:
                iterations += 1

                response = self.provider.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                content = response.get("content")
                tool_calls = response.get("tool_calls")

                if tool_calls and use_tools and self.tool_executor:
                    # Model wants to call tools
                    # Build an assistant message with tool_calls for the conversation
                    assistant_msg = {"role": "assistant", "content": content or ""}
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ]
                    messages.append(assistant_msg)

                    # Execute each tool call
                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_args = tc["arguments"]
                        logger.info("Agent calling tool: %s(%s)", tool_name, tool_args)

                        result = self.tool_executor.execute(tool_name, tool_args)

                        if on_tool_call:
                            on_tool_call(tool_name, result)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tool_name,
                            "content": result,
                        })

                    # Continue the loop to get the model's final response
                    continue

                # No tool calls — we have the final text response
                final_text = content or ""
                self.conversation.append({"role": "assistant", "content": final_text})

                if on_token:
                    on_token(final_text)
                if on_done:
                    on_done(final_text)
                return

            # If we hit max iterations, return what we have
            fallback = "I've performed several actions. Let me know if you need anything else."
            self.conversation.append({"role": "assistant", "content": fallback})
            if on_token:
                on_token(fallback)
            if on_done:
                on_done(fallback)

        except RateLimitError as exc:
            err = str(exc)
            if on_error:
                on_error(err)
        except LLMError as exc:
            err = str(exc)
            if on_error:
                on_error(err)
        except Exception as exc:
            err = f"Unexpected error: {exc}"
            logger.error("Agent error: %s", exc, exc_info=True)
            if on_error:
                on_error(err)
        finally:
            self._is_busy = False
