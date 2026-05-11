"""
XAI Studio — Agent Service (Orchestrator)
==========================================
Manages the agentic conversation loop with Ask / Agent / Plan modes.
Dispatches tool calls to PipelineService via the ToolExecutor.
"""

import json
import os
import time
import uuid
import threading
from datetime import datetime

from config.settings import BASE_DIR
from services.llm_client import create_all_providers, LLMError, RateLimitError
from services.agent_tools import TOOL_SCHEMAS, ToolExecutor
from services.pipeline_service import PipelineService
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize chats directory
CHATS_DIR = os.path.join(BASE_DIR, "data", "chats")
os.makedirs(CHATS_DIR, exist_ok=True)

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

        "═══ CRITICAL DIRECTIVE — VISUAL PIPELINE CONSTRUCTION ═══\n"
        "When a user asks you to create, suggest, recommend, or build a preprocessing "
        "pipeline, you MUST physically populate the Dynamic Pipeline table in the UI. "
        "NEVER just output a text list of steps in the chat.\n\n"

        "MANDATORY WORKFLOW:\n"
        "1. Call `get_data_summary` to understand the dataset (types, missing values, etc.).\n"
        "2. Call `get_preprocessing_catalog` to get valid categories and methods.\n"
        "3. Call `get_columns` to get the available column names.\n"
        "4. Call `get_preprocessing_issues` to detect data quality problems.\n"
        "5. For EACH preprocessing step you want to add, call `add_preprocessing_step` "
        "   with the correct category, method, and target columns. This tool visually "
        "   sets the Category dropdown, Method dropdown, selects columns, and clicks "
        "   'Add to Pipeline' in the UI — the user watches each step appear.\n"
        "6. After all steps are added, summarize the pipeline you built.\n\n"

        "RULES:\n"
        "- Do NOT call `run_preprocessing` to silently apply transforms. Use `add_preprocessing_step`.\n"
        "- Call `add_preprocessing_step` once per step — do NOT batch multiple steps.\n"
        "- Use only categories/methods from the catalog. Use only columns from `get_columns`.\n"
        "- Order steps logically: cleaning → outlier handling → encoding → scaling → selection.\n"
        "- If a step fails, explain the error and suggest an alternative.\n\n"

        "GENERAL GUIDELINES:\n"
        "- First check the pipeline status to understand the current state before taking action.\n"
        "- Explain what you're doing before and after calling tools.\n"
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
        self.session_id = str(uuid.uuid4())

    # ── configuration ────────────────────────────────────────────────

    def configure(self, groq_key: str = "", gemini_key: str = "",
                  groq_model: str = "llama-3.3-70b-versatile",
                  gemini_model: str = "gemini-2.5-flash",
                  navigate_fn=None,
                  preferred_provider: str = "",
                  get_preprocessing_view=None):
        """Configure all available LLM providers and select the active one."""
        self._providers = create_all_providers(groq_key, gemini_key, groq_model, gemini_model)
        self.tool_executor = ToolExecutor(
            self.pipeline, navigate_fn, get_preprocessing_view=get_preprocessing_view,
        )

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
        self.session_id = str(uuid.uuid4())
        logger.info("Agent conversation cleared")

    # ── history management ───────────────────────────────────────────

    def save_conversation(self):
        """Save the current conversation to disk."""
        if not self.conversation:
            return
        
        # Determine title from first user message
        title = "New Chat"
        for msg in self.conversation:
            if msg.get("role") == "user":
                title = msg.get("content", "")[:30] + ("..." if len(msg.get("content", "")) > 30 else "")
                break

        filepath = os.path.join(CHATS_DIR, f"{self.session_id}.json")
        data = {
            "session_id": self.session_id,
            "title": title,
            "updated_at": datetime.now().isoformat(),
            "messages": self.conversation
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    @staticmethod
    def list_conversations() -> list[dict]:
        """List all saved conversations, sorted by most recent first."""
        chats = []
        if not os.path.exists(CHATS_DIR):
            return chats
            
        for filename in os.listdir(CHATS_DIR):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(CHATS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chats.append({
                        "session_id": data.get("session_id", filename[:-5]),
                        "title": data.get("title", "Unknown"),
                        "updated_at": data.get("updated_at", "")
                    })
            except Exception as e:
                logger.error(f"Failed to read chat {filename}: {e}")
                
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats

    def load_conversation(self, session_id: str) -> list[dict]:
        """Load a conversation from disk."""
        filepath = os.path.join(CHATS_DIR, f"{session_id}.json")
        if not os.path.exists(filepath):
            logger.error(f"Conversation {session_id} not found.")
            return []
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.session_id = session_id
                self.conversation = data.get("messages", [])
                logger.info(f"Loaded conversation {session_id}")
                return self.conversation
        except Exception as e:
            logger.error(f"Failed to load conversation {session_id}: {e}")
            return []

    @staticmethod
    def delete_conversation(session_id: str) -> bool:
        """Delete a conversation from disk."""
        filepath = os.path.join(CHATS_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Deleted conversation {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete conversation {session_id}: {e}")
        return False

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
        """Execute a complete agent turn (potentially with multiple tool calls).

        Rate-limit errors are retried automatically with exponential back-off
        (up to ``MAX_RATE_LIMIT_RETRIES`` times).  If a secondary provider is
        available, it will be tried as a last-resort fallback.
        """
        MAX_RATE_LIMIT_RETRIES = 3
        BASE_BACKOFF_SECS = 5           # 5 → 10 → 20 s

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

            active_provider = self.provider
            iterations = 0
            while iterations < MAX_TOOL_ITERATIONS:
                iterations += 1

                # ── call LLM with retry on rate-limit ────────────────
                response = None
                for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
                    try:
                        response = active_provider.chat(
                            messages=messages,
                            tools=tools,
                            tool_choice=tool_choice,
                        )
                        break  # success
                    except RateLimitError:
                        if attempt < MAX_RATE_LIMIT_RETRIES:
                            wait = BASE_BACKOFF_SECS * (2 ** (attempt - 1))
                            logger.warning(
                                "Rate-limited by %s — retrying in %ds (attempt %d/%d)",
                                active_provider.name, wait, attempt, MAX_RATE_LIMIT_RETRIES,
                            )
                            if on_tool_call:
                                on_tool_call(
                                    "rate_limit",
                                    f"⏳ {active_provider.name} rate-limited — retrying in {wait}s…",
                                )
                            time.sleep(wait)
                        else:
                            # Last attempt failed — try fallback provider
                            fallback = self._get_fallback_provider(active_provider)
                            if fallback is not None:
                                logger.info(
                                    "Switching to fallback provider: %s", fallback.name,
                                )
                                if on_tool_call:
                                    on_tool_call(
                                        "provider_switch",
                                        f"🔄 Switching to {fallback.name} (rate limit on {active_provider.name})",
                                    )
                                active_provider = fallback
                                try:
                                    response = active_provider.chat(
                                        messages=messages,
                                        tools=tools,
                                        tool_choice=tool_choice,
                                    )
                                    break
                                except RateLimitError:
                                    raise RateLimitError(
                                        f"Rate limit reached on both {self.provider.name} "
                                        f"and {fallback.name}. Please wait a minute and try again."
                                    )
                            else:
                                raise RateLimitError(
                                    f"{active_provider.name} rate limit reached after "
                                    f"{MAX_RATE_LIMIT_RETRIES} retries. Please wait a minute and try again."
                                )

                if response is None:
                    raise LLMError("Failed to get a response from the LLM.")

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

                    # We save mid-turn in case the user closes the app while tool chaining
                    self.save_conversation()

                    # Continue the loop to get the model's final response
                    continue

                # No tool calls — we have the final text response
                final_text = content or ""
                self.conversation.append({"role": "assistant", "content": final_text})
                self.save_conversation()

                if on_token:
                    on_token(final_text)
                if on_done:
                    on_done(final_text)
                return

            # If we hit max iterations, return what we have
            fallback = "I've performed several actions. Let me know if you need anything else."
            self.conversation.append({"role": "assistant", "content": fallback})
            self.save_conversation()
            
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

    def _get_fallback_provider(self, current_provider):
        """Return an alternative provider if one is configured, else None."""
        for name, provider in self._providers.items():
            if provider is not current_provider:
                return provider
        return None
