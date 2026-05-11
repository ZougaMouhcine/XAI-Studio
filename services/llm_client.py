"""
XAI Studio — Unified LLM Client (Groq + Gemini)
==================================================
Lightweight HTTP client for Groq and Gemini chat-completion APIs
with native tool-calling support.  Uses only stdlib (urllib) so
no extra pip dependencies are needed.
"""

import json
import urllib.request
import urllib.error
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Groq Provider ────────────────────────────────────────────────────

class GroqProvider:
    """Groq cloud chat-completions (OpenAI-compatible endpoint)."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.name = "Groq"

    # ── public ───────────────────────────────────────────────────────
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> dict:
        """Send a chat-completion request and return the parsed response.

        Returns
        -------
        dict with keys:
            content   : str | None        — text reply (if any)
            tool_calls: list[dict] | None — requested tool invocations
            raw       : dict              — full API response
        """
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.BASE_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "XAI-Studio/1.0",
            },
            method="POST",
        )

        try:
            with _urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode() if exc.fp else ""
            logger.error("Groq API HTTP %s: %s", exc.code, body_text)
            if exc.code == 429:
                raise RateLimitError("Groq rate limit reached. Please wait a moment.") from exc
            raise LLMError(f"Groq API error ({exc.code}): {body_text}") from exc
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc

        choice = raw.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return {
            "content": msg.get("content"),
            "tool_calls": self._normalize_tool_calls(msg.get("tool_calls")),
            "raw": raw,
        }

    def test_connection(self) -> bool:
        """Quick connectivity test."""
        try:
            self.chat([{"role": "user", "content": "ping"}])
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Fetch available model IDs from Groq."""
        return groq_list_models(self.api_key)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _normalize_tool_calls(tool_calls) -> list[dict] | None:
        if not tool_calls:
            return None
        result = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            result.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            })
        return result


# ── Gemini Provider ──────────────────────────────────────────────────

class GeminiProvider:
    """Google Gemini REST API with function-calling support."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.name = "Gemini"

    # ── public ───────────────────────────────────────────────────────
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> dict:
        """Send a generateContent request and return normalized output."""
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        # Convert OpenAI-style messages → Gemini format
        contents = self._to_gemini_contents(messages)
        body: dict = {"contents": contents}

        if tools:
            body["tools"] = [{"function_declarations": self._to_gemini_tools(tools)}]
            if tool_choice == "none":
                body["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
            elif tool_choice == "required":
                body["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            else:
                body["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "XAI-Studio/1.0",
            },
            method="POST",
        )

        try:
            with _urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode() if exc.fp else ""
            logger.error("Gemini API HTTP %s: %s", exc.code, body_text)
            if exc.code == 429:
                raise RateLimitError("Gemini rate limit reached. Please wait.") from exc
            raise LLMError(f"Gemini API error ({exc.code}): {body_text}") from exc
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        # Parse Gemini response
        candidates = raw.get("candidates", [{}])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []

        content = None
        tool_calls = None
        for part in parts:
            if "text" in part:
                content = (content or "") + part["text"]
            if "functionCall" in part:
                fc = part["functionCall"]
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": f"call_{fc['name']}",
                    "name": fc["name"],
                    "arguments": fc.get("args", {}),
                })

        return {"content": content, "tool_calls": tool_calls, "raw": raw}

    def test_connection(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}])
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Fetch available model IDs from Gemini."""
        return gemini_list_models(self.api_key)

    # ── format converters ────────────────────────────────────────────
    @staticmethod
    def _to_gemini_contents(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-style messages to Gemini contents format."""
        contents = []
        system_text = ""

        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_text += msg.get("content", "") + "\n"
                continue
            elif role == "assistant":
                gemini_role = "model"
            elif role == "tool":
                # Tool result → send as functionResponse
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.get("name", msg.get("tool_call_id", "tool")),
                            "response": {"result": msg.get("content", "")},
                        }
                    }],
                })
                continue
            else:
                gemini_role = "user"

            text = msg.get("content", "")
            if role == "user" and system_text and not contents:
                text = f"[System Instructions: {system_text.strip()}]\n\n{text}"
                system_text = ""

            # Handle assistant messages with tool_calls
            if role == "assistant" and msg.get("tool_calls"):
                parts = []
                if text:
                    parts.append({"text": text})
                for tc in msg["tool_calls"]:
                    fn = tc if "name" in tc else tc.get("function", {})
                    args_raw = fn.get("arguments", {})
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = args_raw

                    parts.append({
                        "functionCall": {
                            "name": fn.get("name", ""),
                            "args": args,
                        }
                    })
                contents.append({"role": "model", "parts": parts})
                continue

            if text:
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        # Prepend system text if there were no user messages yet
        if system_text and contents:
            first = contents[0]
            if first["role"] == "user" and first["parts"] and "text" in first["parts"][0]:
                first["parts"][0]["text"] = (
                    f"[System Instructions: {system_text.strip()}]\n\n"
                    + first["parts"][0]["text"]
                )

        return contents

    @staticmethod
    def _to_gemini_tools(openai_tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool schema → Gemini function declarations."""
        declarations = []
        for tool in openai_tools:
            fn = tool.get("function", {})
            decl = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
            }
            params = fn.get("parameters")
            if params:
                decl["parameters"] = params
            declarations.append(decl)
        return declarations


# ── Exceptions ───────────────────────────────────────────────────────

class LLMError(Exception):
    """General LLM API error."""


class RateLimitError(LLMError):
    """Rate limit exceeded."""


# ── Standalone helpers (usable without creating a full provider) ─────

def _get_ssl_context():
    """Create an SSL context that works on Windows.

    urllib on Windows often fails with certificate errors because
    the system cert store isn't always picked up.  We try:
      1. Default context (works on most systems).
      2. certifi bundle if installed.
      3. Unverified context as last resort.
    """
    import ssl
    try:
        ctx = ssl.create_default_context()
        return ctx
    except Exception:
        pass
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except Exception:
        pass
    # Last resort: skip verification (still encrypted, just no cert check)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _urlopen(req, timeout=15):
    """urllib.request.urlopen wrapper with SSL fallback."""
    import ssl
    # Try default first
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except ssl.SSLError:
        pass
    except urllib.error.URLError as e:
        if "SSL" in str(e) or "CERTIFICATE" in str(e).upper():
            pass
        else:
            raise
    # Retry with permissive SSL
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def groq_test_key(api_key: str) -> tuple[bool, str]:
    """Test a Groq API key."""
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "XAI-Studio/1.0",
            },
            method="GET",
        )
        with _urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid key"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Invalid key (Unauthorized)"
        return False, f"HTTP error {exc.code}"
    except Exception as exc:
        logger.warning("Groq key test failed: %s", exc)
        return False, f"Connection error: {type(exc).__name__}"


# Well-known Groq models (fallback when /models endpoint is restricted)
_GROQ_FALLBACK_MODELS = [
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]


def groq_list_models(api_key: str) -> list[str]:
    """Fetch available model IDs from Groq API, with fallback."""
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "XAI-Studio/1.0",
            },
            method="GET",
        )
        with _urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        models = [m["id"] for m in data.get("data", []) if m.get("id")]
        return sorted(models) if models else _GROQ_FALLBACK_MODELS
    except Exception as exc:
        logger.warning("Failed to list Groq models (using fallback): %s", exc)
        return _GROQ_FALLBACK_MODELS


def gemini_test_key(api_key: str) -> tuple[bool, str]:
    """Test a Gemini API key. Returns (success, message)."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=1"
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "XAI-Studio/1.0"},
            method="GET",
        )
        with _urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid key"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 400 or exc.code == 403:
            return False, "Invalid key"
        return False, f"HTTP error {exc.code}"
    except Exception as exc:
        logger.warning("Gemini key test failed: %s", exc)
        return False, f"Connection error: {type(exc).__name__}"


def gemini_list_models(api_key: str) -> list[str]:
    """Fetch available model IDs from Gemini API."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "XAI-Studio/1.0"},
            method="GET",
        )
        with _urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            model_id = name.replace("models/", "") if name.startswith("models/") else name
            if model_id and "generateContent" in str(m.get("supportedGenerationMethods", [])):
                models.append(model_id)
        return sorted(models)
    except Exception as exc:
        logger.warning("Failed to list Gemini models: %s", exc)
        return []


# ── Factory ──────────────────────────────────────────────────────────

def create_provider(groq_key: str = "", gemini_key: str = "",
                    groq_model: str = "llama-3.3-70b-versatile",
                    gemini_model: str = "gemini-2.5-flash"):
    """Create the appropriate LLM provider based on available keys.

    Priority: Groq first (better tool-calling with Llama), then Gemini.
    """
    if groq_key:
        logger.info("Using Groq provider (model: %s)", groq_model)
        return GroqProvider(groq_key, groq_model)
    if gemini_key:
        logger.info("Using Gemini provider (model: %s)", gemini_model)
        return GeminiProvider(gemini_key, gemini_model)
    return None


def create_all_providers(groq_key: str = "", gemini_key: str = "",
                         groq_model: str = "llama-3.3-70b-versatile",
                         gemini_model: str = "gemini-2.5-flash") -> dict:
    """Create all available providers and return them keyed by name.

    Returns e.g. {"Groq": GroqProvider(...), "Gemini": GeminiProvider(...)}.
    """
    providers = {}
    if groq_key:
        providers["Groq"] = GroqProvider(groq_key, groq_model)
    if gemini_key:
        providers["Gemini"] = GeminiProvider(gemini_key, gemini_model)
    return providers
