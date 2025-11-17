# agent.py

import os
import json
import logging

from dotenv import load_dotenv
from google import genai
from huggingface_hub import InferenceClient

from tools import TOOLS

# ========== PATHS & LOGGING SETUP ==========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "agent.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("jarvis_agent")

# ========== ENV & BACKEND SETUP ==========

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
# backends: "mock" (local), "hf" (Hugging Face), "gemini" (Gemini)
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock").lower()

logger.info("Selected backend: %s", LLM_BACKEND)

# Gemini client (optional, used only if LLM_BACKEND == "gemini")
gemini_client = None
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-1219"

if LLM_BACKEND == "gemini":
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file")
    logger.info("Initializing Gemini client")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Hugging Face client (optional, used only if LLM_BACKEND == "hf")
hf_client = None
HF_MODEL_NAME = "gpt2"  # simple, widely supported text-generation model

if LLM_BACKEND == "hf":
    if not HF_API_KEY:
        raise ValueError("❌ HF_API_KEY not found in .env file")
    logger.info("Initializing Hugging Face client with model=%s", HF_MODEL_NAME)
    hf_client = InferenceClient(model=HF_MODEL_NAME, token=HF_API_KEY)

# ========== SYSTEM PROMPT ==========

SYSTEM_PROMPT = """
You are Jarvis, an AI concierge agent helping a developer named Madhu.

You are powered by an LLM backend (Gemini or Hugging Face) and you must ALWAYS produce
your primary reasoning output in pure JSON following this schema:

1) To answer the user directly (no tools):
{
  "action": "respond",
  "content": "<your natural language reply to the user>"
}

2) To call a tool:
{
  "action": "tool",
  "name": "<tool_name>",
  "args": { ... }
}

Available tools and their arguments:

1) web_search
   - description: Search the web for recent or dynamic information.
   - args: { "query": "search text" }

2) read_file
   - description: Read a small text file from the local project folder.
   - args: { "path": "relative/path/to/file.txt" }

3) remember_info
   - description: Save an important note to long-term memory.
   - args: { "note": "text to remember" }

4) recall_memory
   - description: Show previously saved notes from long-term memory.
   - args: { }

5) set_reminder
   - description: Create a reminder N minutes from now.
   - args: { "message": "reminder text", "minutes_from_now": 10 }

6) check_reminders
   - description: Check which reminders are due now.
   - args: { }

Rules:
- If the user asks for current or changing information, strongly consider using web_search.
- Use remember_info when the user explicitly says to remember something.
- Use recall_memory when the user wants to recall past notes.
- Use set_reminder/check_reminders for reminder workflows.
- If a tool is not needed, use "action": "respond".

Context Engineering:
- The system may compact older parts of the conversation to keep only the most relevant
  recent turns. When this happens, you still respond normally, assuming a short summary
  of prior context is sufficient.

Return ONLY valid JSON when asked to follow the JSON schema above.
"""


class GeminiAgent:
    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        max_dialogue_chars: int = 8000,
    ):
        self.model_name = model_name
        self.max_dialogue_chars = max_dialogue_chars
        self.dialogue = SYSTEM_PROMPT.strip() + "\n\nConversation so far:\n"
        logger.info(
            "GeminiAgent initialized with model=%s, max_dialogue_chars=%d",
            self.model_name,
            self.max_dialogue_chars,
        )

    # ---------- Context Compaction ----------

    def _compact_context(self):
        """
        Simple context compaction:
        - If dialogue exceeds max_dialogue_chars, keep only the last ~4000 chars,
          with a note that earlier conversation was compacted.
        """
        if len(self.dialogue) <= self.max_dialogue_chars:
            return

        tail = self.dialogue[-4000:]
        self.dialogue = (
            SYSTEM_PROMPT.strip()
            + "\n\n[Older conversation compacted for brevity. "
              "Only the most recent turns are kept here.]\n\n"
            + tail
        )
        logger.info(
            "Context compacted. New dialogue length=%d chars",
            len(self.dialogue),
        )

    # ---------- Real LLM Call Helper (HF + Gemini) ----------

    def _call_model(self, prompt: str) -> str:
        """
        Call the selected LLM backend (Gemini or Hugging Face) and return text.
        Not used when LLM_BACKEND == 'mock'.
        """
        self._compact_context()

        logger.info(
            "Calling LLM backend '%s' with prompt length=%d chars",
            LLM_BACKEND,
            len(prompt),
        )

        # ------- HUGGING FACE BACKEND -------
        if LLM_BACKEND == "hf":
            try:
                response_text = hf_client.text_generation(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.3,
                    stop=None,
                )
                text = (response_text or "").strip()
                logger.info(
                    "HF response length=%d chars (first 120 chars=%r)",
                    len(text),
                    text[:120],
                )
                return text

            except Exception as e:
                logger.error("Error calling Hugging Face model: %s", e)
                # Fallback: JSON "respond" action so the app doesn't crash
                return json.dumps({
                    "action": "respond",
                    "content": (
                        "I couldn't reach the Hugging Face model due to an error.\n\n"
                        f"Technical details: {str(e)}"
                    )
                })

        # ------- GEMINI BACKEND -------
        if LLM_BACKEND == "gemini":
            try:
                response = gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
            except Exception as e:
                logger.error("Error calling Gemini model: %s", e)
                # Fallback: JSON "respond" action
                return json.dumps({
                    "action": "respond",
                    "content": (
                        "I couldn't reach the Gemini API because of a quota or network error.\n\n"
                        f"Technical details: {str(e)}"
                    )
                })

            text = getattr(response, "text", None)
            if not text:
                try:
                    parts = []
                    for cand in response.candidates:
                        for part in cand.content.parts:
                            if getattr(part, "text", None):
                                parts.append(part.text)
                    text = "\n".join(parts)
                except Exception:
                    text = ""

            text = (text or "").strip()
            logger.info(
                "Gemini response length=%d chars (first 120 chars=%r)",
                len(text),
                text[:120],
            )
            return text

        # ------- UNKNOWN BACKEND -------
        logger.error("Unknown LLM_BACKEND=%r", LLM_BACKEND)
        return json.dumps({
            "action": "respond",
            "content": "LLM backend is misconfigured. Please set LLM_BACKEND to 'mock', 'hf' or 'gemini' in .env."
        })

    # ---------- Tool Execution ----------

    def _run_tool(self, name: str, args: dict) -> str:
        """
        Execute a tool from the TOOLS registry and return its result as string.
        """
        logger.info("Running tool '%s' with args=%r", name, args)

        if name not in TOOLS:
            msg = f"Error: tool '{name}' is not implemented."
            logger.warning(msg)
            return msg

        func = TOOLS[name]
        try:
            result = func(**args) if args else func()
        except TypeError as e:
            msg = f"Error calling tool '{name}': {e}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"Tool '{name}' raised an error: {e}"
            logger.error(msg)
            return msg

        result_str = str(result)
        logger.info(
            "Tool '%s' completed. Result length=%d chars",
            name,
            len(result_str),
        )
        return result_str

    # ---------- MOCK BACKEND (No external API needed) ----------

    def _mock_decide_action(self, user_message: str) -> dict:
        """
        Simple rule-based planner that returns JSON actions
        without calling any external LLM.
        This is used when LLM_BACKEND == 'mock'.
        """
        msg = user_message.lower()

        # check reminders FIRST (more specific)
        if "check reminders" in msg or "any reminders" in msg or "show my reminders" in msg:
            return {
                "action": "tool",
                "name": "check_reminders",
                "args": {},
            }

        # remember something
        if "remember" in msg:
            note = user_message.replace("remember that", "").strip()
            if not note:
                note = user_message
            return {
                "action": "tool",
                "name": "remember_info",
                "args": {"note": note},
            }

        # recall memory
        if "recall" in msg or "what do you remember" in msg or "memory" in msg:
            return {
                "action": "tool",
                "name": "recall_memory",
                "args": {},
            }

        # search
        if "search" in msg or "news" in msg:
            return {
                "action": "tool",
                "name": "web_search",
                "args": {"query": user_message},
            }

        # set reminder
        if "remind" in msg or "reminder" in msg or "set a reminder" in msg:
            return {
                "action": "tool",
                "name": "set_reminder",
                "args": {
                    "message": user_message,
                    "minutes_from_now": 5,
                },
            }

        # default response
        return {
            "action": "respond",
            "content": "Hi, I'm Jarvis running in offline mock mode. I received: " + user_message,
        }

    # ---------- Main Chat Loop (Loop Agent) ----------

    def chat(self, user_message: str) -> str:
        """
        One full turn:
        - If backend='mock': use rule-based planner
        - Else: call real LLM (_call_model) to get JSON plan
        - Execute tools if needed
        - Return final natural-language reply
        """
        logger.info("User message: %r", user_message)

        # ---------- MOCK BACKEND PATH (no external calls) ----------
        if LLM_BACKEND == "mock":
            action = self._mock_decide_action(user_message)
            logger.info("Mock action: %r", action)

            if action["action"] == "respond":
                reply_text = action.get("content", "").strip() or "I tried to respond, but my content was empty."
                self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
                return reply_text

            if action["action"] == "tool":
                tool_name = action.get("name")
                args = action.get("args") or {}
                tool_result = self._run_tool(tool_name, args)
                reply_text = f"(Mock mode) I used tool '{tool_name}' and got this result:\n\n{tool_result}"
                self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
                return reply_text

            # fallback
            reply_text = "I produced an unknown action in mock mode."
            self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
            return reply_text

        # ---------- REAL LLM BACKEND PATH (hf / gemini) ----------
        prompt = (
            self.dialogue
            + f"User: {user_message}\n"
            + "Assistant (reply ONLY in JSON as specified above):"
        )

        raw = self._call_model(prompt)

        # Try to parse JSON
        try:
            action = json.loads(raw)
            logger.info("Parsed JSON action: %r", action)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse JSON, returning raw response instead. Raw=%r",
                raw[:200],
            )
            reply_text = raw or "I could not understand my own JSON output."
            self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
            return reply_text

        if not isinstance(action, dict) or "action" not in action:
            logger.warning("JSON did not contain 'action' key. Raw=%r", raw[:200])
            reply_text = raw or "I produced an invalid JSON action."
            self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
            return reply_text

        # ---------- Direct Response Path ----------
        if action["action"] == "respond":
            reply_text = (action.get("content") or "").strip()
            if not reply_text:
                reply_text = "I tried to respond, but my content was empty."
                logger.warning("Empty 'content' in respond action")

            self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
            logger.info("Responding directly to user. Length=%d chars", len(reply_text))
            return reply_text

        # ---------- Tool Call Path ----------
        if action["action"] == "tool":
            tool_name = action.get("name")
            args = action.get("args") or {}
            logger.info("Assistant chose tool '%s' with args=%r", tool_name, args)

            tool_result = self._run_tool(tool_name, args)

            followup_prompt = (
                self.dialogue
                + f"User: {user_message}\n"
                + "Assistant decided to call a tool.\n"
                + f"Tool name: {tool_name}\n"
                + f"Tool args: {json.dumps(args, ensure_ascii=False)}\n"
                + f"Tool result: {tool_result}\n\n"
                + "Now Assistant, give the final, user-friendly answer (no JSON, just text):"
            )

            final_text = self._call_model(followup_prompt)
            final_text = final_text.strip() or "I used a tool but couldn't form a response."

            self.dialogue += f"User: {user_message}\nAssistant: {final_text}\n"
            logger.info(
                "Returning final answer after tool. Length=%d chars",
                len(final_text),
            )
            return final_text

        # ---------- Unknown Action ----------
        logger.warning("Unknown action type: %r", action.get("action"))
        reply_text = raw or "I produced an unknown action type."
        self.dialogue += f"User: {user_message}\nAssistant: {reply_text}\n"
        return reply_text


def main():
    print("🤖 Jarvis++ (Concierge Agent) is ready! Type 'exit' to quit.\n")
    agent = GeminiAgent()

    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Bye buddy! 👋")
            break

        if user_msg.lower().strip() in {"exit", "quit"}:
            print("Jarvis: Bye buddy! 👋")
            break

        try:
            reply = agent.chat(user_msg)
            print(f"Jarvis: {reply}\n")
        except Exception as e:
            logger.exception("Fatal error in main loop: %s", e)
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
