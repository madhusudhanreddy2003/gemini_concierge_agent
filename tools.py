# tools.py

import os
import json
import datetime as dt
import logging

# Try to import requests (for web_search)
try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("jarvis_agent.tools")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
REMINDERS_FILE = os.path.join(BASE_DIR, "reminders.json")


def _ensure_json_file(path, default):
    """Create a JSON file with default content if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        logger.info("Created JSON file at %s", path)


def _load_json(path, default):
    _ensure_json_file(path, default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("JSON decode error for %s, resetting to default", path)
        return default
    except Exception as e:
        logger.error("Error loading JSON from %s: %s", path, e)
        return default


def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved JSON to %s", path)
    except Exception as e:
        logger.error("Error saving JSON to %s: %s", path, e)


# ========== TOOL 1: SIMPLE WEB SEARCH (DuckDuckGo HTML) ==========

def web_search(query: str) -> str:
    """
    Do a very simple web search using DuckDuckGo HTML page.
    Returns top few result titles + URLs as text.
    NOTE: This is a lightweight example, not a full search API.
    """
    logger.info("Tool web_search called with query=%r", query)

    if requests is None:
        msg = (
            "Web search is not available because the 'requests' package is missing. "
            "Install it with: pip install requests"
        )
        logger.warning(msg)
        return msg

    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
        )
    except Exception as e:
        logger.error("Web search request failed: %s", e)
        return f"Web search failed: {e}"

    if resp.status_code != 200:
        logger.error("Web search status code: %s", resp.status_code)
        return f"Web search failed with status code {resp.status_code}"

    text = resp.text
    results = []
    marker = 'class="result__a"'
    start = 0

    while len(results) < 5:
        idx = text.find(marker, start)
        if idx == -1:
            break

        href_start = text.rfind('href="', 0, idx)
        if href_start == -1:
            break
        href_start += len('href="')
        href_end = text.find('"', href_start)
        url = text[href_start:href_end]

        title_start = text.find(">", idx) + 1
        title_end = text.find("</a>", title_start)
        title = text[title_start:title_end]
        title = title.replace("&amp;", "&").replace("&quot;", '"')

        results.append(f"{len(results) + 1}. {title}\n   {url}")
        start = title_end

    if not results:
        msg = f"No results found for: {query}"
        logger.info(msg)
        return msg

    final = "Top search results:\n\n" + "\n\n".join(results)
    logger.info("web_search returning %d results", len(results))
    return final


# ========== TOOL 2: FILE READING ==========

def read_file(path: str) -> str:
    """
    Read a small text file from disk.
    Security: only allow files inside the project folder and limit size.
    """
    logger.info("Tool read_file called with path=%r", path)

    full_path = os.path.abspath(os.path.join(BASE_DIR, path))

    if not full_path.startswith(BASE_DIR):
        msg = "Access denied: you can only read files inside the project folder."
        logger.warning("read_file blocked: %s", msg)
        return msg

    if not os.path.exists(full_path):
        msg = f"File not found: {path}"
        logger.warning(msg)
        return msg

    if os.path.getsize(full_path) > 200 * 1024:
        msg = "File is too large to read (limit: 200KB)."
        logger.warning(msg)
        return msg

    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        logger.error("Error reading file %s: %s", full_path, e)
        return f"Error reading file: {e}"

    if len(content) > 4000:
        content = content[:4000] + "\n\n[Content truncated...]"

    return f"Content of {path}:\n\n{content}"


# ========== TOOL 3: LONG-TERM MEMORY ==========

def remember_info(note: str) -> str:
    """
    Save a note to memory.json for long-term recall.
    """
    logger.info("Tool remember_info called with note=%r", note)
    data = _load_json(MEMORY_FILE, default=[])
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    data.append({"timestamp": timestamp, "note": note})
    _save_json(MEMORY_FILE, data)
    msg = f"I've saved this to memory at {timestamp}."
    logger.info("remember_info saved note at %s", timestamp)
    return msg


def recall_memory() -> str:
    """
    Return all saved notes from memory.json in a readable format.
    """
    logger.info("Tool recall_memory called")
    data = _load_json(MEMORY_FILE, default=[])
    if not data:
        msg = "I don't have any saved memory yet."
        logger.info(msg)
        return msg

    lines = []
    for item in data[-20:]:  # last 20 notes
        lines.append(f"- [{item['timestamp']}] {item['note']}")
    result = "Here are the memories I have:\n\n" + "\n".join(lines)
    logger.info("recall_memory returning %d notes", len(lines))
    return result


# ========== TOOL 4: REMINDERS (STORED LOCALLY) ==========

def set_reminder(message: str, minutes_from_now: int) -> str:
    """
    Create a reminder that will become 'due' after given minutes.
    (User should later ask: 'check my reminders')
    """
    logger.info(
        "Tool set_reminder called with message=%r, minutes_from_now=%r",
        message,
        minutes_from_now,
    )

    reminders = _load_json(REMINDERS_FILE, default=[])
    now = dt.datetime.now()
    due_time = now + dt.timedelta(minutes=int(minutes_from_now))

    reminder = {
        "message": message,
        "created_at": now.isoformat(timespec="seconds"),
        "due_at": due_time.isoformat(timespec="seconds"),
        "delivered": False,
    }
    reminders.append(reminder)
    _save_json(REMINDERS_FILE, reminders)

    msg = (
        f"Reminder set: '{message}' in about {minutes_from_now} minute(s), "
        f"around {due_time.strftime('%Y-%m-%d %H:%M:%S')}."
    )
    logger.info("set_reminder created reminder due at %s", reminder["due_at"])
    return msg


def check_reminders() -> str:
    """
    Check which reminders are due and mark them as delivered.
    Returns text summary of due reminders.
    """
    logger.info("Tool check_reminders called")
    reminders = _load_json(REMINDERS_FILE, default=[])
    now = dt.datetime.now()

    due = []
    for r in reminders:
        if not r.get("delivered", False):
            try:
                due_time = dt.datetime.fromisoformat(r["due_at"])
            except Exception:
                continue
            if due_time <= now:
                due.append(r)
                r["delivered"] = True

    _save_json(REMINDERS_FILE, reminders)

    if not due:
        msg = "You don't have any reminders due right now."
        logger.info("check_reminders: no due reminders")
        return msg

    lines = ["Here are your due reminders:"]
    for r in due:
        lines.append(f"- {r['message']} (due at {r['due_at']})")
    logger.info("check_reminders: %d reminders due", len(due))
    return "\n".join(lines)


# ========== TOOL REGISTRY ==========

TOOLS = {
    "web_search": web_search,
    "read_file": read_file,
    "remember_info": remember_info,
    "recall_memory": recall_memory,
    "set_reminder": set_reminder,
    "check_reminders": check_reminders,
}
