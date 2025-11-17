# Jarvis++ – AI Concierge Agent
###Capstone Project – Agents Intensive Program
Author: Kethari Madhu Sudhan Reddy

##Problem Statement
Developers and knowledge workers often waste significant time on repetitive tasks:
- Searching for information
- Storing notes and personal reminders
- Keeping track of ongoing tasks
- Remembering previous context in long workflows
- Switching between apps (search → notes → reminders)

Existing assistants do not maintain persistent memory, cannot trigger tools, and lack multi-step agentic reasoning.

---

##The challenge:
Can we build an agent that behaves like a personal “Jarvis-style” concierge—capable of remembering, searching, assisting, and executing actions intelligently?
<br>This is important because such an agent can drastically reduce mental load and improve daily productivity.

---

##Why Agents?
Agents are the right solution because:

-  They can plan :
Jarvis++ uses JSON-based action planning to decide whether to respond directly or call a tool.
-  They can use tools :
Tools allow actions (search, remember, remind) that exceed the limitations of pure LLM text generation.
-  They maintain state :
Agents store long-term memory (memory.json) and manage reminders (reminders.json).
-  They operate autonomously :
The agent loops through:

Understanding the user intent

- Selecting a tool
- Executing the tool
- Reasoning over the result
- Producing the final answer
- Traditional chatbots cannot do this.

Agents mimic real productivity workflows
Jarvis++ automates the exact tasks users perform daily:
- Notes → Memory
- Timers → Reminders
- Googling → Search tool
- Task continuation → Context compaction

This makes agents the perfect architecture for this problem.

---

## What Was Created (Architecture Overview)
Jarvis++ is a multi-agent AI Concierge System that includes:

### 1. Planner Agent (LLM-Based or Mock)
Decides the next action using:

- Gemini 2.0
- Hugging Face LLM
- Offline Mock Mode (default due to billing limits)

It outputs structured JSON, such as:
```bash
json
{
  "action": "tool",
  "name": "web_search",
  "args": {"query": "latest AI news"}
}
   ```
### 2. Tool Executor
Executes the action selected by the planner.
Tools include:
- web_search
- remember_info
- recall_memory
- set_reminder
- check_reminders
- read_file

### 3. Long-term Memory Layer
Stored in:
   ```bash
memory.json
   ```

### 4. Reminder Engine
Stored in:
   ```bash
reminders.json
   ```

###  5. Web Search Layer
Chrome HTML scraping → returns summary results.

### 6. Context Manager
Automatically compacts old conversation history → prevents overflow.

### 7. Observability Layer
   ```agent.log ```

### 8. Planner decisions
- Tool calls
- Errors
- Final responses

---

## Offline Mode Explanation
Due to quota and billing limitations, the project defaults to offline mock mode.<br>
This still demonstrates:
- Multi-agent reasoning
- Tool usage
- Memory & reminders
- File operations
- Observability
- Deterministic JSON planner behavior
- Context compaction

The codebase fully supports real LLMs via Gemini and HuggingFace, and runs perfectly with valid API keys.

---

## The Build (Tools & Tech Used):

### 1. Languages & Libraries

   ```bash
- Python 3
- google-genai (Gemini backend)
- huggingface-hub (HF backend)
- requests
- python-dotenv
- reportlab (PDF generation)

   ```
### 2. Tools Implemented
- web_search – HTML-based Chrome search
- remember_info – save notes
- recall_memory – retrieve notes
- set_reminder – schedule reminders
- check_reminders – fetch due reminders
- read_file – read text files

### 3. LLM Backends

- Gemini 2.0 Flash Thinking 
- HuggingFace GPT-2
- Mock Reasoning System (Offline) – default and recommended for submission

### 4. Data Storage
   ```bash
memory.json
   ```
   ```bash
reminders.json
   ```

### 5. Observability
All interactions logged into:
   ```bash
agent.log
   ```

----

## If I Had More Time, This Is What I'd Do: 

### 1. Add a real database
SQLite or MongoDB for scalable memory and reminders.

### 2. Use stronger LLMs
Integrate Gemini 2.0 Pro, Mistral 8x7B, or Llama 3 models.

### 3. Add voice
Speech-to-text + TTS for a fully voice-enabled Jarvis.

### 4. Web or Desktop UI
Create an interactive frontend using React, FastAPI, or Electron.

### 5. Add more tools
- Calendar integration
- WhatsApp / email automation
- Browser automation
- YouTube video uploader for my Tech TrendX channel

### 6. Better JSON enforcement
Improve robustness for noisy LLM output.

## Final Note
Jarvis++ demonstrates a complete agentic system with:
- Multi-agent planning
- Tool execution
- Memory
- Reminders
- Context engineering
- Observability
- Multi-backend LLM support

All while running locally, offline, reliably, and cost-free for reproducible evaluation.

----
### 👨‍💻 Author
Kethari Madhu Sudhan Reddy <br>
Python Developer • ML Enthusiast • Automation Lover<br>
Contact me : maddoxer143@gmail.com

### 📜 License
This project is Open Source.<br>
Feel free to modify, enhance, or reuse it.

