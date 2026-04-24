"""
Autonomous tool-calling agent.

The LLM emits tool calls in the form:
    TOOL: <name> | <args>

The agent runs the tool silently, appends the result as an OBSERVATION, and
re-queries the model. The loop continues until the model produces a plain
answer (no TOOL: line). Only the final answer is shown to the user.
"""

import os
import re
import asyncio
import requests
from config import Config
from tools import bash_tool, pdf_tool, fact_check_tool

DEBUG = os.environ.get("AGENT_DEBUG") == "1"

# Match a line like: TOOL: bash | ls -la  (case-insensitive for small models)
TOOL_PATTERN = re.compile(
    r"^\s*TOOL:\s*(\w+)\s*\|\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

MAX_STEPS = 4

# Words that strongly imply the model should have used a tool.
TOOL_TRIGGERS = (
    # bash-like
    "run ", "nproc", "uname", "git ", "ls ", "find ", "grep ", "cat ",
    "ps ", "df ", "du ", "pwd", "wc ",
    "how many", "list files", "list the", "show me", "what files",
    "check ", "count ", "search for",
    "directory", "folder",
    # fact-check
    "fact check", "fact-check", "is it true", "is this true",
    "verify ", "true or false",
    # pdf
    ".pdf", "pdf file", "read the pdf",
)


def _looks_like_tool_question(question: str) -> bool:
    q = question.lower()
    return any(trigger in q for trigger in TOOL_TRIGGERS)


class ToolAgent:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client

    # ------------- tool dispatch -------------
    def _run_tool(self, name: str, args: str) -> str:
        name = name.lower().strip()
        if name == "bash":
            return bash_tool(args)
        if name == "pdf":
            return pdf_tool(args)
        if name == "fact":
            return fact_check_tool(args, self.ollama_client, Config.DEFAULT_MODEL)
        return f"[error] unknown tool: {name}"

    # ------------- LLM helpers -------------
    def _generate(self, prompt: str, model_name: str) -> str:
        """Non-streaming generate. Stops at 'OBSERVATION:' so the model can't
        hallucinate its own tool results, and at 'User:' so it doesn't
        roleplay the next user turn."""
        resp = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"stop": ["\nOBSERVATION:", "\nUser:"]},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        # Strip leading role-marker echoes like "Assistant:" or "OBSERVATION:".
        for prefix in ("assistant:", "observation:"):
            if text.lower().startswith(prefix):
                text = text.split(":", 1)[1].strip()
                break
        return text

    async def _generate_async(self, prompt: str, model_name: str) -> str:
        return await asyncio.to_thread(self._generate, prompt, model_name)

    # ------------- main loop -------------
    async def run(self, base_prompt: str, model_name: str, user_question: str = "") -> str:
        """Run the agent loop. Returns the final answer string."""
        context = base_prompt
        last_observation = ""
        last_call = None  # (tool_name, tool_args) of previous step
        nudged = False

        for _ in range(MAX_STEPS):
            out = await self._generate_async(context, model_name)

            match = TOOL_PATTERN.search(out)
            if not match:
                # Model didn't call a tool. If the user's question clearly
                # needs one, nudge the model to use it (once).
                if (
                    not nudged
                    and user_question
                    and _looks_like_tool_question(user_question)
                    and not last_observation  # haven't run any tool yet
                ):
                    if DEBUG:
                        print("[agent] no TOOL call, nudging model to use one")
                    nudged = True
                    context = (
                        base_prompt
                        + "\n\nReminder: the user asked about the real system. "
                        "You MUST call a tool. Output a single TOOL: line "
                        "now and nothing else.\nAssistant:"
                    )
                    continue
                # Otherwise this is the final answer.
                return out.strip()

            tool_name = match.group(1)
            tool_args = match.group(2)
            call = (tool_name.lower(), tool_args.strip())

            # If the model repeats the exact same call, force a final answer
            # by appending a strong finalize instruction instead of re-running.
            if call == last_call and last_observation:
                if DEBUG:
                    print("[agent] repeat tool call -> forcing final answer")
                finalize_prompt = (
                    context
                    + "\nDo NOT call any more tools. Using the OBSERVATION "
                    "above, write the final answer for the user in 1-3 "
                    "sentences, including the actual value(s) observed.\n"
                    "Assistant:"
                )
                try:
                    final = await self._generate_async(finalize_prompt, model_name)
                    if final and not TOOL_PATTERN.search(final):
                        return final.strip()
                except Exception:
                    pass
                return last_observation

            if DEBUG:
                print(f"[agent] TOOL: {tool_name} | {tool_args}")
            result = self._run_tool(tool_name, tool_args)
            last_observation = result
            last_call = call
            if DEBUG:
                truncated = result if len(result) < 200 else result[:200] + "..."
                print(f"[agent] OBSERVATION: {truncated}")

            # Feed the tool's output back so the model can finish the answer.
            # Use an explicit instruction instead of just "Assistant:" so the
            # model knows to write a user-facing answer now.
            context = (
                context
                + "\n"
                + out
                + f"\nOBSERVATION: {result}\n"
                + "Now write the final answer to the user, citing the actual "
                "values from the OBSERVATION above. If you need more info, "
                "you may call another TOOL.\nAssistant:"
            )

        # Hit step limit without a plain answer -> surface the observation.
        return last_observation or "(no answer)"
