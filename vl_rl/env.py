"""LLM env wrapper — injects a goal + action-space description into a prompt,
or prepares the goal string for Jev-style typed-question agents (d1-3B).

Supported output_format values
-----------------------------
- "boxed" : free-text reasoning ending in \\boxed{...}       (Ollama / vLLM)
- "json"  : strict JSON {"explanation": ..., "action": ...}  (Ollama / vLLM)
- "d1"    : typed questions generated at call time from the
            action space. Returns (env, goal_string) — NOT a
            text prompt. d1_agent builds the questions.

Notes
-----
For "boxed"/"json", the second return value is a full text prompt that you
feed directly to `ollama_agent` / `vllm_agent`.

For "d1", the second return value is only the *goal string*. `d1_agent`
injects it as the "Goal: ..." context of every question it synthesizes
from `env.action_space`.
"""

from .spaces import space_type, describe_space


# Space families for which d1_agent._questions_for_space() is implemented.
# Keep this in sync with d1_agent.py if you extend it.
_D1_SUPPORTED = {"discrete", "multibinary", "multidiscrete", "continuous", "dict"}


def LLM_env(env, description="", action_description="", output_format="boxed"):
    """Wrap any RL env into an (env, prompt-or-goal) pair.

    Parameters
    ----------
    env : gym.Env | gymnasium.Env
        The environment to wrap.
    description : str
        Natural-language goal, e.g. "Find and mine a diamond".
    action_description : str
        Optional human hints about the action space / strategy.
    output_format : {"boxed", "json", "d1"}
        Determines the second return value (see module docstring).
    """
    if output_format == "d1":
        return _d1_env(env, description, action_description)

    # ── boxed / json ──────────────────────────────────────────
    space = getattr(env, "action_space", None)
    detected = (
        f"# DETECTED ACTION FORMAT\n"
        f"Type: {space_type(space)}\n"
        f"Expected output: {describe_space(space)}"
        if space is not None else ""
    )

    if output_format == "json":
        format_instruction = (
            "Output EXACTLY this JSON format, starting with {:\n"
            '{"explanation": "describe what you see + reasoning", '
            '"action": <action in the expected format>}\n'
            "Do NOT output anything before or after the JSON."
        )
    elif output_format == "boxed":
        format_instruction = (
            "Think step by step in plain text about:\n"
            "1. What you observe in the current state (must be **PRECISE**)\n"
            "2. Relevant objects, obstacles, and their positions\n"
            "3. Your current position and state\n"
            "4. Best action to take given your goal\n\n"
            "IMPORTANT:\n"
            "- End your response with your action in \\boxed{...} "
            "using EXACTLY the expected format above.\n"
            "- Do not **hallucinate** objects that are not in the image.\n\n"
            "Example (discrete space):\n"
            "I see an obstacle ahead at center. I am on flat ground moving forward.\n"
            "I need to jump to clear it safely.\n"
            "\\boxed{4}"
        )
    else:
        raise ValueError(
            f"Unknown output_format: {output_format!r}. "
            "Use 'boxed', 'json', or 'd1'."
        )

    prompt = f"""# GOAL
{description}

# ACTION SPACE
{action_description}

{detected}

# INSTRUCTION
At each step, analyze the current observation and choose the action that best advances your goal.

{format_instruction}"""

    return env, prompt


# ────────────────────────────────────────────────────────────
#  d1 mode
# ────────────────────────────────────────────────────────────

def _d1_env(env, description, action_description):
    """Prepare the goal string consumed by d1_agent.

    d1-3B has no text-output channel: it answers typed questions.
    d1_agent auto-generates those questions from env.action_space
    and uses this goal string as per-question context.
    """
    space = getattr(env, "action_space", None)
    if space is None:
        raise ValueError(
            "LLM_env(..., output_format='d1') requires env.action_space "
            "so that d1_agent can generate typed questions from it."
        )

    st = space_type(space)
    if st not in _D1_SUPPORTED:
        raise ValueError(
            f"d1-3B does not support action space type "
            f"{type(space).__name__!r} (detected as {st!r}). "
            f"Supported: {sorted(_D1_SUPPORTED)}."
        )

    # Compose the goal string injected into every generated question.
    parts = []
    if description:
        parts.append(description.strip())
    if action_description:
        parts.append(f"Action hints: {action_description.strip()}")
    goal = " | ".join(parts) if parts else "Choose the best next action."

    return env, goal
