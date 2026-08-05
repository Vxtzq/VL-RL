from .spaces import space_type, describe_space


def LLM_env(env, description="", action_description="", output_format="boxed"):
    """Wrap n'importe quel env RL en env LLM avec prompt structuré.
    L'action space est auto-détecté et le format attendu injecté dans le prompt."""

    space = getattr(env, "action_space", None)
    detected = (
        f"# DETECTED ACTION FORMAT\n"
        f"Type: {space_type(space)}\n"
        f"Expected output: {describe_space(space)}"
        if space is not None else ""
    )

    if output_format == "json":
        format_instruction = """Output EXACTLY this JSON format, starting with {:
{"explanation": "describe what you see + reasoning", "action": <action in the expected format>}
Do NOT output anything before or after the JSON."""

    elif output_format == "boxed":
        format_instruction = """Think step by step in plain text about:
1. What you observe in the current state (can be anything, must be **PRECISE**)
2. Relevant objects, obstacles, and their positions
3. Your current position and state
4. Best action to take given your goal

IMPORTANT: 
- End your response with your action in \\boxed{...} using EXACTLY the expected format above.
- Do not **hallucinate** objects (or any type of content) that is not in the given image

Example (discrete space):
I see an obstacle ahead at center. I am on flat ground moving forward.
I need to jump to clear it safely.
\\boxed{4}"""

    else:
        raise ValueError(f"Unknown output_format: {output_format}. Use 'json' or 'boxed'.")

    prompt = f"""# GOAL
{description}

# ACTION SPACE
{action_description}

{detected}

# INSTRUCTION
At each step, analyze the current observation and choose the action that best advances your goal.

{format_instruction}"""

    return env, prompt
