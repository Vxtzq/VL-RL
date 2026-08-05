"""Generic RL environment wrapper for LLM/VLM agents."""


def LLM_env(env, description="", action_description="", output_format="json"):
    """
    Wrap any RL env with a structured prompt for LLM/VLM agents.
    
    Args:
        env: Any gymnasium/gym environment
        description: Goal/task description
        action_description: Available actions and their meanings
        output_format: "json" or "boxed"
    
    Returns:
        Tuple of (env, prompt_string)
    """
    
    if output_format == "json":
        format_instruction = """Output EXACTLY this JSON format, starting with {:
{"explanation": "describe what you see + reasoning", "action": <int>}
Do NOT output anything before or after the JSON."""
    
    elif output_format == "boxed":
        format_instruction = """Think step by step in plain text about:
1. What you observe in the current state
2. Relevant objects, obstacles, and their positions
3. Your current position and state
4. Best action to take given your goal

IMPORTANT: 
- Do NOT output JSON.
- Do NOT repeat yourself. Keep reasoning under 300 words.
- End your response with your action in \\boxed{N}.

Example:
I see an obstacle ahead at center. I am on flat ground moving forward.
I need to jump to clear it safely.
\\boxed{4}"""
    
    else:
        raise ValueError(f"Unknown output_format: {output_format}. Use 'json' or 'boxed'.")
    
    prompt = f"""# GOAL
{description}

# ACTION SPACE
{action_description}

# INSTRUCTION
At each step, analyze the current observation and choose the action that best advances your goal.

{format_instruction}"""
    
    return env, prompt
