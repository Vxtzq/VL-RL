def LLM_env(env, description="", action_description="", output_format="json"):
    """Wrap n'importe quel env RL en env LLM avec un prompt structuré."""
    
    if output_format == "json":
        format_instruction = """Output EXACTLY this JSON format, starting with {:
{"explanation": "describe what you see + reasoning", "action": <int>}
Do NOT output anything before or after the JSON."""
    
    elif output_format == "boxed":
        format_instruction = """Think step by step in plain text about:
1. What you observe in the current state (can be anything, must be **PRECISE**)
2. Relevant objects, obstacles, and their positions
3. Your current position and state
4. Best action to take given your goal

IMPORTANT: 
- End your response with your action in \\boxed{N}.
- Do not **hallucinate** objects (or any type of content) that is not in the given image

Example:
I see an obstacle ahead at center. I am on flat ground moving forward.
I need to jump to clear it safely.
\\boxed{4}"""
    
    prompt = f"""# GOAL
{description}

# ACTION SPACE
{action_description}

# INSTRUCTION
At each step, analyze the current observation and choose the action that best advances your goal.

{format_instruction}"""
    
    return env, prompt
