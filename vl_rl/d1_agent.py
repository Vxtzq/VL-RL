"""LiquidAI/d1-3B agent — Jev-style typed-question decision model.

Replaces the free-text + \boxed{} pipeline with structured questions that
map 1:1 onto a gym/gymnasium action space. One forward pass per step.
"""

import numpy as np
import torch
from PIL import Image

from .spaces import space_type, to_action, neutral_action

_d1_model = None


def init_d1(model_name="LiquidAI/d1-3B", device=None, dtype=None):
    """Load d1-3B once at startup."""
    global _d1_model
    from transformers import AutoModel

    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)

    print(f"🔧 Loading d1-3B on {device} ({dtype})...")
    _d1_model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype
    ).to(device)
    print("✅ d1-3B ready.")
    return _d1_model


# ────────────────────────────────────────────────────────────
#  Space → questions
# ────────────────────────────────────────────────────────────

def _questions_for_space(space, goal=""):
    """Turn an action space into d1-3B typed questions."""
    t = space_type(space)
    q = {}

    if t == "discrete":
        # Single choice over the full action set.
        q["action"] = {
            "type": "choice",
            "instructions": f"Pick the best next action. Goal: {goal}",
            "criteria": {str(i): f"Action {i}" for i in range(space.n)},
        }

    elif t == "multibinary":
        # One yes/no per button — d1's sweet spot.
        for i in range(space.n):
            q[f"button_{i}"] = {
                "type": "noul",
                "instructions": f"Press button {i} now? Goal: {goal}",
            }

    elif t == "multidiscrete":
        nvec = np.asarray(space.nvec).flatten()
        for i, n in enumerate(nvec):
            if n <= 8:
                # Small ordered range → choice
                q[f"dim_{i}"] = {
                    "type": "choice",
                    "instructions": f"Dimension {i} (0..{n-1}). Goal: {goal}",
                    "criteria": {str(j): f"Value {j}" for j in range(n)},
                }
            else:
                # Larger ordered range → score (rank from neutral to max)
                q[f"dim_{i}"] = {
                    "type": "score",
                    "instructions": f"Rate dimension {i} along its natural order. Goal: {goal}",
                    "criteria": [f"Level {j}" for j in range(n)],
                }

    elif t == "continuous":
        size = int(np.prod(space.shape))
        lo = np.asarray(space.low, dtype=float).flatten()
        hi = np.asarray(space.high, dtype=float).flatten()
        for i in range(size):
            # score over 5 buckets; map linearly back to [lo, hi]
            q[f"cont_{i}"] = {
                "type": "score",
                "instructions": f"Value of dim {i}, low={lo[i]:.2f} high={hi[i]:.2f}. Goal: {goal}",
                "criteria": ["min", "low", "mid", "high", "max"],
            }

    elif t == "dict":
        for k, s in space.spaces.items():
            sub = _questions_for_space(s, goal)
            for qn, qv in sub.items():
                q[f"{k}.{qn}"] = qv

    else:
        raise ValueError(f"Unsupported space: {type(space).__name__}")

    return q


def _assemble_action(answers, space):
    """Re-assemble d1-3B answers into a valid gym action."""
    t = space_type(space)

    if t == "discrete":
        # answer is likely the criterion key ("3") or its index
        val = answers["action"]
        idx = int(val) if str(val).isdigit() else list(range(space.n))[int(val)]
        return to_action(idx, space)

    if t == "multibinary":
        bits = [int(bool(answers[f"button_{i}"])) for i in range(space.n)]
        return to_action(bits, space)

    if t == "multidiscrete":
        nvec = np.asarray(space.nvec).flatten()
        vals = []
        for i, n in enumerate(nvec):
            v = answers[f"dim_{i}"]
            if isinstance(v, str) and v.startswith("Level "):
                v = int(v.split()[1])
            vals.append(int(v))
        return to_action(vals, space)

    if t == "continuous":
        size = int(np.prod(space.shape))
        lo = np.asarray(space.low, dtype=float).flatten()
        hi = np.asarray(space.high, dtype=float).flatten()
        buckets = {"min": 0.0, "low": 0.25, "mid": 0.5, "high": 0.75, "max": 1.0}
        vals = []
        for i in range(size):
            v = answers[f"cont_{i}"]
            frac = buckets.get(str(v), 0.5)
            vals.append(lo[i] + frac * (hi[i] - lo[i]))
        return to_action(vals, space)

    if t == "dict":
        out = {}
        for k, s in space.spaces.items():
            sub_answers = {
                qn.split(".", 1)[1]: av
                for qn, av in answers.items() if qn.startswith(f"{k}.")
            }
            out[k] = _assemble_action(sub_answers, s)
        return out

    return neutral_action(space)


# ────────────────────────────────────────────────────────────
#  Public agent entry
# ────────────────────────────────────────────────────────────

def d1_agent(obs, action_space, description="",
             history_frames=None, history_actions=None, prompt=None):
    """
    Jev-style step. `prompt` is accepted for API compat and ignored —
    d1-3B doesn't need the free-text prompt; it needs (space → questions).

    Returns (action, explanation_dict).
    """
    if _d1_model is None:
        raise RuntimeError("Call init_d1() first!")

    if not isinstance(obs, Image.Image):
        obs = Image.fromarray(obs)

    # History (if provided) is folded into the description text —
    # d1-3B is trained on single-image states, not video.
    ctx = description
    if history_actions:
        ctx += f" | Recent actions: {list(history_actions)[-3:]}"

    questions = _questions_for_space(action_space, goal=ctx)
    answers = _d1_model.system_one(None, questions, images=[obs])
    action = _assemble_action(answers, action_space)

    print(f"🤖 [d1] {action} | {answers}")
    return action, answers
