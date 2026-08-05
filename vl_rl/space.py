"""Action-space detection, prompt formatting and LLM-output parsing.
Works with both `gym` and `gymnasium` spaces via duck typing."""

import ast
import json
import re

import numpy as np


# ────────────────────────────────────────────────
# 1. DÉTECTION
# ────────────────────────────────────────────────

def space_type(space):
    """Tag décrivant la famille du space (marche avec gym ET gymnasium)."""
    return {
        "Discrete": "discrete",
        "MultiBinary": "multibinary",
        "MultiDiscrete": "multidiscrete",
        "Box": "continuous",
        "Tuple": "tuple",
        "Dict": "dict",
    }.get(type(space).__name__, "unknown")


def describe_space(space):
    """Format de sortie attendu, lisible par le LLM, avec exemple."""
    t = space_type(space)

    if t == "discrete":
        return (f"single integer in [0, {space.n - 1}] "
                f"-> \\boxed{{3}}")

    if t == "multibinary":
        return (f"list of {space.n} bits (0 or 1), one per button "
                f"-> \\boxed{{[0, 1, 0, 1]}}")

    if t == "multidiscrete":
        nvec = np.asarray(space.nvec).flatten()
        ranges = ", ".join(f"[0, {n - 1}]" for n in nvec)
        return (f"list of {nvec.size} integers, ranges {ranges} "
                f"-> \\boxed{{[1, 0, 2]}}")

    if t == "continuous":
        size = int(np.prod(space.shape))
        lo = np.round(np.asarray(space.low, dtype=float).flatten(), 3)
        hi = np.round(np.asarray(space.high, dtype=float).flatten(), 3)
        lo = lo.tolist() if lo.size > 1 else float(lo[0])
        hi = hi.tolist() if hi.size > 1 else float(hi[0])
        return (f"list of {size} floats, low={lo}, high={hi} "
                f"-> \\boxed{{[0.5, -0.2]}}")

    if t == "tuple":
        parts = "\n".join(
            f"    [{i}] {describe_space(s)}" for i, s in enumerate(space.spaces))
        return f"list with one entry per sub-space, in order:\n{parts}"

    if t == "dict":
        parts = "\n".join(
            f'    "{k}": {describe_space(s)}' for k, s in space.spaces.items())
        return f"JSON object with these keys:\n{parts}"

    return "free-form action"


# ────────────────────────────────────────────────
# 2. PARSING : n'importe quel output -> action valide
# ────────────────────────────────────────────────

def extract_boxed(content):
    """Contenu du dernier \\boxed{...} (accolades équilibrées), ou None."""
    idx = content.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start:i]
    return None


def parse_value(raw):
    """Conversion best-effort d'une string en int / float / list / dict."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw)
        except Exception:
            continue
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", raw)]
    return nums or None


def neutral_action(space):
    """Action fallback déterministe si le parsing échoue totalement."""
    t = space_type(space)
    if t == "discrete":
        return 0
    if t == "multibinary":
        return np.zeros(space.n, dtype=np.int8)
    if t == "multidiscrete":
        return np.zeros_like(space.nvec)
    if t == "continuous":
        return np.zeros(space.shape, dtype=space.dtype)
    if t == "tuple":
        return tuple(neutral_action(s) for s in space.spaces)
    if t == "dict":
        return {k: neutral_action(s) for k, s in space.spaces.items()}
    return 0


def to_action(value, space):
    """Force une valeur parsée dans le space (clip, cast, reshape, pad)."""
    t = space_type(space)
    try:
        if t == "discrete":
            if isinstance(value, (list, tuple)):
                value = value[0] if len(value) else 0
            return int(np.clip(int(value), 0, space.n - 1))

        if t == "multibinary":
            arr = np.asarray(value, dtype=float).flatten()
            arr = np.pad(arr, (0, max(0, space.n - arr.size)))[:space.n]
            return (arr > 0).astype(np.int8)

        if t == "multidiscrete":
            nvec = np.asarray(space.nvec).flatten()
            arr = np.asarray(value, dtype=float).flatten()
            arr = np.pad(arr, (0, max(0, nvec.size - arr.size)))[:nvec.size]
            return np.clip(arr, 0, nvec - 1).astype(space.nvec.dtype).reshape(space.nvec.shape)

        if t == "continuous":
            size = int(np.prod(space.shape))
            arr = np.asarray(value, dtype=float).flatten()
            arr = np.pad(arr, (0, max(0, size - arr.size)))[:size]
            arr = np.clip(arr,
                          np.asarray(space.low, dtype=float).flatten(),
                          np.asarray(space.high, dtype=float).flatten())
            return arr.reshape(space.shape).astype(space.dtype)

        if t == "tuple":
            return tuple(to_action(v, s) for v, s in zip(value, space.spaces))

        if t == "dict":
            return {k: to_action(value.get(k), s) for k, s in space.spaces.items()}
    except Exception:
        pass
    return neutral_action(space)


def parse_action(content, space):
    """Pipeline complet : texte LLM brut -> (action valide, explanation)."""
    boxed = extract_boxed(content)
    if boxed is not None:
        explanation = content[:content.rfind("\\boxed{")].strip()
        raw = boxed
    else:
        explanation = content.strip()
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
        raw = lines[-1] if lines else content
    return to_action(parse_value(raw), space), explanation
