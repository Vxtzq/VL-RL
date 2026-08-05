import json
import base64
import io
import re
import requests
from PIL import Image
from collections import deque
from .spaces import parse_action  # ⬅️ nouvel import en haut

def _image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════
#  PARSING COMMUN
# ═══════════════════════════════════════════════════════════

def parse_response(content):
    """Parse \\boxed{N} ou fallback sur un nombre isolé."""
    match = re.search(r'\\boxed\{(\d+)\}', content)
    if match:
        action = int(match.group(1))
        explanation = content[:match.start()].strip()
        return action, explanation

    lines = content.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^\d{1,2}$', line):
            action = int(line)
            explanation = content[:content.rfind(line)].strip()
            return action, explanation

    return 3, content.strip()


# ═══════════════════════════════════════════════════════════
#  OLLAMA AGENT (avec auto-pull)
# ═══════════════════════════════════════════════════════════

_ollama_history_frames = deque(maxlen=1)
_ollama_history_actions = deque(maxlen=1)

OLLAMA_URL = "http://localhost:11434"


def _ollama_model_exists(model_name):
    """Vérifie si le modèle est déjà disponible localement."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        return model_name in available
    except Exception:
        return False


def _ollama_pull(model_name):
    """Télécharge le modèle depuis le registry Ollama."""
    print(f"📥 Modèle '{model_name}' non trouvé. Téléchargement en cours...")
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name, "stream": True},
            stream=True,
            timeout=None
        )
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                completed = data.get("completed", 0)
                total = data.get("total", 0)
                if total > 0:
                    pct = 100 * completed / total
                    print(f"\r   {status}: {pct:.1f}%", end="", flush=True)
                else:
                    print(f"\r   {status}", end="", flush=True)
        print("\n   ✅ Téléchargement terminé.")
    except Exception as e:
        print(f"\n   ❌ Erreur pull: {e}")
        raise


def ollama_agent(obs, prompt, model_name="qwen3-vl:8b",
                 think=True, num_predict=2048, temperature=0.4,
                 history_frames=None, history_actions=None, action_space=None):
    """
    Appelle Ollama avec une image + prompt.
    Si le modèle n'existe pas localement, il est téléchargé automatiquement.
    Retourne (action: int, explanation: str).
    """
    # ⬅️ Historique : utilise les deques passés ou les globals
    h_frames = history_frames if history_frames is not None else _ollama_history_frames
    h_actions = history_actions if history_actions is not None else _ollama_history_actions

    # ⬅️ Auto-pull si modèle absent
    if not _ollama_model_exists(model_name):
        _ollama_pull(model_name)

    img_b64 = _image_to_base64(obs)

    messages = [{"role": "system", "content": prompt}]

    for i, hist_b64 in enumerate(h_frames):
        messages.append({
            "role": "user",
            "content": f"Frame {i+1} from the past.",
            "images": [hist_b64]
        })
        actions_list = list(h_actions)
        messages.append({
            "role": "assistant",
            "content": f"Action taken: {actions_list[i]}" if i < len(actions_list) else "Acknowledged."
        })

    messages.append({
        "role": "user",
        "content": "Current frame. Analyze carefully and decide the action.",
        "images": [img_b64]
    })

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model_name,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            }
        }, timeout=600)

        data = resp.json()

        if resp.status_code != 200 or "message" not in data:
            print(f"⚠️ Ollama error ({resp.status_code}): {data.get('error', data)}")
            return 3, "(ollama error fallback)"

        content = data["message"]["content"]
        thinking = data["message"].get("thinking", "")

        if not content.strip() and not thinking.strip():
            print(f"⚠️ Empty response! Full: {json.dumps(data)[:300]}")
            return 3, "(empty response fallback)"

        if action_space is not None:
            action, explanation = parse_action(content, action_space)
        else:
            action, explanation = parse_response(content)

        if thinking:
            print(f"🧠 Thinking: {thinking[:120]}...")
        print(f"🤖 Action: {action} | {explanation[:100]}...")

        h_frames.append(img_b64)
        h_actions.append(action)

        return action, explanation

    except requests.exceptions.ConnectionError:
        print("⚠️ Ollama not running! Start with: ollama serve")
        return 3, "(connection error)"
    except requests.exceptions.ReadTimeout:
        print("⚠️ Ollama timeout (>600s)")
        return 3, "(timeout fallback)"
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return 3, "(error fallback)"


# ═══════════════════════════════════════════════════════════
#  VLLM AGENT
# ═══════════════════════════════════════════════════════════

_vllm_engine = None
_vllm_sampling_params = None
_vllm_processor = None


def init_vllm(model_name="Qwen/Qwen3-VL-8B-Instruct",
              max_model_len=4096, gpu_memory_utilization=0.9,
              dtype="half", action_space=None):
    """
    Initialise le moteur vLLM (à appeler UNE seule fois au démarrage).
    Nécessite: pip install vllm transformers
    """
    global _vllm_engine, _vllm_sampling_params, _vllm_processor

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"🔧 Chargement vLLM: {model_name}...")
    _vllm_engine = LLM(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 4}
    )
    _vllm_sampling_params = SamplingParams(
        temperature=0.4,
        max_tokens=2048,
        skip_special_tokens=False
    )
    _vllm_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("✅ vLLM prêt.")


def vllm_agent(obs, prompt, model_name=None,
               history_frames_pil=None, history_actions=None):
    """
    Appelle vLLM en local (pas de serveur HTTP, inférence directe).
    obs: PIL Image de la frame actuelle
    Retourne (action: int, explanation: str).
    """
    global _vllm_engine, _vllm_sampling_params, _vllm_processor

    if _vllm_engine is None:
        raise RuntimeError("Appelle init_vllm() d'abord !")

    from PIL import Image as PILImage

    # ⬅️ Construire la liste d'images (historique + actuelle)
    images = []
    if history_frames_pil:
        images.extend(list(history_frames_pil))
    images.append(obs if isinstance(obs, PILImage.Image) else PILImage.fromarray(obs))

    # ⬅️ Construire le prompt au format chat Qwen-VL
    messages = [{"role": "system", "content": prompt}]

    if history_frames_pil and history_actions:
        for i, act in enumerate(history_actions):
            messages.append({
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": f"Frame {i+1} from the past."}]
            })
            messages.append({
                "role": "assistant",
                "content": f"Action taken: {act}"
            })

    messages.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Current frame. Analyze carefully and decide the action."}]
    })

    # ⬅️ Appliquer le template chat du modèle
    text_prompt = _vllm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ⬅️ Préparer les inputs multimodaux
    inputs = [{
        "prompt": text_prompt,
        "multi_modal_data": {"image": images}
    }]

    # ⬅️ Inférence
    outputs = _vllm_engine.generate(inputs, _vllm_sampling_params)
    content = outputs[0].outputs[0].text

    if action_space is not None:
            action, explanation = parse_action(content, action_space)
        else:
            action, explanation = parse_response(content)

    print(f"🤖 [vLLM] Action: {action} | {explanation[:100]}...")

    return action, explanation


# ═══════════════════════════════════════════════════════════
#  RESET HISTORIQUE (à appeler entre les runs)
# ═══════════════════════════════════════════════════════════

def reset_ollama_history():
    _ollama_history_frames.clear()
    _ollama_history_actions.clear()
