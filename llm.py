import os
import time

import requests


def generate(prompt, hf_token=None, openai_key=None, model_name=None):
    """Generate a response using HF API, OpenAI, local transformers, or a safe fallback."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "I did not catch that.", 0.0

    # 1) Hugging Face Inference API
    if hf_token:
        start = time.time()
        model = model_name or os.environ.get("HF_MODEL", "google/flan-t5-small")
        url = os.environ.get("HF_MODEL_URL", f"https://api-inference.huggingface.co/models/{model}")
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128}}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text") or data[0].get("summary_text") or str(data[0])
                elif isinstance(data, dict):
                    text = data.get("generated_text") or data.get("summary_text") or str(data)
                else:
                    text = str(data)
                return text.strip(), time.time() - start
            return f"HF error {resp.status_code}: {resp.text}", time.time() - start
        except Exception as exc:
            return f"HF request failed: {exc}", time.time() - start

    # 2) Local Transformers fallback (optional)
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        start = time.time()
        model = model_name or os.environ.get("HF_MODEL", "google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained(model)
        net = AutoModelForSeq2SeqLM.from_pretrained(model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(device)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = net.generate(**inputs, max_new_tokens=128)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return text.strip(), time.time() - start
    except Exception:
        pass

    # 3) OpenAI fallback
    if openai_key:
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=128,
            )
            text = resp.choices[0].message.content or ""
            return text.strip(), time.time() - start
        except Exception as exc:
            return f"OpenAI error: {exc}", time.time() - start

    # 4) Safe local fallback
    return f"You said: {prompt}", 0.0
