# src/llm/ollama_client.py
"""Expert-level Ollama client with robust error handling + faster keep-alive session."""
import os
import time
from typing import Dict, Any, Optional, List

import requests

# Reuse TCP connections (BIG speedup for multi-call agentic)
_SESSION = requests.Session()

def _get_connect_timeout() -> float:
    return float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "10"))

def _get_read_timeout() -> float:
    return float(os.environ.get("OLLAMA_READ_TIMEOUT", "180"))

def _get_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

_DIAG_CACHE: Optional[Dict[str, Any]] = None
_DIAG_CACHE_TIME: float = 0

def diagnose_ollama() -> Dict[str, Any]:
    global _DIAG_CACHE, _DIAG_CACHE_TIME

    if _DIAG_CACHE and (time.time() - _DIAG_CACHE_TIME) < 30:
        return _DIAG_CACHE

    base_url = _get_base_url()
    result = {"running": False, "models": [], "base_url": base_url, "error_message": "", "suggested_fix": ""}

    try:
        resp = _SESSION.get(f"{base_url}/api/tags", timeout=5)
        if resp.ok:
            models = [m.get("name") for m in resp.json().get("models", [])]
            result = {"running": True, "models": models, "base_url": base_url, "error_message": "", "suggested_fix": ""}
        else:
            result["error_message"] = f"HTTP {resp.status_code}"
            result["suggested_fix"] = "Ollama running but returned error. Check ollama logs."
    except requests.exceptions.ConnectionError:
        result["error_message"] = "Connection refused"
        result["suggested_fix"] = (
            "Ollama not running:\n"
            "1) Install Ollama\n"
            "2) Start: ollama serve\n"
            "3) Pull model: ollama pull llama3.2\n"
            "4) Test: curl http://localhost:11434/api/tags"
        )
    except requests.exceptions.Timeout:
        result["error_message"] = "Timeout"
        result["suggested_fix"] = "Ollama slow to respond. Check system resources."
    except Exception as e:
        result["error_message"] = str(e)
        result["suggested_fix"] = "Unexpected error. Check Ollama installation."

    _DIAG_CACHE, _DIAG_CACHE_TIME = result, time.time()
    return result

def ask_ollama(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.1,
    num_predict: int = 200,
    raise_on_fail: bool = False,
    retries: int = 3,
) -> str:
    base_url = _get_base_url()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    for attempt in range(retries):
        try:
            resp = _SESSION.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=(_get_connect_timeout(), _get_read_timeout()),
            )

            if resp.ok:
                return resp.json().get("message", {}).get("content", "") or ""

            if resp.status_code >= 500 and attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue

            if raise_on_fail:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            return ""

        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            if raise_on_fail:
                raise Exception("Ollama timeout after retries")
            return ""

        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            if raise_on_fail:
                raise Exception("Cannot connect to Ollama")
            return ""

        except Exception:
            if raise_on_fail:
                raise
            return ""

    return ""
