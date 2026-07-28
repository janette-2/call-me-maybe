# Call Me Maybe — Project Source for NotebookLM

## Estructura de carpetas

```
call_me_maybe/
├── pyproject.toml
├── uv.lock
├── Makefile
├── .python-version
├── .gitignore
├── README.md
├── call_me_maybe.subject.md
│
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
│
├── llm_sdk/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── llm_sdk/
│       └── __init__.py
│
└── src/
    ├── __init__.py
    ├── __main__.py
    └── main.py
```

---

## pyproject.toml (raíz)

```toml
[project]
name = "call-me-maybe-janette"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "flake8>=7.3.0",
    "llm-sdk",
    "mypy>=2.2.0",
    "numpy>=2.2.6",
    "pydantic>=2.13.4",
]

[tool.uv.workspace]
members = ["llm_sdk"]

[tool.uv.sources]
llm-sdk = { workspace = true }
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cpu"
name = "pytorch-cpu"
```

---

## Makefile

```makefile
install:
	uv sync

run: src/main.py
	uv run python -m src

debug: src/main.py
	uv run python -m pdb src/main.py

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

lint:
	uv run flake8 src/.
	uv run mypy src/. --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/.
	uv run mypy src/. --strict
```

---

## llm_sdk/pyproject.toml

```toml
[project]
name = "llm-sdk"
version = "0.1.0"
description = "LLM SDK for local model inference"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "huggingface-hub>=0.20.0",
]
```

---

## llm_sdk/llm_sdk/__init__.py (completo, NO modificar)

```python
# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download
import os


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM for fast, low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically select ``mps`` when available on macOS,
        ``cuda`` when available, otherwise we fall back to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS we default to ``float16`` to keep memory
        usage reasonable; on CPU we keep ``float32`` for maximum compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a 2-D ``input_ids`` tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """
        Given a list of input token ids, return the raw logits (no softmax) for the next token.
        """
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocab_file(self) -> str:
        vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path

    def get_path_to_merges_file(self) -> str:
        merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path

    def get_path_to_tokenizer_file(self) -> str:
        tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path
```

### API del SDK — Qué métodos hay y para qué sirven

- `encode(text: str) -> Tensor`: Convierte texto a tokens (tensor 2D).
- `decode(ids: Tensor | list[int]) -> str`: Convierte tokens a texto.
- `get_logits_from_input_ids(input_ids: list[int]) -> list[float]`: Dada una lista de token ids, devuelve los logits raw (~150K floats, uno por cada token del vocabulario) para predecir el siguiente token. Esta es la función principal para implementar decodificación restringida.
- `get_path_to_tokenizer_file() -> str`: Devuelve la ruta al archivo tokenizer.json descargado. Útil para cargar el vocabulario completo y saber qué tokens existen.
- `get_path_to_vocab_file() -> str`: Ruta al vocab.txt.
- `get_path_to_merges_file() -> str`: Ruta al merges.txt.

---

## src/main.py (código actual)

```python
from llm_sdk import Small_LLM_Model


def main():
    prompt = input("Prompt: ")
    model = Small_LLM_Model()
    token = model.encode(prompt)
    print(f"Token: {token}")


if __name__ == "__main__":
    main()
```

---

## data/input/functions_definition.json

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": { "type": "string" }
    },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": { "type": "string" },
      "regex": { "type": "string" },
      "replacement": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

---

## data/input/function_calling_tests.json

```json
[
  "What is the sum of 2 and 3?",
  "What is the sum of 40 and 2?",
  "Reverse the string 'hello'",
  "Reverse the string 'Python'",
  "Calculate the square root of 64",
  "Calculate the square root of 2",
  "Greet John by name",
  "Greet Alice by name",
  "Replace all vowels in 'hello world' with '*'",
  "Replace all digits in 'abc123def456' with '#'",
  "What is the sum of -10 and 5?"
]
```

---

## Enunciado del proyecto (call_me_maybe.subject.md) — Puntos clave

### Objetivo
Desarrollar una herramienta de function calling que traduzca peticiones en lenguaje natural en llamadas a funciones estructuradas. Dada una pregunta como "What is the sum of 2 and 3?", la solución NO debe devolver "5", sino:

```json
{
  "fn_name": "fn_add_numbers",
  "args": {"a": 2, "b": 3}
}
```

### Decodificación restringida (OBLIGATORIO)
La implementación DEBE usar decodificación restringida para garantizar un JSON válido al 100%. Esto implica enmascarar tokens inválidos en cada paso de la generación para forzar que el modelo solo genere JSON válido.

### Archivos de entrada
- `data/input/function_calling_tests.json`: Array de prompts en lenguaje natural.
- `data/input/functions_definition.json`: Definiciones de funciones disponibles (nombre, parámetros, tipos, retorno).

### Archivo de salida
`output/function_calling_results.json`: Array de objetos, cada uno con:
- `prompt` (string): petición original
- `fn_name` (string): nombre de la función a llamar
- `args` (object): argumentos con tipos correctos

### Reglas de validación
- JSON válido (sin comas finales, sin comentarios)
- Claves y tipos deben coincidir exactamente con function_definitions.json
- No se permiten claves adicionales ni texto libre
- Todos los argumentos requeridos deben estar presentes
- Los tipos deben coincidir (number, string, boolean, etc.)

### Rendimiento esperado
- >95% precisión en selección de función y argumentos
- 100% JSON válido en toda la salida
- Procesar todos los prompts en <5 minutos
- Gestión robusta de errores (JSON malformado, archivos ausentes)

### Cómo ejecutar
```bash
make run
# Equivalente a: uv run python -m src
```

### Archivos que debe contener el repositorio
- `src/` con la implementación
- `pyproject.toml` y `uv.lock`
- `llm_sdk/` (copiado del zip proporcionado, NO modificar)
- `data/input/` con archivos de prueba
- `README.md` con documentación completa
- NO incluir `output/` en el repositorio (se genera durante la evaluación)

### Notas importantes
- El modelo Qwen3-0.6B tiene solo 500M parámetros
- Con una decodificación restringida adecuada, puede alcanzar fiabilidad comparable a modelos mucho más grandes
- Los archivos de entrada proporcionados pueden cambiar durante la evaluación
- No hardcodear soluciones basadas en los ejemplos dados
- Se probará con prompts y conjuntos de funciones diferentes
