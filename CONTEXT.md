# Contexto de la sesión - call_me_maybe

## Dónde quedamos
Paso 3 (super-prompt) **TERMINADO**. `build_super_prompt(dict_functions, input_call)` construye el prompt (rol + reglas + listado de funciones + `User: ...` + `Output: `) y se prueba con `make run`. El formato de salida quedó limpio escribiendo el contenido de los strings a **columna 0** dentro de las comillas triples (en vez de usar `textwrap.dedent()`), y se eliminaron las variables intermedias `intro`/`rules`/`functions` que eran renombres muertos. Docstrings y README (desafíos 12 y 13) actualizados. `make lint` pasa. Al retomar, toca el **Paso 4**: loop con decodificación restringida.

## Plan de implementación (5 pasos)

```
Paso 1: Token discovery          → HECHO
Paso 2: Cargar funciones JSON    → HECHO
Paso 3: Construir super-prompt   → HECHO
Paso 4: Loop + autómata de estados → SIGUIENTE
Paso 5: Procesar prompts y escribir output
```

## Visión completa del flujo

```
PROMPT DEL USUARIO: "What is the sum of 2 and 3?"
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│  PASO 3: SUPER-PROMPT                                   │
│  "Available functions:                                  │
│   - fn_add_numbers: Add two numbers...                  │
│   - fn_greet: Generate a greeting...                    │
│   ...                                                  │
│  User: What is the sum of 2 and 3?                     │
│  Output: "                                             │
│  PROPÓSITO: dar contexto para que el modelo decida      │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│  PASO 4: LOOP CON DECODIFICACIÓN RESTRINGIDA             │
│  { → "fn_name" → " → [modelo elige función] → ...       │
│  FORZADO / FORZADO / FORZADO / EL MODELO ELIGE           │
│  PROPÓSITO: forzar JSON 100% válido                     │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│  PASO 5: OUTPUT                                         │
│  output/function_calling_results.json                   │
│  [{"prompt": ..., "fn_name": ..., "args": {...}}]       │
└──────────────────────────────────────────────────────────┘
```

## Estado de la implementación (código en src/main.py)

### Hecho

**1. `functions_info()`** (antes `functions_parameters()`) → devuelve:
```python
{
  "fn_add_numbers": {
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "description": "Add two numbers together and return their sum.",
    "returns": {"type": "number"}
  },
  ...
}
```
- Maneja JSON inválido/ausente con try/except (devuelve {} e imprime el error)
- Aprendizaje: usar dict con claves nombradas (`["parameters"]`, `["description"]`) en vez de lista por índice

**2. `fixed_ids(model, dict_functions)`** → devuelve:
```python
{
  "{": [90], "}": [92], "fn_add_numbers": [8822, 2891, 32964],
  "a": [64], ...
}
```
- Usa `model.encode().flatten().tolist()`: flatten aplana tensor 2D→1D, tolist lo convierte a lista de ints
- Orden correcto: `.flatten().tolist()` (flatten es de tensores, tolist convierte a lista)
- 18 entradas: 6 caracteres fijos + 5 funciones + 7 parámetros

**3. `build_super_prompt(dict_functions, input_call)`** → construye el texto con rol, reglas, listado de funciones y el prompt del usuario al final (termina en `Output: `). La salida queda limpia escribiendo el contenido de los strings a columna 0 (las comillas triples son literales: cualquier indentación dentro del string se cuela en el output). Se descartó `textwrap.dedent()`: con `template_rules` no funcionaba porque la línea `Rules:` tiene 0 espacios (mínimo común = 0 → no quita nada). El parámetro se llama `input_call` (evita ocultar el builtin `input`).

### Datos de token discovery (Qwen3-0.6B)

```
'{' → [90]        '}' → [92]        ':' → [25]
',' → [11]        ' ' → [220]       '\n' → [198]
'"' → [1]         'fn_name' → [8822, 1269]     'args' → [2116]

'fn_add_numbers'    → [8822, 2891, 32964]
'fn_greet'          → [8822, 1889, 3744]
'fn_reverse_string' → [8822, 43277, 3904]
'fn_get_square_root'→ [8822, 3062, 39794, 12993]
'fn_substitute_string_with_regex' → [8822, 5228, 7660, 3904, 6615, 41832]

'a' → [64]    'b' → [65]    'name' → [606]    's' → [82]
'source_string' → [2427, 3904]    'regex' → [26387]    'replacement' → [83631]
```

Observación clave: todas las funciones empiezan por [8822] (token "fn").

### Infraestructura

- **swap de 8 GB creado** (también 1 GB de swap original = 9 GB total): el modelo Qwen3-0.6B (1.5 GB) no cabía en 3.8 GB de RAM con VS Code abierto
- VS Code consume ~1.5 GB de RAM: mejor cerrarlo antes de `make run` (swap evita que pete pero es lento)
- `make lint` pasa (flake8 + mypy). Config mypy en pyproject.toml para ignorar llm_sdk (no tiene anotaciones, no se modifica): `mypy_path = "llm_sdk"`, override con `ignore_errors = true` y `follow_imports = "skip"`
- Docstrings en inglés, formato Google-style, en las 3 funciones

## Decisión de diseño clave

- **Plantilla JSON fija**: el programa FORZA las partes invariables del JSON (llaves, comillas, comas, "fn_name", "args") con IDs conocidos. El LLM solo ELIGE las partes variables (nombre de función y valores de argumentos).
- La separación fue un dilema resuelto: no generamos JSON completo con el LLM, sino que el LLM decide los valores y nosotros construimos la estructura.

## Errores resueltos (aprendizajes)

1. `import Exception` → no es un módulo, es builtin de Python
2. `model.encode(fixed).tolist().flatten()` → error: flatten() es de tensores, no de listas. Orden correcto: `.flatten().tolist()`
3. `dictionary["return"]` → KeyError: el campo JSON se llama `"returns"` (con s)
4. Estructura en lista `[parameters, description, returns]` → mala práctica: acceso por índice oscuro. Mejor dict con claves nombradas
5. `Small_LLM_Model.encode(fixed)` sin instanciar → AttributeError: encode es método de instancia, hay que crear el objeto primero

## Conceptos clave entendidos

### Logits
- El modelo Qwen3-0.6B produce ~150,000 logits (uno por cada token del vocabulario)
- Cada logit es una puntuación cruda de cuán probable es que ese token sea el siguiente
- `get_logits_from_input_ids` devuelve estos logits para el siguiente token
- Un logit alto = el modelo cree que ese token es probable
- Un logit bajo = el modelo cree que ese token es improbable

### Decodificación restringida (lo que nosotros hacemos)
- El modelo produce los 150,000 logits originales (no los cambiamos)
- **ANTES** de elegir token, nosotros modificamos los logits:
  - Tokens válidos → se quedan igual
  - Tokens inválidos → ponerles `-inf`
- Después del softmax, solo los tokens válidos tienen probabilidad > 0
- El modelo **está obligado** a elegir entre los tokens válidos

### Propósito del LLM en el proyecto
- El LLM NO responde la pregunta (no devuelve "5"), identifica qué función llamar y con qué argumentos
- Se usa como motor de inferencia semántica, no como chatbot
- Sin LLM: regex/parseo tradicional no es flexible para lenguaje natural variado
- El super-prompt orienta (señal débil), el logit masking fuerza (garantía fuerte)

### Momento del decode
- encode → texto a IDs (una vez al inicio)
- decode → IDs a texto (una vez al final)
- El loop entero trabaja con IDs numéricos, nunca con strings

## Siguiente paso
Paso 3 (cont.): limpiar el formato del super-prompt con `textwrap.dedent()` (ver "Dónde quedamos" arriba) y luego empezar el Paso 4: el loop con decodificación restringida.

## Notas importantes
- El SDK tiene: `encode`, `get_logits_from_input_ids`, `decode`
- NO necesitamos: `get_path_to_*`
- Los archivos de entrada están en `data/input/`
- El output debe ser JSON válido al 100% con decodificación restringida
- Vocabulario de Qwen3: ~150,000 tokens
