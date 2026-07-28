# Contexto de la sesión - call_me_maybe

## Dónde quedamos
Repasando el proceso interno del LLM (qué pasa con los logits). El usuario quería entender a fondo el paso 2: "pasar por el modelo".

## Conceptos clave entendidos

### El proceso completo del LLM (paso a paso)

```
Prompt: "What is the sum of 2 and 3?"
                ↓
    ┌───────────────────────┐
    │  1. ENCODE (tokenizar)│
    │  → tensor de token ids│
    │  "What"→3923, "is"→310│
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │  2. PASAR POR EL MODELO│
    │                       │
    │  Capa Embedding:      │
    │  Cada token → vector denso (2048 dims)│
    │                       │
    │  Capa Transformer:    │
    │  28 bloques de Self-Attention + FFN│
    │  Cada token "mira" a los otros│
    │  para entender contexto│
    │                       │
    │  Capa Salida (LM Head):│
    │  Vector del ÚLTIMO token se proyecta│
    │  al vocabulario completo│
    │  → ~150,000 floats (logits)│
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │  3. SOFTMAX → probabilidades│
    │  "3" → 0.72, "4" → 0.15, ...│
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │  4. ELEGIR TOKEN      │
    │  argmax o sampling    │
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │  5. AÑADIR A SECUENCIA│
    │  → repetir desde paso 1│
    └───────────────────────┘
```

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

### Loop de generación
- El LLM genera texto en loop: logits → filtrar → elegir token → añadir → repetir
- `generate()` (que el SDK no expone) hace esto automáticamente
- Necesitamos implementar el loop nosotros con `get_logits_from_input_ids`

### Gramática dinámica / Autómata de estados
- Las reglas de restricción cambian según la función elegida
- Llevamos un "estado" que nos dice qué tokens son válidos en cada paso
- Ejemplo: después de elegir `fn_add_numbers`, solo se permiten parámetros `a` (number) y `b` (number)
- Si el modelo intenta generar algo fuera de la gramática → le ponemos `-inf` y no puede

## Siguiente paso
Diseñar el autómata/gramática concreto para el proyecto.

## Notas importantes
- El SDK tiene: `encode`, `get_logits_from_input_ids`, `decode`
- NO necesitamos: `get_path_to_*`
- Los archivos de entrada están en `data/input/`
- El output debe ser JSON válido al 100% con decodificación restringida
- Vocabulario de Qwen3: ~150,000 tokens
