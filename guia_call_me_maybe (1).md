# Guía de mentoría — `call_me_maybe` (versión final)

Esta guía recorre tu código pieza por pieza, explicando **qué hace, por qué existe y qué objetivo del subject cubre**. Al final tienes una sección de puntos a vigilar: cosas que he detectado al leer el código que conviene que pruebes antes de darlo por cerrado.

---

## 1. Visión general: las 3 fases del programa

Todo el código gira en torno a un único proceso, repetido por cada prompt de `function_calling_tests.json`:

1. **Construir el prompt** (`build_super_prompt`): le explicamos al modelo qué funciones existen.
2. **Decodificación restringida** (`loop_prompt_output` y sus ayudantes): construimos el JSON de salida token a token, dejando que el modelo elija solo cuando es imprescindible, y decidiendo nosotros mismos (determinismo) siempre que se pueda.
3. **Validación y guardado** (`main`): comprobamos que el resultado tiene la forma correcta antes de escribirlo.

La idea central de esta versión "best-of-three" es: **cuantas menos veces le preguntes algo al modelo, más rápido y más fiable es tu programa** — porque cada llamada a `get_logits_from_input_ids` cuesta tiempo, y cada vez que dejas "elegir" al modelo introduces la posibilidad de que se equivoque. Por eso casi todo el código nuevo (frases entrecomilladas, números, patrones regex, palabra tras "with") busca **resolver el valor sin preguntarle nada al modelo**, y solo como último recurso se le pregunta.

---

## 2. Los modelos de pydantic

```python
class ParametroFuncion(BaseModel):
    type: str

class DefinicionFuncion(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametroFuncion]
    returns: dict[str, str]

class ResultadoLlamada(BaseModel):
    prompt: str
    fn_name: str | None
    args: dict
```

**Objetivo del subject que cubren:** "gestión robusta de errores... JSON no válido" y "las claves y tipos deben coincidir exactamente con el esquema".

- `DefinicionFuncion` se usa al **leer** `functions_definition.json`: si una función viene mal formada (falta una clave, un tipo raro), pydantic lanza `ValidationError` y la descartas con un aviso, en vez de que el programa reviente más adelante con un `KeyError` inesperado.
- `ResultadoLlamada` se usa al **escribir** cada resultado: garantiza que `prompt` es texto, `fn_name` es texto o `None`, y `args` es un diccionario — la forma exacta que pide el subject en la sección de formato de salida.

No validan los *valores* dentro de `args` (por ejemplo, que `a` sea de verdad un número) — eso lo hace la línea de coerción de tipos en `main()`, más abajo.

---

## 3. `CONCEPTO_A_REGEX`

```python
CONCEPTO_A_REGEX = {
    "number": r"\d+", ...
    "vowel": r"[aeiouAEIOU]", ...
    ...
    "asterisks": "*",
}
```

**Objetivo:** resolver argumentos cuyo valor **no está escrito literalmente en el prompt** — como el patrón regex de "replace all numbers" o "replace all vowels". El modelo no puede "copiar" algo que no existe como texto, así que aquí no hay decodificación: es una tabla de consulta.

Fíjate que también incluye `"asterisks": "*"` — esto no es para un `regex`, es para el argumento `replacement` cuando el usuario dice "with asterisks" en vez de escribir el símbolo directamente. Vale la pena que confirmes con qué argumento se está usando esta entrada exactamente, porque la tabla mezcla dos propósitos distintos (patrones de búsqueda y valores de reemplazo) bajo el mismo nombre — funciona, pero si mañana añades más "atajos de reemplazo" sería más claro tener un segundo diccionario separado (`CONCEPTO_A_VALOR_LITERAL`, por ejemplo).

---

## 4. Utilidades base: `_encode_ids` y `masked_argmax`

```python
def _encode_ids(model, texto) -> list[int]:
    return [int(x) for x in model.encode(texto).flatten().tolist()]

def masked_argmax(llm_logits, allowed_ids) -> int:
    logits_array = np.asarray(llm_logits)
    mascara = np.full(logits_array.shape, -np.inf)
    mascara[allowed_ids] = logits_array[allowed_ids]
    return int(np.argmax(mascara))
```

- `_encode_ids` es solo un envoltorio de `model.encode(...).flatten().tolist()` para no repetir esa cadena de tres llamadas por todo el código, y para asegurar que el resultado es `list[int]` de verdad (por eso el `int(x)` — protege de que algún valor venga como `numpy.int64` en vez de `int` nativo, lo que podría dar problemas raros más adelante al comparar o serializar).
- `masked_argmax` es la pieza central de la "decodificación restringida" que pide el subject: coge el array completo de logits (~150k valores, uno por token del vocabulario), pone `-inf` en todo lo que NO está permitido, y se queda con el índice del valor más alto de lo que sí está permitido. Es la traducción literal a numpy de la idea "en cada paso, ponemos -inf en los tokens no permitidos".

---

## 5. Carga y preparación: `fixed_ids` y `functions_info`

`fixed_ids` tokeniza **una sola vez, al principio**, todas las piezas fijas del JSON (`{`, `}`, `:`, nombres de función, nombres de parámetro...) y las guarda en un diccionario `texto -> tokens`. Esto evita volver a tokenizar "fn_add_numbers" cada vez que aparece — se calcula una vez y se reutiliza.

`functions_info` lee `functions_definition.json`, distingue explícitamente entre "el archivo no existe" (`FileNotFoundError`) y "el archivo existe pero no es JSON válido" (`json.JSONDecodeError`) — el subject pide manejar ambos casos por separado. Cada función se valida con `DefinicionFuncion`; si falla, se descarta esa función concreta con un aviso, pero el programa sigue con las demás.

---

## 6. `build_super_prompt`

Construye el texto que ve el modelo: rol, reglas de salida, lista de funciones disponibles (nombre + descripción + parámetros) y el prompt del usuario al final. No decide nada por sí mismo — es solo el "contexto" sobre el que luego se hace la decodificación restringida. Las reglas del prompt (no inventar valores, extraer frases citadas completas, etc.) ayudan al modelo quando SÍ tiene que decidir algo (en el último recurso de `logit_masking_string`), pero la mayor parte de la precisión ya no depende de que el modelo "entienda" estas reglas, sino de la lógica determinista que ves a continuación.

---

## 7. Extracción determinista — el corazón de esta versión

Estas cinco funciones analizan el **texto plano** del prompt de usuario, sin tocar el modelo, para extraer pistas:

| Función | Qué busca | Ejemplo |
|---|---|---|
| `extraer_frases_entrecomilladas` | Texto entre comillas simples o dobles, en orden de aparición | `'hello'` → `"hello"` |
| `extraer_palabras` | Palabras sueltas alfanuméricas | `"Greet shrek"` → `["Greet", "shrek"]` |
| `extraer_numeros` | Secuencias de dígitos, en orden | `"sum of 2 and 3"` → `["2", "3"]` |
| `extraer_palabra_tras_with` | La palabra justo después de "with"/"con" | `"...with NUMBERS"` → `"NUMBERS"` |
| `inferir_patron_regex` | Si alguna palabra del prompt coincide con `CONCEPTO_A_REGEX` | `"vowels"` → `r"[aeiouAEIOU]"` |

Todas están escritas sin `import re`, recorriendo el texto carácter a carácter — cumple la restricción del subject de no usar ese módulo.

**Detalle importante de `extraer_frases_entrecomilladas`:** procesa el texto de izquierda a derecha manteniendo un único puntero de "¿estoy dentro de una comilla ahora mismo?" (`quote_char`). Esto es mejor que la versión anterior que procesaba primero todas las comillas dobles y luego todas las simples por separado, porque ahora las frases salen **en el mismo orden en que aparecen en el prompt real** — algo esencial para el paso 8, donde se reparten frases a argumentos en orden.

También ignora los apóstrofos de contracciones (`I'm`, `don't`): antes de tratar un `'` como comilla de apertura/cierre, comprueba que no tenga letras pegadas a ambos lados.

---

## 8. `codificar_valor_string` vs `logit_masking_string` — la diferencia clave

Esta es la distinción más importante de todo el código:

- **`codificar_valor_string(model, vocab, texto)`**: el valor **ya se decidió** por lógica determinista (una frase citada, un patrón regex, la palabra tras "with"...). Esta función solo lo convierte a tokens y le pone comillas alrededor. **Cero llamadas al modelo.**
- **`logit_masking_string(vocab, model, init_prompt_ids, candidates_list)`**: se usa solo cuando ninguna pista determinista funcionó (por ejemplo, `"Greet shrek"`, donde no hay comillas ni "with"). Aquí sí se le pregunta al modelo, pero con **una única llamada** a `get_logits_from_input_ids`, no una por token como en las versiones anteriores: se puntúa cada candidato sumando los logits de sus tokens en esa única consulta, y se elige el candidato con la suma más alta.

**Por qué esto importa para la velocidad:** en tu conjunto de 11 prompts, la mayoría de los argumentos string tienen comillas o un "with" — así que `logit_masking_string` casi nunca se ejecuta, y cuando lo hace, es una sola llamada en vez de "una llamada por cada letra del valor" como en versiones anteriores. Ahí está el grueso de la mejora de rendimiento.

**Peaje que pagas por esa velocidad (ver sección 12):** sumar logits de una única consulta para "puntuar" una secuencia completa de varios tokens no es matemáticamente lo mismo que generarla token a token (donde cada token se elige *sabiendo* qué se generó antes). Es una aproximación razonable para desempatar entre palabras candidatas, pero conviene que la pruebes específicamente en el caso que antes fallaba (`"Greet shrek"`) para confirmar que sigue dando `"shrek"` y no algo raro.

---

## 9. `resolver_falla_fn` — red de seguridad para `fn_name`

Si el modelo no logra identificar ninguna función válida (o la decisión determinista de `_escoge_fn` no encuentra coincidencia), esta función entra como último recurso: cuenta cuántas palabras del prompt (de más de 2 letras) aparecen también en el `name` + `description` de cada función, y se queda con la que más coincidencias tiene.

**Por qué es genérico y no hardcodeo:** no compara contra tus 11 prompts de ejemplo ni contra nombres de función concretos — usa el campo `description`, que viene del propio `functions_definition.json`. Si en la evaluación cambian las funciones, esta red de seguridad se adapta sola siempre que las descripciones sigan siendo palabras relacionadas con lo que hace la función.

---

## 10. `loop_prompt_output` — cómo se ensamblan todas las piezas

Aquí es donde todo se junta, en este orden:

1. **Fuerza la estructura fija** del JSON (`{"fn_name": "`) — igual que siempre.
2. **`_escoge_fn()`** decide el nombre de la función (ver sección 12 sobre esto — es la parte que más quiero que revises).
3. **Fuerza el resto de estructura fija** hasta llegar a `"args": {`.
4. **Prepara las pistas UNA VEZ por prompt**, antes de entrar en el bucle de argumentos:
   - `frases_disponibles`: todas las frases citadas, la más larga reservada para el "argumento de texto principal" (el primer argumento string que no se llama `regex`/`pattern`).
   - `numeros_prompt`: todos los números del prompt, en orden, para repartir uno por cada argumento numérico según van apareciendo.
5. **Recorre cada argumento** (`for idx, arg in enumerate(args_fn)`) y, según su tipo, decide el valor:
   - `number` → si queda algún número sin usar de `numeros_prompt`, se codifica directamente (0 llamadas al modelo); si no, cae al `logit_masking_number` de siempre.
   - `boolean` → sin cambios, sigue siendo `logit_masking_boolean`.
   - `string` → sigue la cadena de prioridad de la sección 8: regex → frase principal → siguiente frase disponible → palabra tras "with" → último recurso con el modelo.

Este orden de prioridad es la clave de por qué ahora `regex`, `replacement` y `source_string` deberían resolverse bien en el mismo prompt: cada uno "consume" una pista distinta y no compiten entre sí por el mismo texto.

---

## 11. `main()` — flujo completo

Además de leer/escribir archivos con manejo de errores (ausencia de archivo, JSON inválido, prompt vacío), añade dos pasos nuevos al final de cada iteración:

1. **Fallback de `fn_name`**: si el `fn_name` que salió del JSON generado no está en `dict_functions`, se sustituye por `resolver_falla_fn(...)`. Esto es lo que debería eliminar los `fn_name: null` que viste en la ejecución anterior.
2. **Coerción de tipos**: recorre `args` y convierte a `float` cualquier argumento cuyo tipo declarado sea `"number"` — así garantizas `2.0` en vez de `2`, como pide el ejemplo del subject.

Solo después de estos dos pasos se valida con `ResultadoLlamada` y se añade a `results`.

---

## 12. ⚠️ Puntos a vigilar antes de dar el proyecto por cerrado

Como mentor, esto es lo que más me preocupa de la versión actual — no son errores de estilo, son cosas que podrían afectar a tu 95% de precisión:

### A. `_escoge_fn()` — posible problema con la posición de decisión

```python
posicion = 0
for posicion in range(longitud_max):
    tokens_pos = {t[posicion] for t in fn_names_tokens if len(t) > posicion}
    if len(tokens_pos) > 1:
        break

llm_logits = model.get_logits_from_input_ids(init_prompt_ids)
allowed = [t[posicion] for t in fn_names_tokens if len(t) > posicion]
best_id = masked_argmax(llm_logits, allowed)
```

Esto encuentra la primera posición donde los nombres de función dejan de compartir el mismo token — hasta ahí, es correcto. El problema es lo que viene después: **solo hace UNA llamada a `get_logits_from_input_ids(init_prompt_ids)`**, sin haber añadido antes al contexto los tokens del prefijo compartido (posiciones `0` a `posicion - 1`).

Una llamada a `get_logits_from_input_ids` solo te da la predicción del **siguiente** token inmediato al contexto que le pasas. Si `posicion` es `0`, esto es correcto por casualidad (el "siguiente token" y "la posición de decisión" coinciden). Pero si todas las funciones comparten, por ejemplo, un primer token (`fn` o `fn_`) y solo divergen en el segundo o tercer token, `posicion` será mayor que `0` — y en ese caso estás pidiéndole al modelo que "adivine" el token de la posición 2 usando los logits de la posición 0, lo cual no es válido.

**Cómo comprobarlo tú misma:** añade un `print(posicion)` justo antes de la llamada a `get_logits_from_input_ids`. Si siempre te sale `0` con tus 5 funciones actuales, no pasa nada ahora mismo — pero es un riesgo latente en cuanto la evaluación meta funciones con nombres que compartan más prefijo (`fn_get_square_root` y `fn_get_cube_root`, por ejemplo).

**Fix si hace falta:** antes de la llamada a `get_logits_from_input_ids`, añade a una copia del contexto los primeros `posicion` tokens (que son iguales en todas las funciones, por ejemplo `fn_names_tokens[0][:posicion]`), y pásale esa copia extendida en vez de `init_prompt_ids` directamente.

### B. El fallback de `_escoge_fn` puede devolver un valor "basura" que rompe el programa

```python
fn = resolver_falla_fn(user_prompt, dict_functions)
if fn is not None:
    return dict_fixed_chars[fn]
return dict_fixed_chars["fn_name"]  # placeholder, discarded later
```

Si `resolver_falla_fn` tampoco encuentra ninguna coincidencia (devuelve `None`), la función devuelve los tokens de la palabra literal `"fn_name"` como si fuera un nombre de función válido. Más adelante:

```python
fn = model.decode(fn_tokens)
args_fn = [arg for arg in dict_functions[fn]["parameters"]]
```

Si `fn` termina siendo la cadena `"fn_name"`, `dict_functions["fn_name"]` lanza un `KeyError` y el programa se cae para ese prompt (no hay ningún `try/except` alrededor de esta parte). El comentario dice "discarded later", pero en realidad no hay ningún sitio que lo descarte antes del `KeyError`.

**Es un caso extremo** (solo ocurre si ni el modelo ni el fallback por descripción encuentran nada, lo cual con vuestras 5 funciones y 11 prompts probablemente nunca pase) — pero como el subject pide explícitamente probar "prompts ambiguos" como caso límite, te recomendaría envolver esa parte en un `try/except KeyError` en `main()`, o hacer que `_escoge_fn` devuelva `None` en vez de un placeholder falso, y que `loop_prompt_output` lo maneje explícitamente.

### C. `logit_masking_string` (último recurso) puntúa candidatos con una sola consulta

Como se explica en la sección 8: sumar los logits de varios tokens obtenidos de una única llamada no es lo mismo que generarlos uno a uno con el contexto actualizado. Es un riesgo concreto justo para el caso que más os costó (`"Greet shrek"`). Te recomiendo que, al probar, mires específicamente ese resultado con lupa.

### D. `extraer_numeros` asume que el orden textual coincide con el orden de los parámetros

Para `"What is the sum of 2 and 3?"` con parámetros `a, b` en ese orden, funciona porque el texto también dice "2... 3" en ese orden. Es una heurística razonable para vuestros casos actuales, pero si la evaluación mete un prompt donde el orden textual y el orden de los parámetros no coincidan, fallaría. Merece la pena que lo tengas en mente como limitación conocida, no como bug a arreglar necesariamente.

---

## 13. Checklist para validar antes de entregar

1. Ejecuta con los 11 prompts actuales y compara cada resultado contra el esperado (con los nombres reales de función/parámetro).
2. Mide el tiempo total y, si sigue por encima de 5 min, añade un `print` con el número total de llamadas a `get_logits_from_input_ids` para saber cuántas quedan y dónde.
3. Prueba a mano un caso con nombres de función que compartan prefijo largo (para forzar el punto A) y confirma que `fn_name` sigue saliendo bien.
4. Prueba un prompt que no coincida con ninguna función a propósito, para confirmar que no se cae por el punto B.
5. Revisa específicamente el resultado de `"Greet shrek"` y de cualquier prompt sin comillas ni "with".
