from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError
import json
import numpy as np


# ---------------------------------------------------------------------------
# Modelos de pydantic
# ---------------------------------------------------------------------------

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


CONCEPTO_A_REGEX = {
    "number": r"\d+", "numbers": r"\d+", "digit": r"\d+", "digits": r"\d+",
    "vowel": r"[aeiouAEIOU]", "vowels": r"[aeiouAEIOU]",
    "consonant": r"[^aeiouAEIOU\s]", "consonants": r"[^aeiouAEIOU\s]",
    "letter": r"[a-zA-Z]", "letters": r"[a-zA-Z]",
    "whitespace": r"\s+", "space": r"\s+", "spaces": r"\s+",
    "punctuation": r"[^\w\s]",
    "uppercase": r"[A-Z]", "lowercase": r"[a-z]",
}


def masked_argmax(llm_logits, allowed_ids: list[int]) -> int:
    """Devuelve el id permitido con el logit más alto, con numpy."""
    logits_array = np.asarray(llm_logits)
    mascara = np.full(logits_array.shape, -np.inf)
    mascara[allowed_ids] = logits_array[allowed_ids]
    return int(np.argmax(mascara))


def fixed_ids(model: Small_LLM_Model, dict_functions: dict) -> dict:
    dict_fixed = {}
    chars_fixed = ["{", "}", ":", ",", " ", "fn_name", "args", "\"", "\n"]

    for fixed in chars_fixed:
        dict_fixed[fixed] = model.encode(fixed).flatten().tolist()

    for fun in dict_functions:
        dict_fixed[fun] = model.encode(fun).flatten().tolist()
        for params in dict_functions[fun].get("parameters"):
            dict_fixed[params] = model.encode(params).flatten().tolist()

    return dict_fixed


def functions_info() -> dict:
    dict_functions = {}

    try:
        with open("data/input/functions_definition.json") as f:
            f_content = json.load(f)
    except FileNotFoundError:
        print("Aviso: no se encontró data/input/functions_definition.json")
        return dict_functions
    except json.JSONDecodeError as e:
        print(f"Aviso: functions_definition.json no es un JSON válido: {e}")
        return dict_functions

    for dictionary in f_content:
        try:
            funcion_validada = DefinicionFuncion(**dictionary)
        except ValidationError as e:
            print(f"Aviso: función descartada por formato inválido: {e}")
            continue

        dict_functions[funcion_validada.name] = {
            "parameters": dictionary["parameters"],
            "description": funcion_validada.description,
            "returns": dictionary["returns"],
        }

    return dict_functions


def build_super_prompt(dict_functions: dict, input_call: str) -> str:
    template_intro = """
You are a function-calling assistant. Your task is to
analyze a user request and respond with a valid JSON object that specifies
which available function to call and with what arguments.\n\n"""

    template_rules = """Rules:
- Output ONLY the JSON object. Do not add explanations, greetings or
any other text before or after it.
- The JSON must follow this exact structure:
{"fn_name": "<function name>", "args": {<argument name>: <value>, ...}}
- "fn_name" must be one of the available functions listed below.
- "args" must contain every required argument for that function,
and only those, with the correct type (number, string, boolean).
- Extract every argument value from the exact words or numbers written in
the User request below. Never invent values, never repeat the instruction
words, and never copy the function name or the JSON labels.
- If an argument is a quoted phrase in the request, take the whole phrase as
the value. Strip any surrounding quotes or spaces.
- When a function has more than one value to fill, give each argument the
value the User mentions that matches its labelled role, in the order the
User refers to them.\n\n"""

    template_functions = """Available functions:\n"""
    for func in dict_functions:
        dict_parameters = dict_functions[func].get("parameters")
        list_parameters = []

        for param in dict_parameters:
            temp = f"{param} ({dict_parameters[param].get('type')})"
            list_parameters.append(temp)

        func_description = f"""- {func}: {dict_functions[func].get(
         "description")}\nParameters: {", ".join(list_parameters)}\n"""

        template_functions += func_description

    prompt = (template_intro
              + template_rules
              + template_functions
              + f"\nUser: {input_call}\nOutput: ")

    return prompt


# ---------------------------------------------------------------------------
# Extracción determinista de pistas del texto (sin `re`)
# ---------------------------------------------------------------------------

def extraer_frases_entrecomilladas(texto: str) -> list[str]:
    """Extrae las frases entre comillas (simples o dobles) EN EL ORDEN en
    que aparecen en el texto (a diferencia de vuestra versión anterior, que
    procesaba primero todas las dobles y luego todas las simples, perdiendo
    el orden real de aparición).

    Ignora los apóstrofos de contracciones (I'm, don't) tratándolos como
    parte de la palabra, no como comilla de apertura/cierre.
    """
    frases: list[str] = []
    i, n = 0, len(texto)
    quote_char = None
    current: list[str] = []

    while i < n:
        char = texto[i]

        if quote_char is None:
            if char == '"':
                quote_char = '"'
                current = []
            elif char == "'":
                es_contraccion = (0 < i < n - 1
                                  and texto[i - 1].isalpha()
                                  and texto[i + 1].isalpha())
                if not es_contraccion:
                    quote_char = "'"
                    current = []
        else:
            if char == quote_char:
                frase = "".join(current).strip()
                if frase:
                    frases.append(frase)
                quote_char = None
            else:
                current.append(char)

        i += 1

    return frases


def extraer_palabras(texto: str) -> list[str]:
    """Palabras sueltas del texto (alfanuméricas), sin usar `re`."""
    palabras: list[str] = []
    actual: list[str] = []
    for ch in texto:
        if ch.isalnum() or ch == "_":
            actual.append(ch)
        else:
            if actual:
                palabras.append("".join(actual))
                actual = []
    if actual:
        palabras.append("".join(actual))
    return palabras


def extraer_palabra_tras_with(texto: str) -> str | None:
    """Busca 'with'/'con' y devuelve la palabra siguiente.

    Heurística genérica del patrón "replace/substitute X with Y", que es
    justo el que usa la función fn_substitute_string_with_regex.
    """
    palabras = extraer_palabras(texto)
    for idx, palabra in enumerate(palabras):
        if palabra.lower() in ("with", "con") and idx + 1 < len(palabras):
            return palabras[idx + 1]
    return None


def inferir_patron_regex(user_prompt: str) -> str | None:
    """Busca una palabra-concepto conocida (numbers, vowels...) y devuelve
    el patrón regex asociado. Tabla cerrada: si no reconoce el concepto,
    devuelve None y el argumento cae al resto de heurísticas."""
    for palabra in extraer_palabras(user_prompt.lower()):
        if palabra in CONCEPTO_A_REGEX:
            return CONCEPTO_A_REGEX[palabra]
    return None


def codificar_valor_string(model: Small_LLM_Model, vocab: dict,
                           texto: str) -> list[int]:
    """Codifica un valor string YA DECIDIDO (no elegido por el modelo).

    Duplicamos las barras invertidas antes de codificar: un patrón como
    "\\d+" debe aparecer en el JSON final como \\\\d+ (dos caracteres:
    barra invertida + d) para que json.loads lo interprete como el
    carácter '\\' seguido de 'd', y no como un escape JSON inválido.
    """
    texto_seguro = texto.replace("\\", "\\\\")
    contenido_ids = model.encode(texto_seguro).flatten().tolist()
    return [vocab["\""]] + contenido_ids + [vocab["\""]]


def logit_masking_string(vocab: dict, model: Small_LLM_Model,
                         init_prompt_ids: list[int],
                         candidates_list: list[str],
                         max_chars: int = 32) -> list[int]:
    """Último recurso: cuando no hay comillas, ni concepto regex, ni "with"
    que den la pista. El modelo elige entre las PALABRAS SUELTAS del
    prompt (nunca el texto completo, para no arrastrar comillas ni
    inflar el número de pasos).
    """
    context = init_prompt_ids.copy()
    next_id = [vocab["\""]]

    possible_tokens = [model.encode(c).flatten().tolist()
                       for c in candidates_list]
    possible_tokens = [t for t in possible_tokens if t]

    remaining_opt = possible_tokens.copy()
    i = 0
    while remaining_opt and i < max_chars:
        llm_logits = model.get_logits_from_input_ids(context)
        allowed = [t[i] for t in remaining_opt if len(t) > i]
        if not allowed:
            break
        best_id = masked_argmax(llm_logits, allowed)
        next_id.append(best_id)
        context.append(best_id)
        remaining_opt = [t for t in remaining_opt
                         if len(t) > i and t[i] == best_id]
        i += 1
        if len(remaining_opt) == 1 and len(remaining_opt[0]) == i:
            break

    next_id.append(vocab["\""])
    return next_id


def logit_masking_number(vocab: dict, model: Small_LLM_Model,
                         init_prompt_ids: list[int]) -> list[int]:
    context = init_prompt_ids.copy()
    numbers = "0123456789"
    ids_numbers = [vocab[num] for num in numbers]
    ids_parada = [vocab[","], vocab["}"]]
    candidates = ids_numbers + ids_parada

    next_id = []
    flag = True
    while flag:
        llm_logits = model.get_logits_from_input_ids(context)
        best_id = masked_argmax(llm_logits, candidates)

        if best_id in ids_parada:
            flag = False
        else:
            next_id.append(best_id)
            context.append(best_id)

    return next_id


def logit_masking_boolean(vocab: dict, model: Small_LLM_Model,
                          init_prompt_ids: list[int]) -> list[int]:
    context = init_prompt_ids.copy()
    ids_booleans = [vocab["true"], vocab["false"]]
    ids_parada = [vocab[","], vocab["}"]]
    candidates = ids_booleans + ids_parada

    next_id = []
    flag = True
    while flag:
        llm_logits = model.get_logits_from_input_ids(context)
        best_id = masked_argmax(llm_logits, candidates)

        if best_id in ids_parada:
            flag = False
        else:
            next_id.append(best_id)
            context.append(best_id)

    return next_id


def loop_prompt_output(input: str, model: Small_LLM_Model,
                       dict_fixed_chars: dict,
                       dict_functions: dict,
                       user_prompt: str) -> list[int]:
    init_prompt_ids: list[int] = model.encode(input).flatten().tolist()

    init_prompt_ids.extend(dict_fixed_chars["{"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars["fn_name"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[":"])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["\""])

    fn_names_tokens: list[list[int]] = []
    for name in dict_functions:
        fn_names_tokens.append(dict_fixed_chars[name])

    i = 0
    llm_ids = []
    temp_prompt = init_prompt_ids.copy()
    while len(fn_names_tokens) > 1:
        llm_logits = model.get_logits_from_input_ids(temp_prompt)
        allowed = [ids[i] for ids in fn_names_tokens if len(ids) > i]
        next_id = masked_argmax(llm_logits, allowed)

        llm_ids.extend([next_id])

        fn_ids = fn_names_tokens.copy()
        for ids in fn_ids:
            if len(ids) > i + 1 and ids[i] == llm_ids[i]:
                continue
            else:
                fn_names_tokens.remove(ids)

        temp_prompt.extend([next_id])
        i += 1

    init_prompt_ids.extend(fn_names_tokens[0])

    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[","])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars["args"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[":"])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["{"])

    fn = model.decode(fn_names_tokens[0])
    args_fn = [arg for arg in dict_functions[fn]["parameters"]]

    path_vocab = model.get_path_to_vocab_file()
    with open(path_vocab) as file:
        vocab = json.load(file)

    # --- Preparación de pistas deterministas, UNA VEZ por prompt ---
    string_args = [a for a in args_fn
                   if dict_functions[fn]["parameters"][a]["type"] == "string"]
    # El argumento "principal" es el primer string que NO parece un
    # nombre de patrón (regex/pattern); ese es el que se queda con la
    # frase citada más larga (normalmente el texto sobre el que se opera).
    args_texto_principal = [a for a in string_args
                            if "regex" not in a and "pattern" not in a]
    primer_arg_texto = args_texto_principal[0] if args_texto_principal else None

    frases_disponibles = extraer_frases_entrecomilladas(user_prompt)
    frase_principal = None
    if frases_disponibles and primer_arg_texto is not None:
        frase_principal = max(frases_disponibles, key=len)
        frases_disponibles.remove(frase_principal)

    with_usado = False

    n_args = len(args_fn)
    for idx, arg in enumerate(args_fn):
        arg_type = dict_functions[fn]["parameters"][arg]["type"]
        is_last = (idx == n_args - 1)

        init_prompt_ids.extend(dict_fixed_chars["\""])
        init_prompt_ids.extend(dict_fixed_chars[arg])
        init_prompt_ids.extend(dict_fixed_chars["\""])
        init_prompt_ids.extend(dict_fixed_chars[":"])
        init_prompt_ids.extend(dict_fixed_chars[" "])

        if arg_type == "number":
            value_ids = logit_masking_number(vocab, model, init_prompt_ids)
        elif arg_type == "boolean":
            value_ids = logit_masking_boolean(vocab, model, init_prompt_ids)
        else:
            # Orden de prioridad para decidir el valor de un argumento string:
            patron = None
            if "regex" in arg or "pattern" in arg:
                patron = inferir_patron_regex(user_prompt)

            if patron is not None:
                # 1. Concepto regex reconocido (numbers, vowels...)
                value_ids = codificar_valor_string(model, vocab, patron)
            elif arg == primer_arg_texto and frase_principal is not None:
                # 2. Frase citada más larga -> argumento de texto principal
                value_ids = codificar_valor_string(model, vocab, frase_principal)
            elif frases_disponibles:
                # 3. Siguiente frase citada disponible, en orden de aparición
                siguiente_frase = frases_disponibles.pop(0)
                value_ids = codificar_valor_string(model, vocab, siguiente_frase)
            elif not with_usado and extraer_palabra_tras_with(user_prompt):
                # 4. Palabra tras "with"/"con" (patrón "replace X with Y")
                with_usado = True
                palabra = extraer_palabra_tras_with(user_prompt)
                value_ids = codificar_valor_string(model, vocab, palabra)
            else:
                # 5. Último recurso: el modelo elige entre las palabras
                #    sueltas del prompt (nunca el texto completo).
                candidates_list = extraer_palabras(user_prompt)
                value_ids = logit_masking_string(vocab, model,
                                                 init_prompt_ids,
                                                 candidates_list)

        init_prompt_ids.extend(value_ids)

        if is_last:
            init_prompt_ids.extend(dict_fixed_chars["}"])
        else:
            init_prompt_ids.extend(dict_fixed_chars[","])
            init_prompt_ids.extend(dict_fixed_chars[" "])

    init_prompt_ids.extend(dict_fixed_chars["}"])
    return init_prompt_ids


def main() -> None:
    model = Small_LLM_Model()
    dict_functions = functions_info()

    if not dict_functions:
        print("Aviso: no hay funciones disponibles, no se puede continuar.")
        with open("output/function_calling_results.json", "w",
                  encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return

    dict_fixed_chars = fixed_ids(model, dict_functions)

    try:
        with open("data/input/function_calling_tests.json") as file:
            tests = json.load(file)
    except FileNotFoundError:
        print("Aviso: no se encontró data/input/function_calling_tests.json")
        tests = []
    except json.JSONDecodeError as e:
        print(f"Aviso: function_calling_tests.json no es un JSON válido: {e}")
        tests = []

    results = []
    for test in tests:
        user_prompt = test.get("prompt")
        if not user_prompt:
            print(f"Aviso: entrada sin 'prompt', se omite: {test}")
            continue

        super_prompt = build_super_prompt(dict_functions, user_prompt)
        final_prompt_ids = loop_prompt_output(super_prompt, model,
                                              dict_fixed_chars,
                                              dict_functions, user_prompt)
        final_output = model.decode(final_prompt_ids)

        try:
            result_json = final_output.split("Output: ", 1)[1]
            result = json.loads(result_json)
        except (IndexError, json.JSONDecodeError):
            result = {"fn_name": None, "args": {}}

        fn_name = result.get("fn_name")
        args = result.get("args") or {}

        # Fuerza los tipos numéricos a float, tal y como pide el subject
        # (ej. "a": 2.0, no "a": 2), usando la definición de la función.
        if fn_name in dict_functions:
            parametros = dict_functions[fn_name]["parameters"]
            for nombre_arg, valor in list(args.items()):
                tipo = parametros.get(nombre_arg, {}).get("type")
                if tipo == "number" and isinstance(valor, (int, float)):
                    args[nombre_arg] = float(valor)

        try:
            resultado_validado = ResultadoLlamada(
                prompt=user_prompt, fn_name=fn_name, args=args)
            results.append(resultado_validado.model_dump())
        except ValidationError as e:
            print(f"Aviso: resultado inválido para '{user_prompt}': {e}")
            results.append({"prompt": user_prompt, "fn_name": None, "args": {}})

    with open("output/function_calling_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()