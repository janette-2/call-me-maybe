"""Moulinette estricta para call_me_maybe.

Organizacion:

1. PRUEBAS RAPIDAS (sin modelo): funciones deterministas de extraccion de
   texto. Ampliadas con casos adversariales: comillas sin cerrar, "with"
   repetido, mayusculas, "with" dentro de una frase citada, numeros
   negativos/decimales, cadenas vacias.

2. PRUEBAS COMPLETAS (con modelo), divididas en dos niveles de severidad:
   - CRITICO: casos que verifican una frase EXPLICITA del subject (JSON
     valido, tipos correctos, archivos ausentes/invalidos, limite de
     tiempo). Si uno de estos falla, el proyecto NO deberia entregarse asi.
   - ROBUSTEZ: casos adversariales que el subject no pide literalmente pero
     que "ir a putear" revela: booleanos, numeros negativos/decimales, mas
     numeros que argumentos, "ultimo recurso" sin comillas, funciones
     duplicadas, prompts en mayusculas/con acentos, comillas anidadas mal
     puestas, consistencia entre ejecuciones repetidas.

ADAPTABILIDAD: si existe un archivo `escenarios_adicionales.json` en la raiz
del proyecto (lista de objetos {"nombre", "funciones", "tests"}), se cargan
automaticamente como casos extra de nivel ROBUSTEZ, sin tocar este script.
Sirve para cuando el subject cambie de funciones/prompts en la evaluacion:
en vez de editar el .py, escribes un JSON con el escenario nuevo.

COMO USARLO:
    1. Copia este archivo en la raiz de tu proyecto.
    2. Ajusta MODULO_A_PROBAR con el nombre real de tu modulo.
    3. Ejecuta:
        uv run python moulinette_call_me_maybe.py                (todo)
        uv run python moulinette_call_me_maybe.py --rapidas       (solo 1)
        uv run python moulinette_call_me_maybe.py --completas     (solo 2)
        uv run python moulinette_call_me_maybe.py --solo-critico  (solo lo
                                                    que exige el subject)
"""

import argparse
import importlib
import json
import os
import shutil
import time

MODULO_A_PROBAR = "src.main_best"  # <-- AJUSTA ESTO a tu modulo real

DIR_INPUT = "data/input"
DIR_OUTPUT = "output"
ARCHIVO_FUNCIONES = os.path.join(DIR_INPUT, "functions_definition.json")
ARCHIVO_TESTS = os.path.join(DIR_INPUT, "function_calling_tests.json")
ARCHIVO_RESULTADOS = os.path.join(DIR_OUTPUT, "function_calling_results.json")
ARCHIVO_ESCENARIOS_EXTRA = "escenarios_adicionales.json"

DIR_BACKUP = ".moulinette_backup"
LIMITE_SEGUNDOS = 5 * 60


# ---------------------------------------------------------------------------
# PARTE 1: pruebas rapidas de las funciones deterministas (sin modelo)
# ---------------------------------------------------------------------------

def pruebas_rapidas(modulo) -> list[tuple[str, bool, str]]:
    """Prueba las funciones de extraccion de texto sin llamar al modelo."""
    resultados: list[tuple[str, bool, str]] = []

    def check(nombre: str, obtenido, esperado) -> None:
        ok = obtenido == esperado
        detalle = "" if ok else f"esperado={esperado!r} obtenido={obtenido!r}"
        resultados.append((nombre, ok, detalle))

    def check_no_revienta(nombre: str, fn, *args) -> None:
        """Solo comprueba que la llamada no lance una excepcion."""
        try:
            fn(*args)
            resultados.append((nombre, True, ""))
        except Exception as e:  # noqa: BLE001
            resultados.append(
                (nombre, False, f"lanzo {type(e).__name__}: {e}"))

    # --- extraer_frases_entrecomilladas ---
    check(
        "frases: comillas simples basicas",
        modulo.extraer_frases_entrecomilladas("Reverse the string 'hello'"),
        ["hello"],
    )
    check(
        "frases: contraccion (I'm) no cuenta como comilla",
        modulo.extraer_frases_entrecomilladas("Hello 34 I'm 233 years old"),
        [],
    )
    check(
        "frases: varias frases, en orden de aparicion",
        modulo.extraer_frases_entrecomilladas(
            "Substitute the word 'cat' with 'dog' in 'The cat sat'"),
        ["cat", "dog", "The cat sat"],
    )
    check(
        "frases: comillas dobles",
        modulo.extraer_frases_entrecomilladas(
            'Replace all numbers in "Hello 34" with NUMBERS'),
        ["Hello 34"],
    )
    check(
        "frases: sin comillas -> lista vacia",
        modulo.extraer_frases_entrecomilladas("Greet shrek"),
        [],
    )
    check_no_revienta(
        "frases: comilla sin cerrar no revienta",
        modulo.extraer_frases_entrecomilladas, "Reverse the string 'hello")
    check_no_revienta(
        "frases: comillas anidadas/mal puestas no revientan",
        modulo.extraer_frases_entrecomilladas,
        "Replace 'a \"b' c\" with 'd'")
    check(
        "frases: 'with' DENTRO de una frase citada no se confunde con separador",
        modulo.extraer_frases_entrecomilladas("Say 'goodbye with style'"),
        ["goodbye with style"],
    )
    check_no_revienta(
        "frases: cadena vacia no revienta",
        modulo.extraer_frases_entrecomilladas, "")

    # --- extraer_numeros ---
    check(
        "numeros: dos numeros en orden",
        modulo.extraer_numeros("What is the sum of 265 and 345?"),
        ["265", "345"],
    )
    check(
        "numeros: sin numeros -> lista vacia",
        modulo.extraer_numeros("Greet shrek"),
        [],
    )
    check(
        "numeros: numero muy grande",
        modulo.extraer_numeros("What is the sum of 999999999 and 1?"),
        ["999999999", "1"],
    )
    check(
        "numeros: mas numeros en el texto que argumentos (3 numeros)",
        modulo.extraer_numeros("Add 1, 2 and 3 together"),
        ["1", "2", "3"],
    )
    check_no_revienta(
        "numeros: cadena vacia no revienta",
        modulo.extraer_numeros, "")

    # --- extraer_palabra_tras_with ---
    check(
        "with: caso normal",
        modulo.extraer_palabra_tras_with(
            "Replace all numbers in \"Hello 34\" with NUMBERS"),
        "NUMBERS",
    )
    check(
        "with: no aparece -> None",
        modulo.extraer_palabra_tras_with("Greet shrek"),
        None,
    )
    check(
        "with: es la ultima palabra -> None (no revienta)",
        modulo.extraer_palabra_tras_with("do something with"),
        None,
    )
    check(
        "with: 'With' en mayuscula al inicio de frase tambien cuenta",
        modulo.extraer_palabra_tras_with("With NUMBERS replace everything"),
        "NUMBERS",
    )
    check(
        "with: dos apariciones de 'with', se queda con la primera",
        modulo.extraer_palabra_tras_with(
            "Replace X with Y and also with Z"),
        "Y",
    )

    # --- inferir_patron_regex ---
    check(
        "regex: concepto reconocido (numbers)",
        modulo.inferir_patron_regex(
            "Replace all numbers in \"Hello 34\" with NUMBERS"),
        r"\d+",
    )
    check(
        "regex: concepto reconocido (vowels)",
        modulo.inferir_patron_regex(
            "Replace all vowels in 'Programming is fun' with asterisks"),
        r"[aeiouAEIOU]",
    )
    check(
        "regex: concepto NO reconocido -> None",
        modulo.inferir_patron_regex("Substitute the word 'cat' with 'dog'"),
        None,
    )
    check(
        "regex: concepto en mayusculas tambien se reconoce",
        modulo.inferir_patron_regex("Replace all NUMBERS in this text"),
        r"\d+",
    )

    # --- extraer_palabras ---
    check(
        "palabras: separa por puntuacion",
        modulo.extraer_palabras("Greet shrek!"),
        ["Greet", "shrek"],
    )
    check(
        "palabras: cadena vacia -> lista vacia",
        modulo.extraer_palabras(""),
        [],
    )
    check(
        "palabras: acentos/unicode no revientan",
        modulo.extraer_palabras("Saluda a José-María"),
        ["Saluda", "a", "José", "María"],
    )

    return resultados


# ---------------------------------------------------------------------------
# PARTE 2: pruebas completas, ejecutando main() de verdad
# ---------------------------------------------------------------------------

FUNCIONES_BASE = [
    {"name": "fn_add_numbers", "description": "Add two numbers together and return their sum.",
     "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
     "returns": {"type": "number"}},
    {"name": "fn_greet", "description": "Generate a greeting message for a person by name.",
     "parameters": {"name": {"type": "string"}},
     "returns": {"type": "string"}},
    {"name": "fn_reverse_string", "description": "Reverse a string and return the reversed result.",
     "parameters": {"s": {"type": "string"}},
     "returns": {"type": "string"}},
    {"name": "fn_get_square_root", "description": "Calculate the square root of a number.",
     "parameters": {"a": {"type": "number"}},
     "returns": {"type": "number"}},
    {"name": "fn_substitute_string_with_regex",
     "description": "Replace all occurrences matching a regex pattern in a string.",
     "parameters": {"source_string": {"type": "string"}, "regex": {"type": "string"},
                    "replacement": {"type": "string"}},
     "returns": {"type": "string"}},
]

TESTS_BASE = [
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "What is the sum of 265 and 345?"},
    {"prompt": "Greet shrek"},
    {"prompt": "Greet john"},
    {"prompt": "Reverse the string 'hello'"},
    {"prompt": "Reverse the string 'world'"},
    {"prompt": "What is the square root of 16?"},
    {"prompt": "Calculate the square root of 144"},
    {"prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"},
    {"prompt": "Replace all vowels in 'Programming is fun' with asterisks"},
    {"prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"},
]

FUNCIONES_PREFIJO_COMPARTIDO = [
    {"name": "fn_get_square_root", "description": "Calculate the square root of a number.",
     "parameters": {"a": {"type": "number"}},
     "returns": {"type": "number"}},
    {"name": "fn_get_cube_root", "description": "Calculate the cube root of a number.",
     "parameters": {"a": {"type": "number"}},
     "returns": {"type": "number"}},
]

FUNCIONES_CON_BOOLEANO = [
    {"name": "fn_is_positive", "description": "Check whether a number is positive.",
     "parameters": {"n": {"type": "number"}, "strict": {"type": "boolean"}},
     "returns": {"type": "boolean"}},
]

FUNCIONES_DUPLICADAS = FUNCIONES_BASE + [
    {"name": "fn_add_numbers", "description": "A duplicate of fn_add_numbers.",
     "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
     "returns": {"type": "number"}},
]

# Cada caso tiene una "severidad": "critico" (verifica algo que el subject
# exige explicitamente) o "robustez" (adversarial, no exigido literalmente
# pero conviene saber como se comporta).
CASOS: list[dict] = [
    # --- CRITICO: lo que el subject pide explicitamente ---
    {
        "nombre": "[CRITICO] caso base (11 prompts, 5 funciones)",
        "severidad": "critico",
        "funciones": FUNCIONES_BASE, "tests": TESTS_BASE,
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] archivo de funciones ausente",
        "severidad": "critico",
        "funciones": None, "tests": TESTS_BASE,
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": True, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] JSON de funciones invalido (sintaxis rota)",
        "severidad": "critico",
        "funciones": None, "tests": TESTS_BASE,
        "funciones_texto": '[{"name": "fn_add_numbers", "parameters": ',
        "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] archivo de tests ausente",
        "severidad": "critico",
        "funciones": FUNCIONES_BASE, "tests": None,
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": True,
    },
    {
        "nombre": "[CRITICO] JSON de tests invalido (sintaxis rota)",
        "severidad": "critico",
        "funciones": FUNCIONES_BASE, "tests": None,
        "funciones_texto": None,
        "tests_texto": '[{"prompt": "Greet shrek"',
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] lista de tests vacia",
        "severidad": "critico",
        "funciones": FUNCIONES_BASE, "tests": [],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] entrada de test sin la clave 'prompt'",
        "severidad": "critico",
        "funciones": FUNCIONES_BASE,
        "tests": [{"pregunta": "esto no tiene la clave prompt"},
                  {"prompt": "Greet john"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] prompt vacio",
        "severidad": "critico",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": ""}, {"prompt": "Greet john"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[CRITICO] funciones_definition.json vacio ([])",
        "severidad": "critico",
        "funciones": [], "tests": TESTS_BASE,
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },

    # --- ROBUSTEZ: a putear ---
    {
        "nombre": "[ROBUSTEZ] funciones con prefijo de nombre compartido",
        "severidad": "robustez",
        "funciones": FUNCIONES_PREFIJO_COMPARTIDO,
        "tests": [{"prompt": "What is the square root of 81?"},
                  {"prompt": "What is the cube root of 27?"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] prompt ambiguo, sin funcion relacionada",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "What is the current phase of the moon?"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] caracteres especiales y unicode en la cadena",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "Reverse the string 'a@b#c!'"},
                  {"prompt": "Greet José-María"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] numeros negativos y decimales",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "What is the sum of -5 and 3?"},
                  {"prompt": "What is the sum of 2.5 and 1.5?"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] mas numeros en el texto que argumentos numericos",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "The sum of 2 and 3 was calculated in year 2024"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] argumento string sin comillas ni 'with' (ultimo recurso)",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "Greet shrek"}, {"prompt": "Greet the wizard Gandalf"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] funcion con argumento booleano",
        "severidad": "robustez",
        "funciones": FUNCIONES_CON_BOOLEANO,
        "tests": [{"prompt": "Check if 7 is positive, strictly"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] funciones con nombre duplicado en la definicion",
        "severidad": "robustez",
        "funciones": FUNCIONES_DUPLICADAS,
        "tests": [{"prompt": "What is the sum of 2 and 3?"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] prompt en mayusculas completas",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "GREET SHREK"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] prompt muy largo",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "Reverse the string '" + ("supercali" * 20) + "'"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
    {
        "nombre": "[ROBUSTEZ] mas de una frase citada para un solo argumento string",
        "severidad": "robustez",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "Reverse the string 'hello' not 'world'"}],
        "funciones_texto": None, "tests_texto": None,
        "borrar_funciones": False, "borrar_tests": False,
    },
]


def cargar_escenarios_adicionales() -> list[dict]:
    """Carga escenarios extra desde ARCHIVO_ESCENARIOS_EXTRA si existe.

    Formato esperado: lista de objetos con "nombre", "funciones", "tests".
    Se marcan siempre como severidad "robustez" (son casos que anadiste tu,
    no exigencias literales del subject).
    """
    if not os.path.exists(ARCHIVO_ESCENARIOS_EXTRA):
        return []

    try:
        with open(ARCHIVO_ESCENARIOS_EXTRA, encoding="utf-8") as f:
            extra = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Aviso: {ARCHIVO_ESCENARIOS_EXTRA} no es JSON valido: {e}")
        return []

    casos_extra = []
    for item in extra:
        casos_extra.append({
            "nombre": f"[ROBUSTEZ][extra] {item.get('nombre', 'sin nombre')}",
            "severidad": "robustez",
            "funciones": item.get("funciones"),
            "tests": item.get("tests"),
            "funciones_texto": None, "tests_texto": None,
            "borrar_funciones": False, "borrar_tests": False,
        })
    return casos_extra


def preparar_archivos(caso: dict) -> None:
    os.makedirs(DIR_INPUT, exist_ok=True)

    if caso["borrar_funciones"]:
        if os.path.exists(ARCHIVO_FUNCIONES):
            os.remove(ARCHIVO_FUNCIONES)
    elif caso["funciones_texto"] is not None:
        with open(ARCHIVO_FUNCIONES, "w", encoding="utf-8") as f:
            f.write(caso["funciones_texto"])
    else:
        with open(ARCHIVO_FUNCIONES, "w", encoding="utf-8") as f:
            json.dump(caso["funciones"], f, indent=2)

    if caso["borrar_tests"]:
        if os.path.exists(ARCHIVO_TESTS):
            os.remove(ARCHIVO_TESTS)
    elif caso["tests_texto"] is not None:
        with open(ARCHIVO_TESTS, "w", encoding="utf-8") as f:
            f.write(caso["tests_texto"])
    else:
        with open(ARCHIVO_TESTS, "w", encoding="utf-8") as f:
            json.dump(caso["tests"], f, indent=2)

    if os.path.exists(ARCHIVO_RESULTADOS):
        os.remove(ARCHIVO_RESULTADOS)


def validar_resultado(caso: dict, resultados, duracion: float) -> list[str]:
    problemas: list[str] = []

    if not isinstance(resultados, list):
        problemas.append("la salida no es una lista JSON")
        return problemas

    nombres_funciones_validos = set()
    if caso["funciones"]:
        nombres_funciones_validos = {f["name"] for f in caso["funciones"]}

    for item in resultados:
        if not isinstance(item, dict):
            problemas.append(f"un elemento de la lista no es un objeto: {item!r}")
            continue

        claves = set(item.keys())
        if claves != {"prompt", "fn_name", "args"}:
            problemas.append(
                f"claves incorrectas en {item!r} (esperado prompt/fn_name/args)")

        if item.get("fn_name") is not None:
            if item["fn_name"] not in nombres_funciones_validos:
                problemas.append(
                    f"fn_name={item['fn_name']!r} no esta entre las funciones "
                    f"disponibles {sorted(nombres_funciones_validos)}")

        if "args" in item and not isinstance(item["args"], dict):
            problemas.append(f"args no es un diccionario en {item!r}")

        if item.get("fn_name") in nombres_funciones_validos and caso["funciones"]:
            definicion = next(
                f for f in caso["funciones"] if f["name"] == item["fn_name"])
            for nombre_arg, valor in item.get("args", {}).items():
                tipo_esperado = definicion["parameters"].get(
                    nombre_arg, {}).get("type")
                if tipo_esperado == "number" and not isinstance(valor, float):
                    problemas.append(
                        f"'{nombre_arg}' deberia ser float, salio "
                        f"{type(valor).__name__} ({valor!r}) en {item!r}")
                if tipo_esperado == "boolean" and not isinstance(valor, bool):
                    problemas.append(
                        f"'{nombre_arg}' deberia ser bool, salio "
                        f"{type(valor).__name__} ({valor!r}) en {item!r}")
                if tipo_esperado == "string" and not isinstance(valor, str):
                    problemas.append(
                        f"'{nombre_arg}' deberia ser str, salio "
                        f"{type(valor).__name__} ({valor!r}) en {item!r}")

    if duracion > LIMITE_SEGUNDOS:
        problemas.append(
            f"tardo {duracion:.1f}s, por encima del limite de "
            f"{LIMITE_SEGUNDOS}s (5 min)")

    return problemas


def ejecutar_caso(modulo, caso: dict) -> tuple[bool, list[str], float]:
    preparar_archivos(caso)

    inicio = time.time()
    try:
        modulo.main()
    except Exception as e:  # noqa: BLE001
        duracion = time.time() - inicio
        return False, [f"main() lanzo una excepcion: {type(e).__name__}: {e}"], duracion
    duracion = time.time() - inicio

    if not os.path.exists(ARCHIVO_RESULTADOS):
        return False, ["no se genero output/function_calling_results.json"], duracion

    try:
        with open(ARCHIVO_RESULTADOS, encoding="utf-8") as f:
            resultados = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"el output no es JSON valido: {e}"], duracion

    problemas = validar_resultado(caso, resultados, duracion)
    return len(problemas) == 0, problemas, duracion


def prueba_consistencia(modulo) -> tuple[bool, list[str], float]:
    """Corre el caso base DOS VECES y comprueba que el resultado es
    identico. Vuestra decodificacion es siempre "argmax" (elige el logit
    mas alto, sin muestreo aleatorio), asi que dos ejecuciones con la misma
    entrada deberian dar exactamente el mismo output. Si no coincide, hay
    algo no determinista en el pipeline (peligroso de cara a evaluacion,
    porque el resultado dejaria de ser reproducible).
    """
    caso = CASOS[0]  # caso base
    inicio = time.time()

    preparar_archivos(caso)
    modulo.main()
    with open(ARCHIVO_RESULTADOS, encoding="utf-8") as f:
        primera_pasada = json.load(f)

    preparar_archivos(caso)
    modulo.main()
    with open(ARCHIVO_RESULTADOS, encoding="utf-8") as f:
        segunda_pasada = json.load(f)

    duracion = time.time() - inicio

    if primera_pasada != segunda_pasada:
        return False, ["dos ejecuciones identicas dieron resultados "
                       "distintos: revisa si hay algo no determinista "
                       "(orden de dict, muestreo aleatorio, etc.)"], duracion
    return True, [], duracion


def hacer_backup() -> None:
    if os.path.exists(DIR_BACKUP):
        shutil.rmtree(DIR_BACKUP)
    os.makedirs(DIR_BACKUP, exist_ok=True)

    if os.path.exists(DIR_INPUT):
        shutil.copytree(DIR_INPUT, os.path.join(DIR_BACKUP, "input"))
    if os.path.exists(DIR_OUTPUT):
        shutil.copytree(DIR_OUTPUT, os.path.join(DIR_BACKUP, "output"))


def restaurar_backup() -> None:
    if os.path.exists(DIR_INPUT):
        shutil.rmtree(DIR_INPUT)
    if os.path.exists(DIR_OUTPUT):
        shutil.rmtree(DIR_OUTPUT)

    backup_input = os.path.join(DIR_BACKUP, "input")
    backup_output = os.path.join(DIR_BACKUP, "output")
    if os.path.exists(backup_input):
        shutil.copytree(backup_input, DIR_INPUT)
    if os.path.exists(backup_output):
        shutil.copytree(backup_output, DIR_OUTPUT)

    shutil.rmtree(DIR_BACKUP)


def pruebas_completas(modulo, solo_critico: bool) -> list[tuple[str, str, bool, list[str], float]]:
    """Corre todos los CASOS (mas escenarios adicionales) contra main().

    Returns:
        Lista de (nombre, severidad, paso, problemas, duracion).
    """
    todos_los_casos = CASOS + cargar_escenarios_adicionales()
    if solo_critico:
        todos_los_casos = [c for c in todos_los_casos if c["severidad"] == "critico"]

    resultados: list[tuple[str, str, bool, list[str], float]] = []

    hacer_backup()
    try:
        for caso in todos_los_casos:
            paso, problemas, duracion = ejecutar_caso(modulo, caso)
            resultados.append((caso["nombre"], caso["severidad"], paso, problemas, duracion))

        if not solo_critico:
            paso, problemas, duracion = prueba_consistencia(modulo)
            resultados.append(
                ("[ROBUSTEZ] consistencia entre ejecuciones repetidas",
                 "robustez", paso, problemas, duracion))
    finally:
        restaurar_backup()

    return resultados


# ---------------------------------------------------------------------------
# Informe final
# ---------------------------------------------------------------------------

def imprimir_informe_rapidas(resultados: list[tuple[str, bool, str]]) -> int:
    print("\n=== PRUEBAS RAPIDAS (sin modelo) ===\n")
    fallos = 0
    for nombre, ok, detalle in resultados:
        marca = "OK  " if ok else "FAIL"
        print(f"[{marca}] {nombre}")
        if not ok:
            print(f"        {detalle}")
            fallos += 1
    total = len(resultados)
    print(f"\n{total - fallos}/{total} pruebas rapidas superadas.")
    return fallos


def imprimir_informe_completas(
        resultados: list[tuple[str, str, bool, list[str], float]]) -> tuple[int, int]:
    print("\n=== PRUEBAS COMPLETAS (con modelo) ===\n")
    fallos_criticos = 0
    fallos_robustez = 0
    tiempo_total = 0.0

    for nombre, severidad, ok, problemas, duracion in resultados:
        marca = "OK  " if ok else "FAIL"
        print(f"[{marca}] {nombre}  ({duracion:.1f}s)")
        for problema in problemas:
            print(f"        - {problema}")
        if not ok:
            if severidad == "critico":
                fallos_criticos += 1
            else:
                fallos_robustez += 1
        tiempo_total += duracion

    total = len(resultados)
    print(f"\n{total - fallos_criticos - fallos_robustez}/{total} casos superados. "
         f"Tiempo total: {tiempo_total:.1f}s.")
    print(f"  - Fallos CRITICOS (violan el subject): {fallos_criticos}")
    print(f"  - Fallos de ROBUSTEZ (adversariales, no exigidos literalmente): "
         f"{fallos_robustez}")
    return fallos_criticos, fallos_robustez


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapidas", action="store_true",
                        help="Correr solo las pruebas rapidas (sin modelo).")
    parser.add_argument("--completas", action="store_true",
                        help="Correr solo las pruebas completas (con modelo).")
    parser.add_argument("--solo-critico", action="store_true",
                        help="En las pruebas completas, correr solo los "
                             "casos CRITICO (lo que exige el subject).")
    args = parser.parse_args()

    correr_rapidas = args.rapidas or not (args.rapidas or args.completas)
    correr_completas = args.completas or not (args.rapidas or args.completas)

    modulo = importlib.import_module(MODULO_A_PROBAR)

    fallos_rapidas = 0
    fallos_criticos = 0
    fallos_robustez = 0

    if correr_rapidas:
        fallos_rapidas = imprimir_informe_rapidas(pruebas_rapidas(modulo))

    if correr_completas:
        fallos_criticos, fallos_robustez = imprimir_informe_completas(
            pruebas_completas(modulo, args.solo_critico))

    print()
    if fallos_rapidas == 0 and fallos_criticos == 0 and fallos_robustez == 0:
        print("TODO OK. Listo para evaluacion.")
    elif fallos_criticos > 0:
        print(f"BLOQUEANTE: {fallos_criticos} fallo(s) critico(s) que "
             f"violan requisitos explicitos del subject. Arregla esto "
             f"antes de entregar.")
    else:
        print(f"Sin fallos criticos, pero hay {fallos_rapidas + fallos_robustez} "
             f"aviso(s) de robustez/rapidas a revisar (no bloquean el "
             f"subject, pero pueden salir en la evaluacion entre pares).")


if __name__ == "__main__":
    main()
