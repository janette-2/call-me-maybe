"""Moulinette de pruebas para call_me_maybe.

Dos partes, pensadas para poder iterar rápido:

1. PRUEBAS RAPIDAS (sin modelo): comprueban las funciones deterministas de
   extracción de texto (comillas, numeros, "with", regex). Se ejecutan en
   milisegundos: corrélas cada vez que cambies algo, sin esperar al modelo.

2. PRUEBAS COMPLETAS (con modelo): montan distintos function_definitions.json
   y function_calling_tests.json (caso base, archivos ausentes, JSON
   invalido, prompts ambiguos, funciones con nombres parecidos...),
   ejecutan tu main() de verdad y validan el output real: estructura,
   tipos, manejo de errores y tiempo total.

COMO USARLO:
    1. Copia este archivo en la raiz de tu proyecto (junto a src/, data/...).
    2. Ajusta MODULO_A_PROBAR mas abajo con el nombre real de tu modulo
       (ej. "src.main_best").
    3. Ejecuta:
        uv run python moulinette_call_me_maybe.py            (las dos partes)
        uv run python moulinette_call_me_maybe.py --rapidas   (solo la 1)
        uv run python moulinette_call_me_maybe.py --completas (solo la 2)

IMPORTANTE: la parte 2 SOBREESCRIBE temporalmente tus archivos en
data/input/ y output/. El script guarda una copia antes de empezar y la
restaura al terminar (incluso si algo falla a mitad), pero aun asi haz un
commit antes de correrlo por si acaso.
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

DIR_BACKUP = ".moulinette_backup"
LIMITE_SEGUNDOS = 5 * 60


# ---------------------------------------------------------------------------
# PARTE 1: pruebas rapidas de las funciones deterministas (sin modelo)
# ---------------------------------------------------------------------------

def pruebas_rapidas(modulo) -> list[tuple[str, bool, str]]:
    """Prueba las funciones de extraccion de texto sin llamar al modelo.

    Returns:
        Lista de tuplas (nombre_prueba, ha_pasado, detalle_si_ha_fallado).
    """
    resultados: list[tuple[str, bool, str]] = []

    def check(nombre: str, obtenido, esperado) -> None:
        ok = obtenido == esperado
        detalle = "" if ok else f"esperado={esperado!r} obtenido={obtenido!r}"
        resultados.append((nombre, ok, detalle))

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
    check(
        "frases: comilla sin cerrar -> lista vacia (no revienta)",
        modulo.extraer_frases_entrecomilladas("Reverse the string 'hello"),
        [],
    )

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
        "numeros: numeros pegados a texto",
        modulo.extraer_numeros("Hello 34 I'm 233 years old"),
        ["34", "233"],
    )
    check(
        "numeros: numero muy grande",
        modulo.extraer_numeros("What is the sum of 999999999 and 1?"),
        ["999999999", "1"],
    )

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

# Dos funciones que comparten un prefijo largo a proposito: sirve para
# comprobar el punto A (la posicion de decision en _escoge_fn).
FUNCIONES_PREFIJO_COMPARTIDO = [
    {"name": "fn_get_square_root", "description": "Calculate the square root of a number.",
     "parameters": {"a": {"type": "number"}},
     "returns": {"type": "number"}},
    {"name": "fn_get_cube_root", "description": "Calculate the cube root of a number.",
     "parameters": {"a": {"type": "number"}},
     "returns": {"type": "number"}},
]

TESTS_PREFIJO_COMPARTIDO = [
    {"prompt": "What is the square root of 81?"},
    {"prompt": "What is the cube root of 27?"},
]

# Funciones sin ninguna relacion con el prompt de test: sirve para
# comprobar el punto B (que no reviente si nada coincide).
TESTS_AMBIGUOS = [
    {"prompt": "What is the current phase of the moon?"},
]

CASOS: list[dict] = [
    {
        "nombre": "caso base (11 prompts, 5 funciones)",
        "funciones": FUNCIONES_BASE,
        "tests": TESTS_BASE,
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "archivo de funciones ausente",
        "funciones": None,
        "tests": TESTS_BASE,
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": True,
        "borrar_tests": False,
    },
    {
        "nombre": "JSON de funciones invalido (sintaxis rota)",
        "funciones": None,
        "tests": TESTS_BASE,
        "funciones_texto": '[{"name": "fn_add_numbers", "parameters": ',
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "archivo de tests ausente",
        "funciones": FUNCIONES_BASE,
        "tests": None,
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": True,
    },
    {
        "nombre": "JSON de tests invalido (sintaxis rota)",
        "funciones": FUNCIONES_BASE,
        "tests": None,
        "funciones_texto": None,
        "tests_texto": '[{"prompt": "Greet shrek"',
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "lista de tests vacia",
        "funciones": FUNCIONES_BASE,
        "tests": [],
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "entrada de test sin la clave 'prompt'",
        "funciones": FUNCIONES_BASE,
        "tests": [{"pregunta": "esto no tiene la clave prompt"},
                  {"prompt": "Greet john"}],
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "prompt vacio",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": ""}, {"prompt": "Greet john"}],
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "funciones con nombres de prefijo compartido (punto A)",
        "funciones": FUNCIONES_PREFIJO_COMPARTIDO,
        "tests": TESTS_PREFIJO_COMPARTIDO,
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "prompt ambiguo, sin funcion relacionada (punto B)",
        "funciones": FUNCIONES_BASE,
        "tests": TESTS_AMBIGUOS,
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
    {
        "nombre": "caracteres especiales en la cadena",
        "funciones": FUNCIONES_BASE,
        "tests": [{"prompt": "Reverse the string 'a@b#c!'"}],
        "funciones_texto": None,
        "tests_texto": None,
        "borrar_funciones": False,
        "borrar_tests": False,
    },
]


def preparar_archivos(caso: dict) -> None:
    """Escribe (o borra) los archivos de entrada segun lo que pide el caso."""
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


def validar_resultado(caso: dict, resultados: list, duracion: float) -> list[str]:
    """Comprueba la forma y el contenido de output/function_calling_results.json.

    Returns:
        Lista de problemas encontrados. Vacia si todo esta bien.
    """
    problemas: list[str] = []

    if not isinstance(resultados, list):
        problemas.append("la salida no es una lista JSON")
        return problemas

    nombres_funciones_validos = set()
    if caso["funciones"] is not None:
        nombres_funciones_validos = {f["name"] for f in caso["funciones"]}

    for item in resultados:
        if not isinstance(item, dict):
            problemas.append(f"un elemento de la lista no es un objeto: {item!r}")
            continue

        claves = set(item.keys())
        if claves != {"prompt", "fn_name", "args"}:
            problemas.append(
                f"claves incorrectas en {item!r} (esperado prompt/fn_name/args)")

        if "fn_name" in item and item["fn_name"] is not None:
            if item["fn_name"] not in nombres_funciones_validos:
                problemas.append(
                    f"fn_name={item['fn_name']!r} no esta entre las funciones "
                    f"disponibles {sorted(nombres_funciones_validos)}")

        if "args" in item and not isinstance(item["args"], dict):
            problemas.append(f"args no es un diccionario en {item!r}")

        # Los numeros deben salir como float (2.0, no 2), segun el subject.
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

    if duracion > LIMITE_SEGUNDOS:
        problemas.append(
            f"tardo {duracion:.1f}s, por encima del limite de "
            f"{LIMITE_SEGUNDOS}s (5 min)")

    return problemas


def ejecutar_caso(modulo, caso: dict) -> tuple[bool, list[str], float]:
    """Ejecuta un caso completo: prepara archivos, corre main(), valida.

    Returns:
        (paso, lista_de_problemas, duracion_en_segundos)
    """
    preparar_archivos(caso)

    inicio = time.time()
    try:
        modulo.main()
    except Exception as e:  # noqa: BLE001 - queremos capturar CUALQUIER caida
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


def pruebas_completas(modulo) -> list[tuple[str, bool, list[str], float]]:
    """Corre todos los CASOS contra main() de verdad, con backup/restore."""
    resultados: list[tuple[str, bool, list[str], float]] = []

    hacer_backup()
    try:
        for caso in CASOS:
            paso, problemas, duracion = ejecutar_caso(modulo, caso)
            resultados.append((caso["nombre"], paso, problemas, duracion))
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
        resultados: list[tuple[str, bool, list[str], float]]) -> int:
    print("\n=== PRUEBAS COMPLETAS (con modelo) ===\n")
    fallos = 0
    tiempo_total = 0.0
    for nombre, ok, problemas, duracion in resultados:
        marca = "OK  " if ok else "FAIL"
        print(f"[{marca}] {nombre}  ({duracion:.1f}s)")
        for problema in problemas:
            print(f"        - {problema}")
        if not ok:
            fallos += 1
        tiempo_total += duracion
    total = len(resultados)
    print(f"\n{total - fallos}/{total} casos superados. "
         f"Tiempo total: {tiempo_total:.1f}s "
         f"(limite por caso: {LIMITE_SEGUNDOS}s).")
    return fallos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapidas", action="store_true",
                        help="Correr solo las pruebas rapidas (sin modelo).")
    parser.add_argument("--completas", action="store_true",
                        help="Correr solo las pruebas completas (con modelo).")
    args = parser.parse_args()

    correr_rapidas = args.rapidas or not (args.rapidas or args.completas)
    correr_completas = args.completas or not (args.rapidas or args.completas)

    modulo = importlib.import_module(MODULO_A_PROBAR)

    fallos_totales = 0

    if correr_rapidas:
        fallos_totales += imprimir_informe_rapidas(pruebas_rapidas(modulo))

    if correr_completas:
        fallos_totales += imprimir_informe_completas(pruebas_completas(modulo))

    print()
    if fallos_totales == 0:
        print("TODO OK. Listo para evaluacion.")
    else:
        print(f"{fallos_totales} problema(s) encontrado(s). Revisa el detalle arriba.")


if __name__ == "__main__":
    main()
