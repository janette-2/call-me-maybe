from llm_sdk import Small_LLM_Model
import json


def fixed_ids(model: Small_LLM_Model, dict_functions: dict) -> dict:
    """Map every piece of the output JSON to its token IDs.

    The LLM does not work with text but with numeric IDs. This function uses
    ``model.encode()`` to find, once at the start, the IDs of the fixed parts
    of the JSON (braces, commas, function names and parameter names). These
    IDs are reused later in the constrained-decoding loop to force which
    tokens the model may generate at each step.

    ``model.encode()`` returns a 2-D tensor; ``flatten()`` collapses it to a
    1-D sequence and ``tolist()`` converts it to a plain Python ``list[int]``
    so the values can be compared with integers during generation.

    Args:
        model: Instance of the Qwen3-0.6B model (used only for tokenizing).
        dict_functions: Dictionary ``{function_name: {"parameters": {...},
            "description": str, "returns": {...}}}`` loaded from
            ``functions_definition.json``.

    Returns:
        Dictionary where each key is a piece of text (e.g. ``"{"``,
        ``"fn_add_numbers"``) and each value is its flat list of token IDs.
    """
    dict_fixed = {}
    chars_fixed = ["{",
                   "}",
                   ":",
                   ",",
                   " ",
                   "\n",
                   ]

    for fixed in chars_fixed:
        dict_fixed[fixed] = model.encode(fixed).flatten().tolist()

    # Adding the different functions found in the
    # 'functions_definitions.json' file
    for fun in dict_functions:
        dict_fixed[fun] = model.encode(fun).flatten().tolist()

        # Get the parameters of each function
        for params in dict_functions[fun].get("parameters"):
            dict_fixed[params] = model.encode(params).flatten().tolist()

    return dict_fixed


def functions_info() -> dict:
    """Load function definitions from the input file.

    Reads ``data/input/functions_definition.json`` and transforms it into a
    dictionary accessible by function name. This information is the basis of
    the super-prompt (so the model knows which functions exist) and of the
    constraint automaton (to know which parameters and types to allow).

    The file may contain invalid JSON or not exist; this is handled with a
    ``try``/``except`` that prints the error and returns an empty dictionary.

    Returns:
        Dictionary ``{function_name: {"parameters": {param: {"type": type}},
        "description": str, "returns": {"type": type}}}``, e.g.
        ``{"fn_add_numbers": {"parameters": {"a": {"type": "number"}},
        "description": "Add two numbers...", "returns": {"type": "number"}}}``.
        Empty if the file is missing or malformed.
    """
    dict_func_parameters = {}
    try:
        with open("data/input/functions_definition.json") as f:
            f_content = json.load(f)

        for dictionary in f_content:
            dict_func_parameters[dictionary["name"]] = {
                "parameters": dictionary["parameters"],
                "description": dictionary["description"],
                "returns": dictionary["returns"]
            }

    except Exception as e:
        print(f"{e}")

    return dict_func_parameters


def main() -> None:
    """Entry point: load the model and build the token map.

    Steps:
        1. Load the Qwen3-0.6B model (``Small_LLM_Model``).
        2. Load the function definitions from the input file.
        3. Build the mapping of every fixed JSON piece to its token IDs.
    """
    model = Small_LLM_Model()
    dict_functions = functions_info()
    dict_final = fixed_ids(model, dict_functions)
    print(dict_final)


if __name__ == "__main__":
    main()
