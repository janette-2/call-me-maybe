from llm_sdk import Small_LLM_Model
import json


def fixed_ids(model: Small_LLM_Model,
              dict_functions: dict, user_in: str) -> dict:
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
                   "prompt",
                   "fn_name",
                   "args",
                   f"{user_in}",
                   "\"",
                   "\n",
                   "\t",
                   "\v",
                   ]

    for fixed in chars_fixed:
        dict_fixed[fixed] = model.encode(fixed).flatten().tolist()

    # Adding the different functions found in the
    # 'functions_definitions.json' file
    for fun in dict_functions:
        # Get the name of each function
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
        Dictionary keyed by function name. Each value is itself a dictionary
        with exactly three keys: ``"parameters"`` (dict of
        ``{param_name: {"type": type}}``), ``"description"`` (str) and
        ``"returns"`` (dict ``{"type": type}``). Example::

            {
                "fn_add_numbers": {
                    "parameters": {"a": {"type": "number"},
                                   "b": {"type": "number"}},
                    "description": "Add two numbers together and return their "
                                   "sum.",
                    "returns": {"type": "number"},
                },
                ...
            }

        To access the type of parameter ``"a"`` of ``"fn_add_numbers"`` use
        ``result["fn_add_numbers"]["parameters"]["a"]["type"]``. Empty if
        the file is missing or malformed.
    """
    dict_functions = {}
    try:
        with open("data/input/functions_definition.json") as f:
            f_content = json.load(f)

        for dictionary in f_content:
            dict_functions[dictionary["name"]] = {
                "parameters": dictionary["parameters"],
                "description": dictionary["description"],
                "returns": dictionary["returns"]
            }

    except Exception as e:
        print(f"{e}")

    return dict_functions


def build_super_prompt(dict_functions: dict, input_call: str) -> str:
    """Build the prompt that tells the model which functions are available.

    The model is a general language model: it does not know our functions
    until we list them. This function assembles the full prompt with an
    introductory role, the output rules, the list of available functions
    (name, description and parameters) and the user request placeholder.

    The template strings keep their content at column 0 (no indentation inside
    the triple quotes). Python string literals are exact: any indentation
    written between the quotes becomes real characters in the prompt, so
    writing the content at column 0 keeps the output clean without needing
    ``textwrap.dedent()``.

    Args:
        dict_functions: Dictionary ``{function_name: {"parameters": {...},
            "description": str, "returns": {...}}}`` as returned by
            ``functions_info()``.
        input_call: The user request in natural language, inserted at the
            end of the prompt so the model can map it to a function call.

    Returns:
        The complete prompt text, ready to be tokenized and fed to the model.
    """
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
and only those, with the correct type (number, string, boolean).\n\n"""

    # Loop to create the list of current functions
    template_functions = """Available functions:\n"""
    for func in dict_functions:
        dict_parameters = dict_functions[func].get("parameters")
        list_parameters = []

        for param in dict_parameters:
            temp = f"{param} ({dict_parameters[param].get("type")})"
            list_parameters.append(temp)

        func_description = f"""- {func}: {dict_functions[func].get(
         "description")}\nParameters: {", ".join(list_parameters)}\n"""

        template_functions += func_description

    prompt = (template_intro
              + template_rules
              + template_functions
              + f"\nUser: {input_call}\nOutput: ")

    return prompt


def loop_prompt_output(input: str, model: Small_LLM_Model) -> str:
    init_prompt_ids = model.encode(input).flatten().tolist()
    print(init_prompt_ids)
    return ("")


def main() -> None:
    """Entry point: load the model and build the super-prompt.

    Steps:
        1. Load the Qwen3-0.6B model (``Small_LLM_Model``).
        2. Load the function definitions from the input file.
        3. Build the mapping of every fixed JSON piece to its token IDs.
        4. Build the super-prompt with a sample user request and print it.
    """
    model = Small_LLM_Model()
    dict_functions = functions_info()
    dict_fixed_chars = fixed_ids(model, dict_functions, "What is the"
                                 " sum of 2 and 3?")
    prompt = build_super_prompt(dict_functions, "What is the sum of 2 and 3?")
    loop_prompt_output(prompt, model)
    print("")
    user_input = "What is the sum of 2 and 3?"
    template_0 = [dict_fixed_chars["{"],
                  dict_fixed_chars["\v"],
                  dict_fixed_chars["\t"],
                  dict_fixed_chars["\""],
                  dict_fixed_chars["prompt"],
                  dict_fixed_chars["\""],
                  dict_fixed_chars[":"],
                  dict_fixed_chars[" "],
                  dict_fixed_chars["\""],
                  dict_fixed_chars[f"{user_input}"],
                  dict_fixed_chars["\""],
                  dict_fixed_chars[","],
                  dict_fixed_chars["\v"],
                  dict_fixed_chars["\t"],
                  dict_fixed_chars["\""],
                  dict_fixed_chars["fn_name"],
                  dict_fixed_chars["\""],
                  dict_fixed_chars[":"],
                  ]
    print(template_0)


if __name__ == "__main__":
    main()
