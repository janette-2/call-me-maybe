from llm_sdk import Small_LLM_Model
import json


def fixed_ids(model: Small_LLM_Model,
              dict_functions: dict) -> dict:
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
                   "fn_name",
                   "args",
                   "\"",
                   "\n",
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
and only those, with the correct type (number, string, boolean).
- Extract every argument value from the exact words or numbers written in
the User request below. Never invent values, never repeat the instruction
words, and never copy the function name or the JSON labels.
- If an argument is a quoted phrase in the request, take the whole phrase as
the value. Strip any surrounding quotes or spaces.
- When a function has more than one value to fill, give each argument the
value the User mentions that matches its labelled role, in the order the
User refers to them.\n\n"""

    # Loop to create the list of current functions
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


def loop_prompt_output(input: str, model: Small_LLM_Model,
                       dict_fixed_chars: dict,
                       dict_functions: dict,
                       user_prompt: str) -> list[int]:
    """Run constrained decoding to produce a valid function-call JSON.

    This is the core of the project. It builds the JSON output in three
    phases:

    1. **Structure forcing**: appends the fixed JSON skeleton (``{``, ``"``,
       ``fn_name``, ``"``, ``:``, `` ``, ``"``) to the tokenised prompt.
       These are tokens the program decides, not the LLM.

    2. **Function identification**: lets the model predict one token at a
       time and eliminates function-name candidates that do not match.
       Uses a *copy* of the candidate list for iteration so that removals
       during the loop do not skip elements.  Stops when exactly one
       candidate remains (or zero, which is an error).

    3. **Argument generation**: for each required parameter of the chosen
       function, forces the key (``"param_name": ``) and lets the model
       predict the value token.

    The function appends directly to ``init_prompt_ids`` at every step so
    that each model prediction sees the full accumulated context.  A
    separate ``temp_prompt`` copy is used inside the function-identification
    loop to avoid double-counting tokens.

    Args:
        input: The raw text prompt (already tokenised and extended with
            the fixed structure before calling this function).
        model: Instance of the Qwen3-0.6B model (for encoding, decoding
            and logit retrieval).
        dict_fixed_chars: Dictionary mapping text fragments to their flat
            list of token IDs (from ``fixed_ids()``).
        dict_functions: Dictionary of function definitions (from
            ``functions_info()``).

    Returns:
        Flat list of token IDs representing the full prompt plus the
        generated JSON output, ready to be passed to ``model.decode()``.
    """
    init_prompt_ids: list[int] = model.encode(input).flatten().tolist()

    # Forces the fixed structure tokens to help the LLM predict the 'fn_name'
    init_prompt_ids.extend(dict_fixed_chars["{"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars["fn_name"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[":"])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["\""])

    # Gets list of the function names with their converted tokens:
    fn_names_tokens: list[list[int]] = []
    for name in dict_functions:
        fn_names_tokens.append(dict_fixed_chars[name])
        # Appends a dict value of the form [x, y, z] (list of token IDs)

    # FOR VIEWING THE FUNCTIONS IN TOKENS, FOR TESTING
    # print(fn_names_tokens)

    i = 0
    llm_ids = []
    temp_prompt = init_prompt_ids.copy()
    while len(fn_names_tokens) > 1:
        # Catch the new token that the LLM predicts after the prompt input
        llm_logits = model.get_logits_from_input_ids(temp_prompt)
        allowed = [ids[i] for ids in fn_names_tokens if len(ids) > i]
        # Takes the first id in the different fn_tokens if hey have len > i
        next_id = max(allowed, key=lambda tid: llm_logits[tid])
        # Captures the max of the logits from the fn_names_tokens[i]

        # Store the predicted ids into a list
        llm_ids.extend([next_id])

        # Compare each functions ids with the given next_id
        fn_ids = fn_names_tokens.copy()
        for ids in fn_ids:
            # Check if the ids have enough indexes and if the retrieved
            # llm_id[i] matches the ids[i]
            if len(ids) > i + 1 and ids[i] == llm_ids[i]:
                continue
            else:
                fn_names_tokens.remove(ids)

        temp_prompt.extend([next_id])
        i += 1

    # Outside the while, when we have len(fn_names_tokens) == 1
    # we append the final function to the init_prompt_ids
    init_prompt_ids.extend(fn_names_tokens[0])

    # Also, extend with the preparations to the arguments
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[","])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars["args"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[":"])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["{"])

    # Retrieve, through the obtained fn, its arguments
    fn = model.decode(fn_names_tokens[0])
    args_fn = [arg for arg in dict_functions[fn]["parameters"]]

    # Loop to print the list of arguments stored
    i = 0

    # Vocab to quickly associate "chrs" -> ID [index in LLM]
    path_vocab = model.get_path_to_vocab_file()
    with open(path_vocab) as file:
        vocab = json.load(file)     # dict {texto: ID}

    n_args = len(args_fn)
    for idx, arg in enumerate(args_fn):
        arg_type = dict_functions[fn]["parameters"][arg]["type"]
        is_last = (idx == n_args - 1)

        # Fixed prefix of each argument:
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
            candidates_list = extract_candidate_spans(user_prompt)
            value_ids = logit_masking_string(vocab,
                                             model,
                                             init_prompt_ids,
                                             candidates_list)

        init_prompt_ids.extend(value_ids)

        # Join with the next argument or close the object.
        if is_last:
            init_prompt_ids.extend(dict_fixed_chars["}"])
        else:
            init_prompt_ids.extend(dict_fixed_chars[","])
            init_prompt_ids.extend(dict_fixed_chars[" "])

    # Close the outermost object of the JSON.
    init_prompt_ids.extend(dict_fixed_chars["}"])
    return init_prompt_ids


def logit_masking_number(vocab: dict, model: Small_LLM_Model,
                         init_prompt_ids: list[int]) -> list[int]:

    context = init_prompt_ids.copy()
    numbers = "0123456789"
    ids_numbers = [vocab[num] for num in numbers]
    ids_parada = [vocab[","], vocab["}"]]
    candidates = ids_numbers + ids_parada
    # Lists can be added, the second will extend the first

    next_id = []
    flag = True
    while flag:

        llm_logits = model.get_logits_from_input_ids(context)
        bigger_stats_token = float("-inf")  # REVISAR PA QUE
        best_id = 0

        for id in candidates:
            if llm_logits[id] > bigger_stats_token:
                bigger_stats_token = llm_logits[id]
                best_id = id

        if best_id in ids_parada:  # Found delimeters (last two)
            flag = False
        else:
            next_id.append(best_id)
            context.append(best_id)

    return next_id


def logit_masking_boolean(vocab: dict, model: Small_LLM_Model,
                          init_prompt_ids: list[int]) -> list[int]:

    context = init_prompt_ids.copy()
    ids_booleans = [vocab["true"],  vocab["false"]]
    ids_parada = [vocab[","], vocab["}"]]
    candidates = ids_booleans + ids_parada
    # Lists can be added, the second will extend the first

    next_id = []
    flag = True
    while flag:

        # List of stats of predictibility
        llm_logits = model.get_logits_from_input_ids(context)
        bigger_stats_token = float("-inf")  # REVISAR PA QUE
        best_id = candidates[0]  # To make sure only allowed ids are considered

        for id in candidates:
            if llm_logits[id] > bigger_stats_token:
                bigger_stats_token = llm_logits[id]
                best_id = id

        if best_id in ids_parada:  # Found delimeters (last two)
            flag = False
        else:
            next_id.append(best_id)
            context.append(best_id)

    return next_id


def extract_candidate_spans(user_segment: str) -> list[str]:
    """Extract individual words, quoted phrases, and full text."""
    texto = user_segment.strip()  # Clean leading and trailing whitespace
    candidatos: list[str] = [texto]  # Initialize with full text

    # DOUBLE QUOTES: Process text within double quotes
    if texto.count('"') >= 2:
        partes_dobles = texto.split('"')
        # Odd indices contain text inside quotes
        for i in range(1, len(partes_dobles), 2):
            contenido = partes_dobles[i].strip()
            if contenido:
                candidatos.append(contenido)

    # SINGLE QUOTES: Process character by character to
    # ignore contractions (e.g. I'm)
    i, n, in_quote, current_span = 0, len(texto), False, []
    while i < n:
        char = texto[i]
        if char == "'":
            # Checks if the quote is not the first/last char
            #  if after/before the quote there is text
            is_contraction = i > 0 and i < n - 1 and texto[
                i - 1].isalpha() and texto[i + 1].isalpha()
            if not is_contraction:
                if in_quote:  # Closing quote found
                    span_str = "".join(current_span).strip()
                    if span_str:
                        candidatos.append(span_str)
                    current_span, in_quote = [], False
                else:  # Opening quote found
                    in_quote = True
            elif in_quote:
                current_span.append(char)
        elif in_quote:
            current_span.append(char)
        i += 1

    # INDIVIDUAL WORDS AND CONTRACTIONS
    for palabra in texto.split():
        palabra_limpia = palabra.strip(".,;:!?\"'")  # Strip outer punctuation
        if palabra_limpia.startswith("'") and palabra_limpia.endswith("'"):
            palabra_limpia = palabra_limpia[1:-1]
        if palabra_limpia:
            candidatos.append(palabra_limpia)

    # REMOVE DUPLICATES AND SORT BY LENGTH
    sin_repetidos: list[str] = []
    for elem in candidatos:
        if elem not in sin_repetidos:
            sin_repetidos.append(elem)

    # Return longest items first
    return sorted(sin_repetidos, key=len, reverse=True)


def logit_masking_string(vocab: dict[str, int],
                         model: Small_LLM_Model,
                         init_prompt_ids: list[int],
                         candidates_list: list[str]) -> list[int]:
    """Generate the tokens of a string argument until a closing delimiter.

    Unlike the number/boolean masks (closed sets of digits / booleans), a
    string has no natural closed alphabet. The trick used here is to close the
    space to the tokens that actually appear in the *user request*: any value
    the user asked for is spelled with those exact tokens, so the model is
    forced to pick the real words of the prompt instead of wandering through
    the 150k-token vocabulary (``get_logits`` considers every id; we only let
    it choose among ``ids_content``).

    The model does not emit a bare ``"`` to close the string; it emits compound
    closing tokens such as ``","`` (id 497, used when more arguments follow) or
    ``"}`` (id 9207, used when it is the last argument). These live in
    ``ids_closing``. When the model picks one we stop; the caller is then
    responsible for emitting the exact delimiter this argument needs.

    A ``max_chars`` guard prevents infinite loops: if the model never picks a
    closer, we stop anyway and let the caller close.

    Args:
        model: Instance of the Qwen3-0.6B model.
        init_prompt_ids: The token IDs of the prompt already including the
            opening ``"`` of the string value.
        ids_content: Closed set of token IDs that may form the value (the token
            IDs of the ``User`` segment of the prompt, plus the closing ids).
        ids_closing: List of token IDs that mark the end of the string value.
        max_chars: Maximum number of content tokens before stopping.

    Returns:
        List of token IDs of the string value (content only, without the
        closing quote). The caller is responsible for appending the closing
        ``"`` and the ``,``/``}`` delimiter that follows.
    """

    context = init_prompt_ids.copy()
    possible_tokens = []
    next_id = [vocab["\""]]

    candidates = candidates_list + ['"']
    for candidate in candidates:
        ids = model.encode(candidate).flatten().tolist()
        possible_tokens.append(ids)

    remaining_opt = possible_tokens.copy()
    i = 0
    max_longitud = max((len(t) for t in possible_tokens), default=0)
    while remaining_opt and i < max_longitud:
        llm_logits = model.get_logits_from_input_ids(context)
        # candidates[i] for every candidate not the entire ids list
        allowed = [t[i] for t in remaining_opt if len(t) > i]
        if not allowed:
            break
        best_id = max(allowed, key=lambda tid: llm_logits[tid])
        # To make sure only allowed ids are considered
        next_id.append(best_id)
        context.append(best_id)
        remaining_opt = [t for t in remaining_opt
                         if len(t) > i and t[i] == best_id]
        # Gets the entire ids list if the remainig[i] matches the predicted
        i += 1
        if len(remaining_opt) == 1 and len(remaining_opt[0]) == i:  # REVISAR EL MOTIVO
            break

    next_id.append(vocab['"'])
    return next_id


"""
def logit_masking_string(model, init_prompt_ids, candidates_text,
                         ids_closing, max_chars=64):
    cand_tokens = [model.encode(c).flatten().tolist() for c in candidates_text]
    cand_tokens = [t for t in cand_tokens if t]

    context = init_prompt_ids.copy()
    remaining = cand_tokens.copy()
    next_id: list[int] = []
    i = 0

    while remaining and i < max_chars:
        llm_logits = model.get_logits_from_input_ids(context)
         allowed = {t[i] for t in remaining if len(t) > i}
        if not allowed:
            break

        best_id = max(allowed, key=lambda tid: llm_logits[tid])
        next_id.append(best_id)
        context.append(best_id)

        remaining = [t for t in remaining if len(t) > i and t[i] == best_id]
        i += 1

        if len(remaining) == 1 and len(remaining[0]) == i:
            break

    return next_id
 """


def main() -> None:
    """Entry point: process every prompt in the test file and write results.

    Steps:
        1. Load the Qwen3-0.6B model (``Small_LLM_Model``).
        2. Load the function definitions from the input file.
        3. Build the mapping of every fixed JSON piece to its token IDs.
        4. For every prompt in ``data/input/function_calling_tests.json``,
           build its super-prompt and run constrained decoding.
        5. Extract the generated JSON from each decode and collect the
           prompt / fn_name / args triple.
        6. Write ``output/function_calling_results.json``.
    """
    model = Small_LLM_Model()
    dict_functions = functions_info()
    dict_fixed_chars = fixed_ids(model, dict_functions)

    with open("data/input/function_calling_tests.json") as file:
        tests = json.load(file)

    results = []
    for test in tests:
        user_prompt = test["prompt"]
        super_prompt = build_super_prompt(dict_functions, user_prompt)
        final_prompt_ids = loop_prompt_output(super_prompt,
                                              model, dict_fixed_chars,
                                              dict_functions,
                                              user_prompt)
        final_output = model.decode(final_prompt_ids)

        # Extract the JSON after the "Output: " marker.
        result_json = final_output.split("Output: ", 1)[1]
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            # Keep a placeholder so this prompt is still reported.
            result = {"fn_name": None, "args": {}}

        results.append({
            "prompt": user_prompt,
            "fn_name": result.get("fn_name"),
            "args": result.get("args"),
        })

    with open("output/function_calling_results.json",
              "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
    # main_tests()
