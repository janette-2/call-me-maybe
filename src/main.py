from llm_sdk import Small_LLM_Model


def fixed_ids(model: Small_LLM_Model) -> dict:
    dict_fixed = {}
    chars_fixed = ["{",
                   "}",
                   ":",
                   ",",
                   " ",
                   "\n",
                   "fn_name",
                   "args",
                   "fn_add_numbers",
                   "fn_greet",
                   "fn_reverse_string",
                   "fn_get_square_root",
                   "fn_substitute_string_with_regex",
                   "a",
                   "b",
                   "name",
                   "s",
                   "source_string",
                   "regex",
                   "replacement"]

    for fixed in chars_fixed:
        dict_fixed[fixed] = model.encode(fixed)

    return dict_fixed


def main():
    model = Small_LLM_Model()
    dict_fix = fixed_ids(model)
    print(dict_fix)


if __name__ == "__main__":
    main()
