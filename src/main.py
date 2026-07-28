from llm_sdk import Small_LLM_Model


def main():
    functions = ["fn_add_numbers",
                 "fn_greet",
                 "fn_reverse_string",
                 "fn_get_square_root",
                 "fn_substitute_string_with_regex",
                 ]
    model = Small_LLM_Model()


if __name__ == "__main__":
    main()
