"""Temporary test script for the function-calling loop.

Runs several user prompts through the real pipeline and reports, for each
one, the raw LLM output and whether it parses as valid JSON.
"""

import json
import sys

from llm_sdk import Small_LLM_Model

import main as m

sys.setrecursionlimit(100000)


def run_prompt(model, dict_functions, dict_fixed_chars, user_call):
    prompt = m.build_super_prompt(dict_functions, user_call)
    final_ids = m.loop_prompt_output(prompt, model,
                                     dict_fixed_chars, dict_functions)
    final_output = model.decode(final_ids)
    try:
        return final_output.split("Output: ", 1)[1]
    except IndexError:
        return "<<<NO 'Output:' FOUND>>>"
    except Exception as exc:  # pragma: no cover
        return f"<<<ERROR: {exc}>>>"


TESTS = [
    "What is the sum of 2 and 3?",
    "What is the sum of 265 and 345?",
    "What is the square root of 16?",
    "Greet shrek",
    "Greet john",
    "Reverse the string 'hello'",
    "Reverse the string 'world'",
    "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "Replace all vowels in 'Programming is fun' with asterisks",
    "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
]


def main():
    model = Small_LLM_Model()
    dict_functions = m.functions_info()
    dict_fixed_chars = m.fixed_ids(model, dict_functions)

    for test in TESTS:
        print("=" * 70)
        print("PROMPT :", test)
        result = run_prompt(model, dict_functions, dict_fixed_chars, test)
        print("OUTPUT :", repr(result))
        try:
            data = json.loads(result)
            print("JSON   : OK ->", data)
        except Exception as exc:
            print("JSON   : FAIL ->", exc)


if __name__ == "__main__":
    main()
