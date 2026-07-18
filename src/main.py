from llm_sdk import Small_LLM_Model


def main():
    prompt = input("Prompt: ")
    model = Small_LLM_Model()
    token = model.encode(prompt)
    print(f"Token: {token}")


if __name__ == "__main__":
    main()
