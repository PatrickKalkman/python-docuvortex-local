from query.vortex_query import VortexQuery
import textwrap

DEFAULT_WIDTH = 110

def wrap_text_preserve_newlines(text, width=DEFAULT_WIDTH):
    """Wrap text preserving new lines. 

    Args:
        text (str): Text to wrap.
        width (int, optional): The maximum length of wrapped lines. Defaults to DEFAULT_WIDTH.

    Returns:
        str: Wrapped text preserving new lines.
    """
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return '\n'.join(wrapped_lines)

def display_response(llm_response):
    """Prints formatted LLM response.

    Args:
        llm_response (dict): Response from LLM.
    """
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def main():
    """Main function to prompt user for question and print response."""
    vortex_query = VortexQuery()

    while True:
        print()
        question = input("Question: ")
        answer = vortex_query.ask_question(question)
        display_response(answer)


if __name__ == "__main__":
    main()
