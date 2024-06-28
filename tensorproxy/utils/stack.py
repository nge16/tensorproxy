import inspect


def inspect_stack() -> str:
    stack = inspect.stack()
    functions = [
        f"{context.strip()}, file={frame.filename}, line={frame.lineno}"
        for frame in stack
        for context in frame.code_context  # type: ignore
    ]
    functions = "\n\t".join(functions)
    return functions
