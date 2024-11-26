from distilabel_steps_library import hey

if hey() == "hi":
    print("Test passed!")
else:
    RuntimeError("Failed smoke test.")
