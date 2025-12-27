def initialize_deepspeed():
    """Attempt to import DeepSpeed integration from transformers and report status."""
    try:
        import transformers  # noqa: F401
        import transformers.deepspeed  # noqa: F401
    except ModuleNotFoundError:
        print("DeepSpeed module is not installed.")
    else:
        print("DeepSpeed integration initialized")


if __name__ == "__main__":
    initialize_deepspeed()