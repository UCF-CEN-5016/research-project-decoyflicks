try:
    import transformers
    import transformers.deepspeed
    print("DeepSpeed integration initialized")
except ModuleNotFoundError:
    print("DeepSpeed module is not installed.")