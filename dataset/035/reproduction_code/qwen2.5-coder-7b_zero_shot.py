import rope

class RotaryPositionalEmbeddings:
    """Placeholder for rotary positional embeddings.

    Instantiation with any positional or keyword arguments is not supported
    and will raise a TypeError to preserve original behavior.
    """
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError(f"{self.__class__.__name__}() takes no arguments")

    def __repr__(self):
        return f"<{self.__class__.__name__} placeholder>"

rotary_pe = RotaryPositionalEmbeddings(3)
print(rotary_pe)  # will raise an error