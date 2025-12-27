import torch

def gelu(x, approximate):
    if not isinstance(approximate, str):
        raise TypeError("gelu(): argument 'approximate' must be str, not bool")

def main():
    x = torch.randn(10)
    try:
        gelu(x, approximate="True")  # Passing a string instead of a boolean
    except TypeError as e:
        print(e)

if __name__ == "__main__":
    main()