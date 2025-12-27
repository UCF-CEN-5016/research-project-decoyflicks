import torch

def main():
    # Attempting to access the 'mps' attribute
    try:
        if torch.backends.mps.is_available():
            print("MPS backend is available")
    except AttributeError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()