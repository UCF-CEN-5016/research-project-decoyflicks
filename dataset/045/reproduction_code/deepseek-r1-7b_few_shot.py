class ExampleClass:
    pass

def main():
    obj = ExampleClass()
    try:
        print(obj.input_feat_per_channel)
    except AttributeError as e:
        print(f"AttributeError: {e}")

if __name__ == "__main__":
    main()