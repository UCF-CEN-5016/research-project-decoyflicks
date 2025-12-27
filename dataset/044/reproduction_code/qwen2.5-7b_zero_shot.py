def filter_oov(text, allowed):
    return ''.join([c for c in text if c in allowed])

def main():
    allowed = {'a', 'b', 'c'}
    input_text = 'defg'
    filtered = filter_oov(input_text, allowed)
    print(filtered)

if __name__ == "__main__":
    main()