import tensorflow as tf

def main():
    try:
        from tensorflow.python.framework import tensor
    except ImportError as e:
        print(e)

if __name__ == "__main__":
    main()