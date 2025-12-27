import matplotlib.pyplot as plt

def display_matplotlib_backend():
    print("Before importing tensorflow_models, backend:", plt.get_backend())

def plot_data():
    plt.plot([1, 2, 3], [4, 5, 1])
    plt.show()

def main():
    display_matplotlib_backend()
    import tensorflow_models
    print("After importing tensorflow_models, backend:", plt.get_backend())
    plot_data()

if __name__ == "__main__":
    main()