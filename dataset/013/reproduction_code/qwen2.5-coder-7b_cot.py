import matplotlib.pyplot as plt

def set_interactive_backend(backend: str = 'TkAgg') -> None:
    """Force matplotlib to use the specified interactive backend."""
    plt.switch_backend(backend)

def plot_values(values) -> None:
    """Plot the provided sequence of values and display the figure."""
    plt.plot(values)
    plt.show()

def main() -> None:
    set_interactive_backend('TkAgg')
    data = [1, 2, 3]
    plot_values(data)

if __name__ == '__main__':
    main()