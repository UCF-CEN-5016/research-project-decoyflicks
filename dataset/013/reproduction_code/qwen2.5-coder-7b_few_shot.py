def _prepare_plotting():
    # Import pyplot first (as in the original code), then set backend and import other modules.
    import matplotlib.pyplot as plt_alias
    import matplotlib
    matplotlib.use('TkAgg')  # Set to interactive backend before importing
    import tensorflow_models  # This may change the matplotlib backend
    return plt_alias

def _create_test_plot(plotting_lib):
    plotting_lib.plot([1, 2, 3])
    plotting_lib.title("Test Plot")
    plotting_lib.show()

def main():
    plt = _prepare_plotting()
    _create_test_plot(plt)

if __name__ == "__main__":
    main()