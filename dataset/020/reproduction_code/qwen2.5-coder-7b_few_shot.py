def merge_using_update(source_dict: dict, additional_dict: dict) -> dict:
    """
    Merge two dictionaries using dict.update().
    Returns a new dictionary containing keys from source_dict updated with additional_dict.
    """
    result = dict(source_dict)
    result.update(additional_dict)
    return result


def merge_using_unpacking(first_dict: dict, second_dict: dict) -> dict:
    """
    Merge two dictionaries using dictionary unpacking (**).
    Returns a new dictionary containing keys from both dictionaries.
    """
    return {**first_dict, **second_dict}


def main() -> None:
    # Example 1: merge using update()
    a = {'a': 1}
    b = {'b': 2}
    merged = merge_using_update(a, b)
    print("Merged dictionary:", merged)

    # Example 2: merge using unpacking
    c = {'a': 1, 'b': 2}
    d = {'c': 3, 'd': 4}
    merged = merge_using_unpacking(c, d)
    print("Merged dictionary:", merged)


if __name__ == "__main__":
    main()