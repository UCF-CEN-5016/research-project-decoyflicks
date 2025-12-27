import os
from dominoai import domino as cd

DB_ENV_KEY = 'DB_URI'
DB_CONNECTION_STRING = 'your-database-connection-string'


def set_database_env(key=DB_ENV_KEY, value=DB_CONNECTION_STRING):
    os.environ[key] = value


def initialize_cd():
    try:
        cd.init()
    except Exception as e:
        print(f"Initialization failed: {e}")


def safe_all_reduce(tensor):
    try:
        return cd.all_reduce(tensor)
    except AttributeError as e:
        print(f"AttributeError occurred: {e}")
        return None


def main():
    set_database_env()
    initialize_cd()
    tensor = [[1, 2], [3, 4]]
    _ = safe_all_reduce(tensor)


if __name__ == '__main__':
    main()