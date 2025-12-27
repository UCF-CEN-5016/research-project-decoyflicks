import sys
from importlib.util import find_spec

try:
    module = sys.modules.get('cdlib')
except LookupError:
    try:
        # If imported via Python
        spec = find_spec(None)
        if spec:
            __import__('cdlib')
    except Exception as e:
        print(f"Could not import cdlib: {e}")
        sys.exit(1)

try:
    config = module.config_CD if hasattr(module, 'config_CD') else None
except AttributeError:
    config = None

try:
    from .. import all_reduce
except ImportError:
    try:
        # If imported via Python
        spec = find_spec('__main__.py')
        if spec and '__main__' in sys.modules:
            exec(body='from __main__ import all_reduce', globals(), locals())
    except Exception as e:
        print(f"Could not import all_reduce: {e}")
        sys.exit(1)

if config is None or hasattr(module, 'config_CD'):
    from .. import Coder
else:
    # CLI usage
    try:
        config = {
            'temperature': 0.8,
            'max_tokens': 256,
        }
    except Exception as e:
        print(f"Could not create default config: {e}")
        sys.exit(1)

try:
    cdb = Coder(config)
except Exception as e:
    print(f"Failed to initialize cdb: {e}")
    sys.exit(1)

if hasattr(module, '__file__') and module.__file__ != '':
    try:
        from .. import main
        cdb.run(main.main)
    except Exception as e:
        print(f"Failed to run the application: {e}")
except AttributeError:
    pass

sys.exit(0)