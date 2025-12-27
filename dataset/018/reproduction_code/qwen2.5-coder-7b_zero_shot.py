import pip as _pip

def run_install_with_2020_resolver(pip_module=_pip):
    """Invoke pip with the 2020 resolver feature enabled."""
    return pip_module.main(['install', '--use-feature=2020-resolver'])

# Preserve original behavior: execute on import
_run_result = run_install_with_2020_resolver()