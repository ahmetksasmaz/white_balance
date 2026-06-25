import os

DATASET_ENV_VARS = {
    "cubepp":       ("DATASET_CUBEPP_ROOT",    "Cube++"),
    "gehler":       ("DATASET_GEHLER_ROOT",    "ShiGehler"),
    "nus8":         ("DATASET_NUS8_ROOT",      "NUS8"),
    "nus8extended": ("DATASET_NUS8_ROOT",      "NUS8 (extended)"),
    "lsmi":         ("DATASET_LSMI_ROOT",      "LSMI"),
    "miniature":    ("DATASET_MINIATURE_ROOT", "Miniature"),
    "reallife":     ("DATASET_REALLIFE_ROOT",  "Reallife"),
}


def load_dotenv_if_present(env_path=None):
    if env_path is None:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def validate_dataset_paths(active_datasets):
    errors = []
    seen = set()
    for key in active_datasets:
        if key not in DATASET_ENV_VARS:
            continue
        env_var, name = DATASET_ENV_VARS[key]
        if env_var in seen:
            continue
        seen.add(env_var)
        root = os.environ.get(env_var) or ""
        if not root:
            errors.append(
                f"  [{key}] '{env_var}' is not set.\n"
                f"         export {env_var}=/path/to/{name}"
            )
        elif not os.path.isdir(root):
            errors.append(
                f"  [{key}] {env_var}='{root}' does not exist or is not a directory."
            )
    if errors:
        raise SystemExit(
            "\nDataset path validation failed:\n" + "\n".join(errors) +
            "\n\nSee .env.example for the full list of required variables.\n"
        )
