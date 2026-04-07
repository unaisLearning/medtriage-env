from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN", "")
api = HfApi(token=token)
REPO_ID = "unaisdev/medtriage-env"

files = [
    ("Dockerfile", "Dockerfile"),
    ("inference.py", "inference.py"),
    ("pyproject.toml", "pyproject.toml"),
    ("README.md", "README.md"),
    ("test_medtriage.py", "test_medtriage.py"),
    ("medtriage_env/__init__.py", "medtriage_env/__init__.py"),
    ("medtriage_env/models.py", "medtriage_env/models.py"),
    ("medtriage_env/scenarios.py", "medtriage_env/scenarios.py"),
    ("medtriage_env/graders.py", "medtriage_env/graders.py"),
    ("medtriage_env/client.py", "medtriage_env/client.py"),
    ("medtriage_env/openenv.yaml", "medtriage_env/openenv.yaml"),
    ("medtriage_env/server/__init__.py", "medtriage_env/server/__init__.py"),
    ("medtriage_env/server/app.py", "medtriage_env/server/app.py"),
    ("medtriage_env/server/environment.py", "medtriage_env/server/environment.py"),
    ("medtriage_env/server/requirements.txt", "medtriage_env/server/requirements.txt"),
]

print("Uploading files to " + REPO_ID)
for local, remote in files:
    if os.path.exists(local):
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO_ID, repo_type="space")
        print("  OK: " + remote)
    else:
        print("  MISSING: " + local)

print("Done! https://huggingface.co/spaces/" + REPO_ID)
