from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("ERROR: HF_TOKEN not set")
    exit(1)

api = HfApi(token=token)
REPO_ID = "unaisdev/medtriage-env"

print(f"Creating space: {REPO_ID}")
api.create_repo(
    repo_id=REPO_ID,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)
print("Space created!")
print(f"URL: https://huggingface.co/spaces/{REPO_ID}")
