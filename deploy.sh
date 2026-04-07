#!/usr/bin/env bash
# ============================================================
# MedTriageEnv — Hugging Face Spaces Deployment Script
# ============================================================
# Usage:
#   export HF_USERNAME=your_hf_username
#   export HF_TOKEN=your_hf_token
#   bash deploy.sh
#
# What this does:
#   1. Creates a new HF Space (Docker SDK)
#   2. Pushes all environment files
#   3. Prints the Space URL
# ============================================================

set -euo pipefail

HF_USERNAME="${HF_USERNAME:?Set HF_USERNAME}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
SPACE_NAME="medtriage-env"
REPO_ID="${HF_USERNAME}/${SPACE_NAME}"

echo "============================================================"
echo "  Deploying MedTriageEnv to Hugging Face Spaces"
echo "  Repo: ${REPO_ID}"
echo "============================================================"

# Install huggingface_hub if needed
pip install huggingface_hub --quiet --break-system-packages 2>/dev/null || \
pip install huggingface_hub --quiet

# Login
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

# Create space (skip if exists)
python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        repo_id="${REPO_ID}",
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("Space created (or already exists): ${REPO_ID}")
except Exception as e:
    print(f"Note: {e}")
EOF

# Merge HF Spaces README with main README
cat SPACES_README.md README.md > /tmp/hf_readme.md

# Upload files
python3 - <<EOF
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "${REPO_ID}"

files_to_upload = [
    ("/tmp/hf_readme.md", "README.md"),
    ("Dockerfile", "Dockerfile"),
    ("inference.py", "inference.py"),
    ("pyproject.toml", "pyproject.toml"),
    ("LICENSE", "LICENSE"),
    # Environment package
    ("medtriage_env/__init__.py",                   "medtriage_env/__init__.py"),
    ("medtriage_env/models.py",                     "medtriage_env/models.py"),
    ("medtriage_env/scenarios.py",                  "medtriage_env/scenarios.py"),
    ("medtriage_env/graders.py",                    "medtriage_env/graders.py"),
    ("medtriage_env/client.py",                     "medtriage_env/client.py"),
    ("medtriage_env/openenv.yaml",                  "medtriage_env/openenv.yaml"),
    ("medtriage_env/server/__init__.py",             "medtriage_env/server/__init__.py"),
    ("medtriage_env/server/app.py",                 "medtriage_env/server/app.py"),
    ("medtriage_env/server/environment.py",         "medtriage_env/server/environment.py"),
    ("medtriage_env/server/requirements.txt",       "medtriage_env/server/requirements.txt"),
    # Tests
    ("tests/test_medtriage.py",                     "tests/test_medtriage.py"),
]

for local_path, remote_path in files_to_upload:
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="space",
        )
        print(f"  Uploaded: {remote_path}")
    else:
        print(f"  Skipped (not found): {local_path}")

print("\nAll files uploaded!")
EOF

echo ""
echo "============================================================"
echo "  Deployment complete!"
echo "  Space URL: https://huggingface.co/spaces/${REPO_ID}"
echo "  API URL:   https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo ""
echo "  Test with:"
echo "    curl https://${HF_USERNAME}-${SPACE_NAME}.hf.space/health"
echo "============================================================"
