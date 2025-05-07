# This script updates the CHANGELOG.md file with new commits since the last recorded commit hash.
# 5268e00

import subprocess
from pathlib import Path

# Define the path to your changelog
CHANGELOG_PATH = Path("CHANGELOG.md")

# Replace this with your actual last recorded commit hash
LAST_COMMIT_HASH = "5268e00"

# Define the git log format
git_log_format = "## %h\n#### %ad\n\n%s\n\n%b\n"

# Fetch new commits since the last recorded commit
new_commits = subprocess.run(
    ["git", "log", f"{LAST_COMMIT_HASH}..HEAD", "--pretty=format:" + git_log_format, "--date=short"],
    capture_output=True,
    text=True,
    encoding="utf-8"
).stdout

# Read the existing changelog content
if CHANGELOG_PATH.exists():
    existing_content = CHANGELOG_PATH.read_text(encoding="utf-8")
    # Remove BOM if present
    if existing_content.startswith('\ufeff'):
        existing_content = existing_content.lstrip('\ufeff')
else:
    existing_content = ""

# Prepend new commits to the existing content
updated_content = new_commits.strip() + "\n\n" + existing_content.strip()

# Write the updated content back to the changelog
CHANGELOG_PATH.write_text(updated_content, encoding="utf-8")
