#!/bin/bash

# Replace with your actual last commit hash
LAST_COMMIT_HASH="4a02fe9"

# Generate new commits
git log --pretty=format:"## %h%n#### %ad%n%n%s%n%n%b%n" --date=short $LAST_COMMIT_HASH..HEAD > temp_changelog.md

# Prepend to existing CHANGELOG.md
cat temp_changelog.md CHANGELOG.md > new_changelog.md && mv new_changelog.md CHANGELOG.md

# Clean up
rm temp_changelog.md
echo "CHANGELOG.md has been updated with new commits since $LAST_COMMIT_HASH."