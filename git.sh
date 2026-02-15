#!/bin/bash
set -e

# ====== USER CONFIGURATION ======
REPO_NAME="IITD_Feb26_AAIPL"
COMMIT_MESSAGE="Final Submission: Optimized Agents and Cleanup"
GITHUB_USERNAME="tusharchaudharryy"
# =================================

# 1. Initialize Git if missing
if [ ! -d .git ]; then
    git init
fi

# 2. Configure User
git config user.name "$GITHUB_USERNAME"
git config user.email "chaudharytushar477@gmail.com"

# 3. Create .gitignore (Safety Net)
echo "hf_models/" > .gitignore
echo "data/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.log" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore

# 4. Add & Commit
git add .
git commit -m "$COMMIT_MESSAGE" || echo "Nothing new to commit"
git branch -M main

# 5. Set Remote & Push
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
git remote set-url origin "$REMOTE_URL" 2>/dev/null || git remote add origin "$REMOTE_URL"

echo "Pushing to GitHub..."
git push -u origin main

echo "✅ DONE! Verify at: $REMOTE_URL"
