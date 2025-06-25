# Migrating from Cloned Repository to Fork with Version Upgrade

## Situation
- Working with a cloned version of someone else's repository (Phono3py 3.14)
- Made local modifications to files
- Want to upgrade to newer version (3.16) while preserving changes
- Need proper fork workflow for future contributions

## Step-by-Step Migration Process

### 1. Backup Your Current Changes

```bash
# Navigate to your current repository
cd /path/to/your/current/repo

# Check what files you've modified
git status
git diff --name-only

# Create a patch file with all your changes
git diff > my_modifications.patch

# Also create a stash as backup
git stash push -m "All my changes before migration to fork"

# Optional: Create a backup branch
git checkout -b backup-before-migration
git add .
git commit -m "Backup of all modifications before migration"
git checkout main
```

### 2. Fork the Original Repository

1. Go to the original repository on GitHub (e.g., https://github.com/phonopy/phono3py)
2. Click the **"Fork"** button in the top right
3. This creates your own copy at `https://github.com/yourusername/phono3py.git`

### 3. Set Up New Repository with Your Fork

```bash
# Navigate to a clean directory (outside your current repo)
cd /path/to/parent/directory

# Clone YOUR fork (replace with your actual fork URL)
git clone https://github.com/yourusername/phono3py.git phono3py_fork
cd phono3py_fork

# Add the original repo as upstream to track updates
git remote add upstream https://github.com/phonopy/phono3py.git

# Verify remotes
git remote -v
# Should show:
# origin    https://github.com/yourusername/phono3py.git (fetch)
# origin    https://github.com/yourusername/phono3py.git (push)
# upstream  https://github.com/phonopy/phono3py.git (fetch)
# upstream  https://github.com/phonopy/phono3py.git (push)

# Fetch the latest version from upstream
git fetch upstream

# Ensure your main branch matches the latest upstream
git checkout main
git reset --hard upstream/main
git push origin main
```

### 4. Create Feature Branch and Apply Changes

```bash
# Create a feature branch for your modifications
git checkout -b feature/my-modifications

# Copy your patch file to the new repository
cp /path/to/old/repo/my_modifications.patch .

# Try to apply your changes automatically
git apply my_modifications.patch

# Check what was applied
git status
git diff
```

### 5. Handle Conflicts (if patch doesn't apply cleanly)

```bash
# If the patch fails due to version differences, manually copy files
# First, check which files you actually modified in the original repo

# Manual copy method:
# Copy modified files from old repo to new repo
cp /path/to/old/repo/path/to/modified/file.py corresponding/new/path/

# Example for your case:
cp /path/to/old/repo/phono3py/phonon3/*.py phono3py/phonon3/
cp /path/to/old/repo/phonon_unfolding/plotting.py phonon_unfolding/

# Review each file to ensure compatibility with new version
git diff
```

### 6. Test and Commit Your Changes

```bash
# Review all changes carefully
git status
git diff

# Test your modifications (run any relevant tests)
# python -m pytest tests/  # if applicable

# Stage and commit your changes
git add .
git commit -m "Port modifications from v3.14 to v3.16

- Modified files: [list your modified files]
- Changes include: [brief description of your modifications]
- Tested compatibility with v3.16"

# Push to your fork
git push origin feature/my-modifications
```

### 7. Future Workflow - Staying Up to Date

#### Before starting new work:
```bash
# Always sync with upstream first
git checkout main
git fetch upstream
git reset --hard upstream/main
git push origin main

# Create a new feature branch for new work
git checkout -b feature/new-feature-name
```

#### Regular updates to your feature branch:
```bash
# Update main branch
git checkout main
git fetch upstream
git reset --hard upstream/main
git push origin main

# Rebase your feature branch on latest main
git checkout feature/my-modifications
git rebase main

# If conflicts occur, resolve them and continue
git add .
git rebase --continue

# Force push to update your feature branch (after rebase)
git push origin feature/my-modifications --force-with-lease
```

#### Contributing back to the original project:
1. Push your feature branch to your fork
2. Create a Pull Request from your fork to the original repository
3. Follow the project's contribution guidelines

## Best Practices for Fork Workflow

### Do's ✅
- **Always work on feature branches**, never directly on main
- **Keep your fork's main branch synced** with upstream
- **Write descriptive commit messages**
- **Test your changes** before committing
- **Follow the project's coding standards**
- **Create focused, single-purpose branches**

### Don'ts ❌
- **Never push directly to main branch**
- **Don't work on multiple unrelated features in one branch**
- **Don't force push to main branch**
- **Don't ignore merge conflicts**

## Troubleshooting

### If patch application fails:
```bash
# Check what failed
git apply --check my_modifications.patch

# Apply what you can and handle conflicts manually
git apply --reject my_modifications.patch

# This creates .rej files showing what couldn't be applied
# Manually merge these changes
```

### If your changes conflict with new version:
1. Carefully review each conflict
2. Test functionality after resolving conflicts
3. Consider if your changes are still necessary/relevant
4. Update your changes to work with new APIs if needed

### If you need to start over:
```bash
# Delete your feature branch and start fresh
git checkout main
git branch -D feature/my-modifications
git checkout -b feature/my-modifications
# Re-apply changes carefully
```

## Quick Reference Commands

```bash
# Sync with upstream
git fetch upstream && git checkout main && git reset --hard upstream/main && git push origin main

# Update feature branch
git checkout feature/my-modifications && git rebase main

# Create new feature branch
git checkout main && git checkout -b feature/new-feature

# Check repository status
git remote -v && git status && git log --oneline -5
```

## File Organization

Keep this structure for easy reference:
```
your-project/
├── original-repo-backup/          # Your old cloned repo
├── your-fork/                     # Your new forked repo
├── migration-patches/             # Backup patches
│   ├── my_modifications.patch
│   └── backup-notes.md
└── migration-guide.md            # This document
```

---

**Note**: Replace `yourusername`, `phono3py`, and file paths with your actual values. Always test your changes thoroughly after migration.