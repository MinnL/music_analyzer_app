# Git Workflow for Music Analyzer App

This document provides a basic Git workflow for working with the Music Analyzer App repository.

## Setting Up the Repository

If you're starting from scratch:

```bash
# Clone the repository
git clone https://github.com/your-username/music_analyzer_app.git
cd music_analyzer_app

# Set up your local environment
cp .env.example .env
# Edit .env with your preferred settings
```

## Basic Git Commands

### Check Status

To see which files have been changed:

```bash
git status
```

### Add Changes

To stage changes for commit:

```bash
# Add specific files
git add filename.py

# Add all changes
git add .
```

### Commit Changes

To commit your staged changes:

```bash
git commit -m "Brief description of your changes"
```

### View Commit History

To see the commit history:

```bash
# Show all commits
git log

# Show compact version
git log --oneline

# Show graph of branches
git log --graph --oneline --all
```

## Feature Development Workflow

When working on a new feature:

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit your changes
git add .
git commit -m "Implement your feature"

# Push to your remote repository
git push -u origin feature/your-feature-name
```

## Bug Fix Workflow

When fixing a bug:

```bash
# Create a new branch
git checkout -b fix/bug-description

# Make your changes
# ...

# Commit your changes
git add .
git commit -m "Fix: description of the bug fix"

# Push to your remote repository
git push -u origin fix/bug-description
```

## Best Practices

1. **Commit Often**: Make small, focused commits that do one thing well.
2. **Write Clear Commit Messages**: Use descriptive commit messages that explain what and why.
3. **Keep Branches Updated**: Regularly pull changes from the main branch.
4. **Clean Up Branches**: Delete branches after they've been merged.

## Common Git Issues

### Resolving Merge Conflicts

If you encounter merge conflicts:

```bash
# After a merge conflict occurs
git status  # Shows files with conflicts

# Edit the files to resolve conflicts
# Files will have markers showing conflict areas

# After resolving, stage the files
git add .

# Complete the merge
git commit -m "Resolve merge conflicts"
```

### Undoing Changes

To undo uncommitted changes:

```bash
# Discard changes in working directory
git checkout -- filename.py

# Unstage a file
git reset HEAD filename.py

# Undo the last commit but keep changes
git reset --soft HEAD~1

# Completely undo the last commit
git reset --hard HEAD~1
```

## Git Hooks

The repository includes pre-commit hooks to ensure code quality. These will run automatically when you commit code.

If a hook fails, address the issues before committing again.

## Continuous Integration

The repository uses GitHub Actions for continuous integration. When you push changes, automated tests will run.

Check the Actions tab on GitHub to see test results. 