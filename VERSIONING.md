# Versioning System Documentation

This document describes the versioning system for the UoM Deep Learning Workshop repository.

## How the Versioning System Works

The versioning system uses Git tags and commits to automatically update version information in the codebase. 
The version consists of two parts:
1. **Version number**: Derived from Git tags (e.g., `v1.2.3`)
2. **Commit hash**: The short hash of the current commit

The full version string is formatted as `{version}+{commit}`, for example: `1.2.3+a1b2c3d`.

## Version Information Storage

The version information is stored in `utils/__version__.py` and contains:
- `__version__`: The current version number (from Git tag)
- `__commit__`: The current commit hash
- `__version_full__`: The full version string (`{version}+{commit}`)

## Automatic Updates via GitHub Actions

A GitHub Actions workflow automatically updates the version file on:
- Pushes to the main/master branch
- New tag creation
- Pull request submissions to the main/master branch

### Setup Instructions for GitHub Actions

1. Make sure you have the `.github/workflows/update_version.yml` file in your repository.

2. **Important**: Configure your repository to allow GitHub Actions to push changes:
   - Go to your repository on GitHub
   - Navigate to Settings > Actions > General
   - Scroll down to "Workflow permissions"
   - Select "Read and write permissions"
   - Click "Save"

3. **Set up a Personal Access Token (PAT) for more secure workflow**:
   - Go to your GitHub account settings
   - Navigate to Developer Settings > Personal access tokens > Tokens (classic)
   - Click "Generate new token"
   - Give it a descriptive name like "Version Update Token"
   - Select the `repo` scope
   - Click "Generate token"
   - **Important**: Copy the token value immediately - you won't be able to see it again!
   - Add the token to your repository secrets:
     - Go to your repository settings
     - Navigate to Secrets and variables > Actions
     - Click "New repository secret"
     - Name: `VERSION_UPDATE_TOKEN`
     - Value: Paste your token
   
   The workflow file is already configured to use this token for pushing changes:
   ```yaml
   # In checkout action:
   - uses: actions/checkout@v3
     with:
       fetch-depth: 0
       token: ${{ secrets.VERSION_UPDATE_TOKEN }}
       
   # And in the push step:
   git push https://${{ secrets.VERSION_UPDATE_TOKEN }}@github.com/${{ github.repository }}.git HEAD:${{ github.ref }}
   ```

## Manual Version Updates

To manually update the version file:

1. Run the `update_version.py` script:
   ```
   python update_version.py
   ```

2. This script will:
   - Retrieve the latest tag from Git (or use default if none exists)
   - Get the current commit hash
   - Update the `utils/__version__.py` file

## Creating Version Tags

To create a new version:

1. Tag your commit with a version number:
   ```
   git tag -a v1.2.3 -m "Version 1.2.3"
   ```

2. Push the tag to the remote repository:
   ```
   git push origin v1.2.3
   ```

3. The GitHub Actions workflow will automatically update the version file based on this new tag.

## Accessing Version Information

In Python code, you can access the version information:

```python
from utils.__version__ import __version__, __commit__, __version_full__

print(f"Version: {__version__}")
print(f"Commit: {__commit__}")
print(f"Full version: {__version_full__}")
```