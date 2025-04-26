"""
Version information for the UoM Deep Learning Workshop utilities package.

This module contains version information which is used throughout the package
for version checking, logging, and compatibility information.
"""

import os
import subprocess
from typing import Tuple, Optional

def get_version_info() -> Tuple[str, Optional[str]]:
    """
    Get version information from Git.
    
    Returns:
        Tuple containing (version_string, git_commit_hash)
        If not in a git repository, commit hash will be None
    """
    # Default version if not in a Git repository
    version = '0.9.0'
    commit_hash = 'c02fe65'
    
    try:
        # Try to get the current tag
        tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        if tag:
            version = tag.lstrip('v')  # Remove 'v' prefix if present
        
        # Get the current commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('utf-8').strip()
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repository or git not installed
        pass
    
    return version, commit_hash

# Get version information
__version__, __commit__ = get_version_info()

# Full version string including commit hash if available
__version_full__ = __version__
if __commit__:
    __version_full__ = f"{__version__}+{__commit__}"
