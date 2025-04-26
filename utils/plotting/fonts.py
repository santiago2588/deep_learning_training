"""
Font management utilities for plotting in the UoM Deep Learning Workshop.

This module provides functions for downloading, managing and applying
consistent fonts across all visualizations in the workshop materials.
It handles temporary font file management and cleanup automatically.
"""

from tempfile import NamedTemporaryFile
import urllib3
import matplotlib.font_manager as fm
import os
import atexit

__all__ = ['load_font', 'FONTS_URLS']

FONTS_URLS = {
    "Roboto Mono": 'https://github.com/google/fonts/blob/main/ofl/spacemono/SpaceMono-Regular.ttf',
    "Share Tech": 'https://github.com/google/fonts/blob/main/ofl/sharetech/ShareTech-Regular.ttf'
}

# Store temporary file paths for cleanup
_temp_files = []

def cleanup_temp_files():
    """
    Delete all temporary font files at exit.
    
    This function is registered with atexit to ensure proper cleanup
    of downloaded font files when the Python interpreter exits.
    """
    for file_path in _temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

def load_font(location=FONTS_URLS["Share Tech"]):
    """
    Download and load a custom font for matplotlib visualizations.
    
    This function handles downloading, caching, and loading fonts for
    consistent typography in all workshop visualizations. It attempts
    to reuse previously downloaded fonts when possible.
    
    Args:
        location (str): URL of the font to download, defaults to Share Tech
    
    Returns:
        FontProperties: A matplotlib FontProperties object configured with the loaded font
        
    Note:
        Downloaded fonts are stored in temporary files that are automatically
        removed when the Python interpreter exits.
    """
    font_url = location + "?raw=true"
    
    # Check if we already have this font downloaded
    for temp_file in _temp_files:
        if os.path.exists(temp_file):
            try:
                return fm.FontProperties(fname=temp_file, size=12)
            except:
                # If font file is corrupted, continue to download
                pass

    # Download and save the font
    http = urllib3.PoolManager()
    response = http.request("GET", font_url, preload_content=False)
    f = NamedTemporaryFile(delete=False, suffix=".ttf")
    f.write(response.read())
    f.close()
    
    # Add to cleanup list
    _temp_files.append(f.name)
    
    return fm.FontProperties(fname=f.name, size=12)

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)