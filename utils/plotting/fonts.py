from tempfile import NamedTemporaryFile
import urllib3
import matplotlib.font_manager as fm

__all__ = ['load_font', 'FONTS_URLS']

FONTS_URLS = {
    "Roboto Mono": 'https://github.com/google/fonts/blob/main/ofl/spacemono/SpaceMono-Regular.ttf',
    "Share Tech": 'https://github.com/google/fonts/blob/main/ofl/sharetech/ShareTech-Regular.ttf'
}

def load_font(location=FONTS_URLS["Share Tech"]):
    font_url = location + "?raw=true"

    http = urllib3.PoolManager()
    response = http.request("GET", font_url, preload_content=False)
    f = NamedTemporaryFile(delete=False, suffix=".ttf")
    f.write(response.read())
    f.close()

    return fm.FontProperties(fname=f.name, size=12)