"""Export the OPC Code Atlas (a published artifact's body fragment) to an A4 PDF with Windows Chrome (headless, from
WSL). The snapshot is docs/opc_code_atlas.pdf.

Usage: python scripts/export_atlas_pdf.py FRAGMENT.html docs/opc_code_atlas.pdf "v11, 5 Oct 2026"

FRAGMENT is the page content as published (or as an Artifact read returns it, with the publish skeleton: everything up
to <body> and the closing tags are stripped). The IBM Plex faces are inlined as static woff2 files: Google serves
Plex Sans as a variable font, which Chrome embeds in a PDF as Type3.
"""
import base64
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

FONT_DIR = Path.home() / ".cache" / "opc_atlas_fonts"
# static woff2 faces, pinned (the versions of the 2026-10-04 and -05 exports)
FONT_URL = {"IBMPlexSans": "https://cdn.jsdelivr.net/npm/@ibm/plex-sans@1.1.0/fonts/complete/woff2/{}.woff2",
            "IBMPlexSansCondensed": "https://cdn.jsdelivr.net/npm/@ibm/plex-sans-condensed@2.0.0/fonts/complete/woff2/{}.woff2",
            "IBMPlexMono": "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@2.5.0/fonts/complete/woff2/{}.woff2"}
CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
WIN_TMP = Path("/mnt/c/Temp/atlas_export")
FACES = [("IBM Plex Sans", 400, "normal", "IBMPlexSans-Regular"), ("IBM Plex Sans", 500, "normal", "IBMPlexSans-Medium"),
         ("IBM Plex Sans", 600, "normal", "IBMPlexSans-SemiBold"), ("IBM Plex Sans", 400, "italic", "IBMPlexSans-Italic"),
         ("IBM Plex Sans Condensed", 500, "normal", "IBMPlexSansCondensed-Medium"),
         ("IBM Plex Sans Condensed", 600, "normal", "IBMPlexSansCondensed-SemiBold"),
         ("IBM Plex Mono", 400, "normal", "IBMPlexMono-Regular"), ("IBM Plex Mono", 500, "normal", "IBMPlexMono-Medium")]


def fragment(text: str) -> str:
    if "<body>" in text:
        text = text.split("<body>", 1)[1]
    text = re.sub(r"</body>\s*</html>\s*$", "", text.strip())
    # the Google Fonts links: the faces are inlined below
    return re.sub(r'<link rel="(?:preconnect|stylesheet)"[^>]*>\s*', "", text)


def _font(name: str) -> bytes:
    path = FONT_DIR / f"{name}.woff2"
    if not path.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONT_URL[name.split("-")[0]].format(name), timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def font_css() -> str:
    out = []
    for family, weight, style, name in FACES:
        data = base64.b64encode(_font(name)).decode()
        out.append(f'@font-face{{font-family:"{family}";font-weight:{weight};font-style:{style};'
                   f'src:url(data:font/woff2;base64,{data}) format("woff2")}}')
    return "\n".join(out)


PRINT_CSS = """
@page{size:A4;margin:14mm 12mm 16mm;
  @bottom-center{content:"OPC Code Atlas · %LABEL% · " counter(page) " / " counter(pages);
    font:8.5px "IBM Plex Sans",sans-serif;color:#5B6862}}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact;zoom:.78}
body{font-size:12.5px !important;background:#F4F6F3}
.wrap{display:block !important;max-width:none !important;padding:0 !important}
nav.toc{margin-bottom:18px}
nav.toc ol{flex-direction:row !important}
main{gap:26px !important}
figure .scroll,.tbl{overflow:visible !important}
figure svg{min-width:0 !important}
td.num,td.step,.fx,.chip{white-space:normal !important}
th,td{padding:5px 7px !important}
table{font-size:11.5px !important}
.panel,.risk,.decision,.layer,figure,tr{break-inside:avoid}
h2,h3{break-after:avoid}
"""


def main():
    src, out, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    body = fragment(src.read_text(encoding="utf-8"))
    title = re.search(r"<title>(.*?)</title>", body)
    page = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<title>{title.group(1) if title else 'OPC Code Atlas'}</title>"
            f"<style>{font_css()}</style><style>{PRINT_CSS.replace('%LABEL%', label)}</style></head><body>"
            f"{body}</body></html>")
    WIN_TMP.mkdir(parents=True, exist_ok=True)
    html = WIN_TMP / "atlas_print.html"
    html.write_text(page, encoding="utf-8")
    pdf = WIN_TMP / "atlas.pdf"
    if pdf.exists():
        pdf.unlink()
    win = lambda p: subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()
    profile = WIN_TMP / "chrome_profile"  # our own profile: without it the call goes to the user's running Chrome
    cmd = [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline",
           f"--user-data-dir={win(profile)}", f"--print-to-pdf={win(pdf)}", "file:///" + win(html).replace("\\", "/")]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    shutil.copy(pdf, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
