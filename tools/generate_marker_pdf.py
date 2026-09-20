"""Printable A4 PDF of labelled ArUco markers for every product and shelf in config/locations.json.

  python tools/generate_marker_pdf.py --all             # markers/pdf/all_markers.pdf
  python tools/generate_marker_pdf.py --products        # markers/pdf/product_markers.pdf
  python tools/generate_marker_pdf.py --shelves         # markers/pdf/shelf_markers.pdf
  python tools/generate_marker_pdf.py --product-ids 101 102 --shelf-ids 203
  python tools/generate_marker_pdf.py --all --product-size-cm 4 --shelf-size-cm 10

Selection: --all / --products / --shelves pick what to print; --product-ids / --shelf-ids (marker IDs)
restrict a type to those IDs and imply that type. With no selection at all, everything is printed.
Files that use --product-ids / --shelf-ids get a "_selected" suffix so they never overwrite the full set.

The marker is drawn as vector squares at the exact physical size (the black square, without the white
quiet zone, is the requested size), using the dictionary named in the config, so it matches the detector.
Nothing is written to the final path until every marker has been detected by the project's own detector,
both from the marker bitmaps and from the rendered PDF pages (which also checks page size and marker size).

PRINT AT 100% / ACTUAL SIZE. The PDF asks viewers not to scale, but only the printer dialog decides.
"""
import argparse
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from reportlab.lib.colors import Color, black
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)
from aruco.detector import SUPPORTED_DICTIONARIES  # noqa: E402
from config.loader import DEFAULT_CONFIG_PATH, ConfigError, load_config  # noqa: E402
from verify_markers import check_single_marker, detect_markers, rasterize_pdf  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(ROOT, "markers", "pdf")
KINDS = ("product", "shelf")

MM = 72.0 / 25.4                       # PDF points per millimetre
PAGE_W, PAGE_H = A4
MARGIN = 7 * MM                        # most printers cannot print closer than ~5 mm to the edge
HEADER_H = 22 * MM
FOOTER_H = 7 * MM
MIN_CELL_W = 42 * MM                   # keeps text readable when the marker is small
MIN_QUIET = 3 * MM                     # smallest white border we accept inside a label
TEXT_PT = {"type": 8.5, "name": 11.0, "id": 9.5}   # base font sizes; scaled up for large markers
LEADING = 1.3
TEXT_PAD = 2 * MM
MAX_TEXT_SCALE = 1.8
VERIFY_DPI = 200
GRAY = Color(0.55, 0.55, 0.55)


class GenerationError(Exception):
    pass


@dataclass(frozen=True)
class Label:
    kind: str
    name: str
    marker_id: int
    dictionary: str
    size_pt: float


@dataclass(frozen=True)
class Layout:
    size: float          # marker (black square) side, points
    quiet: float         # white border around the marker inside the label, points
    cell_w: float
    cell_h: float
    text_scale: float
    cols: int
    rows: int


# ---------------------------------------------------------------- selecting and validating labels

def _dictionary(name):
    return cv2.aruco.getPredefinedDictionary(SUPPORTED_DICTIONARIES[name])


def marker_grid(dictionary_name, marker_id):
    """The marker as a square 0/255 array, one pixel per module, black border included."""
    dictionary = _dictionary(dictionary_name)
    if not 0 <= marker_id < dictionary.bytesList.shape[0]:
        raise GenerationError("ID {} does not exist in dictionary {} (0-{})".format(
            marker_id, dictionary_name, dictionary.bytesList.shape[0] - 1))
    return cv2.aruco.generateImageMarker(dictionary, marker_id, dictionary.markerSize + 2)


def _pick_ids(kind, requested, configured, marker_type, other_kind_ids):
    """Validates requested marker IDs against the config; returns them in the order given, without repeats."""
    if requested is None:
        return sorted(configured)
    picked = []
    for marker_id in requested:
        if marker_id not in configured:
            hint = " (that is a {} marker ID; use --{}-ids)".format(*other_kind_ids[marker_id]) \
                if marker_id in other_kind_ids else ""
            raise GenerationError("{} marker ID {} is not in config/locations.json{}. Configured {} IDs: {}".format(
                kind, marker_id, hint, kind, ", ".join(str(i) for i in sorted(configured)) or "none"))
        if not marker_type.owns(marker_type.dictionary, marker_id):
            raise GenerationError("{} marker ID {} is outside the configured {} ID range {}-{}".format(
                kind, marker_id, kind, marker_type.id_min, marker_type.id_max))
        if marker_id not in picked:
            picked.append(marker_id)
    return picked


def select_labels(config, kinds, product_ids, shelf_ids, sizes_cm):
    """Builds the labels to print, straight from the config. `sizes_cm` maps kind -> physical size."""
    product_t, shelf_t = config.marker_types["product"], config.marker_types["shelf"]
    products = {p.marker_id: p for p in config.products.all()}
    shelves = {s.marker_id: s for s in config.shelves.all()}
    labels = {kind: [] for kind in KINDS}
    if "product" in kinds:
        for marker_id in _pick_ids("product", product_ids, products, product_t, {i: ("shelf", "shelf") for i in shelves}):
            labels["product"].append(Label("product", products[marker_id].name, marker_id, product_t.dictionary,
                                           sizes_cm["product"] * 10 * MM))
    if "shelf" in kinds:
        for marker_id in _pick_ids("shelf", shelf_ids, shelves, shelf_t, {i: ("product", "product") for i in products}):
            labels["shelf"].append(Label("shelf", "Shelf {}".format(shelves[marker_id].shelf_id), marker_id,
                                         shelf_t.dictionary, sizes_cm["shelf"] * 10 * MM))
    for kind in kinds:
        if not labels[kind]:
            raise GenerationError("Nothing to print: no {}s are configured in config/locations.json".format(kind))
    return labels


# ---------------------------------------------------------------- layout

def _text_height(scale):
    return sum(TEXT_PT.values()) * LEADING * scale + 2 * TEXT_PAD


def _try_layout(size, scale):
    avail_w = PAGE_W - 2 * MARGIN
    avail_h = PAGE_H - 2 * MARGIN - HEADER_H - FOOTER_H
    quiet = min(size / 7, (avail_w - size) / 2)      # one module of white, less if the page is too narrow
    if quiet < MIN_QUIET:
        return None
    cell_w = max(size + 2 * quiet, MIN_CELL_W)
    cell_h = 2 * quiet + size + _text_height(scale)
    if cell_h > avail_h:
        return None
    return Layout(size, quiet, cell_w, cell_h, scale, int(avail_w / cell_w + 1e-6), int(avail_h / cell_h + 1e-6))


def plan_layout(size_pt, kind):
    layout = None
    scale = min(max(size_pt / (5 * 10 * MM), 1.0), MAX_TEXT_SCALE)     # bigger text next to bigger markers
    while layout is None and scale >= 1.0 - 1e-9:
        layout = _try_layout(size_pt, scale)
        scale -= 0.1
    if layout is None:
        low, high = 0.0, 500.0 * MM                                     # largest marker that still fits, in mm
        for _ in range(30):
            mid = (low + high) / 2
            low, high = (mid, high) if _try_layout(mid, 1.0) else (low, mid)
        raise GenerationError("A {:.1f} cm {} marker does not fit on an A4 page with its label; "
                              "the largest that does is about {:.1f} cm".format(
                                  size_pt / MM / 10, kind, low / MM / 10 - 0.05))
    return layout


# ---------------------------------------------------------------- drawing

def _fit_text(c, text, font, size, max_w):
    """Shrinks the font (down to 5 pt) and finally truncates so `text` fits in `max_w`."""
    text = text.encode("latin-1", "replace").decode("latin-1")
    while stringWidth(text, font, size) > max_w and size > 5:
        size -= 0.5
    while stringWidth(text, font, size) > max_w and len(text) > 1:
        text = text[:-2].rstrip() + "."
    return text, size


def _draw_marker(c, grid, x, y, size):
    """Draws the marker as one filled path (no seams between touching modules); runs of black modules share a rectangle."""
    n = grid.shape[0]
    module = size / n
    path = c.beginPath()
    for row in range(n):
        col = 0
        while col < n:
            if grid[row, col] != 0:
                col += 1
                continue
            start = col
            while col < n and grid[row, col] == 0:
                col += 1
            path.rect(x + start * module, y + size - (row + 1) * module, (col - start) * module, module)
    c.setFillColor(black)
    c.drawPath(path, stroke=0, fill=1)


def _draw_label(c, layout, label, grid, x, y_top):
    """Draws one cut-out label with its top-left corner at (x, y_top); returns the marker's bottom-left corner."""
    c.setStrokeColor(GRAY)
    c.setLineWidth(0.4)
    c.setDash(2, 2)
    c.rect(x, y_top - layout.cell_h, layout.cell_w, layout.cell_h, stroke=1, fill=0)   # cut line
    c.setDash()
    marker_x = x + (layout.cell_w - layout.size) / 2
    marker_y = y_top - layout.quiet - layout.size
    _draw_marker(c, grid, marker_x, marker_y, layout.size)

    s = layout.text_scale
    cx = x + layout.cell_w / 2
    text_w = layout.cell_w - 4 * MM
    baseline = marker_y - layout.quiet - TEXT_PAD
    c.setFillColor(black)
    for text, font, base in ((label.kind.upper(), "Helvetica-Bold", TEXT_PT["type"]),
                             (label.name, "Helvetica-Bold", TEXT_PT["name"]),
                             ("Marker ID: {}".format(label.marker_id), "Helvetica", TEXT_PT["id"])):
        text, size = _fit_text(c, text, font, base * s, text_w)
        baseline -= base * s * LEADING
        c.setFont(font, size)
        c.drawCentredString(cx, baseline + base * s * (LEADING - 1) * 0.3, text)
    return marker_x, marker_y


def _draw_header(c):
    top = PAGE_H - MARGIN
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN, top - 15, "ArUco Robot Pick & Place")
    c.setFont("Helvetica", 10.5)
    c.drawString(MARGIN, top - 29, "Product / Shelf Marker Labels")
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(Color(0.75, 0.1, 0.1))
    c.drawString(MARGIN, top - 43, "Print at 100% / Actual Size. Do not use Fit to Page.")

    bar_x, bar_y = PAGE_W - MARGIN - 100 * MM, top - 30
    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    c.line(bar_x, bar_y, bar_x + 100 * MM, bar_y)
    for tick in range(11):
        c.line(bar_x + tick * 10 * MM, bar_y, bar_x + tick * 10 * MM, bar_y + (3 if tick in (0, 10) else 1.5) * 1.5)
    c.setFillColor(black)
    c.setFont("Helvetica", 7)
    c.drawRightString(bar_x + 100 * MM, bar_y + 9, "Scale check: this bar must measure exactly 10 cm")
    c.setStrokeColor(GRAY)
    c.setLineWidth(0.4)
    c.line(MARGIN, PAGE_H - MARGIN - HEADER_H + 3 * MM, PAGE_W - MARGIN, PAGE_H - MARGIN - HEADER_H + 3 * MM)


def _draw_footer(c, text):
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 7)
    c.drawString(MARGIN, MARGIN * 0.6, text)


def build_pdf(path, sections):
    """Writes the PDF. `sections` is [(kind, layout, labels)]. Returns [(page_index, label, x, y, size)] in points."""
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle("ArUco Robot Pick & Place - Product / Shelf Marker Labels")
    c.setCreator("tools/generate_marker_pdf.py")
    c.setViewerPreference("PrintScaling", "None")     # ask viewers/printers not to scale the page
    pages = sum(-(-len(labels) // (layout.cols * layout.rows)) for _, layout, labels in sections)
    placements = []
    page = 0
    for kind, layout, labels in sections:
        per_page = layout.cols * layout.rows
        x0 = MARGIN + (PAGE_W - 2 * MARGIN - layout.cols * layout.cell_w) / 2
        top = PAGE_H - MARGIN - HEADER_H
        for start in range(0, len(labels), per_page):
            _draw_header(c)
            _draw_footer(c, "{} markers | dictionary {} | black square {:.1f} cm | page {} of {}".format(
                kind.capitalize(), labels[0].dictionary, layout.size / MM / 10, page + 1, pages))
            for i, label in enumerate(labels[start:start + per_page]):
                row, col = divmod(i, layout.cols)
                mx, my = _draw_label(c, layout, label, marker_grid(label.dictionary, label.marker_id),
                                     x0 + col * layout.cell_w, top - row * layout.cell_h)
                placements.append((page, label, mx, my, layout.size))
            c.showPage()
            page += 1
    c.save()
    return placements


# ---------------------------------------------------------------- verification

def check_bitmaps(labels):
    """Detects every marker from its bitmap (1 module of white around it) before any PDF is written."""
    for label in labels:
        bitmap = np.kron(marker_grid(label.dictionary, label.marker_id), np.full((20, 20), 1, np.uint8))
        bitmap = cv2.copyMakeBorder(bitmap, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        problem = check_single_marker(bitmap, label.dictionary, label.marker_id)
        if problem:
            raise GenerationError("{} {} (id {}) failed detection: {}".format(label.kind, label.name, label.marker_id, problem))


def verify_pdf(path, placements, expected_pages):
    """Renders the PDF like a viewer would and detects each placed marker at the right spot and size."""
    problems = []
    scale = VERIFY_DPI / 72.0
    by_page = {}
    for placement in placements:
        by_page.setdefault(placement[0], []).append(placement)
    pages_seen = 0
    for index, image, (width, height) in rasterize_pdf(path, VERIFY_DPI):
        pages_seen += 1
        if abs(width - PAGE_W) > 0.5 or abs(height - PAGE_H) > 0.5:
            problems.append("page {} is {:.1f}x{:.1f} mm, not A4".format(index + 1, width / MM, height / MM))
        for _, label, mx, my, size in by_page.get(index, []):
            left, right = mx * scale, (mx + size) * scale
            top, bottom = (PAGE_H - my - size) * scale, (PAGE_H - my) * scale
            hits = [d for d in detect_markers(image, label.dictionary)
                    if d.marker_id == label.marker_id and left <= d.center[0] <= right and top <= d.center[1] <= bottom]
            where = "page {}: {} {} (id {})".format(index + 1, label.kind, label.name, label.marker_id)
            if not hits:
                problems.append("{} was not detected in the rendered PDF".format(where))
                continue
            measured_mm = hits[0].side_px / scale / MM
            expected_mm = size / MM
            if abs(measured_mm - expected_mm) > 0.02 * expected_mm + 0.3:
                problems.append("{} measures {:.2f} mm, expected {:.2f} mm".format(where, measured_mm, expected_mm))
    if pages_seen != expected_pages:
        problems.append("PDF has {} pages, expected {}".format(pages_seen, expected_pages))
    if problems:
        raise GenerationError("PDF verification failed:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- CLI

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="products and shelves")
    p.add_argument("--products", action="store_true", help="product markers")
    p.add_argument("--shelves", action="store_true", help="shelf markers")
    p.add_argument("--product-ids", type=int, nargs="+", metavar="ID", help="only these product marker IDs")
    p.add_argument("--shelf-ids", type=int, nargs="+", metavar="ID", help="only these shelf marker IDs")
    p.add_argument("--product-size-cm", type=float, help="physical size of the product marker's black square (default: config)")
    p.add_argument("--shelf-size-cm", type=float, help="physical size of the shelf marker's black square (default: config)")
    p.add_argument("--output", help="output PDF path (default: markers/pdf/<product|shelf|all>_markers.pdf)")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = p.parse_args(argv)
    for name in ("product_size_cm", "shelf_size_cm"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error("--{} must be greater than 0".format(name.replace("_", "-")))
    return args


def output_path(args, kinds):
    if args.output:
        return args.output
    stem = "all" if len(kinds) == 2 else kinds[0]
    suffix = "_selected" if (args.product_ids or args.shelf_ids) else ""
    return os.path.join(DEFAULT_OUT_DIR, "{}_markers{}.pdf".format(stem, suffix))


def generate(args):
    config = load_config(args.config)
    wanted = set()
    if args.all:
        wanted |= set(KINDS)
    if args.products or args.product_ids:
        wanted.add("product")
    if args.shelves or args.shelf_ids:
        wanted.add("shelf")
    kinds = [k for k in KINDS if k in wanted] or list(KINDS)
    sizes_cm = {
        "product": args.product_size_cm or config.marker_types["product"].size_cm,
        "shelf": args.shelf_size_cm or config.marker_types["shelf"].size_cm,
    }

    labels = select_labels(config, kinds, args.product_ids, args.shelf_ids, sizes_cm)
    sections = [(kind, plan_layout(labels[kind][0].size_pt, kind), labels[kind]) for kind in kinds]
    for kind, layout, _ in sections:
        if layout.quiet < layout.size / 7 - 1e-6:
            print("warning: the {} white border is only {:.1f} mm (one module is {:.1f} mm); do not trim it, "
                  "or use a smaller --{}-size-cm".format(kind, layout.quiet / MM, layout.size / 7 / MM, kind))
    check_bitmaps([label for _, _, group in sections for label in group])

    out = output_path(args, kinds)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    try:
        placements = build_pdf(tmp, sections)
        expected_pages = max(p[0] for p in placements) + 1
        verify_pdf(tmp, placements, expected_pages)
        os.replace(tmp, out)
    except PermissionError:
        raise GenerationError("Cannot write {} - is it open in a PDF viewer?".format(out))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print("Wrote {} ({} page{}, all {} markers detected in the rendered PDF)".format(
        out, expected_pages, "" if expected_pages == 1 else "s", len(placements)))
    for kind, layout, group in sections:
        print("  {}: {} marker{}, black square {:.1f} cm, {} per row x {} rows per page".format(
            kind, len(group), "" if len(group) == 1 else "s", layout.size / MM / 10, layout.cols, layout.rows))
    print("Print at 100% / Actual Size (no Fit to Page) and check the 10 cm scale bar with a ruler.")


def main(argv=None):
    args = parse_args(argv)
    try:
        generate(args)
    except (ConfigError, GenerationError) as e:
        print("error: {}".format(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
