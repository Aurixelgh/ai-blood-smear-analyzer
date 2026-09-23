"""
Shared design system for the AI Blood Smear Analyzer Streamlit app.

Single source of truth for color tokens, typography, and reusable HTML
fragments so every page renders with one consistent visual identity instead
of ad-hoc per-page styling. Injected via st.markdown(unsafe_allow_html=True)
-- Streamlit itself remains the application framework (per the existing
architecture); this module only restyles its chrome and adds small,
purpose-built HTML/CSS fragments (hero, step indicator, badges, cards).

Nothing in this module renders or infers any model output -- it is pure
presentation. Every function that takes real data (counts, confidences,
labels) renders exactly what it is given; it never invents a placeholder
value for missing data. Callers are responsible for passing None/empty
states explicitly (see e.g. render_metric_or_pending()).
"""
import html as _html

import streamlit as st

# ---------------------------------------------------------------------------
# Color tokens -- sophisticated scientific palette (spec: near-black/charcoal
# base, deep burgundy/crimson, restrained blood-red ACCENT only, warm
# off-white, muted pink, neutral scientific grays). Red is never the whole UI.
# ---------------------------------------------------------------------------
BG_PRIMARY = "#120F0E"
BG_SECONDARY = "#1B1614"
BG_TERTIARY = "#241D1A"
BORDER_SUBTLE = "#332924"
BORDER_STRONG = "#4A3B35"
TEXT_PRIMARY = "#F2EAE3"
TEXT_MUTED = "#B4A89F"
TEXT_FAINT = "#7C716A"
BURGUNDY = "#3D1420"
CRIMSON = "#6E1F2B"
ACCENT = "#C13A46"
ACCENT_HOVER = "#D6505C"
ACCENT_SOFT = "rgba(193, 58, 70, 0.14)"
MUTED_PINK = "#D6A6A9"
SCI_GRAY = "#8B8781"
SUCCESS = "#5C9A7C"
WARNING = "#C79A4B"

_FONTS_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap"
)


def html_block(text: str) -> str:
    """Collapse a multi-line HTML/CSS/SVG fragment to one contiguous block
    with every line flush-left and no blank lines.

    Streamlit's st.markdown runs content through a CommonMark parser even
    with unsafe_allow_html=True. Two CommonMark rules otherwise corrupt any
    HTML/CSS built from an indented triple-quoted Python string: (1) a line
    starting a block with >=4 spaces of indentation is parsed as a literal
    indented code block rather than HTML, and (2) a blank line always ends
    an open HTML block, so any interior blank line (e.g. for readability
    between CSS rule groups) splits one <style>/<div> block into several,
    with the later fragments no longer recognized as HTML and rendered as
    visible raw text. Stripping each line and dropping blanks avoids both
    failure modes. Safe for pure HTML/CSS/SVG (no <pre>/<textarea> content)
    since browsers already collapse interior whitespace/newlines in normal
    flow, so this never changes the rendered result -- only prose paragraphs
    that rely on blank lines for paragraph breaks must not use this."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def inject_global_css() -> None:
    """Call once near the top of every page. Idempotent -- Streamlit dedupes
    identical <style> blocks across reruns within a session naturally since
    it's just re-injected markdown, no persistent side effect."""
    st.markdown(
        html_block(f"""
        <link rel="stylesheet" href="{_FONTS_IMPORT}">
        <style>
        :root {{
            --bg-primary: {BG_PRIMARY};
            --bg-secondary: {BG_SECONDARY};
            --bg-tertiary: {BG_TERTIARY};
            --border-subtle: {BORDER_SUBTLE};
            --border-strong: {BORDER_STRONG};
            --text-primary: {TEXT_PRIMARY};
            --text-muted: {TEXT_MUTED};
            --text-faint: {TEXT_FAINT};
            --burgundy: {BURGUNDY};
            --crimson: {CRIMSON};
            --accent: {ACCENT};
            --accent-hover: {ACCENT_HOVER};
            --accent-soft: {ACCENT_SOFT};
            --muted-pink: {MUTED_PINK};
            --sci-gray: {SCI_GRAY};
            --success: {SUCCESS};
            --warning: {WARNING};
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: var(--bg-primary) !important;
            color: var(--text-primary);
        }}
        [data-testid="stHeader"] {{ background-color: transparent !important; }}
        [data-testid="stAppViewContainer"] * {{ font-family: 'Inter', -apple-system, sans-serif; }}
        h1, h2, h3, h4, .sci-heading {{
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
            letter-spacing: -0.01em;
        }}
        [data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; max-width: 1180px; }}

        /* Top navigation (st.navigation position="top") */
        [data-testid="stTopNav"], header[data-testid="stHeader"] {{
            background-color: var(--bg-primary) !important;
            border-bottom: 1px solid var(--border-subtle);
        }}
        [data-testid="stTopNav"] a, [data-testid="stTopNavSection"] a {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-muted) !important;
        }}
        [data-testid="stTopNav"] a[aria-selected="true"] {{ color: var(--text-primary) !important; }}

        /* Sidebar (used for filters, not primary nav) */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-subtle);
        }}

        /* Buttons */
        [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] button,
        button[kind="primary"] {{
            background: linear-gradient(180deg, var(--accent), var(--crimson)) !important;
            border: none !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-size: 0.82rem !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 0 rgba(0,0,0,0.25);
        }}
        [data-testid="stBaseButton-secondary"] button, button[kind="secondary"] {{
            background-color: transparent !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--text-primary) !important;
            border-radius: 6px !important;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            font-size: 0.82rem !important;
        }}

        /* File uploader -> "insert a slide" dropzone */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: var(--bg-secondary) !important;
            border: 1.5px dashed var(--border-strong) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--accent) !important; }}

        /* Metrics */
        [data-testid="stMetric"] {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0.9rem 1rem 0.7rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem !important;
            color: var(--text-muted) !important;
        }}
        [data-testid="stMetricValue"] {{ color: var(--text-primary) !important; font-family: 'Space Grotesk', sans-serif; }}

        /* Tabs */
        [data-testid="stTabs"] [data-baseweb="tab"] {{
            text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.05em;
            color: var(--text-muted);
        }}
        [data-testid="stTabs"] [aria-selected="true"] {{ color: var(--text-primary) !important; }}

        /* Progress bar accent */
        [data-testid="stProgress"] > div > div > div {{ background-color: var(--accent) !important; }}

        /* Dataframe */
        [data-testid="stDataFrame"] {{ border: 1px solid var(--border-subtle); border-radius: 8px; }}

        /* Captions / dividers */
        [data-testid="stCaptionContainer"] {{ color: var(--text-faint) !important; }}
        hr {{ border-color: var(--border-subtle) !important; }}

        /* Custom component classes */
        .sci-label {{
            text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.72rem;
            font-weight: 600; color: var(--text-muted);
        }}
        .sci-card {{
            background: var(--bg-secondary); border: 1px solid var(--border-subtle);
            border-radius: 12px; padding: 1.1rem 1.3rem;
        }}
        .sci-badge {{
            display: inline-flex; align-items: center; gap: 0.35rem;
            padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
        }}
        .sci-badge-accent {{ background: var(--accent-soft); color: var(--muted-pink); border: 1px solid rgba(193,58,70,0.35); }}
        .sci-badge-muted {{ background: var(--bg-tertiary); color: var(--text-muted); border: 1px solid var(--border-subtle); }}
        .sci-badge-warn {{ background: rgba(199,154,75,0.14); color: var(--warning); border: 1px solid rgba(199,154,75,0.35); }}
        .sci-disclaimer {{
            font-size: 0.78rem; color: var(--text-faint); border-top: 1px solid var(--border-subtle);
            padding-top: 0.6rem; margin-top: 0.4rem;
        }}
        .sci-divider {{ height: 1px; background: var(--border-subtle); margin: 1.6rem 0; border: none; }}
        </style>
        """),
        unsafe_allow_html=True,
    )



def esc(value) -> str:
    return _html.escape(str(value))


def section_label(text: str) -> None:
    st.markdown(f"<div class='sci-label'>{esc(text)}</div>", unsafe_allow_html=True)


def disclaimer_line(text: str = "Research & educational use • Not a clinical diagnostic system") -> None:
    st.markdown(f"<div class='sci-disclaimer'>{esc(text)}</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "muted") -> str:
    cls = {"accent": "sci-badge-accent", "muted": "sci-badge-muted", "warn": "sci-badge-warn"}.get(kind, "sci-badge-muted")
    return f"<span class='sci-badge {cls}'>{esc(text)}</span>"


def pipeline_steps(steps: list, active_index: int = None, done_index: int = None) -> None:
    """Renders a horizontal pipeline/step indicator, e.g.
    ["Image Quality", "Cell Detection", "Classification", "Morphology", "Results"].
    `done_index`: steps with index <= done_index render as completed.
    `active_index`: the currently-running step (pulses); None if idle/finished."""
    items = []
    for i, step in enumerate(steps):
        if done_index is not None and i <= done_index and i != active_index:
            state = "done"
        elif active_index is not None and i == active_index:
            state = "active"
        else:
            state = "pending"
        color = {"done": "var(--success)", "active": "var(--accent)", "pending": "var(--border-strong)"}[state]
        text_color = {"done": "var(--text-primary)", "active": "var(--text-primary)", "pending": "var(--text-faint)"}[state]
        pulse = "animation: sci-pulse 1.4s ease-in-out infinite;" if state == "active" else ""
        items.append(
            f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<span style='width:9px;height:9px;border-radius:50%;background:{color};{pulse}'></span>"
            f"<span style='font-size:0.74rem;letter-spacing:0.06em;text-transform:uppercase;color:{text_color};'>{esc(step)}</span>"
            f"</div>"
        )
        if i < len(steps) - 1:
            connector_color = "var(--success)" if (done_index is not None and i < done_index) else "var(--border-subtle)"
            items.append(f"<div style='flex:1;height:1px;background:{connector_color};min-width:14px;'></div>")

    st.markdown(
        html_block(f"""
        <style>
        @keyframes sci-pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}
        @media (prefers-reduced-motion: reduce) {{
            [style*='sci-pulse'] {{ animation: none !important; }}
        }}
        </style>
        <div style='display:flex;align-items:center;gap:0;padding:0.9rem 0;'>{''.join(items)}</div>
        """),
        unsafe_allow_html=True,
    )


def hero_rbc_svg(size: int = 300) -> str:
    """A stylized biconcave-disc RBC illustration (side-profile silhouette),
    not a flat circle -- pure inline SVG, no external asset/dependency."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustrative red blood cell shape">
      <defs>
        <radialGradient id="rbcBody" cx="50%" cy="42%" r="65%">
          <stop offset="0%" stop-color="#D6505C"/>
          <stop offset="55%" stop-color="#A5303C"/>
          <stop offset="100%" stop-color="#5C1620"/>
        </radialGradient>
        <radialGradient id="rbcDimple" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#3D1420" stop-opacity="0.75"/>
          <stop offset="70%" stop-color="#3D1420" stop-opacity="0.15"/>
          <stop offset="100%" stop-color="#3D1420" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <ellipse cx="150" cy="150" rx="128" ry="118" fill="url(#rbcBody)"/>
      <ellipse cx="150" cy="150" rx="128" ry="118" fill="none" stroke="#7C2A32" stroke-width="2" opacity="0.6"/>
      <ellipse cx="150" cy="150" rx="55" ry="46" fill="url(#rbcDimple)"/>
      <ellipse cx="112" cy="108" rx="34" ry="20" fill="#F0A0A6" opacity="0.28"/>
    </svg>
    """


def micro_particles_backdrop(count: int = 14) -> str:
    """A handful of faint, slowly-drifting particles for subtle microscopy
    atmosphere -- CSS-only, no canvas/WebGL, negligible perf cost."""
    import random
    rng = random.Random(7)
    dots = []
    for i in range(count):
        x = rng.randint(0, 100)
        y = rng.randint(0, 100)
        r = rng.uniform(1.2, 3.2)
        dur = rng.uniform(9, 18)
        delay = rng.uniform(0, 6)
        dots.append(
            f"<circle cx='{x}%' cy='{y}%' r='{r:.1f}' fill='#B4A89F' opacity='0.18'>"
            f"<animate attributeName='opacity' values='0.05;0.22;0.05' dur='{dur:.1f}s' begin='{delay:.1f}s' repeatCount='indefinite'/>"
            f"</circle>"
        )
    return f"<svg width='100%' height='100%' style='position:absolute;inset:0;pointer-events:none;'>{''.join(dots)}</svg>"
