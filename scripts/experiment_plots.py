"""Accuracy-vs-tokens chart (P3-21) as a self-contained SVG; no plotting dependency.

One dot per system at (mean tokens per question, accuracy). Whiskers are
the patient-clustered 95% CIs on both axes. Tokens use a log axis because
the direct baseline makes one call while the agents make many. Every dot
is labeled directly and has a hover tooltip. The palette is the first
three categorical slots, checked for color-vision deficiency in light and
dark mode. Colors follow the system, not its rank. The CSV tables written
next to the chart carry the same numbers.
"""
import math
from html import escape

# Categorical slots 1-3 (light, dark), fixed per system
SERIES = {"direct_llm": ("#2a78d6", "#3987e5"), "single_agent": ("#eb6834", "#d95926"),
          "multi_agent": ("#1baf7a", "#199e70")}
LABELS = {"direct_llm": "Direct LLM", "single_agent": "Single agent", "multi_agent": "Multi-agent"}
WIDTH, HEIGHT = 680, 420
LEFT, RIGHT, TOP, BOTTOM = 64, 120, 84, 52


def _style():
    light = "".join(f"--{key}:{value[0]};" for key, value in SERIES.items())
    dark = "".join(f"--{key}:{value[1]};" for key, value in SERIES.items())
    dark_ink = ("color-scheme:dark;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--grid:#2c2c2a;"
                "--axis:#383835;" + dark)
    return f"""<style>
.viz-root{{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;{light}font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
@media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])) .viz-root{{{dark_ink}}}}}
:root[data-theme="dark"] .viz-root{{{dark_ink}}}
.title{{font-size:15px;font-weight:600;fill:var(--ink)}} .sub{{font-size:12px;fill:var(--ink-2)}}
.tick{{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}} .label{{font-size:12px;fill:var(--ink-2)}}
.grid{{stroke:var(--grid);stroke-width:1}} .axis{{stroke:var(--axis);stroke-width:1}}
.point:hover .hit{{fill:var(--grid);fill-opacity:.6}}
</style>"""


def _tokens_label(value):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _nice(low, high):
    """1-2-5 values from just below `low` to just above `high`."""
    return [step * 10 ** exponent for exponent in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1)
            for step in (1, 2, 5)]


def _log_ticks(low, high):
    return [value for value in _nice(low, high) if low <= value <= high]


def accuracy_vs_tokens_svg(points, subtitle=""):
    """`points`: {system: {"accuracy": clustered_mean(...), "tokens": clustered_mean(...)}}."""
    points = {k: v for k, v in points.items() if v["accuracy"]["estimate"] is not None
              and v["tokens"]["estimate"] and v["tokens"]["ci_low"] > 0}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" class="viz-root" viewBox="0 0 {WIDTH} {HEIGHT}" '
             f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t d">', _style(),
             f'<rect width="{WIDTH}" height="{HEIGHT}" fill="var(--surface)"/>',
             '<text id="t" class="title" x="16" y="26">Accuracy vs tokens per question</text>',
             f'<text id="d" class="sub" x="16" y="44">{escape(subtitle or "Held-out questions; whiskers are patient-clustered 95% CIs")}</text>']
    if not points:
        parts.append('<text class="sub" x="16" y="80">No complete arms to plot yet.</text></svg>')
        return "\n".join(parts)

    t_low = min(p["tokens"]["ci_low"] for p in points.values())
    t_high = max(p["tokens"]["ci_high"] for p in points.values())
    # Padded so no dot or whisker sits on the plot edge
    x_min = max(v for v in _nice(t_low, t_high) if v <= t_low / 1.15)
    x_max = min(v for v in _nice(t_low, t_high * 1.15) if v >= t_high * 1.15)
    a_low = min(p["accuracy"]["ci_low"] for p in points.values())
    a_high = max(p["accuracy"]["ci_high"] for p in points.values())
    y_min, y_max = max(0.0, math.floor(a_low * 10) / 10), min(1.0, math.ceil(a_high * 10) / 10)
    if y_max - y_min < 0.2:
        y_min, y_max = max(0.0, y_min - 0.1), min(1.0, y_max + 0.1)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    x = lambda v: LEFT + (math.log10(v) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * plot_w
    y = lambda v: TOP + (y_max - v) / (y_max - y_min) * plot_h

    # Legend row above the plot: a dot keys each system; text stays in ink
    legend_x = 16
    for system in points:
        parts.append(f'<circle cx="{legend_x + 5}" cy="62" r="5" fill="var(--{system})"/>'
                     f'<text class="label" x="{legend_x + 14}" y="66">{LABELS.get(system, system)}</text>')
        legend_x += 30 + 8 * len(LABELS.get(system, system))

    steps = round((y_max - y_min) / 0.1)
    for i in range(steps + 1):
        value = y_min + i * 0.1
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{y(value):.1f}" y2="{y(value):.1f}"/>'
                     f'<text class="tick" x="{LEFT - 8}" y="{y(value) + 4:.1f}" text-anchor="end">{value:.0%}</text>')
    for value in _log_ticks(x_min, x_max):
        parts.append(f'<line class="grid" x1="{x(value):.1f}" x2="{x(value):.1f}" y1="{TOP}" y2="{TOP + plot_h}"/>'
                     f'<text class="tick" x="{x(value):.1f}" y="{TOP + plot_h + 18}" text-anchor="middle">'
                     f'{_tokens_label(value)}</text>')
    parts += [f'<line class="axis" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{TOP + plot_h}" y2="{TOP + plot_h}"/>',
              f'<text class="tick" x="{LEFT + plot_w / 2}" y="{HEIGHT - 12}" text-anchor="middle">'
              'Mean tokens per question (log scale)</text>',
              f'<text class="tick" transform="translate(16 {TOP + plot_h / 2}) rotate(-90)" text-anchor="middle">'
              'Accuracy</text>']

    for system, p in points.items():
        acc, tok = p["accuracy"], p["tokens"]
        cx, cy = x(tok["estimate"]), y(acc["estimate"])
        tip = (f"{LABELS.get(system, system)}: accuracy {acc['estimate']:.1%} "
               f"(95% CI {acc['ci_low']:.1%} to {acc['ci_high']:.1%}); "
               f"{tok['estimate']:,.0f} tokens per question (95% CI {tok['ci_low']:,.0f} to {tok['ci_high']:,.0f}); "
               f"{acc['n']} questions, {acc['patients']} patients")
        parts.append(
            f'<g class="point"><title>{escape(tip)}</title>'
            f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="14" fill="transparent"/>'
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(acc["ci_low"]):.1f}" y2="{y(acc["ci_high"]):.1f}" '
            f'stroke="var(--{system})" stroke-width="2" stroke-linecap="round"/>'
            f'<line x1="{x(tok["ci_low"]):.1f}" x2="{x(tok["ci_high"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" '
            f'stroke="var(--{system})" stroke-width="2" stroke-linecap="round"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="var(--{system})" stroke="var(--surface)" stroke-width="2"/>'
            f'<text class="label" x="{cx + 10:.1f}" y="{cy - 8:.1f}">{LABELS.get(system, system)} '
            f'{acc["estimate"]:.0%}</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)
