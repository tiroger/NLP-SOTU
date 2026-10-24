"""Altair chart definitions shared by the marimo notebooks and the website.

Every function accepts either a pandas DataFrame (notebooks) or an
``alt.UrlData`` / ``alt.NamedData`` (website), so field types are always
spelled out explicitly. Colors are the light-mode palette; the website swaps
each hex for its dark-mode step (see DARK_SWAP and site/app.js).
"""

import altair as alt

# --- Palette (validated with the dataviz skill's palette checker) ----------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PARTY_GROUPS = ["Democratic", "Republican", "Earlier parties"]
PARTY_COLORS = ["#2a78d6", "#e34948", "#1baf7a"]
PARTY_SHAPES = ["circle", "square", "triangle-up"]

# Light hex -> dark hex. Sequential ramp flips so "near zero" recedes into the
# dark surface instead of glowing.
DARK_SWAP = {
    SURFACE: "#1a1a19", INK: "#f0efec", INK_2: "#c3c2b7", GRID: "#2c2c2a", AXIS: "#383835",
    "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
    "#e87ba4": "#d55181", "#6250d6": "#9085e9", "#e34948": "#e66767",
    **dict(zip(SEQUENTIAL, ["#183a64", "#1c4f8a", "#2465b0", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])),
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

EVENTS = [
    (1861.3, "Civil War"), (1917.3, "WWI"), (1929.8, "Great Depression"),
    (1941.9, "WWII"), (2001.7, "9/11"), (2008.7, "Financial crisis"), (2020.2, "COVID-19"),
]


def theme() -> alt.theme.ThemeConfig:
    axis = {
        "domainColor": AXIS, "tickColor": AXIS, "gridColor": GRID, "gridWidth": 1,
        "labelColor": MUTED, "titleColor": INK_2, "labelFont": FONT, "titleFont": FONT,
        "labelFontSize": 11, "titleFontSize": 12, "titleFontWeight": 500,
    }
    return {
        "config": {
            "background": SURFACE,
            "font": FONT,
            "view": {"stroke": None},
            "axis": axis,
            "axisX": {"grid": False},
            "axisY": {"domain": False, "ticks": False},
            "legend": {
                "labelColor": INK_2, "titleColor": INK_2, "labelFont": FONT, "titleFont": FONT,
                "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500, "orient": "top",
            },
            "header": {"labelColor": INK_2, "titleColor": INK_2, "labelFont": FONT, "labelFontSize": 12},
            "title": {"color": INK, "font": FONT, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
            "text": {"color": INK_2, "font": FONT},
            "range": {"category": SERIES},
            "point": {"size": 64},
            "line": {"strokeWidth": 2},
        }
    }


@alt.theme.register("sotu", enable=True)
def _sotu_theme():
    return theme()


# --- Shared encodings ------------------------------------------------------

def _party_scale():
    return alt.Scale(domain=PARTY_GROUPS, range=PARTY_COLORS)


YEAR_SCALE = alt.Scale(domain=[1785, 2030], nice=False, zero=False)


def _year_x(field="when", title=None):
    # Every layer needs the explicit scale, or a layer's default zero=True
    # stretches the shared axis back to year 0.
    return alt.X(f"{field}:Q", title=title, axis=alt.Axis(format="d", tickCount=10), scale=YEAR_SCALE)


SPEECH_TOOLTIP = [
    alt.Tooltip("president:N", title="President"),
    alt.Tooltip("date:N", title="Date"),
    alt.Tooltip("party:N", title="Party"),
    alt.Tooltip("delivery:N", title="Delivered"),
]


def _written_era_band():
    """Shade 1801-1912, when every message was sent to Congress in writing."""
    band = alt.Data(values=[{"start": 1801, "end": 1913, "label": "Written messages only"}])
    base = alt.Chart(band)
    rect = base.mark_rect(color=GRID, opacity=0.6).encode(x=_year_x("start"), x2="end:Q")
    text = base.mark_text(align="left", baseline="top", dx=6, dy=6, fontSize=11, color=MUTED).encode(
        x=_year_x("start"), y=alt.value(0), text="label:N"
    )
    return rect + text


def per_speech(data, field: str, title: str, fmt: str = ".1f", events: bool = False,
               zero_rule: bool = False, height: int = 320) -> alt.Chart:
    """One dot per speech over time, colored and shaped by party, with a LOESS trend."""
    base = alt.Chart(data)
    points = base.mark_point(filled=True, size=56, opacity=0.85, stroke=SURFACE, strokeWidth=1.5).encode(
        x=_year_x(),
        y=alt.Y(f"{field}:Q", title=title),
        color=alt.Color("party_group:N", scale=_party_scale(), title=None),
        shape=alt.Shape("party_group:N", scale=alt.Scale(domain=PARTY_GROUPS, range=PARTY_SHAPES), title=None),
        tooltip=SPEECH_TOOLTIP + [alt.Tooltip(f"{field}:Q", title=title, format=fmt)],
    )
    trend = base.transform_loess("when", field, bandwidth=0.15).mark_line(color=INK_2, strokeWidth=2).encode(
        x=_year_x(), y=f"{field}:Q"
    )
    layers = [_written_era_band(), points, trend]
    if zero_rule:
        layers.insert(1, alt.Chart(alt.Data(values=[{"y": 0}])).mark_rule(color=AXIS).encode(y="y:Q"))
    if events:
        ev = alt.Chart(alt.Data(values=[{"when": w, "label": l} for w, l in EVENTS]))
        layers += [
            ev.mark_rule(color=MUTED, strokeWidth=1, opacity=0.6).encode(x=_year_x()),
            ev.mark_text(angle=270, align="right", baseline="bottom", dx=-4, dy=-2, fontSize=10, color=MUTED)
              .encode(x=_year_x(), y=alt.value(0), text="label:N"),
        ]
    return alt.layer(*layers).properties(width="container", height=height)


def length_chart(data):
    return per_speech(data, "words", "Words", fmt=",d")


def grade_chart(data):
    return per_speech(data, "grade_level", "Flesch–Kincaid grade level")


def diversity_chart(data):
    return per_speech(data, "lexical_diversity", "Lexical diversity (MTLD)")


def tone_chart(data):
    return per_speech(data, "tone", "Net tone (positive − negative)", fmt="+.2f", events=True, zero_rule=True)


def pronoun_chart(data):
    """Decade averages of 'I' vs 'we' per 1,000 words, direct-labeled."""
    long = alt.Chart(data).transform_fold(["i_per_1k", "we_per_1k"], as_=["pronoun", "rate"]).transform_calculate(
        decade="floor(datum.year / 10) * 10",
        label="datum.pronoun == 'i_per_1k' ? 'I / me / my' : 'we / us / our'",
    ).transform_aggregate(rate="mean(rate)", groupby=["decade", "label"])
    color = alt.Color("label:N", title=None, scale=alt.Scale(domain=["we / us / our", "I / me / my"], range=SERIES[:2]))
    lines = long.mark_line(interpolate="monotone").encode(
        x=alt.X("decade:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("rate:Q", title="Per 1,000 words"),
        color=color,
    )
    dots = long.mark_point(filled=True, size=70, opacity=0).encode(
        x="decade:Q", y="rate:Q", color=color,
        tooltip=[alt.Tooltip("decade:Q", title="Decade", format="d"), alt.Tooltip("label:N", title="Pronoun"),
                 alt.Tooltip("rate:Q", title="Per 1,000 words", format=".1f")],
    )
    end_labels = long.transform_filter("datum.decade == 2020").mark_text(align="left", dx=6, fontSize=12, color=INK_2).encode(
        x="decade:Q", y="rate:Q", text="label:N"
    )
    return alt.layer(lines, dots, end_labels).properties(width="container", height=300)


def emotion_chart(data):
    """Small multiples: share of sentences whose strongest emotion is each label, by decade."""
    return alt.Chart(data).mark_line(color=SERIES[0], interpolate="monotone").encode(
        x=alt.X("decade:Q", title=None, axis=alt.Axis(format="d", tickCount=4)),
        y=alt.Y("share:Q", title=None, axis=alt.Axis(format=".0%", tickCount=4)),
        tooltip=[alt.Tooltip("emotion:N", title="Emotion"), alt.Tooltip("decade:Q", title="Decade", format="d"),
                 alt.Tooltip("share:Q", title="Share of sentences", format=".1%")],
    ).properties(width=180, height=110).facet(
        facet=alt.Facet("emotion:N", title=None, sort=["joy", "fear", "anger", "sadness", "surprise", "disgust"]),
        columns=3,
    ).resolve_scale(y="independent")


def topic_heatmap(data, topics: list[str]):
    return alt.Chart(data).mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=2).encode(
        x=alt.X("decade:O", title=None, axis=alt.Axis(labelAngle=-90, labelExpr="datum.value % 20 == 0 ? datum.value : ''")),
        y=alt.Y("topic:N", title=None, sort=topics, axis=alt.Axis(labelLimit=200)),
        color=alt.Color("share:Q", title="Share of words", scale=alt.Scale(range=[SEQUENTIAL[0], SEQUENTIAL[-1]]),
                        legend=alt.Legend(format=".0%", orient="top", gradientLength=180)),
        tooltip=[alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("decade:O", title="Decade"),
                 alt.Tooltip("share:Q", title="Share of words", format=".1%")],
    ).properties(width="container", height=alt.Step(18))


def president_tone_chart(data):
    """Average tone per president with the range of their speeches."""
    sort = alt.EncodingSortField("tone_mean", order="descending")
    base = alt.Chart(data).encode(y=alt.Y("president:N", title=None, sort=sort, axis=alt.Axis(labelLimit=200)))
    rng = base.mark_rule(color=AXIS, strokeWidth=2).encode(x="tone_min:Q", x2="tone_max:Q")
    dot = base.mark_point(filled=True, size=80, stroke=SURFACE, strokeWidth=1.5).encode(
        x=alt.X("tone_mean:Q", title="Net tone (mean, with range across speeches)"),
        color=alt.Color("party_group:N", scale=_party_scale(), title=None),
        shape=alt.Shape("party_group:N", scale=alt.Scale(domain=PARTY_GROUPS, range=PARTY_SHAPES), title=None),
        tooltip=[alt.Tooltip("president:N", title="President"), alt.Tooltip("party:N", title="Party"),
                 alt.Tooltip("speeches:Q", title="Speeches"), alt.Tooltip("tone_mean:Q", title="Mean tone", format="+.2f"),
                 alt.Tooltip("tone_min:Q", title="Lowest", format="+.2f"), alt.Tooltip("tone_max:Q", title="Highest", format="+.2f")],
    )
    return alt.layer(rng, dot).properties(width="container", height=alt.Step(17))


def distinctive_chart(data, presidents: list[str], default: str):
    """Words most over-used by one president relative to all others (log-odds z-score)."""
    pick = alt.param(name="president", value=default,
                     bind=alt.binding_select(options=presidents, name="President "))
    return alt.Chart(data).transform_filter(alt.datum.president == pick).mark_bar(
        color=SERIES[0], cornerRadiusEnd=4, height={"band": 0.7}
    ).encode(
        x=alt.X("z:Q", title="How distinctive (log-odds z-score)"),
        y=alt.Y("word:N", title=None, sort=alt.EncodingSortField("z", order="descending")),
        tooltip=[alt.Tooltip("word:N", title="Word"), alt.Tooltip("count:Q", title="Times used"),
                 alt.Tooltip("z:Q", title="z-score", format=".1f")],
    ).add_params(pick).properties(width="container", height=alt.Step(20))


def party_words_chart(data):
    """Diverging bars: words Democrats favor (left) vs Republicans (right)."""
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("z:Q", title="← More Democratic · More Republican →"),
        y=alt.Y("word:N", title=None, sort=alt.EncodingSortField("z", order="ascending")),
        color=alt.Color("side:N", scale=alt.Scale(domain=PARTY_GROUPS[:2], range=PARTY_COLORS[:2]), title=None),
        tooltip=[alt.Tooltip("word:N", title="Word"), alt.Tooltip("side:N", title="Favored by"),
                 alt.Tooltip("dem:Q", title="Democratic uses"), alt.Tooltip("rep:Q", title="Republican uses"),
                 alt.Tooltip("z:Q", title="z-score", format="+.1f")],
    ).properties(width="container", height=alt.Step(16))


def trend_chart(data=None):
    """Word-frequency explorer; the website fills the 'trend' dataset at runtime."""
    data = data if data is not None else alt.NamedData("trend")
    color = alt.Color("term:N", title=None, scale=alt.Scale(range=SERIES))
    base = alt.Chart(data)
    lines = base.mark_line(interpolate="monotone").encode(
        x=_year_x(), y=alt.Y("rate:Q", title="Uses per 1,000,000 words (5-year average)"), color=color
    )
    hover = alt.selection_point(fields=["when"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    rule = base.mark_rule(color=AXIS).encode(
        x=_year_x(), opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("when:Q", title="Year", format="d"), alt.Tooltip("term:N", title="Term"),
                 alt.Tooltip("rate:Q", title="Per million words", format=",.0f")],
    ).add_params(hover)
    pts = base.mark_point(filled=True, size=50).encode(
        x=_year_x(), y="rate:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    return alt.layer(lines, rule, pts).properties(width="container", height=340)


def map_chart(data):
    """2-D UMAP projection of speech embeddings; nearby speeches read alike."""
    return alt.Chart(data).mark_point(filled=True, size=70, opacity=0.85, stroke=SURFACE, strokeWidth=1.5).encode(
        x=alt.X("map_x:Q", axis=None),
        y=alt.Y("map_y:Q", axis=None),
        color=alt.Color("party_group:N", scale=_party_scale(), title=None),
        shape=alt.Shape("party_group:N", scale=alt.Scale(domain=PARTY_GROUPS, range=PARTY_SHAPES), title=None),
        tooltip=SPEECH_TOOLTIP + [alt.Tooltip("top_topic:N", title="Main topic")],
    ).properties(width="container", height=460)


def arc_chart(data=None):
    """Tone across one speech, from first sentence to last."""
    data = data if data is not None else alt.NamedData("arc")
    base = alt.Chart(data)
    zero = alt.Chart(alt.Data(values=[{"y": 0}])).mark_rule(color=AXIS).encode(y="y:Q")
    line = base.mark_area(
        line={"color": SERIES[0], "strokeWidth": 2}, color=SERIES[0], opacity=0.15, interpolate="monotone"
    ).encode(
        x=alt.X("position:Q", title="Progress through speech", axis=alt.Axis(format=".0%", tickCount=5)),
        y=alt.Y("tone:Q", title="Net tone", scale=alt.Scale(domain=[-1, 1])),
        tooltip=[alt.Tooltip("position:Q", title="Position", format=".0%"), alt.Tooltip("tone:Q", title="Tone", format="+.2f")],
    )
    return alt.layer(zero, line).properties(width="container", height=200)


def speech_topics_chart(data=None):
    data = data if data is not None else alt.NamedData("topics")
    return alt.Chart(data).mark_bar(color=SERIES[0], cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("share:Q", title="Share of words", axis=alt.Axis(format=".0%")),
        y=alt.Y("topic:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200)),
        tooltip=[alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("share:Q", title="Share of words", format=".1%")],
    ).properties(width="container", height=alt.Step(20))
