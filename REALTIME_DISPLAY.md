# Realtime Display

`realtime_display.py` is the fast exhibit-oriented display for UDP detector input.
It keeps the Streamlit app as the rich configuration/prototype UI, while this
Pygame app targets a 10 FPS local HDMI display on Raspberry Pi or a local PC.

## Install

```bash
python -m pip install -r requirements-realtime.txt
```

## Basic Run

```bash
python realtime_display.py --fps 10 --host 127.0.0.1 --port 9900
```

This starts in a normal, resizable window so that the computer remains usable
while adjusting the display. Press `F11` after setup when a full-screen exhibit
is wanted; press `Esc` to return to a window. `Q` quits the app.

The display searches for a Japanese-capable system font. On systems without
one, it uses English labels automatically rather than displaying garbled text.
Use `--font /path/to/japanese-font.ttf` to choose a specific font file.

The default is the public-facing **exhibit view**. It makes the reconstructed
sky the large square panel, keeps all supporting panels square, uses an
exhibit-specific color treatment, and briefly animates a configuration update.
It also uses a modest display-only UV gridding step followed by a compact
CLEAN-style restoration to suppress the strongest dirty-image sidelobes while
retaining configuration-specific differences. The UV panel always shows the
sparse, instantaneous measurement pattern. Press `D` to inspect the raw dirty
image in science view.

## Reconstruction Display Styles

The reconstruction calculation is unchanged when switching styles; only the
large reconstructed-image panel changes. Press `W` in exhibit view to cycle:

- `exhibit`: the existing colorful presentation style
- `clean`: neutral radio-image style with intensity contours and a synthesized
  restoring-beam marker
- `eht`: black, red, orange, and cream high-dynamic-range presentation inspired
  by public EHT imagery
- `residual`: signed CLEAN residual in a blue-to-red diagnostic palette

Choose the initial style at launch:

```bash
python realtime_display.py --reconstruction-style clean
python realtime_display.py --reconstruction-style eht --sample ring
python realtime_display.py --reconstruction-style residual
```

`clean` is the best choice when explaining standard radio-interferometric
imaging. `eht` is a display treatment only: it does not add information or
change the effective angular resolution.

The image and reconstruction grid now defaults to `192 x 192`, and the two
image panels are smoothly scaled in exhibit view. Use `256` on a sufficiently
fast PC for a sharper display, or `96` on a slower Raspberry Pi if necessary:

```bash
python realtime_display.py --fps 10 --grid 256
python realtime_display.py --fps 10 --grid 96
```

Use the analysis-oriented four-panel view when adjusting or inspecting the
simulation:

```bash
python realtime_display.py --display-mode science
```

Choose an exhibit color theme at startup:

```bash
python realtime_display.py --exhibit-theme aurora
python realtime_display.py --exhibit-theme ember
python realtime_display.py --exhibit-theme tide
```

Use `--width` and `--height` if the exhibit monitor needs a different window
size. `--fullscreen` is available for automatic full-screen startup and uses
the monitor's native resolution.

UDP test sender:

```bash
python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --seq
```

Use this when checking whether the reconstructed image visibly changes:

```bash
python udp_sender_test.py --rows 8 --cols 8 --active 12 --interval 1.0 --mode layout_demo --seq
```

`layout_demo` cycles through compact, line, diagonal, outer-edge, clustered,
three-arm, and SKA-like spiral antenna layouts so the uv coverage and dirty
image should change much more clearly than with the smooth moving pattern.

Use this when checking only the SKA-like dense-core plus spiral-arm layout:

```bash
python udp_sender_test.py --rows 8 --cols 8 --active 16 --interval 1.0 --mode ska_spiral --seq
```

## Change Input Image

Use a built-in sample:

```bash
python realtime_display.py --sample gas
python realtime_display.py --sample points
python realtime_display.py --sample bubbles
python realtime_display.py --sample ska
python realtime_display.py --sample double
python realtime_display.py --sample jet
python realtime_display.py --sample ring
python realtime_display.py --sample spiral
python realtime_display.py --sample cluster
python realtime_display.py --sample cross
python realtime_display.py --sample crescent
python realtime_display.py --sample resolution
```

`resolution` and `cross` are particularly useful for making the effect of
different antenna configurations easy to see.

Use an image file:

```bash
python realtime_display.py --image /path/to/input.png
```

Runtime keys:

- `I`: cycle built-in sample images
- `L`: reload the file passed with `--image`
- `H`: show or hide help
- `D`: switch between exhibit and science views
- `W`: cycle reconstruction style: exhibit, CLEAN, EHT-style, residual
- `F11`: switch between full-screen and window display
- `Esc`: leave full-screen, or quit from a window
- `Q`: quit

## Runtime Adjustment Keys

- `C`: cycle the exhibit color theme, or the science-view dirty-image colormap
- `Z` / `X`: decrease / increase asinh stretch
- `N` / `M`: decrease / increase contrast percentile
- `B` / `V`: decrease / increase maximum baseline
- `R` / `T`: decrease / increase reference baseline
- `J` / `U`: decrease / increase uv zoom
- `O` / `P`: decrease / increase uv point size
- `A` / `S`: decrease / increase uv smoothing

The current values are shown along the top of the window.

The exhibit view starts with the `aurora` theme. Press `C` to cycle through
`aurora`, `ember`, `tide`, `violet`, `mint`, `mono`, and `coral`. The
science-view dirty image uses `thermal` by default and accepts the following
colormap options:

```bash
python realtime_display.py --cmap icefire
python realtime_display.py --cmap viridis
python realtime_display.py --cmap RdBu_r
```
