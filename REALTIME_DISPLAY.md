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

The default window is now less wide, and the four display panels are kept
square. Use `--width` and `--height` if the exhibit monitor needs a different
window size.

UDP test sender:

```bash
python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --seq
```

Use this when checking whether the reconstructed image visibly changes:

```bash
python udp_sender_test.py --rows 8 --cols 8 --active 12 --interval 1.0 --mode layout_demo --seq
```

`layout_demo` cycles through compact, line, diagonal, outer-edge, and clustered
antenna layouts so the uv coverage and dirty image should change much more
clearly than with the smooth moving pattern.

## Change Input Image

Use a built-in sample:

```bash
python realtime_display.py --sample gas
python realtime_display.py --sample points
python realtime_display.py --sample bubbles
python realtime_display.py --sample ska
```

Use an image file:

```bash
python realtime_display.py --image /path/to/input.png
```

Runtime keys:

- `I`: cycle built-in sample images
- `L`: reload the file passed with `--image`
- `H`: show or hide help
- `Q` or `Esc`: quit

## Runtime Adjustment Keys

- `C`: cycle dirty-image colormap
- `Z` / `X`: decrease / increase asinh stretch
- `N` / `M`: decrease / increase contrast percentile
- `B` / `V`: decrease / increase maximum baseline
- `R` / `T`: decrease / increase reference baseline
- `J` / `U`: decrease / increase uv zoom
- `O` / `P`: decrease / increase uv point size
- `A` / `S`: decrease / increase uv smoothing

The current values are shown along the top of the window.

The default reconstructed-image palette is `thermal`. Start with another
palette if needed:

```bash
python realtime_display.py --cmap icefire
python realtime_display.py --cmap viridis
python realtime_display.py --cmap RdBu_r
```
