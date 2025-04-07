# SVG to STL Converter

![2025-04-02_20-06](screens/2025-04-02_20-06.png)

![2025-04-02_20-05](screens/2025-04-02_20-05.png)

A Python script for converting SVG files (specifically PCB mask negatives exported from KiCad) to STL 3D models for photopolymer printing.

[Blogpost about photoresist on SLA with this tool](https://positroid.tech/post/caretaker-part-3-photoresist)

[Русская версия документации](readme-ru.md)

## Description

This tool automates the process of converting SVG files exported from KiCad (with black fill representing the PCB mask negative) into accurate 3D models in STL format for use with photopolymer printers like the Photon Mono X6KS.

## Installation & Usage

### Option 1: Pre-built Binaries

Download a ready-to-use executable for your operating system from the [releases section](https://github.com/YOURNAME/svg-to-stl/releases):

- **macOS**: svg2stl-macos
- **Linux**: svg2stl-linux

After downloading:
- **macOS/Linux**: Make the file executable (`chmod +x svg2stl-*`) and then run it through the terminal

> **Note for Linux users**: The binaries are compiled on older Linux systems (Ubuntu 20.04) for maximum compatibility with most distributions. If you still encounter errors, please use the source code method below.

### Option 2: Run from Source Code

Requires Python 3.7 or higher and the following packages:
```
numpy
cairosvg
pillow
numpy-stl
tqdm
```

Install dependencies:
```
pip install -r requirements.txt
```

Run:
```
python svg2stl.py input.svg --thickness 1.0 --pixel_size 0.025
```

## Features

- Convert SVG files to STL 3D models
- Precise extraction of only black elements from SVG (PCB traces and outlines)
- Automatic cropping to actual content boundaries, significantly reducing model size
- Advanced 3D model optimization by merging adjacent pixels into a minimal number of rectangles
- Process individual files or all SVG files in the current directory
- Configurable parameters for thickness and pixel size (resolution)
- Automatic calculation of optimal DPI for rasterization based on pixel size
- Progress indicator for tracking the conversion process

## Command Line Arguments

- `--thickness`: Thickness of the resulting 3D model in mm (default: 1.0)
- `--pixel_size`: Size of each pixel in mm (default: 0.025). Determines rasterization resolution and model detail
- `--debug`: Save debug images for check
- `--all`: Process all SVG files in the current directory
- `--inverted`: Extract white pixels instead of black (useful for negative images or light-on-dark designs)

## Examples

Convert a single SVG file:
```
python svg2stl.py input.svg --thickness 1.0
```

Convert all SVG files in the current directory:
```
python svg2stl.py --all --thickness 1.0
```

Convert a KiCad copper layer export to a 0.8mm thick STL with high resolution (1200 DPI):
```
python svg2stl.py board-F_Cu.svg --thickness 0.8
```

Process white elements from SVG instead of black (for negative/inverted designs):
```
python svg2stl.py negative-design.svg --thickness 1.0 --inverted
```

Process with medium resolution (600 DPI) for faster processing:
```
python svg2stl.py --all --thickness 1.0 --pixel_size 0.05
```

Quick processing of a large file with lower resolution (300 DPI):
```
python svg2stl.py large-board.svg --thickness 1.0 --pixel_size 0.1
```

### Example files

[caretaker-F_Cu.svg](caretaker-F_Cu.svg) > [caretaker-F_Cu.stl](caretaker-F_Cu.stl)
