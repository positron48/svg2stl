# SVG to STL Converter

A Python script for converting SVG files (specifically PCB mask negatives exported from KiCad) to STL 3D models for photopolymer printing.

[Русская версия документации](readme.md)

## Description

This tool automates the process of converting SVG files exported from KiCad (with black fill representing the PCB mask negative) into accurate 3D models in STL format for use with photopolymer printers like the Photon Mono X6KS.

## Installation & Usage

### Option 1: Pre-built Binaries

Download a ready-to-use executable for your operating system from the [releases section](https://github.com/YOURNAME/svg-to-stl/releases):

- **Windows**: svg2stl-windows.exe
- **macOS**: svg2stl-macos
- **Linux**: svg2stl-linux

After downloading:
- **Windows**: Simply double-click the exe file
- **macOS/Linux**: Make the file executable (`chmod +x svg2stl-*`) and then run it through the terminal

> **Note for Windows users**: Some antivirus software, including Windows Defender, may falsely flag the executable as malicious. This is a common issue with Python applications compiled into standalone executables. The application is safe to use, but if you're concerned, you have these options:
> 1. Add an exception in your antivirus for the svg2stl-windows.exe file
> 2. Run the script directly from source code (see Option 2 below)

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
python svg2stl.py input.svg --thickness 1.0 --pixel_size 0.05
```

## Adding an Exception in Windows Defender

If Windows Defender blocks the application, you can add an exception:

1. Open **Windows Security** (search for it in the Start menu)
2. Click on **Virus & threat protection**
3. Under "Virus & threat protection settings", click **Manage settings**
4. Scroll down to **Exclusions** and click **Add or remove exclusions**
5. Click **Add an exclusion** and select **File**
6. Browse to and select the svg2stl-windows.exe file

This will allow the application to run normally on your system.

## Features

- Convert SVG files to high-quality STL 3D models
- Precise extraction of only black elements from SVG (PCB traces and outlines)
- Automatic cropping to actual content boundaries, significantly reducing model size
- Advanced 3D model optimization by merging adjacent pixels into a minimal number of rectangles
- Process individual files or all SVG files in the current directory
- Configurable parameters for thickness and pixel size (resolution)
- Automatic calculation of optimal DPI for rasterization based on pixel size
- Progress indicator for tracking the conversion process

## Command Line Arguments

- `--thickness`: Thickness of the resulting 3D model in mm (default: 1.0)
- `--pixel_size`: Size of each pixel in mm (default: 0.05). Determines rasterization resolution and model detail
- `--debug`: Save debug images for checking
- `--all`: Process all SVG files in the current directory
- `--inverted`: Extract white pixels instead of black (useful for negative images or light-on-dark designs)

## Examples

Convert a single SVG file:
```
svg2stl input.svg --thickness 1.0 --pixel_size 0.05
```

Convert all SVG files in the current directory:
```
svg2stl --all --thickness 1.0 --pixel_size 0.05
```

Convert a KiCad copper layer export to a 0.8mm thick STL with high resolution (600 DPI):
```
svg2stl board-F_Cu.svg --thickness 0.8 --pixel_size 0.05
```

Process white elements from SVG instead of black (for negative/inverted designs):
```
svg2stl negative-design.svg --thickness 1.0 --pixel_size 0.05 --inverted
```

Process all SVG files with very high resolution (1200 DPI):
```
svg2stl --all --thickness 1.0 --pixel_size 0.025
```

Quick processing of a large file with lower resolution (300 DPI):
```
svg2stl large-board.svg --thickness 1.0 --pixel_size 0.1
```

## Performance Recommendations

The pixel size (`pixel_size`) fully determines the resolution and detail of the model:

| Pixel Size | Calculated DPI | Quality | File Size | Processing Time |
|------------|----------------|---------|-----------|-----------------|
| 0.025 mm   | 1200 DPI       | Very High | ~5 MB   | ~4 sec          |
| 0.05 mm    | 600 DPI        | High    | ~3 MB     | ~2 sec          |
| 0.1 mm     | 300 DPI        | Medium  | ~2 MB     | ~1 sec          |
| 0.2 mm     | 150 DPI        | Low     | ~1 MB     | <1 sec          |

The formula used is: `DPI = 30 / pixel_size`

## Building from Source

To create your own binary file from the source code with maximum compatibility:

### Linux (for maximum compatibility)
```
# Using an older distribution is recommended (e.g., Ubuntu 18.04 or 20.04)
pip install pyinstaller
pyinstaller --onefile --clean --name svg2stl \
  --hidden-import=PIL._tkinter_finder \
  --exclude-module=tcl \
  --exclude-module=tk \
  --exclude-module=Tkinter \
  --exclude-module=_tkinter \
  svg2stl.py
```

### Windows and macOS
```
pip install pyinstaller
pyinstaller --onefile svg2stl.py
```

The executable file will be created in the `dist/` folder.

## Troubleshooting

### Linux Compatibility Issues

If you encounter errors like `GLIBC_2.xx not found`, it means the binary was built with a newer version of Linux libraries than what's available on your system. Use one of these solutions:

1. Run the script directly with Python (Option 2)
2. Build your own binary with PyInstaller on your system using the instructions above

## License

This project is licensed under the MIT License - see the LICENSE file for details. 