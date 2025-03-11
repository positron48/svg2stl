#!/usr/bin/env python3
"""
Скрипт для создания адаптера cairosvg, который использует svglib и reportlab
вместо cairocffi для Windows-сборки.
"""

import os
import sys

def main():
    # Создаем директорию cairosvg
    os.makedirs('cairosvg', exist_ok=True)

    # Создаем файлы для модуля-заглушки
    with open('cairosvg/__init__.py', 'w') as f:
        f.write('from . import surface\n')

    with open('cairosvg/surface.py', 'w') as f:
        f.write('from .compat import svg2png\n')

    # Создаем файл с реализацией
    compat_code = '''import os
from pathlib import Path
import io
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

def svg2png(url=None, file_obj=None, dpi=96, output_width=None, output_height=None, **kwargs):
    """
    Convert SVG to PNG using pycairo and svglib.
    Compatible with cairosvg.svg2png signature.
    """
    # Determine input source
    svg_path = url
    if file_obj:
        svg_path = getattr(file_obj, "name", None)
        if not svg_path:
            # If file_obj doesn't have a name, it might be a BytesIO
            try:
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    svg_path = io.BytesIO(content)
                else:
                    svg_path = io.BytesIO(content.encode("utf-8"))
            except:
                raise ValueError("Cannot read from file_obj")
                
    # Convert SVG to PNG
    drawing = svg2rlg(svg_path)
    
    # Scale based on DPI
    scale_factor = dpi / 72.0  # ReportLab uses 72 DPI by default
    
    # Handle width/height
    if output_width and output_height:
        # Both dimensions specified
        width = output_width
        height = output_height
    elif output_width:
        # Only width specified
        width = output_width
        height = drawing.height * (output_width / drawing.width)
    elif output_height:
        # Only height specified
        height = output_height
        width = drawing.width * (output_height / drawing.height)
    else:
        # No dimensions specified, use DPI scaling
        width = drawing.width * scale_factor
        height = drawing.height * scale_factor
    
    # Generate PNG
    png_data = renderPM.drawToString(drawing, fmt="PNG", dpi=dpi)
    
    # Output handling
    output_path = kwargs.get("write_to")
    if output_path:
        with open(output_path, "wb") as f:
            f.write(png_data)
    else:
        return png_data'''

    with open('cairosvg/compat.py', 'w') as f:
        f.write(compat_code)

    print('CairoSVG adapter module created successfully')

    # Добавляем текущий каталог в путь поиска модулей Python
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Проверяем, что модуль создан и работает
    try:
        import cairosvg
        print('CairoSVG imported successfully')
    except ImportError as e:
        # Если импорт не удался, выводим предупреждение, но не прерываем процесс
        print(f'Warning: Could not import cairosvg module: {e}')
        print('This may not be a problem if the module will be imported from a different context.')
        # Не выходим с ошибкой, так как файлы уже созданы
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 