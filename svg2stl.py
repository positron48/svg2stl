#!/usr/bin/env python3
"""
svg2stl.py - Convert SVG files exported from KiCad to STL 3D models
Specifically designed for converting PCB mask negatives to 3D models for photopolymer printing.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import cairosvg
from PIL import Image, ImageChops
from stl import mesh
import time
from tqdm import tqdm

# Константы для преобразования
MM_PER_INCH = 25.4  # 1 дюйм = 25.4 мм

def calculate_dpi(pixel_size):
    """
    Расчет DPI на основе размера пикселя в мм.
    
    Эмпирическая формула на основе примеров:
    pixel_size=0.05, dpi=600
    pixel_size=0.1, dpi=300
    
    Получается соотношение: pixel_size * dpi = 30
    """
    return int(30 / pixel_size)

def crop_to_content(img, threshold=240, debug=False):
    """
    Crop the image to remove empty space around content.
    
    Args:
        img: PIL Image object
        threshold: Pixels with values > threshold are considered empty
        debug: If True, save debug images
        
    Returns:
        Cropped PIL Image
    """
    # Convert to grayscale if not already
    if img.mode != 'L':
        img_gray = img.convert('L')
    else:
        img_gray = img
    
    # Invert image since KiCad PCBs usually have black content on white background
    # We want to detect black content (PCB), so we invert to make content white
    img_array = np.array(img_gray)
    
    # Create binary mask: True for content, False for background
    mask = img_array < threshold  # PCB content is black (low values)
    
    # Find the bounding box of content
    # Get rows and columns with content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find min/max row and column with content
    if not np.any(rows) or not np.any(cols):
        print("Warning: No content found in the image")
        return img
        
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    # Add a small padding
    padding = 10
    row_min = max(0, row_min - padding)
    row_max = min(img_array.shape[0] - 1, row_max + padding)
    col_min = max(0, col_min - padding)
    col_max = min(img_array.shape[1] - 1, col_max + padding)
    
    # Create the bbox (left, upper, right, lower)
    bbox = (col_min, row_min, col_max + 1, row_max + 1)
    
    print(f"Cropping image from {img.size} to {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
    
    # Crop the original image using the bounding box
    cropped = img.crop(bbox)
    
    if debug:
        cropped.save("debug_cropped.png")
        
    return cropped

def find_maximal_rectangles(mask, min_rect_size=2):
    """
    Находит максимальные прямоугольники в бинарной маске.
    
    Этот алгоритм находит прямоугольники, которые могут быть созданы из смежных пикселей,
    что позволяет значительно сократить количество полигонов в итоговой 3D-модели.
    
    Args:
        mask: Бинарная маска (numpy array), True для содержимого
        min_rect_size: Минимальный размер прямоугольника (площадь в пикселях)
        
    Returns:
        Список прямоугольников в формате (x, y, width, height)
    """
    height, width = mask.shape
    print("Finding maximal rectangles...")
    
    # Вспомогательная матрица высот
    heights = np.zeros((height, width), dtype=int)
    
    # Для каждой строки вычисляем "высоту" столбца (сколько последовательных True вверх)
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                if y > 0:
                    heights[y, x] = heights[y-1, x] + 1
                else:
                    heights[y, x] = 1
    
    # Список найденных прямоугольников (x, y, width, height)
    rectangles = []
    
    # Используем подход "максимальных прямоугольников"
    for y in range(height):
        # Для каждой строки ищем максимальные прямоугольники по высотам
        stack = []  # Стек для слежения за потенциальными прямоугольниками
        
        for x in range(width + 1):  # +1 чтобы "закрыть" прямоугольники в последнем столбце
            # Высота в текущей позиции (0 за пределами массива или если элемент не входит в маску)
            h = heights[y, x-1] if x > 0 and x < width and mask[y, x-1] else 0
            
            # Обрабатываем прямоугольники из стека, которые выше текущего
            start_pos = x
            while stack and stack[-1][1] > h:
                pos, height_at_pos = stack.pop()
                width_of_rect = x - pos
                
                # Создаем прямоугольник только если его площадь достаточно большая
                area = width_of_rect * height_at_pos
                if area >= min_rect_size:
                    rectangles.append((pos, y - height_at_pos + 1, width_of_rect, height_at_pos))
                
                start_pos = pos
            
            # Если текущая высота ненулевая, добавляем в стек
            if h > 0:
                stack.append((start_pos, h))
    
    # Выполняем дополнительное объединение прямоугольников, где это возможно
    # (это может сократить количество прямоугольников до 10 раз)
    optimized_rectangles = optimize_rectangles(rectangles, mask, min_rect_size * 2)
    
    print(f"Initial rectangles: {len(rectangles)}, after optimization: {len(optimized_rectangles)}")
    return optimized_rectangles

def optimize_rectangles(rectangles, mask, min_area=4):
    """
    Оптимизирует список прямоугольников, объединяя их в более крупные блоки.
    
    Алгоритм пытается найти прямоугольники, которые можно объединить в более 
    крупные прямоугольники для значительного сокращения количества полигонов.
    
    Args:
        rectangles: Список прямоугольников в формате (x, y, width, height)
        mask: Исходная бинарная маска
        min_area: Минимальная площадь прямоугольника для сохранения
        
    Returns:
        Оптимизированный список прямоугольников
    """
    # Сортируем прямоугольники по площади (от большего к меньшему)
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)
    
    # Создаем маску для отслеживания уже покрытых пикселей
    height, width = mask.shape
    covered = np.zeros((height, width), dtype=bool)
    
    # Итоговый набор оптимизированных прямоугольников
    optimal_rectangles = []
    
    # Рассматриваем каждый прямоугольник
    for rect in rectangles:
        x, y, w, h = rect
        
        # Проверяем, остались ли непокрытые пиксели в этом прямоугольнике
        area_mask = covered[y:y+h, x:x+w]
        if np.all(area_mask):
            # Все пиксели уже покрыты, пропускаем
            continue
        
        # Вычисляем процент непокрытых пикселей
        uncovered_pixels = np.count_nonzero(~area_mask)
        total_pixels = w * h
        
        # Добавляем прямоугольник, если он покрывает достаточно непокрытых пикселей
        # или если он достаточно большой
        if uncovered_pixels / total_pixels > 0.2 or total_pixels >= min_area:
            optimal_rectangles.append(rect)
            # Отмечаем пиксели как покрытые
            covered[y:y+h, x:x+w] = True
    
    return optimal_rectangles

def create_optimized_mesh(mask, thickness, pixel_size):
    """
    Создает оптимизированный меш, объединяя соседние пиксели в прямоугольники.
    
    Args:
        mask: Бинарная маска (numpy array), True для содержимого
        thickness: Толщина 3D-модели в мм
        pixel_size: Размер пикселя в мм
        
    Returns:
        vertices, faces: Массивы вершин и граней для меша
    """
    height, width = mask.shape
    
    # Найти максимальные прямоугольники в маске
    rectangles = find_maximal_rectangles(mask)
    
    # Создать вершины и грани для каждого прямоугольника
    all_vertices = []
    all_faces = []
    vertex_count = 0
    
    for rect in tqdm(rectangles, desc="Creating mesh"):
        x, y, w, h = rect
        
        # Вычисляем координаты вершин в мм
        # Переворачиваем ось Y (в PIL начало координат в верхнем левом углу, 
        # а нам нужно в нижнем левом для 3D)
        x_pos = x * pixel_size
        y_pos = (height - y - h) * pixel_size
        
        # Ширина и высота прямоугольника в мм
        w_mm = w * pixel_size
        h_mm = h * pixel_size
        
        # Создаем 8 вершин для прямоугольного блока (куба)
        # Нижняя грань (z=0)
        v0 = [x_pos, y_pos, 0]
        v1 = [x_pos + w_mm, y_pos, 0]
        v2 = [x_pos + w_mm, y_pos + h_mm, 0]
        v3 = [x_pos, y_pos + h_mm, 0]
        
        # Верхняя грань (z=thickness)
        v4 = [x_pos, y_pos, thickness]
        v5 = [x_pos + w_mm, y_pos, thickness]
        v6 = [x_pos + w_mm, y_pos + h_mm, thickness]
        v7 = [x_pos, y_pos + h_mm, thickness]
        
        vertices = [v0, v1, v2, v3, v4, v5, v6, v7]
        all_vertices.extend(vertices)
        
        # Создаем грани (треугольники)
        # Нижняя грань
        all_faces.append([vertex_count, vertex_count+1, vertex_count+2])
        all_faces.append([vertex_count, vertex_count+2, vertex_count+3])
        
        # Верхняя грань
        all_faces.append([vertex_count+4, vertex_count+6, vertex_count+5])
        all_faces.append([vertex_count+4, vertex_count+7, vertex_count+6])
        
        # Боковые грани
        # Спереди
        all_faces.append([vertex_count+3, vertex_count+2, vertex_count+6])
        all_faces.append([vertex_count+3, vertex_count+6, vertex_count+7])
        
        # Сзади
        all_faces.append([vertex_count+0, vertex_count+5, vertex_count+1])
        all_faces.append([vertex_count+0, vertex_count+4, vertex_count+5])
        
        # Справа
        all_faces.append([vertex_count+1, vertex_count+5, vertex_count+6])
        all_faces.append([vertex_count+1, vertex_count+6, vertex_count+2])
        
        # Слева
        all_faces.append([vertex_count+0, vertex_count+3, vertex_count+7])
        all_faces.append([vertex_count+0, vertex_count+7, vertex_count+4])
        
        vertex_count += 8
    
    # Преобразуем списки в numpy массивы
    return np.array(all_vertices), np.array(all_faces)

def parse_svg_viewbox(svg_file):
    """
    Parse the viewBox attribute from SVG file to get the actual content dimensions.
    """
    try:
        import xml.etree.ElementTree as ET
        
        # Register the SVG namespace
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        
        # Parse the SVG file
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Get the viewBox attribute
        if 'viewBox' in root.attrib:
            viewbox = root.attrib['viewBox']
            x, y, width, height = map(float, viewbox.split())
            return (x, y, width, height)
        
        # If no viewBox, try to get width and height attributes
        elif 'width' in root.attrib and 'height' in root.attrib:
            width = root.attrib['width']
            height = root.attrib['height']
            
            # Remove units if present
            width = float(width.replace('mm', '').replace('px', ''))
            height = float(height.replace('mm', '').replace('px', ''))
            
            return (0, 0, width, height)
    
    except Exception as e:
        print(f"Warning: Could not parse SVG viewBox: {e}")
    
    return None

def svg_to_stl(svg_file, output_file=None, thickness=1.0, pixel_size=0.05, debug=False):
    """
    Convert an SVG file to an STL 3D model.
    
    Args:
        svg_file (str): Path to the SVG file
        output_file (str, optional): Path to the output STL file. If None, derived from the SVG filename.
        thickness (float): Thickness of the resulting 3D model in mm
        pixel_size (float): Size of each pixel in mm
        debug (bool): Whether to save debug images
    
    Returns:
        str: Path to the generated STL file
    """
    start_time = time.time()
    print(f"Converting {svg_file} to STL...")
    
    # Рассчитываем DPI на основе размера пикселя
    dpi = calculate_dpi(pixel_size)
    print(f"Using pixel size {pixel_size}mm (calculated DPI: {dpi})")
    
    # If output_file is not specified, derive it from the input filename
    if output_file is None:
        output_file = str(Path(svg_file).with_suffix('.stl'))
    
    # Get SVG dimensions
    svg_dims = parse_svg_viewbox(svg_file)
    if svg_dims:
        print(f"SVG viewBox: x={svg_dims[0]}, y={svg_dims[1]}, width={svg_dims[2]}, height={svg_dims[3]}")
    
    # Create a temporary PNG file
    temp_png = "temp.png"
    
    # Convert SVG to PNG with calculated resolution
    print(f"Rasterizing SVG at {dpi} DPI...")
    
    # Use cairosvg with specific parameters to preserve transparency
    cairosvg.svg2png(
        url=svg_file,
        write_to=temp_png,
        dpi=dpi,
        background_color="white",  # Set background to white
        scale=1.0
    )
    
    # Load the PNG image and convert to grayscale
    print("Creating binary mask...")
    img = Image.open(temp_png).convert('RGB')  # Convert to RGB first
    
    # Extract only the black parts (PCB traces in KiCad)
    # For KiCad, the PCB traces are black, and we want to turn them into 3D
    r, g, b = img.split()
    
    # Check all channels - black pixels have low values in all channels
    # Create a binary mask where black parts are white (255) and everything else is black (0)
    threshold = 50  # Adjust if needed
    black_mask = np.where((np.array(r) < threshold) & 
                          (np.array(g) < threshold) & 
                          (np.array(b) < threshold), 255, 0).astype(np.uint8)
    
    # Convert back to PIL Image
    mask_img = Image.fromarray(black_mask)
    
    width, height = mask_img.size
    print(f"Original image dimensions: {width}x{height} pixels")
    
    # Now crop the mask to the content area
    print("Cropping image to content area...")
    bbox = mask_img.getbbox()
    
    if bbox is None:
        print("Warning: No black content found in the image. The SVG might not contain black fills.")
        # Use the whole image as fallback
        cropped_mask = mask_img
    else:
        # Add padding
        padding = 10
        width, height = mask_img.size
        bbox = (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(width, bbox[2] + padding),
            min(height, bbox[3] + padding)
        )
        print(f"Cropping image from {mask_img.size} to {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
        cropped_mask = mask_img.crop(bbox)
    
    # Сохраняем только обрезанное изображение, если требуется отладка
    if debug:
        cropped_mask.save("debug_cropped.png")
    
    # Используем обрезанное изображение без дальнейшего масштабирования
    width, height = cropped_mask.size
    print(f"Processing image dimensions: {width}x{height} pixels")
    
    # Create binary mask: True where image has content (white in our mask)
    img_array = np.array(cropped_mask)
    # Convert values > 127 to True, <= 127 to False
    mask = img_array > 127
    
    # Calculate real-world dimensions
    real_width = width * pixel_size
    real_height = height * pixel_size
    print(f"Real dimensions: {real_width:.2f}mm x {real_height:.2f}mm x {thickness:.2f}mm")
    
    # Create 3D model with optimization
    print("Generating optimized 3D model...")
    vertices, faces = create_optimized_mesh(mask, thickness, pixel_size)
    
    if vertices is None or len(faces) == 0 or len(vertices) == 0:
        print("Error: Failed to generate mesh. Check if the SVG contains valid black fill areas.")
        if os.path.exists(temp_png) and not debug:
            os.remove(temp_png)
        return None
    
    # Create the mesh
    print(f"Creating STL mesh with {len(vertices)} vertices and {len(faces)} faces...")
    
    # Create the mesh
    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j]]
    
    # Save the mesh to STL file
    print(f"Saving mesh to {output_file}...")
    stl_mesh.save(output_file)
    
    # Clean up temporary files
    if os.path.exists(temp_png) and not debug:
        os.remove(temp_png)
    
    elapsed_time = time.time() - start_time
    print(f"Conversion complete: {output_file}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Convert SVG files to STL 3D models for photopolymer printing.'
    )
    
    # Input file or --all flag
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('svg_file', nargs='?', help='Input SVG file')
    input_group.add_argument('--all', action='store_true', help='Convert all SVG files in the current directory')
    
    # Optional parameters
    parser.add_argument('--thickness', type=float, default=1.0, help='Thickness of the resulting 3D model in mm (default: 1.0)')
    parser.add_argument('--pixel_size', type=float, default=0.05, help='Size of each pixel in mm (default: 0.05)')
    parser.add_argument('--debug', action='store_true', help='Save debug images and keep temporary files')
    
    args = parser.parse_args()
    
    if args.all:
        # Process all SVG files in the current directory
        svg_files = list(Path('.').glob('*.svg'))
        if not svg_files:
            print("No SVG files found in the current directory.")
            sys.exit(1)
        
        print(f"Found {len(svg_files)} SVG files.")
        for svg_file in svg_files:
            svg_to_stl(
                str(svg_file),
                thickness=args.thickness,
                pixel_size=args.pixel_size,
                debug=args.debug
            )
    else:
        # Process a single SVG file
        if not os.path.exists(args.svg_file):
            print(f"Error: File '{args.svg_file}' not found.")
            sys.exit(1)
        
        svg_to_stl(
            args.svg_file,
            thickness=args.thickness,
            pixel_size=args.pixel_size,
            debug=args.debug
        )

if __name__ == "__main__":
    main()
