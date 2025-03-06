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

def downsample_image(img, max_size=1000):
    """Downsample image if it's too large."""
    width, height = img.size
    
    # Calculate scaling factor to keep aspect ratio
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        print(f"Downsampling image from {width}x{height} to {new_width}x{new_height}")
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    return img

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
    
    if debug:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save("debug_mask.png")
    
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

def create_heightmap_mesh(mask, thickness, pixel_size):
    """Create a mesh using a heightmap approach (more efficient)."""
    height, width = mask.shape
    
    # Create a grid of vertices
    x = np.arange(0, width) * pixel_size
    y = np.arange(0, height) * pixel_size
    xx, yy = np.meshgrid(x, y)
    
    # Flip Y coordinates (PIL origin is top-left, we want bottom-left for 3D)
    yy = (height - 1) * pixel_size - yy
    
    # Create bottom and top vertices
    bottom_vertices = np.column_stack((xx.flatten(), yy.flatten(), np.zeros(width * height)))
    top_vertices = np.column_stack((xx.flatten(), yy.flatten(), np.ones(width * height) * thickness))
    
    # Filter out vertices that are not in the mask
    mask_flat = mask.flatten()
    active_indices = np.where(mask_flat)[0]
    
    if len(active_indices) == 0:
        print("Warning: No pixels found in mask. Check if the SVG has proper black fills.")
        return None, None
    
    # Create a mapping from original indices to filtered vertex indices
    index_map = np.zeros(width * height, dtype=int) - 1
    index_map[active_indices] = np.arange(len(active_indices))
    
    # Filter vertices
    bottom_active = bottom_vertices[active_indices]
    top_active = top_vertices[active_indices]
    
    # Combine all vertices
    vertices = np.vstack((bottom_active, top_active))
    
    # Number of vertices per layer
    n_active = len(active_indices)
    
    # Create faces
    faces = []
    
    # Function to check if a point is in the mask and has a valid index
    def is_valid(x, y):
        if 0 <= x < width and 0 <= y < height:
            index = y * width + x
            return mask_flat[index] and index_map[index] >= 0
        return False
    
    # Progress indicator
    print(f"Creating mesh with {n_active} active vertices...")
    print("Creating top and bottom faces...")
    
    # Create top and bottom faces for each active pixel
    for i in tqdm(range(height - 1)):
        for j in range(width - 1):
            # Check if current pixel is in the mask
            if not mask[i, j]:
                continue
                
            # Get indices of the four corners of this pixel
            idx00 = i * width + j
            idx01 = i * width + (j + 1)
            idx10 = (i + 1) * width + j
            idx11 = (i + 1) * width + (j + 1)
            
            # Check if neighbors are also in the mask
            has_right = mask[i, j + 1] if j + 1 < width else False
            has_bottom = mask[i + 1, j] if i + 1 < height else False
            has_diagonal = mask[i + 1, j + 1] if (i + 1 < height and j + 1 < width) else False
            
            # Create bottom face triangles
            if has_right and has_bottom and has_diagonal:
                # Get the mapped indices
                m00 = index_map[idx00]
                m01 = index_map[idx01]
                m10 = index_map[idx10]
                m11 = index_map[idx11]
                
                # Create two triangles for the bottom face
                faces.append([m00, m01, m11])
                faces.append([m00, m11, m10])
                
                # Create two triangles for the top face
                faces.append([m00 + n_active, m11 + n_active, m01 + n_active])
                faces.append([m00 + n_active, m10 + n_active, m11 + n_active])
    
    print("Creating side faces...")
    
    # Create side faces along the edges of the model
    for y in tqdm(range(height)):
        for x in range(width):
            if not mask[y, x]:
                continue
                
            idx = y * width + x
            m_idx = index_map[idx]
            
            # Check each of the four sides and add faces if it's an edge
            # Right edge
            if x == width - 1 or not mask[y, x + 1]:
                if x < width - 1 and y < height - 1 and mask[y + 1, x]:
                    right_bottom_idx = index_map[(y + 1) * width + x]
                    faces.append([m_idx, m_idx + n_active, right_bottom_idx + n_active])
                    faces.append([m_idx, right_bottom_idx + n_active, right_bottom_idx])
            
            # Left edge
            if x == 0 or not mask[y, x - 1]:
                if x > 0 and y < height - 1 and mask[y + 1, x]:
                    left_bottom_idx = index_map[(y + 1) * width + x]
                    faces.append([m_idx, left_bottom_idx, m_idx + n_active])
                    faces.append([left_bottom_idx, left_bottom_idx + n_active, m_idx + n_active])
            
            # Top edge
            if y == 0 or not mask[y - 1, x]:
                if y > 0 and x < width - 1 and mask[y, x + 1]:
                    top_right_idx = index_map[y * width + (x + 1)]
                    faces.append([m_idx, top_right_idx, m_idx + n_active])
                    faces.append([top_right_idx, top_right_idx + n_active, m_idx + n_active])
            
            # Bottom edge
            if y == height - 1 or not mask[y + 1, x]:
                if y < height - 1 and x < width - 1 and mask[y, x + 1]:
                    bottom_right_idx = index_map[y * width + (x + 1)]
                    faces.append([m_idx, m_idx + n_active, bottom_right_idx])
                    faces.append([m_idx + n_active, bottom_right_idx + n_active, bottom_right_idx])
    
    return vertices, np.array(faces)

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

def svg_to_stl(svg_file, output_file=None, thickness=1.0, pixel_size=0.05, dpi=300, max_image_size=1000, debug=False):
    """
    Convert an SVG file to an STL 3D model.
    
    Args:
        svg_file (str): Path to the SVG file
        output_file (str, optional): Path to the output STL file. If None, derived from the SVG filename.
        thickness (float): Thickness of the resulting 3D model in mm
        pixel_size (float): Size of each pixel in mm
        dpi (int): DPI for rasterization of the SVG
        max_image_size (int): Maximum size for image processing in pixels
        debug (bool): Whether to save debug images
    
    Returns:
        str: Path to the generated STL file
    """
    start_time = time.time()
    print(f"Converting {svg_file} to STL...")
    
    # If output_file is not specified, derive it from the input filename
    if output_file is None:
        output_file = str(Path(svg_file).with_suffix('.stl'))
    
    # Get SVG dimensions
    svg_dims = parse_svg_viewbox(svg_file)
    if svg_dims:
        print(f"SVG viewBox: x={svg_dims[0]}, y={svg_dims[1]}, width={svg_dims[2]}, height={svg_dims[3]}")
    
    # Create a temporary PNG file
    temp_png = "temp.png"
    
    # Convert SVG to PNG with high resolution
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
    
    if debug:
        img.save("debug_original_color.png")
        mask_img.save("debug_black_mask.png")
    
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
    
    if debug:
        cropped_mask.save("debug_cropped_mask.png")
    
    # Downsample image if too large
    scaled_mask = downsample_image(cropped_mask, max_size=max_image_size)
    width, height = scaled_mask.size
    print(f"Processing image dimensions: {width}x{height} pixels")
    
    if debug:
        scaled_mask.save("debug_downsampled_mask.png")
    
    # Create binary mask: True where image has content (white in our mask)
    img_array = np.array(scaled_mask)
    # Convert values > 127 to True, <= 127 to False
    mask = img_array > 127
    
    # Calculate real-world dimensions
    real_width = width * pixel_size
    real_height = height * pixel_size
    print(f"Real dimensions: {real_width:.2f}mm x {real_height:.2f}mm x {thickness:.2f}mm")
    
    # Create 3D model
    print("Generating 3D model...")
    vertices, faces = create_heightmap_mesh(mask, thickness, pixel_size)
    
    if vertices is None or len(faces) == 0:
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
    parser.add_argument('--dpi', type=int, default=300, help='DPI for rasterization of the SVG (default: 300)')
    parser.add_argument('--max_size', type=int, default=1000, help='Maximum image size in pixels (default: 1000)')
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
                dpi=args.dpi,
                max_image_size=args.max_size,
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
            dpi=args.dpi,
            max_image_size=args.max_size,
            debug=args.debug
        )

if __name__ == "__main__":
    main()
