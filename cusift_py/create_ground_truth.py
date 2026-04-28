"""
create_ground_truth.py - A visual tool for creating ground truth keypoint files
for a directory of images, with a Gaussian blur option and display controls.

This script opens a single window to iterate through all .tiff images. A Gaussian
blur can be applied upon loading. All annotations are saved upon closing the window.

Usage:
    python create_ground_truth.py <image_directory> [--output <output_directory>] [--blur-sigma <value>]

Controls (in the image window):
    - Mouse Wheel Scroll: Zoom in/out, centered on the cursor.
    - Middle/Right-Click & Drag: Pan the image when zoomed in.
    - <-- Prev / Next --> Buttons: Navigate between images.
    - Intensity/Contrast Sliders: Adjust image display.
    - Left-click: Add a keypoint.
    - 'u' key:    Undo the last added keypoint.
    - 'c' key:    Clear all keypoints from the current image.
    - Close Window: Saves all annotations and exits the program.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import Slider, Button
from PIL import Image
import numpy as np
# --- MODIFICATION: Import gaussian_filter ---
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None

class DirectoryAnnotator:
    """
    Manages the entire annotation process for a directory of images in a
    single, persistent window with navigation and display controls.
    """
    # --- MODIFICATION: Accept blur_sigma in constructor ---
    def __init__(self, image_paths: list, output_dir: Path, blur_sigma: float):
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.blur_sigma = blur_sigma # Store the blur value
        self.total_images = len(self.image_paths)
        self.current_index = 0
        self.all_annotations = {}
        self.original_image_data = None

        self.fig, self.ax = plt.subplots(figsize=(15, 12))
        self.fig.subplots_adjust(bottom=0.25)
        
        cdict = {
            'red':   [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
            'green': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)],
            'blue':  [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        }
        self.custom_cmap = LinearSegmentedColormap('MagentaGreen', cdict)
        
        self.im_display = None
        self.markers = []
        
        self._setup_widgets()
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def _setup_widgets(self):
        """Initializes all matplotlib widgets (sliders and buttons)."""
        ax_contrast = self.fig.add_axes([0.25, 0.12, 0.5, 0.03])
        ax_intensity = self.fig.add_axes([0.25, 0.07, 0.5, 0.03])
        
        self.contrast_slider = Slider(ax=ax_contrast, label='Contrast', valmin=1, valmax=255, valinit=255)
        self.intensity_slider = Slider(ax=ax_intensity, label='Intensity (%)', valmin=0, valmax=200, valinit=150)
        
        self.contrast_slider.on_changed(self.update_display)
        self.intensity_slider.on_changed(self.update_display)

        ax_prev = self.fig.add_axes([0.7, 0.015, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, '<-- Prev')
        self.btn_prev.on_clicked(self.prev_image)

        ax_next = self.fig.add_axes([0.81, 0.015, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next -->')
        self.btn_next.on_clicked(self.next_image)

    def load_image(self):
        """Loads and displays the image at the current index, resetting the view."""
        self.ax.clear()
        self.markers.clear()
        
        img_path = self.image_paths[self.current_index]
        
        try:
            with Image.open(img_path) as img:
                self.original_image_data = np.array(img, dtype=np.float32)

            # --- MODIFICATION: Apply Gaussian blur if sigma > 0 ---
            if self.blur_sigma > 0:
                print(f"  -> Applying Gaussian blur (sigma={self.blur_sigma})...", end="")
                self.original_image_data = gaussian_filter(self.original_image_data, sigma=self.blur_sigma)
                print(" Done.")
            # --- END MODIFICATION ---

            self.im_display = self.ax.imshow(self.original_image_data, cmap=self.custom_cmap)
            self.update_display()
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error loading image:\n{img_path.name}\n{e}", ha='center', va='center')
            self.im_display = None
            self.original_image_data = None
        
        self.ax.set_title(
            f"Image {self.current_index + 1}/{self.total_images}: {img_path.name}\n"
            f"Click to add | 'u' to undo | 'c' to clear | Close window to save all"
        )
        self.ax.axis('off')
        
        self.reload_annotations()
        self.fig.canvas.draw_idle()

    def update_display(self, val=None):
        if self.im_display is None or self.original_image_data is None:
            return

        contrast = self.contrast_slider.val
        intensity_percent = self.intensity_slider.val
        intensity_scale = intensity_percent / 100.0
        
        modified_data = self.original_image_data * intensity_scale
        
        self.im_display.set_data(modified_data)
        self.im_display.set_clim(-contrast, contrast)
        
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax: return

        base_scale, cur_xlim, cur_ylim = 1.5, self.ax.get_xlim(), self.ax.get_ylim()
        scale_factor = 1 / base_scale if event.step > 0 else base_scale

        xdata, ydata = event.xdata, event.ydata
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
        self.ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])

        self.fig.canvas.draw_idle()

    def change_image(self, new_index: int):
        self.save_current_annotations()
        
        if 0 <= new_index < self.total_images:
            self.current_index = new_index
            self.load_image()
        else:
            print("Reached the end of the directory." if new_index >= self.total_images else "Already at the first image.")

    def next_image(self, event):
        self.change_image(self.current_index + 1)
        
    def prev_image(self, event):
        self.change_image(self.current_index - 1)
        
    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1 and self.im_display:
            x, y = event.xdata, event.ydata
            marker, = self.ax.plot(x, y, 'yx', markersize=8, markeredgewidth=1.5)
            self.markers.append(marker)
            self.fig.canvas.draw_idle()
        
    def on_key_press(self, event):
        if event.key == 'u' and self.markers:
            self.markers.pop().remove()
            self.fig.canvas.draw_idle()
        elif event.key == 'c':
            for marker in self.markers: marker.remove()
            self.markers.clear()
            self.fig.canvas.draw_idle()

    def save_current_annotations(self):
        if self.current_index >= len(self.image_paths): return
        img_path = self.image_paths[self.current_index]
        output_filename = img_path.with_suffix('.json').name
        
        current_points = [{'x': m.get_xdata()[0], 'y': m.get_ydata()[0]} for m in self.markers]
        self.all_annotations[output_filename] = current_points
        if current_points:
             print(f"Stored {len(current_points)} points for {img_path.name}")
    
    def reload_annotations(self):
        img_path = self.image_paths[self.current_index]
        output_filename = img_path.with_suffix('.json').name
        
        points_to_load = self.all_annotations.get(output_filename, [])
        for point in points_to_load:
            marker, = self.ax.plot(point['x'], point['y'], 'yx', markersize=8, markeredgewidth=1.5)
            self.markers.append(marker)

    def on_close(self, event):
        self.save_current_annotations()
        
        print("\n" + "-" * 70 + "\nWindow closed. Saving all annotations...")
        
        for filename, keypoints in self.all_annotations.items():
            output_path = self.output_dir / filename
            if keypoints or (not keypoints and output_path.exists()):
                print(f"  -> Saving {len(keypoints)} points to {output_path}...")
                try:
                    with open(output_path, 'w') as f:
                        json.dump(keypoints, f, indent=2)
                except Exception as e:
                    print(f"     ERROR: Could not write file. Reason: {e}", file=sys.stderr)
            else:
                print(f"  -> Skipping {filename} (no points).")
        
        print("Save complete.\n" + "-" * 70)

    def show(self):
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visual tool to create ground truth keypoint files for a directory of images.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image_dir", type=Path, help="Path to the directory containing .tiff image files.")
    parser.add_argument("-o", "--output", type=Path, dest="output_dir", default=None,
                        help="Directory to save the output JSON files.\nDefaults to the input directory.")
    # --- MODIFICATION: Add command-line argument for blur sigma ---
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian blur applied to the image upon loading. Default: 1.0. Use 0 for no blur."
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        print(f"Error: Input directory not found at '{args.image_dir}'", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir is not None else args.image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(args.image_dir.glob("*.tiff")))
    if not image_paths:
        print(f"No .tiff files found in '{args.image_dir}'.", file=sys.stderr)
        sys.exit(0)

    total_images = len(image_paths)
    print(f"Found {total_images} TIFF images. Launching annotator...")
    
    # --- MODIFICATION: Pass blur_sigma to the annotator ---
    annotator = DirectoryAnnotator(image_paths, output_dir, blur_sigma=args.blur_sigma)
    
    for img_path in image_paths:
        output_filename = img_path.with_suffix('.json').name
        output_path = output_dir / output_filename
        
        if output_path.exists():
            print(f"  -> Found existing annotations for '{output_filename}'. Loading them.")
            try:
                with open(output_path, 'r') as f:
                    annotator.all_annotations[output_filename] = json.load(f)
            except Exception as e:
                print(f"     WARNING: Could not load file. Error: {e}. Starting fresh.", file=sys.stderr)
                annotator.all_annotations[output_filename] = []
        else:
            annotator.all_annotations[output_filename] = []

    annotator.load_image()
    annotator.show()

if __name__ == "__main__":
    main()
