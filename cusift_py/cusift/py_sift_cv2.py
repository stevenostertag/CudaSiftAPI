import cv2
import numpy as np
from pathlib import Path
from typing import Union



class SIFT_CV2:
    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
    
    def _keypoints_descriptors_to_dict(self,keypoints, descriptors):
        keypoints_dict = []
        for kp in keypoints:
            octave, layer = self._unpack_octave(kp)
            kp_dict = {
                "x": kp.pt[0],
                "y": kp.pt[1],
                "size": kp.size,
                "angle": kp.angle,
                "response": kp.response,
                "octave": octave,
                "layer": layer,
                "descriptor": descriptors[keypoints.index(kp)].tolist() if descriptors is not None else None
            }
            keypoints_dict.append(kp_dict)

        return keypoints_dict

    def _remove_duplicate_keypoints(self, keypoints, descriptors, cluster_radius):
        # If cluster_radius is 0 or negative, we skip duplicate removal and return all keypoints and descriptors
        # If cluster radius is positive, we filter out keypoints that are within the specified radius of each other, keeping only one representative keypoint and its descriptor
        # Choosing to keep the keypoint that has the highest response value among the duplicates, as it is likely to be the most distinctive and stable keypoint in that region. This approach helps to retain the most informative keypoints while reducing redundancy.
        if cluster_radius <= 0.0:
            return keypoints, descriptors

        # Sort by response descending so the first kept in each cluster is the strongest
        sorted_indices = sorted(range(len(keypoints)), key=lambda i: keypoints[i].response, reverse=True)
        sorted_kps = [keypoints[i] for i in sorted_indices]
        sorted_descs = descriptors[sorted_indices] if descriptors is not None else None

        unique_keypoints = []
        unique_descriptors = []
        for i, kp in enumerate(sorted_kps):
            is_duplicate = False
            for ukp in unique_keypoints:
                dist = np.sqrt((kp.pt[0] - ukp.pt[0]) ** 2 + (kp.pt[1] - ukp.pt[1]) ** 2)
                if dist < cluster_radius:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_keypoints.append(kp)
                if sorted_descs is not None:
                    unique_descriptors.append(sorted_descs[i])

        return unique_keypoints, np.array(unique_descriptors) if descriptors is not None else None
    
    def _unpack_octave(self, kp):
        octave = kp.octave & 0xFF
        layer = (kp.octave >> 8) & 0xFF
        if octave >= 128:
            octave -= 256
        return octave, layer

    def _extract_image_arg(self, image):
        if isinstance(image, str) or isinstance(image, Path):
            return cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError("Unsupported image type. Must be a file path or a numpy array.")

    def extract(self,
        image: Union[str, Path, np.ndarray],
        *,
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
        min_size=0,
        max_size=float('inf'),
        min_response=0.0,
        max_response=float('inf'),
        cluster_radius=0.0
    ):
        img_data = self._extract_image_arg(image)
        sift = cv2.SIFT_create(nfeatures=nfeatures,
                               nOctaveLayers=nOctaveLayers,
                               contrastThreshold=contrastThreshold,
                               edgeThreshold=edgeThreshold,
                               sigma=sigma)
        keypoints, descriptors = sift.detectAndCompute(img_data, None)

        if keypoints:
            filtered = [(kp, i) for i, kp in enumerate(keypoints) if (min_size <= kp.size <= max_size) and (min_response <= kp.response <= max_response)]
            if filtered:
                keypoints, indices = zip(*filtered)
                keypoints = list(keypoints)
                descriptors = descriptors[list(indices)] if descriptors is not None else None
            else:
                keypoints = []
                descriptors = None
        
        keypoints, descriptors = self._remove_duplicate_keypoints(keypoints, descriptors, cluster_radius)

        return self._keypoints_descriptors_to_dict(keypoints, descriptors)
    
    def draw_keypoints(self,
        image: Union[str, Path, np.ndarray],
        keypoints_list: list,
        outfile: Union[str, Path] = None,
        color: tuple = None,
        thickness: int = 1,
        show_size: bool = True,
        show_angle: bool = True,
        color_bins: int = 16,
    ):
        if isinstance(image, (str, Path)):
            img_data = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        elif isinstance(image, np.ndarray):
            img_data = image
        else:
            raise ValueError("Unsupported image type. Must be a file path or a numpy array.")
        img_out = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR) if len(img_data.shape) == 2 else img_data.copy()

        if not keypoints_list:
            if outfile:
                cv2.imwrite(str(outfile), img_out)
            return img_out

        if color is None:
            sizes = [kp['size'] for kp in keypoints_list]
            min_size, max_size = min(sizes), max(sizes)
            size_range = max_size - min_size if max_size > min_size else 1.0

            def size_to_color(size):
                t = (size - min_size) / size_range
                bin_idx = min(int(t * color_bins), color_bins - 1)
                hue = int(bin_idx * 180 / color_bins)
                color_hsv = np.uint8([[[hue, 255, 255]]])
                bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            size_to_color = lambda size: color

        for kp_dict in keypoints_list:
            x, y = int(round(kp_dict['x'])), int(round(kp_dict['y']))
            radius = int(round(kp_dict['size'] / 2))
            kp_color = size_to_color(kp_dict['size'])

            if show_size:
                cv2.circle(img_out, (x, y), radius, kp_color, thickness, lineType=cv2.LINE_AA)
            else:
                cv2.circle(img_out, (x, y), 3, kp_color, -1, lineType=cv2.LINE_AA)

            if show_angle and kp_dict['angle'] >= 0:
                angle_rad = np.deg2rad(kp_dict['angle'])
                x_end = int(round(x + radius * np.cos(angle_rad)))
                y_end = int(round(y + radius * np.sin(angle_rad)))
                cv2.line(img_out, (x, y), (x_end, y_end), kp_color, thickness, lineType=cv2.LINE_AA)

        if outfile:
            cv2.imwrite(str(outfile), img_out)

        return img_out
    
    def draw_descriptors(self,
        image: Union[str, Path, np.ndarray],
        keypoints_list: list,
        outdir: Union[str, Path] = None,
        snippet_size: int = 128,
    ):
        if isinstance(image, (str, Path)):
            img_data = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        elif isinstance(image, np.ndarray):
            img_data = image
        else:
            raise ValueError("Unsupported image type. Must be a file path or a numpy array.")

        if len(img_data.shape) == 2:
            img_color = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img_data.copy()

        h, w = img_color.shape[:2]

        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)

        results = []
        for idx, kp_dict in enumerate(keypoints_list):
            x, y = kp_dict['x'], kp_dict['y']
            radius = kp_dict['size'] / 2
            pad = max(int(radius * 2), snippet_size // 2)

            # Crop region (clamped to image bounds)
            x1 = max(int(x - pad), 0)
            y1 = max(int(y - pad), 0)
            x2 = min(int(x + pad), w)
            y2 = min(int(y + pad), h)
            snippet = img_color[y1:y2, x1:x2].copy()

            # Draw circle and angle on snippet
            cx, cy = int(x - x1), int(y - y1)
            r = int(round(radius)*3)  # Scale radius for better visibility
            cv2.circle(snippet, (cx, cy), r, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            if kp_dict['angle'] >= 0:
                angle_rad = np.deg2rad(kp_dict['angle'])
                ex = int(round(cx + r * np.cos(angle_rad)))
                ey = int(round(cy + r * np.sin(angle_rad)))
                cv2.line(snippet, (cx, cy), (ex, ey), (0, 255, 0), 1, lineType=cv2.LINE_AA)

            # Resize snippet to fixed height
            snippet_h = snippet_size
            scale = snippet_h / snippet.shape[0] if snippet.shape[0] > 0 else 1
            snippet_w = max(int(snippet.shape[1] * scale), 1)
            snippet_resized = cv2.resize(snippet, (snippet_w, snippet_h))

            # Build descriptor visualization (4x4 grid of 8-bin histograms)
            descriptor = kp_dict.get('descriptor')
            if descriptor is not None:
                desc = np.array(descriptor)
                desc_img = self._draw_descriptor_grid(desc, snippet_h)
            else:
                desc_img = np.zeros((snippet_h, snippet_h, 3), dtype=np.uint8)

            # Combine snippet and descriptor side by side
            combined = np.hstack([snippet_resized, desc_img])

            if outdir is not None:
                cv2.imwrite(str(outdir / f"descriptor_{idx:06d}.png"), combined)

            results.append(combined)

        return results

    def _draw_descriptor_grid(self, descriptor, size):
        """Draw a 4x4 grid where each cell shows an 8-bin orientation histogram."""
        grid = 4
        cell = size // grid
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # SIFT descriptor: 4x4 spatial bins x 8 orientation bins = 128
        bins = descriptor.reshape(grid, grid, 8)
        max_val = bins.max() if bins.max() > 0 else 1.0

        for gy in range(grid):
            for gx in range(grid):
                hist = bins[gy, gx]
                cx = gx * cell + cell // 2
                cy = gy * cell + cell // 2

                # Draw cell border
                cv2.rectangle(img, (gx * cell, gy * cell),
                              ((gx + 1) * cell - 1, (gy + 1) * cell - 1),
                              (40, 40, 40), 1)

                # Draw each orientation bin as a line from center
                for b in range(8):
                    angle = b * (2 * np.pi / 8)
                    magnitude = hist[b] / max_val
                    length = magnitude * (cell // 2 - 2)
                    ex = int(cx + length * np.cos(angle))
                    ey = int(cy + length * np.sin(angle))
                    brightness = int(128 + 127 * magnitude)
                    cv2.line(img, (cx, cy), (ex, ey),
                             (brightness, brightness, 255), 1, lineType=cv2.LINE_AA)

        return img


if __name__ == "__main__":
    img_path = Path(__file__).parent / "share" / "river1.jpg"

    keypoints = SIFT_CV2().extract(cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE), min_size=5, max_size=50, min_response=0.02, max_response=0.5, cluster_radius=10.0, nOctaveLayers=5)

    for kp in keypoints:
        print(f"Keypoint: {kp['x']}, {kp['y']}, Size: {kp['size']}, Angle: {kp['angle']}, Response: {kp['response']}, Octave: {kp['octave']}, Layer: {kp['layer']}")
        #print(f"Descriptor: {kp['descriptor']}")

    # Draw keypoints on the image and save it
    output_path = Path(__file__).parent / "share" / "img_with_keypoints.png"
    SIFT_CV2().draw_keypoints(str(img_path), keypoints, outfile=output_path)

    # Descriptor directory
    descriptor_dir = Path(__file__).parent / "share" / "descriptors"
    SIFT_CV2().draw_descriptors(str(img_path), keypoints, outdir=descriptor_dir)
