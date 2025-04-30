from LineageTree import lineageTree
import numpy as np
from skimage.morphology import ball
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tifffile import imread, imwrite
from scipy.ndimage import sum_labels
from scipy.ndimage import gaussian_filter, median_filter, grey_erosion
from skimage.filters import threshold_otsu, sobel
import argparse

lT = lineageTree.load("/Users/irmakavci/Desktop/STAGE IBDM/LineageTree_Hackathon.lT")

def draw_sph_at_pos(pos_at_t: np.ndarray, shape: tuple[int, int, int], radius: int) -> np.ndarray:
    """
    Draw spheres at specified positions in a 3D segmentation array.
    
    Parameters:

    pos_at_t (np.ndarray): Array of cell positions as (x, y, z) coordinates.
    shape (tuple[int, int, int]): Shape of the segmentation image (x, y, z).
    radius (int): Radius of the spheres to be drawn at each position.
    
    Returns:
    int
        New segmentation array with spheres drawn at cell positions.
    """

    padd_shape = (shape[0] + 2 * radius, shape[1] + 2 * radius, shape[2] + 2 * radius)
    padd_seg = np.zeros(padd_shape, dtype=np.uint16)  

    sphere = ball(radius) 
    mask = 0 < sphere # dtype=bool 

    
    for i, pos in enumerate(pos_at_t, start=1):
        x, y, z = np.round(pos[::-1]).astype(int)  
        x, y, z = x + radius, y + radius, z + radius  

        x_start, x_end = x - radius, x + radius + 1
        y_start, y_end = y - radius, y + radius + 1
        z_start, z_end = z - radius, z + radius + 1

        padd_seg[x_start:x_end, y_start:y_end, z_start:z_end][mask] = i
    
    new_seg = padd_seg[radius:-radius, radius:-radius, radius:-radius]

    return new_seg


def extract_corr_seg_cells(im_seg: np.ndarray,  
                                    min_area=50, max_area=5000, threshold=0.8) -> np.ndarray:
    """
    Extracts correctly segmented cells based on manual positions.

    Parameters:
    - manual_positions: np.ndarray -> Array of manually labeled positions (x, y, z).
    - segmented_image: np.ndarray -> The segmented image with labeled regions.

    Returns:
    int
        np.ndarray -> New segmentation image with only correctly assigned cells.
    """

    filter_seg = np.zeros(im_seg.shape, dtype=np.uint16)

    regions = {"coords": [], "labels": []}
 
    for prop in regionprops(im_seg):
        mask = im_seg == prop.label
        if min_area <= prop.area <= max_area and prop.area > threshold:
            filter_seg[mask] = prop.label #filter_seg[label == prop.label] = prop.label  
            regions["coords"].append(prop.centroid)  
            regions["labels"].append(prop.label)  

    return filter_seg


def region_filter_lsa(im_seg: np.ndarray, pos_at_t: np.ndarray) -> np.ndarray:

    filter_seg = np.zeros(im_seg.shape, dtype=np.uint16)

    regions = {"coords": [], "labels": []}
    points = [pos[::-1] for pos in pos_at_t]

    for prop in regionprops(im_seg):

        regions["coords"].append(prop.centroid)  
        regions["labels"].append(prop.label) 

    cost = cdist(points, regions["coords"])  
    row_ind, col_ind = linear_sum_assignment(cost)
    hits = np.array(regions["labels"])[col_ind]

    corr_seg = np.zeros_like(filter_seg)

    for im_seg_id in hits:
        corr_seg[im_seg == im_seg_id] = im_seg_id
    
    return corr_seg

def ws_adaptive_mask(im, pos_at_t: np.ndarray, r=7, sigma=1.5,
                     min_th_vol=0, max_th_vol=10000,
                     percent_of_th=70, increment=10,
                     edge_thresh=0.008, intensity_floor=25,
                     min_mask_intensity=100,
                     use_mixed_threshold=True, mix_ratio=0.3,
                     min_mask_floor_limit=0):
    """
    Adaptive watershed segmentation using seed positions and image filtering.

    Applies erosion, Gaussian filtering, and an adaptive threshold combining Otsu and intensity mean.
    Iteratively adjusts the mask to recover missed seeds while ignoring low-gradient or low-intensity regions.

    Parameters:
        im (np.ndarray): Input image.
        pos_at_t (np.ndarray): Seed positions (Y, X).
        r (int): Erosion radius.
        sigma (float): Gaussian blur sigma.
        min_th_vol (int): Minimum allowed object volume.
        max_th_vol (int): Maximum allowed object volume.
        percent_of_th (int): Starting threshold percentage.
        increment (int): Decrease step for threshold.
        edge_thresh (float): Gradient threshold for ignoring background.
        intensity_floor (float): Minimum intensity for inclusion.
        min_mask_intensity (float): Minimum mask intensity.
        use_mixed_threshold (bool): Whether to mix Otsu and mean thresholds.
        mix_ratio (float): Weight for mixing threshold methods.
        min_mask_floor_limit (float): Lower bound for intensity thresholding.

    Returns:
        np.ndarray: Labeled segmentation mask.
    """

    pos_array = np.array([p[::-1] for p in pos_at_t]).round().astype(np.uint16)

    im_filtered = gaussian_filter(im, sigma=sigma)
    im_filtered = median_filter(im_filtered, size=3)

    gradient = sobel(im_filtered)
    ignore_mask = (gradient < edge_thresh) & (im_filtered < intensity_floor)

    erosions = {rad: grey_erosion(im_filtered, size=rad) for rad in [5, 7, 10, 15]}
    im_for_ws = im_filtered - erosions[r]
    im_for_ws_gs = gaussian_filter(im_for_ws, sigma=1)

    th_otsu = threshold_otsu(im_for_ws_gs)
    th = (1 - mix_ratio) * th_otsu + mix_ratio * np.mean(im_for_ws_gs)

    seeds = np.zeros_like(im)
    seeds[tuple(pos_array.T)] = np.arange(1, len(pos_at_t) + 1)

    def get_mask(pct, min_intensity):
        thresh_val = max(th * (pct / 100), min_intensity)
        return (im_for_ws_gs > thresh_val) & (~ignore_mask)

    current_min_intensity = min_mask_intensity
    mask = get_mask(percent_of_th, current_min_intensity)
    ws = watershed(np.max(im_for_ws_gs) - im_for_ws_gs, seeds, mask=mask)

    all_seeds = np.unique(seeds)
    missed_seeds = set(all_seeds).difference(np.unique(ws))
    ones_ws = np.ones_like(ws)

    def sum_labels(arr, lbls, label_vals):
        return [np.sum(arr[lbls == v]) for v in label_vals]

    while missed_seeds and percent_of_th > 0:
        percent_of_th -= increment
        current_min_intensity = max(current_min_intensity - 5, min_mask_floor_limit)
        mask = get_mask(percent_of_th, current_min_intensity)
        new_ws = watershed(np.max(im_for_ws_gs) - im_for_ws_gs, seeds, mask=mask)
        label_list = np.arange(1, new_ws.max() + 1)
        volumes = dict(zip(label_list, sum_labels(ones_ws, new_ws, label_list)))
        for s in missed_seeds.intersection(label_list):
            if min_th_vol < volumes[s] < max_th_vol:
                ws[new_ws == s] = s
        missed_seeds = set(all_seeds).difference(np.unique(ws))
        print(f"Th {percent_of_th}% – min_intensity={current_min_intensity} – missing seeds: {len(missed_seeds)}")

    return ws

# optional 
 
def remove_irregular_labels_3d(im, ws, min_vol=500, max_vol=100000, max_ecc=0.99, min_solidity=0.7, min_mean_intensity=50):
    """
    Remove large or irregular segmented objects based on geometry and intensity.

    Filters out regions with too large area or low solidity, if they also have low mean intensity.

    Parameters:
        im (np.ndarray): Original intensity image.
        ws (np.ndarray): Labeled segmentation mask.
        min_vol (int): Minimum volume (unused here).
        max_vol (int): Maximum allowed volume.
        max_ecc (float): Maximum eccentricity (unused here).
        min_solidity (float): Minimum solidity to keep region.
        min_mean_intensity (float): Minimum mean intensity to keep region.

    Returns:
        np.ndarray: Cleaned label mask with irregular regions removed.
    """

    cleaned_ws = np.copy(ws)
    props = regionprops(ws, intensity_image=im)

    for region in props:
        label = region.label
        region_mask = (ws == label)

        mean_intensity = region.mean_intensity if region.intensity_image is not None else 0

        bigger = region.area > max_vol
        irregular = hasattr(region, "solidity") and region.solidity < min_solidity

        if (bigger or irregular) and mean_intensity < min_mean_intensity:
            cleaned_ws[region_mask] = 0  

    return cleaned_ws

def main_function(
        t: int,
        method: str = ["watershed", "sphere", "segmentation"],
        output_path: str = None,
        input_path: str = None,
        lineage_tree_path: str = None):
    im = imread(input_path.format(t=t))
    lT = lineageTree.load(lineage_tree_path)
    pos_at_t = [lT.pos[c] for c in lT.nodes_at_t(t)]
    ws = ws_adaptive_mask(im, pos_at_t)
    cleaned_ws = remove_irregular_labels_3d(im, ws)

    imwrite(output_path, cleaned_ws)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog= "Hybride watershed",
                                     description= "Run a watershed algorithm to create a new ground truth image")
    parser.add_argument("-t", "--time", type=int, help= "Time to process")
    parser.add_argument("-m", "--method", type=str, choices= ["watershed", "sphere", "segmentation"], help= "Methods of creating a new ground truth image")
    parser.add_argument("-o", "--output_path", type=str, help= "Output path for a new ground truth image")
    parser.add_argument("--input_path", type=str, default= ("/Users/irmakavci/Desktop/STAGE IBDM/t{t:05}.tiff"), help= "Input path for pattern with t as time")
    parser.add_argument("--lineage_tree_path", type=str, default= ("/Users/irmakavci/Desktop/STAGE IBDM/LineageTree_Hackathon.lT"), help= "Path to lineage tree file")

    args = parser.parse_args()

main_function(
        t=args.time,
        method=args.method,
        output_path=args.output_path,
        input_path=args.input_path,
        lineage_tree_path=args.lineage_tree_path
    )

# python3 seg.py -t 200 -m watershed -o seg_200_ws.tiff

