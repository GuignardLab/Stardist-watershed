from lineagetree import LineageTree
import numpy as np
from skimage.morphology import ball
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter, median_filter, grey_erosion
from skimage.filters import threshold_otsu, sobel
import argparse
from tqdm import tqdm
import watermasks.utils as utils

def percentile_intensities_sampler(
    image:np.ndarray,
    intensity_thresh_range:list[int]=[10,500], # good for single view, non-deconvolved
    percentages:list[int]=[99.99, 99.9, 99, 97, 95, 92, 90, 80, 50, 30, 10],
    print_thresholds:bool=False
) -> dict:
    """ Outputs intensity values for thresholds using percentiles of intensity (to be used in the adaptive watershed mask). 

    Args:
        image (np.ndarray): The imput image
        intensity_thresh_range (list[int], optional): Values for initial filtering. Defaults to list[int][10,500] (probabbly good for no Deconvolved dataset).
        percentages (list[int], optional): the intensity percentiles to sample. Defaults to [99.99, 99.9, 99, 97, 95, 92, 90, 80, 50, 30, 10].
        print_thresholds (bool, optional): Set to true if you want the percentage and percentile value pairs printed. Defaults to False.

    Returns:
        dict: a dictionary with percentages as keys and percentile intensities values: e.g. {99.99: 523}
    """    

    thresholds = {}
    flat_im = image[(image > intensity_thresh_range[0]) & (image < intensity_thresh_range[1])] # reminder: this creates a butchered flattened 1-D array

    thresholds = {pct : np.percentile(flat_im, pct) for pct in percentages} 
    # this makes the dictionary

    if print_thresholds == True:
        from beautifultable import BeautifulTable
        table = BeautifulTable()
        table.columns.header = ['Percentage', 'Percentile Intensity']
        for first, second in thresholds.items():
             table.rows.append([f'{first} %', second])
        table.set_style(BeautifulTable.STYLE_SEPARATED)
        print(table)
    
    return thresholds


def preprocess_im(
    im,
    gauss_sigma=1,
    median_size=1,
    erosion_radius=7
):
    im_gauss = gaussian_filter(im, sigma=gauss_sigma)
    im_gauss_median = median_filter(im_gauss, size=median_size) # this seems that is not doing much! (inspection by eye)
    erosion = grey_erosion(im_gauss_median, size=erosion_radius)
    im_for_ws = im_gauss_median - erosion
    im_for_ws = gaussian_filter(im_for_ws, sigma = 1)

    return im_for_ws

def get_seeds(lT, tp:int, view:str, R_of_t:np.ndarray, scaling:np.ndarray, raw_image:np.ndarray):
    """Using the lineage tree, this function creates an image containing a different label at each annotation position.
    Positions share the same coordinate system as the raw image (to be used for watershed).

    Args:
        lT (_type_): _description_
        tp (int): _description_
        view (str): _description_
        R_of_t (np.ndarray): _description_
        scaling (np.ndarray): _description_
        raw_image (np.ndarray): _description_

    Returns:
        _type_: _description_
    """    
    reg_pos_at_t = []
    for mastodon_id_t in lT.time_nodes[tp]:
        x, y, z = utils.registered_position_of_id_in_t(mastodon_id_t, lT, tp, R_of_t, view, scaling)
        reg_pos_at_t.append([x, y, z])
    reg_pos_at_t = np.asarray(reg_pos_at_t)

    seeds_pos = np.array([p[::-1] for p in reg_pos_at_t]).round().astype(np.uint16) # the image needs (z, y, x)
    seeds_array = np.zeros_like(raw_image)
    seeds_array[tuple(seeds_pos.T)] = np.arange(1, len(reg_pos_at_t) + 1) # each seed has it's own label

    return seeds_pos, seeds_array

def ws_adaptive_mask(
    im_for_ws:np.ndarray,
    seeds_array:np.ndarray,
    min_int:int=10,
    max_int:int=500,
    min_vol:int=300,
    max_vol:int=1500,
    percentage_list:list = [99.99, 99.9, 99, 97, 95, 92, 90, 80, 50, 30, 10]

)->list :
    """A function that segments an image with the watershed function, using annotations as seeds and utilizing an adaptive mask.
    All preset values are tested against single views, 2nd hdf5 layer.

    Args:
        im_for_ws (np.ndarray): pre-processed image
        pos_at_t (np.ndarray): list of (x,y,z) annotation positions
        max_int (int, optional): minmum intenstiy value for thresholds @ adaptive mask. Defaults to 500.
        min_int (int, optional): maximum intenstiy value for thresholds @ adaptive mask. Defaults to 10.
        min_vol (int, optional): minimum allowed volume of a image segment. Defaults to 300.
        max_vol (int, optional):  maximum allowed volume of a image segment. Defaults to 1500.
        percentage_list (list)
        !!!!!!! Add description


    Returns:
        list: [segmentation mask, stats table] 
    """

    # GET THE SET OF ALL SEEDS VALUES
    all_seeds = set(np.unique(seeds_array)) - {0} # don't include the background

    # MASK (TO BE UPDATED)
    def get_mask(thresh_val): # simpler mask generation, no ingore mask needed
        return (im_for_ws > thresh_val)

    #INITIALIZE VARIABLES
    ws = np.zeros_like(im_for_ws)
    vols = {}
    missed_seeds = all_seeds

    # CALCULATE INSTENSITY TRESHOLDS
    intensities = percentile_intensities_sampler(im_for_ws,intensity_thresh_range=[min_int, max_int], percentages=percentage_list) # calculates the list of thresholds
    intensities[100]=max_int # for first iteration

    # STATS SETUP
    stat_table = []

    # ADAPTIVE MASKS - MAIN LOOP
    for pct, current_threshold in tqdm(sorted(intensities.items())[::-1],
        desc="Processing thresholds for watershed:",
        unit="step"):

        if not missed_seeds: # break if previous step found all seeds
            break

        mask = get_mask(current_threshold)
        new_ws = watershed(np.max(im_for_ws) - im_for_ws, seeds_array, mask=mask)

        labels, volumes = np.unique_counts(new_ws) # labels are consistent!
        vols = dict(zip(labels, volumes))

        for s in missed_seeds.intersection(vols.keys()):
            if s != 0 and min_vol < vols[s] < max_vol: # exclude background
                ws[new_ws == s] = s
        
        # UPDATE MISSED SEEDS & GET STATS
        n_found_now = len(missed_seeds) - (len(all_seeds) - len(set(np.unique(ws)))) - len({0}) # previous missed seeds - (missed seeds now) - background
        missed_seeds = set(all_seeds).difference(np.unique(ws)) #now
        n_missed_seeds = len(missed_seeds)
        stat_table.append([f'{pct} %', current_threshold, n_found_now, n_missed_seeds])

    return ws, all_seeds, missed_seeds, stat_table


def add_lost_seeds(ws_in:np.ndarray,
                    seeds_array,
                    missed_seeds,
                    min_vol=300
):
    """Adds a shperical mask around all missing seeds.

    Args:
        ws_in (np.ndarray): _description_
        seeds_array (_type_): _description_
        missed_seeds (_type_): _description_
        min_vol (int, optional): _description_. Defaults to 300.

    Returns:
        _type_: _description_
    """

    rad = (3*min_vol/(4*np.pi))**(1/3) # the sphere will have the minimum allowed volume
    rad = int(np.round(rad).astype(np.uint8))
    ball(rad)
    sph_mask = ball(rad) > 0 # make a boolean mask with shperical shape

    for ms in missed_seeds:
        pos_ms = np.asarray(np.where(seeds_array == ms), dtype=int).ravel() # this is costly
        padd = ws_in < 0 # this is False everywhere
        padd[pos_ms[0]-rad:pos_ms[0] + rad +1, pos_ms[1]-rad:pos_ms[1]+rad+1, pos_ms[2]-rad:pos_ms[2]+rad+1] = sph_mask
        ws_in[padd] = seeds_array[tuple(pos_ms.T)]

    return ws_in

def remove_irregular_labels_3d(
    im,
    ws,
    min_vol=500,
    max_vol=100000,
    max_ecc=0.99,
    min_solidity=0.7,
    min_mean_intensity=50,
):
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
        region_mask = ws == label

        mean_intensity = (
            region.mean_intensity if region.intensity_image is not None else 0
        )

        bigger = region.area > max_vol
        irregular = hasattr(region, "solidity") and region.solidity < min_solidity

        if (bigger or irregular) and mean_intensity < min_mean_intensity:
            cleaned_ws[region_mask] = 0

    return cleaned_ws


def main_function( # to be updated 
    t: int,
    method: str = ["watershed", "sphere", "segmentation"],
    output_path: str = None,
    input_path: str = None,
    lineage_tree_path: str = None,
):
    im = imread(input_path.format(t=t))
    lT = LineageTree.load(lineage_tree_path, file_type="mastodon") #<--- have changed that
    pos_at_t = [lT.pos[c] for c in lT.nodes_at_t(t)]
    ws = ws_adaptive_mask(im, pos_at_t)
    cleaned_ws = remove_irregular_labels_3d(im, ws)

    imwrite(output_path, cleaned_ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hybrid watershed",
        description="Run a watershed algorithm to create a new ground truth image",
    )
    parser.add_argument("-t", "--time", type=int, help="Time to process")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["watershed", "sphere", "segmentation"],
        help="Methods of creating a new ground truth image",
    )
    parser.add_argument(
        "-o", "--output-path", type=str, help="Output path for a new ground truth image"
    )
    parser.add_argument(
        "-i", "--input-path", type=str, help="Input path for pattern with t as time"
    )
    parser.add_argument(
        "-lt", "--lineage-tree-path", type=str, help="Path to lineage tree file"
    )

    args = parser.parse_args()

    main_function(
        t=args.time,
        method=args.method,
        output_path=args.output_path,
        input_path=args.input_path,
        lineage_tree_path=args.lineage_tree_path,
    )