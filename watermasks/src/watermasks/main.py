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
from scipy.ndimage import distance_transform_edt, distance_transform_cdt
from typing import Sequence

# ~~~~ IMAGE PROCESSING AND DATA MANIPULATION  ~~~~
def preprocess_im(
	im,
	gauss_sigma=1,
	median_size=1,
	erosion_radius=7
):
	"""The sequence of image processing steps that have been giving the best results for the 2nd compression hdf5 layer.

	Args:
		im (_type_): _description_
		gauss_sigma (int, optional): _description_. Defaults to 1.
		median_size (int, optional): _description_. Defaults to 1.
		erosion_radius (int, optional): _description_. Defaults to 7.

	Returns:
		_type_: _description_
	"""    
	im_gauss = gaussian_filter(im, sigma=gauss_sigma)
	im_gauss_median = median_filter(im_gauss, size=median_size) # this seems that is not doing much! (inspection by eye)
	erosion = grey_erosion(im_gauss_median, size=erosion_radius)
	im_for_ws = im_gauss_median - erosion
	im_for_ws = gaussian_filter(im_for_ws, sigma = 1)

	return im_for_ws


def get_seeds(lT, tp:int, view:str, R_of_t:np.ndarray, scaling:np.ndarray, raw_image:np.ndarray, trans_in_rev:bool=False)-> list[np.ndarray, np.ndarray]: 
	"""
	Using the lineage tree, this function creates an image containing a different label at each annotation position.
	Positions share the same coordinate system as the raw image (to be used for watershed).
	Args:
		lT (_type_): Lineage tree input to get annotation positions
		tp (int): Timepoint
		view (str): View in the hdf5
		R_of_t (np.ndarray): Registration matrices, as loaded form the .xml file.
		scaling (np.ndarray): the ratio of the shape of the current image over the shape of 0-th level of hdf5 = 1/downsampling_list. 
								It should be of the form [a, b, c], whith a, b, c <= 1 .
		raw_image (np.ndarray): The loaded image.
		trans_in_rev (bool, optional): Dictates the sequence of application of the registration transofmation matrices, as loaded from the .xml 
										. Defaults to False.

	Raises:
		IndexError: Reminds to the user to change the trans_in_rev variable in case of index error raise.

	Returns:
		list: [all seeds postition (np.ndarray). A seeded image with a unique label at each annotation position (np.ndarray)]
	"""

	reg_pos_at_t = []
	for mastodon_id_t in lT.time_nodes[tp]:
		x, y, z = utils.registered_position_of_id_in_t(mastodon_id_t, lT, tp, R_of_t, view, scaling, Transformations_In_Reverse=trans_in_rev)
		reg_pos_at_t.append([x, y, z]) #floating numbers
	reg_pos_at_t = np.asarray(reg_pos_at_t)

	seeds_pos = np.array([p[::-1] for p in reg_pos_at_t]).round().astype(np.uint16) # pixel positions, given that the image coords are (z, y, x)
	seeds_array = np.zeros_like(raw_image)
	
	try:
		seeds_array[tuple(seeds_pos.T)] = np.arange(1, len(reg_pos_at_t) + 1) # an with a unique label at each annotation pixel position  # an with a unique label at each annotation pixel position
		seeds_pos = dict(zip(np.arange(1, len(reg_pos_at_t) + 1), tuple(seeds_pos)))
	except IndexError:
		raise IndexError(f'Attempted position in array is out of bounds. Try negating "trans_in_rev" variable from False to True or vice versa.')

	return seeds_pos, seeds_array


# ~~~~ WHOLE-IMAGE WATERSHED FUNCTIONS -> Watershed is applied to the whole image per thershold. ~~~~
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


def ws_adaptive_mask(
	im_for_ws:np.ndarray,
	seeds_array:np.ndarray,
	min_int:int=10,
	max_int:int=500,
	min_vol:int=300,
	max_vol:int=1500,
	percentage_list:list = [99.99, 99.9, 99, 97, 95, 92, 90, 80, 50, 30, 10]

)->list[np.ndarray, list] :
	"""A function that segments an image with the watershed function, using annotations as seeds and utilizing an adaptive mask.
	All preset values are tested against single views, 2nd hdf5 layer.

	Args:
		im_for_ws (np.ndarray): pre-processed image
		pos_at_t (np.ndarray): list of (x,y,z) annotation positions
		max_int (int, optional): minmum intenstiy value for thresholds @ adaptive mask. Defaults to 500.
		min_int (int, optional): maximum intenstiy value for thresholds @ adaptive mask. Defaults to 10.
		min_vol (int, optional): minimum allowed volume of a image segment. Defaults to 300.
		max_vol (int, optional):  maximum allowed volume of a image segment. Defaults to 1500.
		percentage_list (list): list of intensity percentiles to be used as thresholds. 

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
			if min_vol < vols[s] < max_vol: # background is already excluded
				ws[new_ws == s] = s
		
		# UPDATE MISSED SEEDS & GET STATS
		n_found_now = len(missed_seeds) - (len(all_seeds) - len(set(np.unique(ws)))) - len({0}) # previous missed seeds - (missed seeds now) - background
		missed_seeds = set(all_seeds).difference(np.unique(ws)) #now
		n_missed_seeds = len(missed_seeds)
		stat_table.append([pct, current_threshold, n_found_now, n_missed_seeds])

	return ws, all_seeds, missed_seeds, stat_table


# ~~~~ CROPPED IMAGE WATERSHED FUNCTIONS -> Watershed is applied to image crops first and then the whole-image segmentation is updated. ~~~~
def crop_around_seed(image_:np.ndarray, seeds_pos:dict[int, np.ndarray], seed_label_:int, radius_:int)->np.ndarray:
	"""Create a boxed crop of an image, centered around an seed in the seed_array, with a width of 2*radius + 1.

	Args:
		image_ (np.ndarray): Large input image to be croped
		seed_label_ (int): Seed label. The given seed will be in the center of the cropped image
		radius_ (int): 1/2 -1 the size of the crop in each direction

	Returns:
		np.ndarray: cropped image around the seed. 
	"""    

	sp = seeds_pos[seed_label_]

	z_min = max(sp[0]-radius_, 0)
	y_min = max(sp[1]-radius_, 0)
	x_min = max(sp[2]-radius_, 0)

	z_max = min(sp[0]+radius_+1, image_.shape[0])
	y_max = min(sp[1]+radius_+1, image_.shape[1])
	x_max = min(sp[2]+radius_+1, image_.shape[2])

	return image_[z_min:z_max, y_min:y_max, x_min:x_max]


def get_n_th_closest_labels(dist_arr:np.ndarray, query_label:int, n:int=4)->list:
	## You could also use the LineageTree function as well, but needs more work.
	"""Find the labels of the n closest seeds to the query label

	Args:
		dist_arr (np.ndarray): the self-distance matrix of a poitn cloud. C_{i, j} = dist(r_i, r_j)
			where r_k is the k-th seed position in the seed array.
		query_label (int): the central label under consideration
		n (int, optional): The number of neighbours you are interested in finding. Defaults to 4.

	Returns:
		list: a list containing all the neighbours labels
	"""	
	dummy_dist = list(dist_arr[query_label-1]) # 0th element refers to label = 1, the 1st to label =2 etc
	full_dist = list(dist_arr[query_label-1])
	label_list = []
	dist_list = []
	
	for i in range(n+1):
		min_val = min(dummy_dist)
		idx_dummy = dummy_dist.index(min_val)
		idx_full = full_dist.index(min_val)
		label_list.append(idx_full + 1)
		dist_list.append(min_val)
		dummy_dist.pop(idx_dummy) # in first interation, it deletes self-distance

	return label_list, dist_list


def crop_to_include_nei(image:np.array, neis_pos:list[list], min_d:int=7)->np.ndarray:
	"""An anisotropic crop designed to include nearby seeds (n-fisrt neighbours)

	Args:
		image (np.array): the image to be cropped
		neis_pos (list[list]): the list of neighbourig seeds postions 
			(the first element on the list, is the position of the 0th neighbour,
			aka label of interest, aka query_point).
		min_d = the crop image will have as the smallest width 2*min_d + 1 in each of the z, y, x directions.

	Returns:
		np.ndarray: A minmally cropped image around the first position that includes the neirby seeds.
	"""
	c_pos = [int(pp) for pp in neis_pos[0]]
	Dz = []
	Dy = []
	Dx = []
	
	for i in range(1, len(neis_pos)):
		n_pos = [int(pp) for pp in neis_pos[i]]
		Dz.append(n_pos[0]-c_pos[0])
		Dy.append(n_pos[1]-c_pos[1])
		Dx.append(n_pos[2]-c_pos[2])
	
	Dz.append(-min_d)
	Dy.append(-min_d)
	Dx.append(-min_d)
	Dz.append(min_d)
	Dy.append(min_d)
	Dx.append(min_d)
	# these are needed to ensure that the seed nucleus is always in the image.
	# print(f'Dz , Dy, Dx = {Dz, Dy, Dx}')
	
	return image[
			max(c_pos[0] + min(Dz), 0):min(c_pos[0] + max(Dz)+1, image.shape[0]),
			max(c_pos[1] + min(Dy), 0):min(c_pos[1] + max(Dy)+1, image.shape[1]), 
			max(c_pos[2] + min(Dx), 0):min(c_pos[2] + max(Dx)+1, image.shape[2])
			]


def update_image(big_image, small_image, spibi, spisi, label_):
	"""
	A funtion that updates a larger segmentation image from a smaller (cropped) segmentation image.
	Only the label = label_ is updated in the larger image. It is assumed that the large image does not contain
	the label in question.

	Args:
		empty_watershed (np.ndarray]): The large input segmetnation mask
		small_image (np.ndarray]): A segmetnation mask of a smaller cropped image.
		spibi (np.array or list): Stands for Seed Position in Big Image := SPIBI.
		spisi (np.array or list): Stands for Seed Position in Small Image := SPISI
		label_ (int): The label of the seed.

	Returns:
		np.ndarray: The updated large (full-sized) segmentation mask.
	"""
	big_mask = big_image < -1 # This is False everywhere
	sp = [int(seep) for seep in spibi]
	form_bell_z = spisi[0]
	form_down_y = spisi[1]
	form_left_x = spisi[2]
	# These may be different (if say, the annotation is near the bounds of the image)
	# this creates anisotropic cropping

	z_min = max(sp[0]-form_bell_z, 0)
	y_min = max(sp[1]-form_down_y, 0)
	x_min = max(sp[2]-form_left_x, 0)

	z_max = min(sp[0]-form_bell_z+small_image.shape[0], big_image.shape[0])
	y_max = min(sp[1]-form_down_y+small_image.shape[1], big_image.shape[1])
	x_max = min(sp[2]-form_left_x+small_image.shape[2], big_image.shape[2])

	small_mask = small_image == label_
	big_mask[z_min: z_max, y_min:y_max, x_min:x_max]=small_mask
	big_image[big_mask] = label_

	return big_image


# ~~~~ GEOMETRY MASKS ~~~~
def find_locus_closest_to_point(query_point_pos, crop_array):
	"""Finds array positions that are the closest to the query point (with respect to the other non-background points) 

	Args:
		query_point_pos (np.ndarray or list): The array position of the query point
		crop_array (np.ndarrat): the seed array, containing non-zero values at the annotated positions.

	Returns:
		np.ndarray[bool]: an array with True values @ positions that are closest to the query point.
	"""    
	locus_mask = crop_array > 0
	indices = distance_transform_cdt(~locus_mask, return_distances=False, return_indices=True)
	# print(f'indices shape = {indices.shape}')
	allz = indices[0,:,:,:] == query_point_pos[0]
	ally = indices[1,:,:,:] == query_point_pos[1]
	allx = indices[2,:,:,:] == query_point_pos[2]
	locus = allz & ally & allx
	
	return locus

def find_median_plane(point_1_position:np.ndarray, point_2_position:np.ndarray, image_shape: Sequence[int])->np.ndarray[bool]:
	"""
	A function that creates a boolean mask containing the locus of all array positions that are eqidistant to a
	pair of array points.

	Args:
		point_1_position (np.ndarray): The aray position of the first point 
		point_2_position (np.ndarray): The aray position of the second point 
		image_shape (Sequence[int]): The desired output shape of the mask

	Returns:
		np.ndarray (bool): A boolean mask with True values for all array positions that satisfy d(point_1) = d(point_2)
			should be planes in 3d space. The plain the median-perpendicular plane to the linearsigment jointing the points.
	"""

	dist_arr_1 = np.zeros(image_shape)
	pad = dist_arr_1 == 0
	dist_arr_1[pad] = 1
	dist_arr_1[tuple(point_1_position.T)] = 0
	dist_arr_1 = distance_transform_edt(dist_arr_1)

	dist_arr_2 = np.zeros(image_shape)
	pad = dist_arr_2 == 0
	dist_arr_2[pad] = 1
	dist_arr_2[tuple(point_2_position.T)] = 0
	dist_arr_2 = distance_transform_edt(dist_arr_2)

	median_plane_mask = np.abs(dist_arr_2 - dist_arr_1) <= 1

	return median_plane_mask


def sphere_mask(array_shape: tuple, center: tuple, radius: float) -> np.ndarray:
	"""
	Creates a boolean mask where True indicates a point is inside a sphere.
	
	Parameters:
		array_shape : Shape of the output mask (e.g., (D, H, W) for 3D)
		center      : Center of the sphere (e.g., (z0, y0, x0))
		radius      : Radius of the sphere
	
	Returns:
		Boolean NumPy array of the given shape.
	"""
	# Build a grid of indices for each dimension
	grids = np.ogrid[tuple(slice(0, s) for s in array_shape)]
	
	# Sum squared distances along each dimension
	dist_sq = sum(
		(grid - c) ** 2
		for grid, c in zip(grids, center)
	)
	
	return dist_sq <= radius ** 2


def bresenham_3d(start, end):
    """
    Returns all grid cells along the line segment connecting two 3D points.
    Uses the dominant axis to drive stepping.
    """
    x1, y1, z1 = start
    x2, y2, z2 = end

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1

    positions = [[x1, y1, z1]]

    # Dominant axis: X
    if dx >= dy and dx >= dz:
        err_y = 2 * dy - dx
        err_z = 2 * dz - dx
        for _ in range(dx):
            if err_y >= 0:
                y1 += sy
                err_y -= 2 * dx
            if err_z >= 0:
                z1 += sz
                err_z -= 2 * dx
            x1 += sx
            err_y += 2 * dy
            err_z += 2 * dz
            positions.append([x1, y1, z1])

    # Dominant axis: Y
    elif dy >= dx and dy >= dz:
        err_x = 2 * dx - dy
        err_z = 2 * dz - dy
        for _ in range(dy):
            if err_x >= 0:
                x1 += sx
                err_x -= 2 * dy
            if err_z >= 0:
                z1 += sz
                err_z -= 2 * dy
            y1 += sy
            err_x += 2 * dx
            err_z += 2 * dz
            positions.append([x1, y1, z1])

    # Dominant axis: Z
    else:
        err_x = 2 * dx - dz
        err_y = 2 * dy - dz
        for _ in range(dz):
            if err_x >= 0:
                x1 += sx
                err_x -= 2 * dz
            if err_y >= 0:
                y1 += sy
                err_y -= 2 * dz
            z1 += sz
            err_x += 2 * dx
            err_y += 2 * dy
            positions.append([x1, y1, z1])

    return positions

def int_inf_rad(image:np.ndarray, label_:int, query_pos:list, pos_list:list[list], seeds_pos:dict)->int:
	"""A function that caclulates the radius of shpere, nessesary to enclose the first cell nucelus,
		using intensity information along the connecting line between two points.
	Args:
		image (np.ndarray): the image containing the intensity data. (better to use the raw data instead of the proccessed data)
		label_ (int): the label of the seed in question
		query_pos (list): the position of the see in question
		pos_list (list[list]): a list of first neghbours, surounding the query point.

	Returns:
		int: the wanted radius, rounded to floor, avoiding sphere overlay.
	"""	
	qpp = [int(query_pos[i]) for i in [0,1,2]]
	sph_rad = []
    
	for p2 in pos_list:
		pp2 = [int(p2[i]) for i in [0,1,2]] # the z,y,x positions of surrounding seeds
		point_int = {} # dictionary with pixel intensity : corresponding pos 
		pnts = [np.asarray(element) for element in bresenham_3d(qpp, pp2)]
		# list of teh pixel positions in the connecting line segement between query point and the rest of the positions

		for i in range(len(pnts)):
			point_int[image[tuple(pnts[i].T)]] = pnts[i]

		p_minint = point_int[min(point_int.keys())]
		sph_rad.append(np.floor(np.sqrt(np.sum([(p_minint[i] - seeds_pos[label_][i])**2 for i in [0,1,2]]))))

	return np.floor(np.min(sph_rad))


# ~~~~ OLDER FUNCTIONS TO ASSESS SEGMENTATION QUALITY AND UPDATE THE SEGMENTATION. To be updated. ~~~~
def add_lost_seeds(ws_in:np.ndarray,
					seeds_pos,
					missed_seeds,
					min_vol=300
):
	"""
	Adds a shperical segmentation mask around all missing seeds with the given labels.

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
	sph_mask = ball(rad) > 0 # make a boolean mask with shperical shape

	for label in missed_seeds:
		pos_ms = seeds_pos[label]
		padd = ws_in < 0 # this is False everywhere
		padd[pos_ms[0]-rad:pos_ms[0] + rad +1, pos_ms[1]-rad:pos_ms[1]+rad+1, pos_ms[2]-rad:pos_ms[2]+rad+1] = sph_mask
		ws_in[padd] = label

	return ws_in


def remove_irregular_labels_3d(
	im:np.ndarray,
	ws:np.ndarray,
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

