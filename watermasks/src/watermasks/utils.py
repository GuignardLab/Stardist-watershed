import numpy as np
from skimage.io import imread
from pathlib import Path
from ipywidgets import IntSlider, interactive, Layout, Button, ToggleButton, Checkbox, HBox, VBox
import matplotlib.pyplot as plt
from matplotlib import colors
from time import perf_counter
from contextlib import contextmanager
from beautifultable import BeautifulTable

# IMPORTING IMAGE DATA AND ANNOTATIONS
def paths(dataset_name:str):
    """Imports paths from a .txt file in the parent directory of the project.

    Returns:
        list: A list with the directories for all nessesary files of the dataset. One must open the .txt to see what is the sequence of directories. 
    """    
    parent = Path.cwd().parent
    paths_file = parent / f"{dataset_name}_paths.txt"
    return paths_file.read_text().splitlines()


def import_registration_mats_one_view(path_to_xml:str, starting_tp : int, final_tp : int, view : str ):
    """Given an xml file with registartion matrices, it outputs a dictionary with the list of the corresponding transformations (in sequence) as values.
    WARNING: Currently working with one view only, preferably the one used for manual tracking.

    Args:
        path_to_xml (str): the path to the xml file
        starting_tp (int): starting timepoint of interest
        final_tp (int): final timepoint of interest
        view (str): Camera view of interest in the xml.

    Returns:
        {t: [[T_1], [T_2], ...]}, whrere T_i are the transformation matrices for that timepoint
    """
    with open(path_to_xml, 'r') as f:
        Lines = f.readlines()

    switch1 = 0
    switch2 = 0
    dt = -1
    matrices = {}

    for line in Lines:
        # you have passed timepoint of interest
        if f'timepoint="{final_tp + 1}' in line:
            break
        # you have ecountered the first timepoint of interest
        if f'timepoint="{starting_tp + dt + 1}' in line:
            dt += 1
            matrices[starting_tp + dt] = []
            switch1 = 1
        # you are below a line with the wrong setup
        if ('setup="' in line) and (line[line.find('setup="') + 7] != str(view)): 
            switch2 = 0
        # you are below a line with the correct setup
        if ('setup="' in line) and (line[line.find('setup="') + 7] == str(view)): 
            switch2 = 1
        # you are in a line that contains the transformation matrix elements for the given setup
        if switch1 == 1 and switch2 ==1:
            if '<affine>' in line:
                elements = line.strip()[8:-9].split(' ') # elements are in a 
                mat = []
                for i in range(3):
                    mat.append([])
                    for j in range(4):
                        mat[i].append([])
                        mat[i][j] = float(elements[i*4+j])
                matrices[starting_tp + dt].append(mat)

    print('transformations extracted from xml')
    return matrices

def import_registration_mats_multiview(path_to_xml:str, starting_tp:int, final_tp:int) -> dict: 
    """A funtion that extracts the transformation (registration) matrices that are applied in sequence per timepoint per view.

    Args:
        path_to_xml (str): file path of the xml
        starting_tp (int): first timepoint to be included
        final_tp (int): last timepoint to be included

    Returns:
        dict: nested dictionary of the form matrices[timepont:int][setup:str] = [[mat1], [mat2],..., [matN]]
    """
    #READ THE XML
    with open(path_to_xml, 'r') as f:
        Lines = f.readlines()   
    
    #INITIALIZE VARIABLES
    switch1 = 0
    dt = -1
    matrices = {}

    #LINE-BY-LINE MANUPULATION
    for line in Lines:
        if f'timepoint="{final_tp + 1}' in line:
        # you have passed timepoint of interest
            break
        if f'timepoint="{starting_tp + dt + 1}' in line:
        # you have ecountered the first timepoint of interest
            dt += 1
            switch1 = 1
            matrices[starting_tp + dt] = {}
        if ('setup="' in line) and switch1 == 1:
        # you are below a line containing the setup (aka view)
            view = line[line.find('setup="') + 7]
            matrices[starting_tp + dt][view] = []
        if switch1 == 1:
            if '<affine>' in line:
        # you are in a line that contains the transformation matrix elements for the given setup
                elements = line.strip()[8:-9].split(' ') # elements are separated by ' ' in the xml
                mat = []
                for i in range(3):
                    mat.append([])
                    for j in range(4):
                        mat[i].append([])
                        mat[i][j] = float(elements[i*4+j])
                matrices[starting_tp + dt][view].append(mat)

    return matrices

def registered_position_of_id_in_t(mastodon_id:int, lT, t_:int, R_of_t:dict, view:str, scaling_:np.ndarray, Transformations_In_Reverse:bool=True):
    if Transformations_In_Reverse ==True:
        mats_t = R_of_t[t_][view][::-1]
    elif Transformations_In_Reverse == False:
        mats_t = R_of_t[t_][view][:]

    x, y, z = lT.pos[mastodon_id][0:3]

    for mat in mats_t:
        
        mat = np.asarray(mat)
        rotation = mat[:,0:3]
        translation = mat[:, 3]
        x, y, z = np.linalg.inv(rotation) @ ( [x - translation[0], y - translation[1] , z - translation[2]] )
        
    zi = int(z * scaling_[0])
    yi = int(y * scaling_[1])
    xi = int(x * scaling_[2])

    return(xi, yi, zi)


# CROP WATERSHED FUNCTIONS
def crop_around_seed(image_:np.ndarray, seed_label_:int, seeds_pos_:dict[int, list], radius_:int)->np.ndarray:
    """Create a boxed crop of an image, centered around an annotation seed for a 

    Args:
        image_ (np.ndarray): Image to be croped.
        seed_label_ (int): Label of the seed to be in the center of the crop.
        seed_pos_ (dict[in, list]): A dictionary that gives the position of a given seed.
        radius_ (int): 1/2 -1 the size of the crop in each direction.

    Returns:
        np.ndarray: cropped image around the seed.
"""

    sp = [int(seep) for seep in seeds_pos_[seed_label_]]
    # this avoids ocasional bugs returning interger sutraction overflow. Unknown cause.

    z_min = max(sp[0]-radius_, 0)
    y_min = max(sp[1]-radius_, 0)
    x_min = max(sp[2]-radius_, 0)

    z_max = min(sp[0]+radius_+1, image_.shape[0])
    y_max = min(sp[1]+radius_+1, image_.shape[1])
    x_max = min(sp[2]+radius_+1, image_.shape[2])

    return image_[z_min:z_max, y_min:y_max, x_min:x_max]


# VISUALIZATIONS PART
def paint_annotations(
    t_:int,
    lT,
    shape_,
    R_of_t:dict,
    scaling_,
    view_:str='0',
    s_an:int=2,
    Transformations_In_Reverse:bool=True
) -> np.ndarray : 
    """
    A funtion that creates an image representation of your registered point annoations in the frame of reference of the input image. point annotations are represented as cubes.

    Args:
        t_ (_type_): _description_
        lT ( frame of reference): _description_
        shape_ (_type_): _description_
        R_of_t (_type_): _description_
        scaling_ (_type_): _description_
        view_ (str, optional): _description_. Defaults to '0'.
        s_an (int, optional): _description_. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """    
    
    an_space = np.zeros(shape_)
    #make an empty image

    # PAINT ANNOTATIONS
    for mastodon_id_t in lT.time_nodes[t_]:

        xi, yi, zi = registered_position_of_id_in_t(mastodon_id_t, lT, t_, R_of_t, view_, scaling_, Transformations_In_Reverse)

        # MAKE SURE POSITIONS ARE WITHING BOUNDS
        z0 = max(zi - s_an, 0)
        z1 = min(zi + s_an, an_space.shape[0])

        y0 = max(yi - s_an, 0)
        y1 = min(yi + s_an + 1, an_space.shape[1])

        x0 = max(xi - s_an, 0)
        x1 = min(xi + s_an + 1, an_space.shape[2])

        an_space[z0:z1, y0:y1, x0:x1] = 1

    print('done painting')
    return an_space

def view_t_f_overlays_3d(volumes, cmaps, alphas=None, custom_mask=colors.ListedColormap([(0,0,0,0), 'lightgreen'])):
    """
    Overlay 3d viewer for a pair of photos

    Parameters
    ----------
    volumes : list of np.ndarray
        List of 3D arrays of identical shape: (X, Y, Z)
    alphas : list of float, optional
        Transparency values for each volume. Default: all 0.5
    cmap : str
        Matplotlib colormap for imshow
    """
    mycmap2 = custom_mask
    
    # --- Safety checks ---
    if not isinstance(volumes, list) or len(volumes) == 0:
        raise ValueError("volumes must be a non-empty list of 3D numpy arrays.")
    shape0 = volumes[0].shape
    if any(np.asarray(vol).shape != shape0 for vol in volumes):
        raise ValueError("All volumes must have the same shape.")

    if alphas is None:
        alphas = [0.5] * len(volumes)     # default transparency
    if len(alphas) != len(volumes):
        raise ValueError("Length of alphas must match length of volumes.")

    nz = shape0[2]
    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(15,17))
    fig.set_figwidth(10)
    fig.set_figheight(10)
    ax.set_title("Overlayed volumes")

    # First image handle
    artists = []
    for vol, cm, alpha in zip(volumes, cmaps, alphas):
        if cm=='bincmap': 
            img = ax.imshow(vol[:, :, 115],
                cmap=mycmap2,
                alpha=alpha)
        else:
            img = ax.imshow(vol[:, :, 115],
                cmap=cm,
                alpha=alpha)
        artists.append(img)

    # --- Update function ---
    def update(z, show):
        for img, vol in zip(artists, volumes):
            slice_ = vol[:, :, z]
            img.set_data(slice_)
            img.set_clim(slice_.min(), slice_.max())
            if show is True and img is not artists[0]:
                img.set_visible(False)
            else:
                img.set_visible(True)

        ax.set_title(f"Overlayed volumes — slice {z}")
        fig.canvas.draw_idle()

    # --- Slider ---
    slider = IntSlider(
        value=115,
        min=0,
        max=nz - 1,
        step=1,
        description="Slice",
        continuous_update=True,
        layout=Layout(width='1000px')
    )

    #--- Show Button ---
    show_button = ToggleButton(
        value=True,
        description='Show/Hide',
        disabled=False,
        continuous_update=False
    )
    
    return interactive(update, z=slider, show=show_button)


def view_overlays_3d_V2(volumes, im_names, cmaps , alphas:None=None, same_contrast:bool=True):
    """Higher perfrmance overlay viewer for MULTIPLE large 3D stacks. Uses checkboxes to toggle visibility of each overlay.

    Args:
        volumes (list): list of volumetric images to be visualised
        im_names (list): list of desctiptions to be displayed 
        cmaps (list): list of prefered cmap names + a custom 'binmap' for bin or boolean arrays
        alphas (None, optional): _description_. Defaults to None.
        same_contrast (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    binmap = colors.ListedColormap([(0,0,0,0), 'lightgreen'])
    # ----------------------------
    # SAFETY CHECKS
    # ----------------------------
    shape0 = volumes[0].shape
    nz = shape0[2]
    n_vols = len(volumes)

    if alphas is None:
        alphas = [0.5] * n_vols

    # ----------------------------
    # PRECOMPUTE GLOBAL CLIMS
    # ----------------------------
    clims = []
    for vol in volumes:
        vmin = np.min(vol)
        vmax = np.max(vol)
        clims.append([vmin, vmax])
    if same_contrast==True:
        for i in range(len(clims)):
            clims[i] = (clims[0][0], clims[0][1])
    # ----------------------------
    # FIGURE SETUP
    # ----------------------------
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("Slice 0")

    # ----------------------------
    # INITIAL IMAGES (ARTISTS)
    # ----------------------------
    artists = []
    for vol, cm, alpha, (vmin, vmax) in zip(volumes, cmaps, alphas, clims):
        if cm!='binmap':
            img = ax.imshow(
                vol[:, :, 206],
                cmap=cm,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            img = ax.imshow(
                vol[:, :, 206],
                cmap=binmap,
                alpha=alpha,
                vmin=0,
                vmax=1
            )
        artists.append(img)

    # ----------------------------
    # CHECKBOXES for visibility
    # ----------------------------
    checkboxes = []
    for i in range(n_vols):
        cb = Checkbox(
            value=True,
            description=f"{im_names[i]}",
            indent=False
        )
        checkboxes.append(cb)

    # Map checkboxes → artists
    checkbox_map = dict(zip(checkboxes, artists))

    # ----------------------------
    # CHECKBOX EVENT HANDLERS
    # ----------------------------
    def on_checkbox_change(change):
        cb = change["owner"]
        img = checkbox_map[cb]
        img.set_visible(cb.value)
        fig.canvas.draw_idle()

    for cb in checkboxes:
        cb.observe(on_checkbox_change, names="value")

    # ----------------------------
    # SLICE UPDATE FUNCTION
    # ----------------------------
    def update(z):
        # Update only the slice data (FAST)
        for img, vol in zip(artists, volumes):
            img.set_data(vol[:, :, z])

        ax.set_title(f"Slice {z}")
        fig.canvas.draw_idle()

    # ----------------------------
    # SLIDER
    # ----------------------------
    slider = IntSlider(
        value=206,
        min=0,
        max=nz - 1,
        step=1,
        description="Slice",
        continuous_update=True,
        layout={"width": "1200px"}
    )

    # ----------------------------
    # RETURN COMPOSITE WIDGET
    # ----------------------------
    return VBox([HBox(checkboxes), interactive(update, z=slider)])

def display_ws_thresholding_results(stat_table, all_seeds):

    table = BeautifulTable()
    table.columns.header = ['Percentage', 'Percentile Value', 'Found Seeds', 'Missed Seeds']
    for i in range(len(stat_table)):
        table.rows.insert(i, [f'{stat_table[i][0]} %', *[stat_table[i][j] for j in range(1, len(stat_table[i]))]])
    print(table)
    print(f'Out of {len(all_seeds)} anotations in total, missed seeds: {stat_table[-1][-1]}({stat_table[-1][-1]/len(all_seeds)*100:.3g}%)')

@contextmanager
def block_timer():
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"-(Time elapsed: {end - start:.6f} s)")