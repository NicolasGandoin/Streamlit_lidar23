"""
lidar-dec-23 project
Lidar las Utility fonctions
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import laspy
import re
import os
import cv2
# import open3d as o3d
import src.IGN_API_utils as api
from src.topo_utils import bdtopo_to_image_slice

from rtree import index
from shapely.geometry import Point
import geopandas as gpd

from scipy.spatial import cKDTree

"""--------------------------------"""
""" Miscelaneous Utility functions """
"""--------------------------------"""

def cut_ranges(ranges_x, ranges_y, n):
    """
        Cut ranges in n equivalent slices 1000mx1000m gives 25 ranges of 200mx200m
     """
    minX, maxX = ranges_x
    minY, maxY = ranges_y
    # print(type(minX), type(minY), type(maxX), type(maxY))
    # Calculer les incréments pour x et y
    increment_x = (maxX - minX) // n
    increment_y = (maxY - minY) // n
    ranges_x = [(minX + i * increment_x, minX + (i + 1) * increment_x) for i in range(n)]
    ranges_y = [(minY + i * increment_y, minY + (i + 1) * increment_y) for i in range(n)]
    ranges = [(range_x, range_y) for range_x in ranges_x for range_y in ranges_y]

    return ranges


def get_bounds_from_range(rangexy):
    """ Get bounds from range xy
    rangexy is tuple of ranges - first is x and last is y
    bounds is minx, miny, maxx, maxy"""
    # print(rangexy)
    return rangexy[0][0], rangexy[1][0], rangexy[0][1], rangexy[1][1]


def get_ranges_from_bounds(bounds):
    """
        Get ranges from bounds
        returns tuple of ranges - first is x and last is y
    """
    return (bounds[0], bounds[2]), (bounds[1], bounds[3])


def get_file_range(filename):
    """
    returns (xmin, xmax, ymin, ymax) from filename assuming tile filesize is 1000mx1000m
    ATTENTION Cette fonction ne concerne que les fichiers LIDAR IGN source
    """
    root_name = filename.split('.')[0]
    xmin = int(root_name.split('_')[2]) * 1000
    ymax = int(root_name.split('_')[3]) * 1000  # le nom du fichier contient ymax ps y min
    return (xmin, xmin + 1000), (ymax - 1000, ymax)


def get_range_from_file(filename):
    """
    returns (xmin, xmax, ymin, ymax) from filename
    buid like 'ORTHOHR-520000-6700000-LA93-0M20-1000x1000.csv'
    Assumed size is 200mx200m
    # TODO manage any size ?
    """
    root_name = filename.split('.')[0]
    parts = root_name.split('-')
    size = 200
    xmin = int(parts[1])
    ymax = int(parts[2])
    return (xmin, xmin + size), (ymax , ymax + size)


def get_las_file_from_point(lidar_dir, point):
    """
        Return las filanme of that contains given point parameter
        Based on fact that filename is build like this:
        LHD_FXX_0519_6702_PTS_C_LAMB93_IGN69.copc.laz
        0519 is the lower X an 6702 is the upper Y size is 1000mx1000m
    """
    import re
    r = re.compile('([0-9]{4})_([0-9]{4})')
    paths = Path().rglob(lidar_dir + '/LAZ/*.laz')
    for path in paths:
        match = [1000 * int(m) for m in r.findall(path.name)[0]]
        if point[0] >= match[0] and point[0] < match[0] + 1000 and point[1] < match[1] and point[1] >= match[1] - 1000:
            return (lidar_dir + 'LAZ/' + path.name)


def to_local(x, _range, size):
    """
        converts x & y to local pixel coordinates - rounds non int pixel coordinates
        May be we xill need better interpolation ?
        _range is a tuple (min, max) on an axis in term of LA93 coords
    """
    _x = int(size * (x - _range[0]) / (_range[1] - _range[0]))

    return int(_x - 1) if _x > 0 else 0


def get_rootfilename(xmin, ymin, size):
    """
        returns normalized filename for tile file
    """
    return f'LHD-{xmin}-{ymin}-LA93-0M20-{size}x{size}'


def get_filename(filename, filetype, extension):
    """ take a file name and return the filename that correspond to same area
        but  with new type  and ext
        ex LHD-520000-6701200-LA93-0M20-1000x1000.png, 'ORTHOHR', 'csv' returns 'ORTHOHR-520000-6701200-LA93-0M20-1000x1000.csv'
        """

    return filetype + '-' + '-'.join((filename.split('.')[0] + extension).split('-')[1:])


def interpolate_zero_values(image):
    """
    Interpolation des points vides avec les valeurs voisines
    """
    from scipy import interpolate
    # on extrait d'abord les point et les valeurs de points
    points = np.argwhere(image != 0)
    values = image[points[:, 0], points[:, 1]]
    # on crée un meshgrid de 1000 par 1000
    grid_x, grid_y = np.meshgrid(np.arange(1000), np.arange(1000))
    # l'image deviendra cette grille
    img = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')

    return img


def interpolate_neg_values(image):
    """
    Interpolation des points vides avec les valeurs voisines
    """
    from scipy import interpolate
    # on extrait d'abord les point et les valeurs de points
    points = np.argwhere(image >= 0)
    values = image[points[:, 0], points[:, 1]]
    # on crée un meshgrid de 1000 par 1000
    grid_x, grid_y = np.meshgrid(np.arange(1000), np.arange(1000))
    # l'image deviendra cette grille
    img = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')

    return img


def convolve_values(image):
    """
    Convolve pixels with 5,5 kernel
    """
    # image = cv2.inpaint(image.astype('uint8'), mask, 3, cv2.INPAINT_TELEA)
    # Pas résussi à interpoler correctement avec cv2
    # Bidouille pour corriger 2 pixels voisins TODO interpolation bilinéaire scikit image ?
    kernel = np.ones((5, 5))
    kernel = kernel / 25
    kernel[2, 2] = 0

    print(kernel)
    img = np.convolve(image.flatten(), kernel.flatten(), 'same').reshape(image.shape)
    img = np.where(image == 0, img, image)
    return img



""" --------- """
""" RGB stuff """
""" --------- """


def add_rgb_column(df, bounds):
    """
    Add RGB column to las dataframe
    :param df: las dataframe
    :param bounds: bounds tuple in meters LA93 !!!
    :return:
    """
    if df.shape[0] == 0:
        df[['R', 'G', 'B']] = [np.nan, np.nan, np.nan]
        return df
    # On va chercher l'image raster via l'api (WMS)
    # la size en pixels de l'image c'est la range * 5 (20cm/pixel)
    img = api.load_bdortho(bounds, 5 * (bounds[2] - bounds[0]),
                           5 * (bounds[3] - bounds[1]))  # definition = 5x bounds # environ 20 sec
    # pour les besoins du mapping image on va alimenter les coordonnées x,y locales
    npixelsx = 5 * (bounds[2] - bounds[0])
    npixelsy = 5 * (bounds[3] - bounds[1])
    rangex, rangey = get_ranges_from_bounds(bounds)

    # Appliquer les transformations en utilisant NumPy
    X_values = df['X'].to_numpy()
    Y_values = df['Y'].to_numpy()
    df['locy'] = np.array([to_local(x / 100, rangex, npixelsx) for x in X_values])
    df['locx'] = np.array([to_local(y / 100, rangey, npixelsy) for y in Y_values])

    # on alimente chaque entrée avec RGB

    # apply est  trop lent on passe par numpy
    x_coords = df['locx'].to_numpy()
    y_coords = df['locy'].to_numpy()
    # Assigner les valeurs RGB aux colonnes du DataFrame
    df[['R', 'G', 'B']] = img[x_coords, y_coords]
    df = df.loc[df.X>0] # remove any 0
    return df.drop(['locy', 'locx'], axis=1)  # pas besoin de garder les coordonnées locales


def add_height_column(df, bounds):
    """
    Add height column to las dataframe based on kdTree lidar of type 'sol'
    :param df: las dataframe
    :param bounds: bounds tuple in meters LA93 !!!
    :return: the df with height calculated with Z - groubd_height

    #  l'appel à RGE ALTI n'est pas concluant - sur certaines zones il y a des distortions importantes de zéro
    #  Envisager un système de recherche du point sol le plus proche - voir add_lidar_height_column pour remplacement
    #  Peux fonctionner comme suit. => générer une carte d'altitudes des points du sol avec interpolation sur les points manquants .

    """
    if df.shape[0] == 0:
        df[['H']] = [np.nan]
        return df

    # Initialisation de H avec 0 pour tous les points
    df['H'] = 0
    try:
        # Extraction des points de type 'sol'
        sol_points = df[df['classification'] == 2]
        sol_coords = sol_points[['X', 'Y', 'Z']].values

        # Création de l'arbre k-d pour les points de type 'sol'
        tree = cKDTree(sol_coords[:, :2])  # Utilisation des coordonnées X et Y pour le k-d tree

        # Extraction des points qui ne sont pas de type 'sol'
        non_sol_mask = df['classification'] != 2
        non_sol_points = df[non_sol_mask]
        non_sol_coords = non_sol_points[['X', 'Y']].values

        # Recherche des points de type 'sol' les plus proches pour chaque point non 'sol'
        distances, indices = tree.query(non_sol_coords)

        # Extraction des hauteurs Z des points de type 'sol' les plus proches
        nearest_sol_heights = sol_coords[indices, 2]

        # Calcul de la hauteur au sol H pour les points non 'sol'
        H_non_sol = non_sol_points['Z'].values - nearest_sol_heights

        # Mise à jour de H dans le DataFrame original uniquement pour les points non 'sol'
        df.loc[non_sol_mask, 'H'] = H_non_sol
    except:
        pass # ignore errors

    return df  # pas besoin de garder les coordonnées locales


def add_lidar_height_column(df, bounds):
    """
    Add height column to las dataframe - DEPRECATED see note
    :param df: las dataframe
    :param bounds: bounds tuple in meters LA93 !!!
    :return: the df with height calculated with Z - groubd_height
    """
    if df.shape[0] == 0:
        df[['H']] = [np.nan]
        return df
    # On va chercher l'image raster MNT via l'api (WMS)
    # la size en pixels de l'image c'est la range * 5 (20cm/pixel)
    # TODO l'appel à RGE ALTI n'est pas concluant - sur certaines zones il y a des distortions importantes de zéro
    #  Envisager un système de recherche du point sol le plus proche
    #  Peux fonctionner comme suit. => générer une carte d'altitudes des points du sol avec interpolation sur les points manquants .
    img = api.load_rgealti(bounds, 5 * (bounds[2] - bounds[0]),
                           5 * (bounds[3] - bounds[1]))  # definition = 5x bounds # environ 20 sec
    # pour les besoins du mapping image on va alimenter les coordonnées x,y locales
    npixelsx = 5 * (bounds[2] - bounds[0])
    npixelsy = 5 * (bounds[3] - bounds[1])
    rangex, rangey = get_ranges_from_bounds(bounds)

    # Appliquer les transformations en utilisant NumPy
    X_values = df['X'].to_numpy()
    Y_values = df['Y'].to_numpy()
    df['locy'] = np.array([to_local(x / 100, rangex, npixelsx) for x in X_values])
    df['locx'] = np.array([to_local(y / 100, rangey, npixelsy) for y in Y_values])

    # applyest  trop lent on passe par numpy
    x_coords = df['locx'].to_numpy()
    y_coords = df['locy'].to_numpy()
    # Assigner les valeurs RGB aux colonnes du DataFrame
    df['H'] = (df['Z'] - img[x_coords, y_coords]*100).astype('int') # H pour Height (attention Z en cm => H en cm)
    df = df.loc[df.X>0] # remove any 0
    return df.drop(['locy', 'locx'], axis=1)  # pas besoin de garder les coordonnées locales


""" --------------------------- """
""" Linking LAS and bdtopo bat  """
""" --------------------------- """

def add_bdtopo_cleabs_column(df, bounds):
    """
        Add bdtopo batid column to las dataframe
        bounds permet de rechercher les batiments par l'api
        le champ identifiant du batiment est cleabs dans BDTOPO
        La recherche s'appuie sur un index spatial construit à partir des polygones
        :param  df est un dataframe lidar X,Y, Z
        :param bounds (x,y,xmax,ymax) (Lambert 93)
    """
    df = df.copy()

    batiments = api.load_batiments(bounds)  # retourne la bdtopo sur les bounds

    if batiments.shape[0] > 0:
        # Créer un index spatial pour les géométries des bâtiments
        spatial_index = index.Index()
        for idx, geom in enumerate(batiments.geometry):
            spatial_index.insert(idx, geom.bounds)

        # Fonction pour obtenir l'ID du bâtiment contenant le point géométrique
        def get_building_id(point, batiments, spatial_index):
            idx_candidates = list(spatial_index.intersection(point.bounds))
            for idx in idx_candidates:
                if batiments.geometry.iloc[idx].contains(point):
                    return batiments.cleabs.iloc[idx]
            return None

        points = np.array([Point(x, y) for x, y in zip(df['X'] / 100, df['Y'] / 100)])
        gdf = gpd.GeoDataFrame(df, geometry=points)  # df avec les Points shapely

        #34 secondes
        df['cleabs'] = np.vectorize(get_building_id, excluded=['batiments', 'spatial_index'])(points, batiments=batiments, spatial_index=spatial_index)
    else:
        df['cleabs'] = np.NaN
    return df


""" ---------------- """
""" Dataframing las"""
""" ---------------- """
def get_lidar_df_from_file(lasfile):
    """  lasfile name (full path) => return dataframe
    """
    las = laspy.read(lasfile)
    return las_to_df(las, del_features=False) # on a besoin de x,y,z


def las_to_df(las, del_features=True):
    """
    Old function create one big pickle for each las
    Converts laspy instance into pandas dataframe
    for this option to work properly you need to provide las file name
    :param las: a laspy instance
    :param del_features: keep only important features (see code)
    20240615 BDA removed filename
    """
    df = pd.DataFrame(np.array(las.x), columns=['x'])
    df['y'] = np.array(las.y)
    df['z'] = np.array(las.z)
    for dimension in las.point_format.dimensions:
        dim = dimension.name
        df[dim] = np.array(las[dim])  # ramène aussi X, Y,  Z en cm
    # on ne garde que ces features significatives (pour le stockage pickle)
    if del_features:
        df = df[['X', 'Y', 'Z', 'intensity', 'return_number',
                 'number_of_returns', 'scan_direction_flag', 'classification',
                 'scan_angle', 'point_source_id', 'gps_time']].copy()
    return df


def las_to_df_rgbh(las, lidargb_dir=None, del_features=True):
    """
    New function create 25 sub parts of one las file 1000mx1000m - 200mx200m each
    Adds RGB and H property for each point
    Finally stores pickles in in lidargb_dir (usually LIDAR_HD/RGB)
    :param las: a laspy instance
    :param las_filename: the filename
    :param lidargb_dir: Directory to store pickle into
    :param del_features: keep only important features (see code)

    """
    df = las_to_df(las) # 1000mx1000m source las file instance
    # TODO voir si on peut vérifier la présence des fichiers avant de lire le las (en se basant sur le nom las)
    #  sinon la regénération est un peu longue
    #cutting in ranges (25)
    rangex = df.X.min() // 100, 1000 + df.X.min() // 100
    rangey = df.Y.min() // 100, 1000 + df.Y.min() // 100
    ranges = cut_ranges(rangex, rangey, 5) # cut in 5 parts each axe
    for range_ in ranges:
        bds = get_bounds_from_range(range_)
        df1name = f'LHD-{bds[0]}-{bds[1]}-LA93-200mx200m.pckl'
        print(f'{lidargb_dir+df1name} ... ', end='')
        if not os.path.exists(lidargb_dir+df1name):
            df1 = df.loc[
                (df.X >= bds[0] * 100) & (df.X < bds[2] * 100) & (df.Y >= bds[1] * 100) & (df.Y <= bds[3] * 100)].copy()
            df1 = add_rgb_column(df1, bds)
            df1 = add_height_column(df1, bds)  # ajout de la hauteur aussi !
            df1.to_pickle(lidargb_dir+df1name)
            print('done')
        else:
            print('already done - skipping')
    print('finished')
    return df


""" ---------------------------------- """
""" Lidar to image - utility functions """
""" ---------------------------------- """

def get_lidar_mask(lidar_dir, bounds, classifs=[3, 4, 5, 6], size_m=200, size_p=1000, shrink=4):
    """
    Returns mask image corresponding to boundaries and size
    The image is rebuild from pickled dataframe that need to be present in LAZ dir.
    Same code as mask generation but live request here
    :param bounds = (xmin, ymin, xmax, ymax)
    """
    # 20240615 BDA remove PCKLE for LAZ
    x, y = bounds[0], bounds[1]
    img_size = size_p // shrink  # we shrink the image by 4 default to minimize pixelling effect (
    # first we need to know in wich file we are in (must be in lidar dir
    lasfile = get_las_file_from_point(lidar_dir, (bounds[0], bounds[1]))  # lower left bound
    df = get_lidar_df_from_file(lasfile)

    slice = df.loc[(df.x >= x) & (df.x < x + size_m) & (df.y >= y) & (df.y <= y + size_m)
                   & (df.classification.isin(classifs))].copy()
    img = np.zeros((img_size, img_size, 1), dtype=np.uint8)  # dtype uint sur  255

    #REM BDA optim sans doute possible en utilisant des numpy array au lieu de apply (pas grave grave)
    slice['locx'] = slice['x'].apply(lambda r: to_local(r, (x, x + size_m), img_size)).astype(int)
    slice['locy'] = slice['y'].apply(lambda r: to_local(r, (y, y + size_m), img_size)).astype(int)
    slice = slice.sort_values(by=['classification'], ascending=False)
    slice.drop_duplicates(subset=['locx', 'locy'], keep='first', inplace=True)
    for idx, row in slice.iterrows():
        if row.classification == 6:
            img[int(row.locy), int(row.locx)] = 255  # BAT => voir data/las_classification
        elif row.classification == 5:
            img[int(row.locy), int(row.locx)] = 128  # VEGE id
        elif row.classification == 4:
            img[int(row.locy), int(row.locx)] = 64  # better be power of 2
        elif row.classification == 3:
            img[int(row.locy), int(row.locx)] = 32
    img = cv2.resize(img, (size_p, size_p), interpolation=cv2.INTER_NEAREST)  # resize to 1000x1000
    return img


def lidar_to_images_slices(laspath, classif=[6], rep='MASK', reload=False, size_m=200, size_p=1000, shrink=4):
    """
        Converts pandas lidar dataframe to N splitted images size_px size_p pixels of 0M20 each
        and store them in a folder 'IMG...' depending on type of image (categorie mask - points altitudes - other ? )
        20240422 As a first DRAFT - simply extract batiments or vegetation points (depending on classif as a list) and print it in nb elements of chosen classif
        20240518 NBL adding multiple classification support
        20240607 BDA parametres de taille et de facteur de rétrécissement pour le masque
        20240615 BDA - remplace picklefile par fichier copc.laz
        20240615 don't use dir as a variable name => dir_
        TODO BDA replace default classif value with immutable
        TODO BDA refactoring call to get_lidar_mask
    """
    base_dir = str(laspath.parent.parent.parent)+'/'  # On devrait mieux gérer les paths.. mais bon ..
    lidar_dir = base_dir+'LIDAR_HD/'
    img_size = size_p // shrink  # we shrink the image by 4 default to minimize pixelling effect (
    xrange, yrange = get_file_range(laspath.name) # Formattage lidar IGN
    print('dealing with ' + laspath.name)
    df = get_lidar_df_from_file(laspath)
    # Vérifier si le chemin d'accès existe
    if not os.path.exists(base_dir + 'IMG/' + str(rep)):
        # Si le chemin d'accès n'existe pas, le créer
        os.makedirs(base_dir + 'IMG/' + str(rep))
    for x in range(*xrange, size_m):
        for y in range(*yrange, size_m):
            filename = get_rootfilename(x, y, size_p)
            bounds = (x, y, x+size_m, y+size_m)  # correx
            dest_file = base_dir + 'IMG/' + str(rep) + '/' + filename + '.png'
            if (reload) or (not os.path.isfile(dest_file)):
                print('generating slice', rep, x, y)
                # On ne prend que les points classifiés dans la liste d'éléments voulus (classif)
                img = get_lidar_mask(lidar_dir, bounds, classifs=classif)
                # DONE interpolation corrige le problème des effets de lissage
                cv2.imwrite(dest_file, img)
            else:
                print(rep, filename, 'already exists ... skipping')

    print('done range', xrange, yrange)
    print('-' * 50)


def lidar_to_images_ALT(laspath, reload=False):
    """
        Converts pandas lidar dataframe to N splitted images 1000x1000 0M20
        and store them in a folder 'IMG...' depending on type of image (categorie mask - points altitudes - other ? )
        TODO BDA replace ALT by HEIGHT - use RGB pickles

    """

    def to_local(x, range_, size):
        """
            converts x & y to local pixel coordinates - rounds non int pixel coordinates
            May be we xill need better interpolation ?
        """
        _x = int(size * (x - range_[0]) / (range_[1] - range_[0]))
        return _x - 1

    base_dir = str(laspath.parent.parent.parent)+'/' # le chemin de path remonté 3 fois
    filename = get_filename(laspath.name,'ALT','.png')
    xrange, yrange = get_range_from_file(laspath.name) # ATTENTION formattage LIDAR IGN
    df = pd.read_pickle(laspath)
    if not os.path.exists(base_dir + 'IMG/ALT/'):
        os.makedirs(base_dir + 'IMG/ALT/')
    dest_file = base_dir + 'IMG/ALT/' + filename
    print('generating', dest_file, end='...')
    if (reload) or (not os.path.isfile(dest_file)):
        img = np.full((1000, 1000), fill_value=-1)
        # conversion en coordonnées locales (Attention X et Y en cm => /100
        df['locx'] = df['X'].apply(lambda r: to_local(r/100, xrange, 1000)).astype(int) # voir avec numpy
        df['locy'] = df['Y'].apply(lambda r: to_local(r/100, yrange, 1000)).astype(int)
        df.sort_values(by='Z').drop_duplicates(subset=['locx', 'locy'], keep='last', inplace=True)

        # on ne garde aucun point sous le sol
        df.loc[df.H <0, 'H'] = 0

        # # clipping extrem values
        # upper_percentile = np.percentile(df['H'], 99)
        # df['H'] = np.minimum(df['H'], upper_percentile)

        # normalisation min max (0,255)
        # df['normH'] = (255 * (df.H - df.H.min()) / (df.H.max() - df.H.min())).astype(int)
        # df['normHint'] = df.normH.astype('int')
        # suppression de la normalisation pour images
        img[df['locx'].to_numpy(dtype=int), df['locy'].to_numpy(dtype=int)] = df['H'].to_numpy()
        img = interpolate_neg_values(img) # initialement on a mis pixels absents à -1
        cv2.imwrite(dest_file, img//10) #la valeur du pixel est donnée en décimètres (tronque à 255 tous les batiments plus hauts que 29 m)
        print('done')
    else:
        print('already exists ... skipping')


def lidar_to_images_slices_mix(dir_, csvpath, classif=[3,4,5], rep='MASK_MIX', reload=False, size_m = 200, size_p = 1000):
    """
        dir: répertoire "VILLE" (car mask mix TOPO/LIDAR)
        Read bdtopo files (csv) : compute interpolated images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
        classif buildings taken from bdtopo.geometry
        Read lidar files (las) : save interpolated images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
        classif :  type of lidar classification (list) except BAT(see las_classification.csv)
        rep : ['MASK_MIX','MASK' or whatever u want]
        20240615 BDA - remove pickle to use laz file instead.
        20240615 NBL - move the test of file existence at the beginning
        20240616 BDA bit of refactoring - move dest to IMG - simplify prints
    """

    # print('dealing with ' + str(csvpath))
    csvfilename = csvpath.name
    filename = get_filename(csvfilename, 'LHD', '.png')
    xrange, yrange = get_range_from_file(filename)
    dest_dir = dir_ +'IMG/'+ str(rep) # sous IMG
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_filename = dest_dir + '/' + filename
    print('generating'+ dest_filename,  end='...')
    if (reload) or (not os.path.isfile(dest_filename)):
        masktopo = bdtopo_to_image_slice(dir_+'ORTHO_WMS/', str(csvpath), xrange = xrange, yrange = yrange)*255
        masklidarvege = get_lidar_mask(dir_+'LIDAR_HD/',(xrange[0],yrange[0]), classifs = [3,4,5])
        img = np.maximum(masktopo, masklidarvege)
        cv2.imwrite(dest_filename, img)
        print('done')
    else:
        print('already exists => skipping')

""" -------------------- """
""" Batch load functions """
""" -------------------- """

def batch_create_las_rgb_pickles(lasdir):
    """
    Reads LIDARHD LAZ content and foe eache file split and save
    pickle dataframes with RGB an H features (IN RGB directory)
    :rgbdir: directory where RGB images are stored (default = RGB)
    """
    paths = Path().rglob(lasdir + '/LAZ/*.laz')
    if not os.path.exists(lasdir + 'RGB/'):
        os.makedirs(lasdir + 'RGB/')
    for path in paths:
        print('--'*50)
        print('batch_create_las_rgb_pickles: loading', path.name)
        las = laspy.read(lasdir + 'LAZ/' + path.name)
        # Call to sub fonction for pickle creation
        las_to_df_rgbh(las, lidargb_dir=lasdir + 'RGB/')
    print('Done with batch_create_las_rgb_pickles - see files in ' + lasdir + 'RGB/')


def batch_create_lidarclassif_images(dir, classif=[6], rep='MASK'):
    """
        Read lidar files (pickle) and
        save interpolated images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
        classif :  type of lidar classification (list) (see las_classification.csv)
        rep : ['BAT' 'MASK' or whatever u want]
        TODO: pass default classif value in unmutable manner
        20240615 change default to MASK
    """
    paths = Path(dir + '/LAZ/').rglob('*.laz')  # les images sont sous LAS (modif BDA 20240615)
    for path in paths:
        lidar_to_images_slices(path, classif, rep)


def batch_create_lidarclassif_images_mix(dir_, classif=[3,4,5], rep='MASK_MIX'):
     """
        Dir: répertoire "VILLE" (car mask mix TOPO/LIDAR)
        Read bdtopo files (csv) : compute interpolated images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
        classif buildings taken from bdtopo.geometry
        Read lidar files (pickle) : save interpolated images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
        classif :  type of lidar classification (list) except BAT(see las_classification.csv)
        rep : ['MASK_MIX','MASK' or whatever u want]
    """
     paths = Path(dir_ + 'ORTHO_WMS/').rglob('*.csv') # les fichiers sont sous ORTHO_WMS
     filtered_paths = [path for path in paths if 'index.csv' not in path.name]
     for path in filtered_paths:
         lidar_to_images_slices_mix(dir_, path)


def batch_create_lidaralt_images(dir_):
    """
        Read lidar RGB files (pickle with Heihg) and
        save corresponding images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
    """
    # TODO BDA - replace ALT by HEIGHT in process use RGB

    paths = Path(dir_ + '/RGB/').rglob('*.pckl')  # les lidar  sont sous LAZ
    for path in paths:
        lidar_to_images_ALT(path)


""" --------------------- """
""" chained run processes """
""" --------------------- """

def run_all_preprocesses(lidardir, classifs=[3, 4, 5, 6]):
    """
     Lance tous les préprocessings lidar après avoir téléchargé les LAZ daans LIDAR_HD/LAZ
    """
    batch_create_lidarclassif_images(lidardir, classif=classifs, rep='MASK') # The one to use for masks (ancien mode)
    basedir = '/'.join(lidardir.split('/')[:-2])+'/'  # bidouille classif mix attends le répertoire  pas lidar
    batch_create_lidarclassif_images_mix(basedir, classif=classifs, rep='MASK_MIX')
    batch_create_las_rgb_pickles(lidardir) # BDA 20260625 nouvelle méthode (25 pickles RGB par batch)
    batch_create_lidaralt_images(lidardir)

""" Main """

if __name__ == '__main__':
    basedir = '../data/TOURS/'
    lidar_dir = os.path.join(basedir, 'LIDAR_HD','')
    # batch_create_lidarclassif_images_mix(basedir, rep='MASK_MIX')
    # batch_create_lidaralt_images(lidar_dir)
    batch_create_lidarclassif_images(lidar_dir, classif=[3, 4, 5, 6], rep='MASK')
    batch_create_las_rgb_pickles(lidar_dir) # BDA 20260625 nouvelle méthode (25 pickles RGB par batch)


    # # DEBUG
    # las = laspy.read('../data/TOURS/LIDAR_HD/LAZ/LHD_FXX_0527_6706_PTS_C_LAMB93_IGN69.copc.laz')
    # # Call to sub fonction for pickle creation
    # las_to_df_rgbh(las, lidargb_dir='../data/TOURS/LIDAR_HD/RGB/')