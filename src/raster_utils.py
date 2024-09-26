import pandas as pd
import numpy as np
import os
import rasterio
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from rasterio import Affine # transformations affines
from src.topo_utils import * # TODO import alias instead
import src.las_utils as lu

class RasterImageObject:
    """
    Class to represent a Raster Image that will be saved to pickle
    """
    def __init__(self):
        self.filename = None
        self.piclefilename = None
        self.width = None
        self.height = None
        self.transform = None  # la transformation affine
        self.bounds = None
        self.resolution = None
        self.crs = None
        self.img = None
    def __str__(self):
        return str(self.__dict__)


def load_and_pickle_raster_image(imgpath):
    """
        Load image from imgpath and save it to imgpath.jp2.pckl
    """
    filename = os.path.basename(imgpath)
    print(filename)
    # On reprend le nom original et on ajoute l'extension pckl
    pickfilename = filename + '.pckl'
    dirname = os.path.dirname(imgpath)
    try:
        f = open(str(imgpath)+'.pckl', 'rb')
        rio = pickle.load(f)
        print('file already exists - quiting ')
    except: # le fichier n'existe pas
        rio = RasterImageObject()  # the object that is to be saved to pickle
        print('loading ' + filename, end='...')
        rio.pickfilename = pickfilename
        src = rasterio.open(imgpath)  # object rasterio # ~ 2 minutes minimum for loading
        rio.img = np.transpose(src.read(), [1, 2, 0]) # transpose et ramène en x, y + couleur
        rio.resolution = src.res  # la résolution
        rio.bounds = src.bounds  # les boundaries
        rio.transform = src.transform # la transformation affine à faire pour passer de local à crs
        rio.crs = src.crs  # la projection
        pickfile = open(str(imgpath) + '.pckl', 'wb')  # l'objet sera à côté de l'image source
        pickle.dump(rio, pickfile) #on sauve l'objet sous pickle (beaucoup plus rapide à charger)
        pickfile.close()
        src.close()
        print('saved ' + rio.pickfilename)

    return rio

def range_to_local(range_, size=1000):
    """ returns loacl rang(in terms of pixel for given LA93 range (in meters)
        range is in normally in meters
        size is in pixels
    """
    return 0, int(size*(range_[1]-range_[0]) / (range_[1] - range_[0]))


def to_local(coord, bounds, image_size):
    """
    retourne les coordonnées passée en paramètre en coordonnées  pixel image
    coord : tuple x,y
    bounds: les boundaries de l'image
    image_size : size of image
    """
    xmin, ymin, xmax, ymax = bounds
    x, y = coord
    x_pixel = (x - xmin) / (xmax - xmin) * image_size[1]  # Notez que OpenCV utilise (height, width) pour la taille
    y_pixel = (y - ymin) / (ymax - ymin) * image_size[0]  # Corriger l'effet de miroir horizontal

    return int(x_pixel), int(y_pixel)


def add_batiment_on_image(img, img_bounds, batiment):
    """
    Ajoute le polygone dans l'image passée en paramètres
    convertit le coordonnées du polygone en local
    les bounds indiquent les coordonnées de l'image exprimées dans la crs du polygone
    batiment est un enregistrment de geo dataframe
    """
    geom = batiment.geometry
    if geom.geom_type == 'Polygon':
        coords = geom.exterior.coords
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            coords = poly.exterior.coords

    polygon_coords = [to_local((x, y), img_bounds, img.shape) for x, y, z in coords]
    polygon_coords = np.array(polygon_coords, np.int32)
    polygon_coords = polygon_coords.reshape((-1, 1, 2))

    cv2.polylines(img, [polygon_coords], isClosed=True, color=(0, 0, 255), thickness=2)  # Couleur rouge en BGR

    return img


def split_raster_image(rio):
    """
    Splits 25kx25K pixel image in 125 subdivisions of 1000x1000 pixels
    :param rio: RasterImageObject instance
    #TODO BDA corriger ça, ça ne fonctionne pas
    """
    range_x = int(rio.bounds.left), int(rio.bounds.right)
    range_y = int(rio.bounds.bottom), int(rio.bounds.top)
    img = np.flipud(rio.img) #ATTENTION l'image est de haut en bas - on la flip de bas en haut !
    minx, miny = range_x[0], range_y[0]
    ranges = lu.cut_ranges(range_x, range_y, 25)
    for rx, ry in ranges:
        coords='{}-{}'.format(rx[0], ry[0])
        print('coords: {}'.format(coords))
        lminx, lminy = 5*(rx[0]-minx), 5*(ry[0]-miny) #on passe en pixels
        slice = img[lminy:lminy+1000, lminx:lminx+1000 ] #ATTENTION dasn les images 0 est l'axe Y !!!

        filename = '../data/ORTHO_JP2/JP2/ORTHOHR-{}-LA93-200mx200m.png'.format(coords, minx, miny)
        cv2.imwrite(filename, slice)

    pass

def load_and_split_raster(jp2path):
    """
    Charge une image raster 25kx25k - la découpe en plusieurs (1000x1000) dans un sous dossier avec convention de nommage
    Les coordonnées de l'image seront accessible dans la convention de nom de fichier
    """
    paths = Path(jp2path).rglob('*.jp2')  # les lidar  sont sous LAZ
    for path in paths:
        rio = load_and_pickle_raster_image(path)
        split_raster_image(rio)
    return


def pickle_sample_from_dir(path, size):
    paths = Path().rglob(path)
    paths = list(paths) # convertir en liste de paths
    """ Takes a list of paths and pick a random sample of size images from it """
    for i in np.random.randint(0, len(paths), size):
        # On charge un sample de 10 images au hasard
        print(paths[i])
        load_and_pickle_raster_image(paths[i])


def show_colors_dominance(img, name=''):
    """
    plot raster image and dominant hue color (5 colors) in it
    :param img: numpy RGB image
    """
    imgh = cv2.blur(img, ksize=(3,3)) # red anomaly blur
    imgh = cv2.cvtColor(imgh, cv2.COLOR_RGB2HSV) # normalise le H entre  0 à 180
    fig = plt.figure(figsize=(19, 6))
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, cmap='hsv')
    ax1.set_ylim(0,img.shape[1]) # reverse image on y
    ax2 = fig.add_subplot(132)
    ax2.imshow(imgh[..., 0], cmap='hsv')
    ax2.set_ylim(0,imgh.shape[1]) # reverse image on y (set 0 on bottom left corner)
    ax3 = plt.subplot(133)
    img_reshaped = imgh.reshape(img.shape[0]* img.shape[1], 3)
    teintes = pd.DataFrame(img_reshaped[:, 0], columns=['teinte'])
    # de 0 à 25 => rouge ,  25 à 35 => Jaune , 35 à 85=> Vert , , 85 à 135 => Bleu , 135 à 180 => Magenta
    teintes.loc[teintes.teinte < 25, 'couleur'] = 'Rouge'
    teintes.loc[(teintes.teinte >= 25) & (teintes.teinte < 35), 'couleur'] = 'Jaune'
    teintes.loc[(teintes.teinte >= 35) & (teintes.teinte < 85), 'couleur'] = 'Vert'
    teintes.loc[(teintes.teinte >= 85) & (teintes.teinte < 135), 'couleur'] = 'Bleu'
    teintes.loc[(teintes.teinte >= 135), 'couleur'] = 'Magenta'
    ax3 = teintes.couleur.value_counts(normalize=True).plot(kind='bar')
    plt.suptitle(name)
    plt.show()


def plot_color_distribution(img):


    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('Distribution des canaux de couleur')


def plot_batiments(dir, filename, ax, showcontours=True):

    bats = gpd_read_csv(dir + filename[:-4]+'.csv') # les batiments en détail
    map_size, xrange, yrange = get_size_and_ranges_from_filename(filename)

    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    if bats.shape[0] > 0:
        if 'geometry' in bats.columns:
            bats['local'] = bats['geometry'].apply(convert_polygon(map_size, xrange, yrange))  # geometry locale
            bats['centroids'] = bats.geometry.apply(convert_centroid(map_size, xrange, yrange))  # les centroids
            if showcontours:
                bats.local.apply(lambda bat: ax.fill(bat[0], bat[1], facecolor='none', edgecolor='red'));  # outlines
            bats.centroids.apply(lambda x: ax.scatter(x[0], x[1], c='r'));  # centroids


def show_ortho_batiments_resume(dir, filename):
    """
        plot une ligne de subplot batiment de l'ortho batiment'
    """
    def plot_batiments(bats):
        if bats.shape[0] > 0:
            if 'geometry' in bats.columns:
                bats['local'] = bats['geometry'].apply(convert_polygon(map_size, xrange, yrange))  # geometry locale
                bats['centroids'] = bats.geometry.apply(convert_centroid(map_size, xrange, yrange))  # les centroids
                bats.local.apply(lambda bat: plt.fill(bat[0], bat[1], facecolor='none', edgecolor='red'));  # outlines
                bats.centroids.apply(lambda x: plt.scatter(x[0], x[1], c='r'));  # centroids

    rootfilename = filename[:-4]  # racine du jpg et csv
    img = cv2.imread(dir + filename)
    map_size, xrange, yrange = get_size_and_ranges_from_filename(rootfilename)
    bats = gpd_read_csv(dir + rootfilename+'.csv') # les batiments en détail
    n_bats = bats.shape[0]
    plt.figure(figsize=(14, 8))
    plt.axis('off')
    plt.subplot(121)
    plt.title(f'{n_bats} Batiments {xrange[0]}-{yrange[0]}')
    plt.imshow(img)
    # affichage batiments
    plot_batiments(bats)
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.subplot(122)
    plot_color_distribution(img)
    plt.show()


def get_size_and_ranges_from_filename(filename: str):
    """
    Parse le nom de fichier filename et renvoie ranges et size de l'emprise correspondante
    """
    elts = filename.split('.')[0].split('-')
    # elts = ['ORTHOHR', '520000', '6703000', 'LA93', '0M20', '1000x1000']
    pixel_size = float(elts[-2].replace('M', '.'))
    map_size = [int(e) for e in elts[-1].split('x')]
    xrange = int(elts[1]), int(int(elts[1]) + pixel_size * map_size[0])
    yrange = int(elts[2]), int(int(elts[2]) + pixel_size * map_size[1])
    return map_size, xrange, yrange


def batch_generate_roofs(dir, batiments):
    """
    Génére une vignette de batiment par par toit dans le répertoire dir '
    :param dir: répertoire de destination
    :param batiments: geodataframe des batiments
    :return: void
    """

    for idx, row in batiments.iterrows():
        width = 5*(row.bds[2]- row.bds[0])
        height = 5*(row.bds[3]- row.bds[1])
        try:
            img =  api.load_bdortho(row.bds, width, height)
            # Ajout du contour de bdtopo pour mieux cerner le toit que l'on cherche à catégoriser
            img = ru.add_batiment_on_image(img, row.bds, row)
            img = cv2.resize(img, (224, 224)) # un peu arbitraire ?
            cv2.imwrite(rootdir + f'data/TOURS/IMG/ROOFS/{row.cleabs}.png', img)
            print('wrote', row.cleabs)
        except Exception as e:
            print('Erreur api' , e)




if __name__ == '__main__':

    """ exécution du script en direct """

    # load_and_split_raster('../data/ORTHO_JP2')

    import src.IGN_API_utils as api

    img = api.load_bdortho((520000, 6700000, 520200, 6700200))
    plt.imshow(img)
    plt.show()
    #
    # ortho_img = '../data/BD_ORTHO_JP2/BDORTHO_1-0_RVB-0M20_JP2-E080_LAMB93_D037_2021-01-01/ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D037_2021-01-01' + \
    #              '/ORTHOHR/1_DONNEES_LIVRAISON_2022-07-00086/OHR_RVB_0M20_JP2-E080_LAMB93_D37-2021/37-2021-0520-6705-LA93-0M20-E080.jp2'
    # load_and_pickle_raster_image(ortho_img)

    # Chemin Spécifique machine BDA
    # ortho_path = '..\\data\\BD_ORTHO_JP2\\BDORTHO_1-0_RVB-0M20_JP2-E080_LAMB93_D037_2021-01-01\\' + \
    #              'ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D037_2021-01-01\\ORTHOHR\\1_DONNEES_LIVRAISON_2022-07-00086\\' + \
    #              'OHR_RVB_0M20_JP2-E080_LAMB93_D37-2021\\*.jp2'


    #
    # ortho_path = '../data/BD_ORTHO_JP2/BDORTHO_1-0_RVB-0M20_JP2-E080_LAMB93_D037_2021-01-01/' + \
    #              'ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D037_2021-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2022-07-00086/' + \
    #              'OHR_RVB_0M20_JP2-E080_LAMB93_D37-2021/*.jp2'
    #
    # # Traitement en masse du répertoire
    # # va lire 10 images au hasard  et créer le pickle correspondant (Attention 2Go par pickle !)
    # pickle_sample_from_dir(ortho_path, size=10)

    # get_size_and_ranges_from_filename('ORTHOHR-520000-6703000-LA93-0M20-1000x1000.jpg')
    # print(get_ortho_index('../data/ORTHO_WMS/'))