import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import re
import os
import glob
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon
from PIL import Image, ImageDraw



def load_and_pickle_shp_file(filename, reload=False):
    """
    Essaie de charger le pickle -
    si échec on charge le shp et on crée le pickle
    si reload == True => recharge et recrée le shp
    # TODO déprécier .Remplacer le recours aux gros pickles.
    """

    def load_and_pickle(filename):
        print('...loading shp ...')
        gdf = gpd.read_file(filename, engine='pyogrio')
        gdf.to_pickle(filename + '.pckl')
        return gdf

    if reload:
        load_and_pickle(filename)
    else:
        try:
            # on essaie de charger le pickle
            print('Trying to load from pickle ')
            gdf = pd.read_pickle(filename + ".pckl")  # use of pd.read
            print("OK loaded")
        except:
            # et si ça plante on charge le shp (et on stocke un pickle)
            print('load pickle failed')
            gdf = load_and_pickle(filename)
    return gdf


def polygons_to_lists(geom):
    """
    renvoie la liste des listes de points [(x,y,z), ...]
    geom est un polygon ou multipolygone =>
    """

    def polygon_tolist(geom):
        """ polygone simple """
        return list(geom.exterior.coords)

    if isinstance(geom, shapely.geometry.polygon.Polygon):
        return [polygon_tolist(geom)]
    else:
        return [polygon_tolist(g) for g in list(geom.geoms)]  # multipolygone


# fonctions utilitaires topo
# reprises de starter kit
def isInMap(xrange, yrange):
    def my_function(polynom):
        x, y = polynom.centroid.x, polynom.centroid.y
        if xrange[0] < x and xrange[1] > x and yrange[0] < y and yrange[1] > y:
            return True
        else:
            return False

    return my_function


def isInMap2(xrange, yrange):
    def my_function(polynom):
        # Définir une géométrie de référence pour le masque (par exemple, un rectangle)
        reference_geometry = Polygon(
            [(xrange[0], yrange[0]), (xrange[0], yrange[1]), (xrange[1], yrange[1]), (xrange[1], yrange[0])])
        # Créer un masque basé sur l'intersection avec la géométrie de référence
        resultat = polynom.intersects(reference_geometry)
        return resultat

    return my_function


def get_polynom_intersect_coords(polygon, xrange, yrange):
    """
    renvoie les coordonnées (x,y) des points délimitant le polygone résultant de l'intersection avec l'emprise (reference_geometry)
    """
    reference_geometry = Polygon(
        [(xrange[0], yrange[0]), (xrange[0], yrange[1]), (xrange[1], yrange[1]), (xrange[1], yrange[0])])
    coords = []
    intersection = polygon.intersection(reference_geometry)
    if not intersection.is_empty:
        if isinstance(intersection, MultiPolygon):
            intersection = [g for g in intersection.geoms][
                0]  # cas à la marge. On prend le premier polygone pour garder le bon nb de bâtiments
        boundaries_coords = list(intersection.exterior.coords)
        coords.append(boundaries_coords)
    return coords


# On fait une interpolation linéaire pour redéfinir les centroîdes en fonction de l'image (conversion en pixel)
# taille de l'image (map-zize): 5000*5000 pixels
def convert_centroid(map_size, xrange, yrange):
    def my_function(polygon):
        x, y = polygon.centroid.x, polygon.centroid.y
        x_new = (x - xrange[0]) / (xrange[1] - xrange[0]) * map_size[0]
        # y_new = map_size[1] - (y - yrange[0])/(yrange[1]-yrange[0])*map_size[1]  # suppression de l'inversion imga de bas vers haut
        y_new = (y - yrange[0]) / (yrange[1] - yrange[0]) * map_size[1]
        return [x_new, y_new]

    return my_function


def convert_polygon(map_size, xrange, yrange):
    """
    Fonction originale starterkit qui renvoie une fonction à utiliser avec apply
    on considère que xrange et yrange sont en mètres et map_size en pixels
    """

    def my_function(geometry):
        """ renvoie la conversion du polygon, multipolygones ou lignes en geometrie CRS
            en une liste  de coordonnées exprimées en pixels de l'image
        """
        if geometry.geom_type.lower() == "polygon":
            x, y = geometry.exterior.coords.xy
            x = x.tolist()
            y = y.tolist()
        elif geometry.geom_type.lower() == "linestring":
            x, y = geometry.coords.xy
            x = x.tolist()
            x += x[::-1]
            y = y.tolist()
            y += y[::-1]
        elif geometry.geom_type.lower() == "multipolygon":
            # BDA  on va chercher le premier  polygone du multipolygone dont on va prendre l'extérieur
            # il n'y a que très peu de vrais multipolygones => ça semble suffisant
            polygons = [g for g in geometry.geoms]
            x, y = polygons[0].exterior.coords.xy  # essai avec un seul
            x = x.tolist()
            y = y.tolist()
        else:
            x = [1, 2]
            y = [1, 2]
        x = np.array(x)
        y = np.array(y)
        x_new = (x - xrange[0]) / (xrange[1] - xrange[0]) * map_size[0]
        # apparemment on doit pas faire
        y_new = (y - yrange[0]) / (yrange[1] - yrange[0]) * map_size[1]
        return [x_new, y_new]

    return my_function


def generate_xy_polygons(bdtopo_area, map_size):
    list_x = []
    for xpoly in bdtopo_area['xpolygon']:
        list_x.extend(xpoly.tolist() + [None])
    list_x = list_x[:-1]

    list_y = []
    for ypoly in bdtopo_area['ypolygon']:
        ypoly = map_size[1] - ypoly
        list_y.extend(ypoly.tolist() + [None])
    list_y = list_y[:-1]

    return list_x, list_y


def generate_x_polygons(xdata):
    list_x = []
    for xpoly in xdata:
        list_x.extend(xpoly.tolist() + [None])
    list_x = list_x[:-1]
    return list_x


def gpd_read_csv(file, crs='EPSG:2154'):
    """
    :param file:
    :param crs:
    Charge un csv et retourne un geodataframe (pas natif)
    Si pas de geometrie retourne le df pandas
    """
    df = pd.read_csv(file)
    if 'geometry' in df.columns:
        gdf = gpd.GeoDataFrame(df.drop('geometry', axis=1), geometry=df.geometry.apply(shapely.wkt.loads), crs=crs)
        return gdf
    else:
        return df


def bdtopo_to_image_slice(dir_, pathfile, xrange, yrange, size_p=1000):
    """
    Converts csv bdtop dataframe to N splitted images size_px size_p pixels of 0M20 each using geometry
    and store them in a folder 'IMG...'
    TODO dir_ is unused  ?  ->  for the unused part(save the mask) which could be activated ... remove if not
    """
    bdtopo = gpd_read_csv(pathfile)
    bdtopo.drop('Unnamed: 0', axis=1, inplace=True)  # specify index col juste au dessus ?
    img = Image.new('L', (size_p, size_p), 0)  # equivalent np.zeros avec pillow
    if len(bdtopo) != 0:
        # On va créer une colonne "type" qui récupère, pour le df bdtopo, le type d'objet à partir du champ cleabs (ici bâtiment)
        # c'est utile pour l'agrégation:
        def extract_initial_letters(text):
            match = re.search(r"^[A-Za-z]+", text)
            return match.group(0) if match else None

        bdtopo['Type'] = bdtopo['cleabs'].apply(extract_initial_letters)
        bdtopo = bdtopo.explode(
            index_parts=False)  # On transforme les  multiobjets en objet simple (multipolygones en polygones). S'il y en a plusieurs, cela créera plusieurs lignes pour le même bâtiment
        map_size = [size_p, size_p]
        bdtopo_zone = bdtopo[bdtopo['geometry'].apply(isInMap2(xrange, yrange))].copy()
        bdtopo_zone['centroid'] = bdtopo_zone['geometry'].apply(convert_centroid(map_size, xrange, yrange))
        bdtopo_zone['xcentroid'] = bdtopo_zone['centroid'].apply(lambda x: x[0])
        bdtopo_zone['ycentroid'] = bdtopo_zone['centroid'].apply(lambda x: x[1])
        bdtopo_point = bdtopo_zone[
            bdtopo['geometry'].apply(lambda x: x.wkt.lower()[:5] == "point")]  # on prevoit aussi le cas des points
        bdtopo_zone['polygon'] = bdtopo_zone['geometry'].apply(convert_polygon(map_size, xrange, yrange))
        bdtopo_zone['xpolygon'] = bdtopo_zone['polygon'].apply(lambda x: x[0])
        bdtopo_zone['ypolygon'] = bdtopo_zone['polygon'].apply(lambda x: x[1])
        bdtopo_zone_agregate = bdtopo_zone.groupby('Type').agg({'xpolygon': list, 'ypolygon': list})
        bdtopo_zone_agregate['xpolygon_ready'] = bdtopo_zone_agregate['xpolygon'].apply(generate_x_polygons)
        bdtopo_zone_agregate['ypolygon_ready'] = bdtopo_zone_agregate['ypolygon'].apply(generate_x_polygons)
        bdtopo_point_agregate = bdtopo_point.groupby('Type').agg({'xcentroid': list, 'ycentroid': list})
        data_mask = [
            list(zip(bdtopo_zone_agregate['xpolygon']['BATIMENT'][i], bdtopo_zone_agregate['ypolygon']['BATIMENT'][i]))
            for i in range(len(bdtopo_zone_agregate['xpolygon']['BATIMENT']))]
        # img = Image.new('L', (size_p, size_p), 0) # equivalent np.zeros avec pillow
        for p in data_mask:
            ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
        mask = np.array(img)
    else:
        mask = np.array(img)
    # #on ne sauvegarde pas ce masque car il va être combiné avec un masque LIDAR sinon on pourra activer:
    # print('dealing with ' + maskfile.name)
    # # Vérifier si le chemin d'accès existe
    # if not os.path.exists(dir + 'IMG/' + str(rep)):
    # # Si le chemin d'accès n'existe pas, le créer
    #     os.makedirs(dir + 'IMG/' + str(rep))
    # for x in range(*xrange, size_m):
    #     for y in range(*yrange, size_m):
    #         filename = get_rootfilename(x, y, size_p)
    #         if (reload) or (not os.path.isfile(dir + 'IMG/' + str(rep) + '/' + filename + '.png')):
    #             print('generating slice', rep, x, y)
    return mask


def bdtopo_bat_yolo_txt(dir_,pathfile,min_area, label,size_p=1000):
    from src.preprocessing_utils import get_roof_label,get_dic_labels
    from src.las_utils import get_range_from_file, get_filename
    """
    create for one image, the associated txt file required by YOLO (annotation)
    The file name is the same as the image (except the extension)
    For each building, there is a line with the following format: class of the object (in our case one class1 for building) and points (X,y)  of the polygone 
    '<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>'
    exemple: 
    0 0.681 0.485 0.670 0.487 0.676 0.487 
    We keep only buildings having an area of at least min_area square meters.
    label : if we are interested in only buildings, label ="buildings"  ("0" in the text  file)
            if we are interested in roofs, label ="roofs" (value of the label in the text file depending on function "get_roof_label")
    the function returns the lebels directory 
    
    """
    pattern = r'/data/([^/]+)/'
    match = re.search(pattern, str(pathfile))
    ville = match.group(1)
    dir = pathfile.parents[2]/'YOLO'/f'{ville}_{label}_minarea{min_area}'  # répertoire d'écriture des fichiers
    bdtopo = gpd_read_csv(pathfile)
    nom_fichier = os.path.basename(pathfile)
    print(f'dealing with {nom_fichier}')
    if not os.path.exists(dir):
        os.makedirs(dir)
    map_size = [size_p, size_p]
    xrange, yrange = get_range_from_file(nom_fichier)
    nom_txt = get_filename(nom_fichier, 'ORTHOHR', '.txt')
    path_txt_file = dir / nom_txt
    if os.path.exists(path_txt_file):
        print('already exists => skipping')
    else:            
        with open(path_txt_file, 'w') as fichier:
            if len(bdtopo) != 0: 
                bdtopo = bdtopo.explode(index_parts=False)  # On transforme les multiobjets en objets simples (multipolygones en polygones). S'il y en a plusieurs, cela créera plusieurs lignes pour le même bâtiment               
                bdtopo = bdtopo.loc[bdtopo.geometry.area >= min_area].reset_index() # on ne garde que les surfaces > 50m2
                for index, geom in enumerate(bdtopo.geometry):
                    id_bat =bdtopo.loc[index,"cleabs"]                                                        
                    boundaries_coord = get_polynom_intersect_coords(geom, xrange, yrange)
                    if label == "buildings":
                        info_text = '0'  # initialisation de la chaine de caractère a écrire avec le numéro de la classe : 0 (bâtiment)
                    elif label == "roofs":
                        datadir = '../data'
                        dic_labels = get_dic_labels(datadir)
                        roof_label = get_roof_label(id_bat, datadir, ignore_labels = [])
                        if roof_label in dic_labels:
                            info_text = str(dic_labels[roof_label])
                        else:
                            continue
                    for point in boundaries_coord[0]:
                        x = str(round((point[0] - xrange[0]) / (xrange[1] - xrange[0]), 3))  # on doit avoir des coordonnées entre 0 et 1
                        y = str(round((point[1] - yrange[0]) / (yrange[1] - yrange[0]), 3))
                        info_text += ' ' + x + ' ' + y
                    fichier.write(info_text + '\n')    
            else:
                pass # on écrit un fichier vide car pas de bâtiments à considérer (>50m2)                        
        print(f'{nom_txt} done')
    return  dir
                   

def get_squared_bounds(geometry, buffer=0):
    """ Renvoie la boite englobante de la geometry rammenés en valeur entière et rendue carrée centrée
    Geometry (polygone ou multipolygone
    buffer: nombre de mètres à ajouter de chaque côté (pour agrandir la fenêtre)
    """
    import math
    bounds = geometry.bounds
    minx, miny, maxx, maxy = [math.floor(bounds[0]),math.floor(bounds[1]), math.ceil(bounds[2]),math.ceil(bounds[3])]
    width = maxx - minx
    height = maxy - miny
    diff = abs(width - height)
    reste = diff%2
    if width < height:
        minx = minx - diff//2
        maxx = maxx + diff//2 +reste
    elif  width > height:
        miny = miny - diff//2
        maxy = maxy + diff//2 + reste
    # on rajoute un buffer de 2m de chaque côté certaines images sont très petites
    return minx-buffer, miny-buffer, maxx+buffer, maxy+buffer



""" -------------------- """
""" Batch load functions """
""" -------------------- """


def batch_create_segyolo_bat_txt(dir_,min_area,label):
    """
       Dir source: répertoire "VILLE" 
       Read bdtopo files (csv) : compute interpolated images to disk (with 1000x1000 pixels of 0M20 -every 200 meters
       classif buildings taken from bdtopo.geometry
       .txt files generated in data/YOLO directory
       Return the path of labels directory to be used in  function yolo_train_test_valid
   """
    paths = Path(dir_ + 'ORTHO_WMS/').rglob('*.csv')  # les fichiers sont sous ORTHO_WMS
    filtered_paths = [path for path in paths if 'index.csv' not in path.name]
    for path in filtered_paths:
       target_dir =bdtopo_bat_yolo_txt(dir_, path, min_area=min_area, label =label)  # second output de la fonction
    txt_files = glob.glob(os.path.join(target_dir, '*.txt'))
    num_txt_files = len(txt_files)
    compteur = 0
    nb_objets_lab = 0
    for file_path in txt_files:
        if os.stat(file_path).st_size != 0:
            compteur += 1
            with open(file_path, 'r') as file:
                num_lines = sum(1 for line in file)
            nb_objets_lab += num_lines
    print(f"Il y a {num_txt_files} fichiers .txt dans le répertoire d'annotations {target_dir} dont {compteur} non vides avec {nb_objets_lab} objets labellisés") 
    return str(target_dir) + '/'


if __name__ == '__main__':
    # tests fonctions polygones

    # from src.IGN_API_utils import *
    #
    # bounds = (520000, 6700000, 520200, 6700200)
    # size = (0.2, 0.2)
    # img = load_bdortho(bounds)
    # bats = load_batiments(bounds)
    # # %%
    # map_size = [1000, 1000]  # size in pixels
    # xrange = bounds[0], bounds[2]
    # yrange = bounds[1], bounds[3]
    # conv = bats.geometry.apply(convert_polygon(map_size, xrange, yrange))
    #
    # print(conv)

    # tests de la partie conversion de multipolygones
    from src.raster_utils import *

    dir = '../data/ORTHO_WMS/'
    rootfilename = 'ORTHOHR-520000-6703000-LA93-0M20-1000x1000'
    map_size, xrange, yrange = get_size_and_ranges_from_filename(rootfilename)
    bats = gpd_read_csv(dir + rootfilename + '.csv')  # les batiments en détail
    bat = bats.iloc[0]
    bats.geometry.apply(convert_polygon(map_size, xrange, yrange))