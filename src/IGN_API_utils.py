import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Polygon
import cv2
import os
import src.las_utils # pour la génération des images lidar
from pathlib import Path
import pyproj

def get_ortho_index(path):
    """
        Retourne un dataframe qui contient l'index du contenu de dossier (path) image_wms (nom de fichier)
        avec 2 colonnes fichier,nb_batiments
        les images sont *.jpg et les batiments sont dans les *.csv
    """
    paths = Path().rglob(path+'*.jpg')
    paths = list(paths) # convertir en liste de paths
    df = pd.DataFrame(columns=['fichier', 'nb_batiments'])
    for file in paths:
        filerootname = file.name[:-4]
        bats = pd.read_csv(path+filerootname + '.csv') #on lit les csv
        nb_batiments = bats.shape[0]
        df.loc[len(df)] = [filerootname, nb_batiments]  # append row
    df.to_csv(path+'index.csv') #refresh index file
    return df


def load_bdortho(bounds, width=1000, height=1000):
    """
    Requête ORTHOHR pour les boundaries passés en paramètre
    Retourne l'image sous forme np.array
    :param height: in pixels
    :param width: in pixels
    :param bounds: tuple 4 coordonnées  - en EPSG2154
    :return: image sous forme np.array
    """
    # 20240618 correction appel API IGN - Passer les paramètres Lambert93
    # DONE vérifier la correction des affichages constatés (photo vs topo) => OK !!
    minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]

    request = 'https://data.geopf.fr/wms-r?LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS'
    request += '&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX='
    request += str(minx)+','+str(miny)+','+str(maxx)+','+str(maxy)  # la bbox en LA93
    request += f'&WIDTH={str(width)}&HEIGHT={str(height)}'
    # print(request)
    response = requests.get(request).content
    orthophoto = Image.open(BytesIO(response))
    orthophoto = ImageOps.flip(orthophoto)
    return np.array(orthophoto)


def load_rgealti(bounds, width=1000, height=1000):
    """
    Requête RGEALTI pour les boundaries passés en paramètre
    Retourne l'image sous forme np.array
    :param bounds: tuple 4 coordonnées  - en EPSG2154
    :return: image sous forme np.array
    """
    # DONE reconstruire la requête avec LA93 au lieu de WSG84 !!!
    minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]
    request = 'https://data.geopf.fr/wms-r?LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES&FORMAT=image/tiff'
    request += '&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX='
    request += str(minx)+','+str(miny)+','+str(maxx)+','+str(maxy)  # la bbox en LA93
    request += f'&WIDTH={str(width)}&HEIGHT={str(height)}'
    response = requests.get(request).content
    orthophoto = Image.open(BytesIO(response))
    orthophoto = ImageOps.flip(orthophoto)
    return np.array(orthophoto)


def load_batiments(bounds):
    """
        Effectue une requête WFS sur service IGN TOPO et retourne un geodataframe des batiments trouvés
        Le paramètre bounds est un tuple minx, miny, maxx, maxy en coordonnées Lambert93 EPSG 2154
        la taille des images est 1000x1000 pixels (OK si bounds est un carré 200m x 200m)
    """
    # BDTOPO ne fonctionne pas avec CRS2154
    minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]
    # on crée un gdf pour la reprojection en lat, lon
    poly_bounds = Polygon([(minx, miny) , (minx, maxy), (maxx, maxy), (maxx, miny)]) # trois points suffisent
    gdf_bounds = gpd.GeoDataFrame({'geometry': [poly_bounds]}, crs="EPSG:2154")

    # on récupère les min max mais en EPSG4326 cette fois
    bounds = gdf_bounds.to_crs("EPSG:4326").bounds
    minx, miny, maxx, maxy = bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]

    # Construction de la requête sur le service TOPO V3 Batiments en WFS
    request = ("https://data.geopf.fr/wfs/ows?SERVICE=WFS&TYPENAMES=BDTOPO_V3:batiment&REQUEST=GetFeature&VERSION=2.0.0&outputformat=application/json&bbox=")
    request += str(miny)+','+str(minx)+','+str(maxy)+','+str(maxx)  # la bbox en lat,lon
    response = requests.get(request)  # appel au service

    # on reprojète les boundaries en EPSG:2154
    bounds = gdf_bounds.to_crs("EPSG:2154").bounds
    minx, miny, maxx, maxy = bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]  # on revient en lambert
    batiments = gpd.GeoDataFrame.from_features(response.json()["features"])
    if batiments.shape[0] > 0:
        batiments = batiments.set_crs("EPSG:4326").to_crs("EPSG:2154")
        cadre = Polygon(((minx, miny),  (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)))
        batiments = batiments[batiments.intersects(cadre)] # on ne garde que les batiments en intersection avec le cadre
        batiments = batiments[~(batiments.geometry.isna() | batiments.geometry.is_empty)]
    return batiments


def batch_load_ortho_files(bds, reload=False, filepath ='../data/ORTHO_WMS/' ):
    """
    Charge BDORTHO WMS - TOPO WFS et stocke l'ensemble de l'emprise dans le sous répertoire ORTHO_WMS
    Enregistre l'image 200mx200m- 1000x1000 pixels
    un csv des bâtiments pour chaque image
    un fichier récap des images + nombre de batiments

    :param bounds: (emprise totale à considérer)
    :param reload: indicates if we want to reload all files or not (default False)
    :return: void
    """
    # 5km sur 5km produit en principe 625 fichiers images sauf qu'on exclut les tuiles sans bâtiments

     # Vérifier si le chemin d'accès existe
    if not os.path.exists(filepath):
    # Si le chemin d'accès n'existe pas, le créer
        os.makedirs(filepath)
    try:
        df = pd.read_csv(filepath + 'index.csv', index_col=0)
    except:
        df = pd.DataFrame(columns=['fichier', 'nb_batiments'])

    for x in range(bds[0], bds[2], 200):
        for y in range(bds[1], bds[3], 200):
            bounds = (x, y, x+200, y+200)
            filerootname = 'ORTHOHR-' + str(bounds[0]) + '-' + str(bounds[1]) + '-LA93-0M20-1000x1000'
            if (reload) or (not os.path.isfile(filepath+filerootname+'.jpg')):
                try:
                    batiments = load_batiments(bounds)
                    nb_batiments = batiments.shape[0]
                    print(filerootname, ':', nb_batiments, 'batiments récupérés')
                    # if (nb_batiments > 0):
                    ortho = load_bdortho(bounds)
                    cv2.imwrite(filepath + filerootname + '.jpg', ortho)
                    batiments.to_csv(filepath + filerootname + '.csv')  # on garde le détail des batiments en csv
                    df.loc[len(df)] = [filerootname, nb_batiments] # append row
                except Exception as e:
                    print('echec avec le message: ' + str(e))
            else:
                print(filerootname, 'file exists => skipping')
    get_ortho_index(filepath)


def batch_load_rgealti(bds, reload=False, filepath = '../data/RGEALTI/'):
    """
    Charge BD RGEALTI (MNT)
    Enregistre sous forme de numpy 200mx200m- 1000x1000 pixels
    ATTENTION les altitudes  sont des float

    :param bounds: (emprise totale à considérer)
    :param reload: indicates if we want to reload all files or not (default False)
    :return: void (store files)
    """
    if not os.path.exists(filepath):
    # Si le chemin d'accès n'existe pas, le créer
        os.makedirs(filepath)
    for x in range(bds[0], bds[2], 200):
        for y in range(bds[1], bds[3], 200):
            bounds = (x, y, x+200, y+200)
            filerootname = 'ALTI-' + str(bounds[0]) + '-' + str(bounds[1]) + '-LA93-0M20-1000x1000'
            if (reload) or (not os.path.isfile(filepath+filerootname+'.jpg')):
                try:
                    alti = load_rgealti(bounds)
                    np.save(filepath + filerootname + '.npy', alti)
                except Exception as e:
                    print('echec avec le message: ' + str(e))
            else:
                print(filerootname, 'file exists => skipping')


def batch_load_lidar_hd(bounds, reload=False, filepath='../data/LIDAR_HD/LAZ/'):
    """
        Télécharge les dalles LIDAR correspondant aux boundaries (en Lambert 93)
        S'appuie pour cela de la liste des urls complètes du fichier  ../data/LIDAR_URL_FRANCE_ENTIERE.csv
        Le nom du fichier contient les coordonnées de la dalle LIDAR - c'est sur cette base qu'on va sélectionner ce qu'on télécharge
        TODO on pourrait prendre le centroid ou l'intersection avec le cadre - à voir ?
    """
    if not os.path.exists(filepath):
    # Si le chemin d'accès n'existe pas, le créer
        os.makedirs(filepath)
    urls = pd.read_csv('../data/LIDAR_URLS.csv')
    # extraction des coordonnées
    urls['x'] = urls.fichier.apply(lambda x: x[8:12]).astype('int')*1000
    urls['y'] = urls.fichier.apply(lambda x: x[13:17]).astype('int')*1000

    # rajout zone sécurité de 1000 pour s'assurer de prendre les urls dont l'origine n'est pas dans les bounds
    # le x y de l'url est le point en haut à gauche de le dalle 1000m 1000m donc il faut ajouter une sécu
    urls = urls.loc[(urls.x >= bounds[0]) & (urls.x <= bounds[2] + 1000) & (urls.y >= bounds[1] - 1000) & (urls.y <= bounds[3])]

    print(urls.shape[0], ' urls sélectionnées ')
    for idx, url in urls.iterrows():
        if not os.path.isfile(filepath + url.fichier) or reload:
            request = url.url
            print('loading', url.fichier, ' ... please wait')
            try:
                response = requests.get(request)  # appel au service
                if response.status_code == 200:
                    with open(filepath+url.fichier, "wb") as file:
                        file.write(response.content)
                        print("File downloaded successfully!")
                        file.close()
                else:
                    print("Failed to download the file.")
            except:
                # on error just print don't stop
                print(f'error during download of {url.fichier}')

        else:
            print(url.fichier, 'already downloaded. skipping...')


def download_las_file(name, url, filepath):
    """
    :param name: le nom du fichier
    :param url: url du fichier
    :param file_path: path où sauvegarder le fichier
    :return: message
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    fullpath = os.path.join(filepath, name)
    if not os.path.isfile(fullpath):
        request = url
        print('loading', name, ' ... please wait' , end = '...')
        try:
            response = requests.get(request)  # appel au service
            if response.status_code == 200:
                with open(fullpath, "wb") as file:
                    file.write(response.content)
                    print("File downloaded successfully!")
                    file.close()
            else:
                print("Failed to download the file.")
        except:
            # on error just print don't stop
            print(f'error during download of {name}')
    else:
        print(f'{name} - using cache file')
    return fullpath


def load_lidar_urls(bounds):
    """"
    Renvoie la liste des urls LIDAR correspondant aux boundaries (passées en en Lambert 93)

    """
    minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]
    request = ("https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=ta_lidar-hd:dalle&outputFormat=application/json&bbox=")
    request += str(minx)+','+str(miny)+','+str(maxx)+','+str(maxy)
    response = requests.get(request)  # appel au service
    urls = []
    for feature in response.json()['features']:
        urls.append(feature['properties'])
    return urls


def load_lidar_url(point, crs=4326):
    """"
    Renvoie l'url du fichier LIDAR correspondant au point (latitude, longitude) (par défaut )passé en paramètre
    :param point (x,y) en crs 4326 par défaut
    :param crs - si point est dans une autre crs
    return dictionnary {'name': ... , 'url': ...  }
    """
    if crs != 2154:
        transformer = pyproj.Transformer.from_crs(crs, 2154) # le point est converti en 2154
        point = transformer.transform(point[0], point[1])
    minx, miny, maxx, maxy = point[0], point[1], point[0], point[1]
    request = ("https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=ta_lidar-hd:dalle&outputFormat=application/json&bbox=")
    request += str(minx)+','+str(miny)+','+str(maxx)+','+str(maxy)
    response = requests.get(request)  # appel au service
    try:
        return response.json()['features'][0]['properties']
    except:
        pass
    return {}


if __name__ == '__main__':
    import sys
    from ast import literal_eval as make_tuple

    mode =''
    if len(sys.argv) > 1:
        bds = make_tuple(sys.argv[1])
        mode = 'all'
        if len(sys.argv) > 2:
            mode = sys.argv[2] # modes = both, ortho, lidar
    else:
        print('usage ', "python IGN_API_utils.py (xmin,ymin,xmax,ymax) ['all','lidar','ortho','alti']\n"
                        "ex: python IGN_API_utils.py '(525000,6700000,530000,6705000)' 'all'" )

    if mode == 'all':
        batch_load_ortho_files(bds)
        batch_load_lidar_hd(bds)

    elif mode == 'lidar':
        batch_load_lidar_hd(bds)
    elif mode == 'ortho':
        batch_load_ortho_files(bds)
    elif mode == 'alti':
        batch_load_rgealti(bds)
    else:
        print('mode must be one of the following: all, lidar,ortho or alti')

    # bds = (520000,6700000, 525000, 6705000) # attention mutiple de 200 requis (1000 c'est même mieux)
    # exemple alti
    # $python ./IGN_API_utils.py "(520000,6700000,530000,6705000)" alti
