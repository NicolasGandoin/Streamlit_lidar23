import pandas as pd
import numpy as np
import cv2
import shapely
import re
import os
import yaml
import glob
import random
import shutil
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon
from PIL import Image, ImageDraw
from src.las_utils import get_filename


def get_roof_label(cleabs, datadir = '../data', ignore_labels = ['0_indetermine']):
    """
    Renvoie l'étiquette de classification manuelle du toit donné par id_toit
    Les images sont considérée comme classifiées si elles sont dans un sous répertoire de data/[VILLE]/IMG/ROOFS/
    Elles sont donc recherchées pas un glob à partir de data "*/IMG/ROOFS/*/batiment_id.png"
    L'avant dernier niveau est donc l'étiquette
    Attention cette fonction renvoie la première occurence de batiment trouvée.
    Si le batiment est présent dans plusieurs sous répertoire cela est ignoré
    :param batiment_id: cleabs from BDTOPO
    :param datadir: path du fichier de data
    :param ignore_labels liste des labels à ignorer
    :return: label (from roof_labels)
    """
    classified_paths = Path(datadir).glob(f'*/IMG/ROOFS/*/{cleabs}.png')  # on cherche les sous répertoires
    # en principe il n'y en a qu'un mais on prend le premier
    try:
        path = next(classified_paths)
        label = path.parent.name
        if label in ignore_labels:
            return None
    except:
        return None # no value in glob

    return label

def get_dic_labels(datadir = '../data',label='roofs'):
     """renvoie le dictionnaire avec comme clé le nom du répertoire et, comme valeur, le numéro du label.
    (attendue dans le fichier txt de config. pour YOLO")
    """
     dic={}
     if label=='roofs':
        label_paths = Path(datadir).glob('*/IMG/ROOFS/*/')
        for index, path in enumerate(label_paths):
            if path.is_dir():
                directory_name = path.name  # Récupère le nom du répertoire
                dic[directory_name] = index
     elif label=='buildings':
        dic={'buildings':0}
     return dic

def clean_and_create_dirs(directories):
        for dir_path in directories:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def yolo_train_test_valid(labels_dir, images_dir,valid_labels_dir=None,valid_images_dir=None,model='yolov8_seg', test_size=0.2, nb_images=None, nb_max_objets=None):
    '''
    La fonction va, à partir du répertoire contenant les fichiers .txt de labellisation et celui 
    des fichiers images associées , créer les datasets train/ test ainsi qu'un dataset de validation si 
    des répertoires pour le jeu de validation sont renseignés.
    La paramètre model contient le modèle utilisé qui sera présent dans le nom
    Le paramètre nb_max_objets permettra de fixer un nombre d'objet (batiments:roofs) maximum par image .
    Les images contenant plus de nb_max_objets ne seront pas prises en compte pour constituer le dataset 
    La fichier retournera le nom du répertoire du dataset.
    '''
    # dir =f'../data/{ville}/'
    # base_dir = f'{dir}ORTHO_WMS/'
    
    # Récupération des paramètres yolo dans le nom du fichier txt
    pattern = r'/YOLO/([^/]+)/'
    match = re.search(pattern, labels_dir)
    labels_dir_name = match.group(1)
    # Répertoires de destination
    base_dest = f'../data/YOLO/datasets/{labels_dir_name}_{model}_nbimg{nb_images}_maxobj{nb_max_objets}'
    train_img_dir = os.path.join(base_dest, 'images', 'train')
    test_img_dir = os.path.join(base_dest, 'images', 'test')
    valid_img_dir = os.path.join(base_dest, 'images', 'valid')
    train_label_dir = os.path.join(base_dest, 'labels', 'train')
    test_label_dir = os.path.join(base_dest, 'labels', 'test')
    valid_label_dir = os.path.join(base_dest, 'labels', 'valid')

    # Créer les répertoires de destination s'ils n'existent pas déjà. S'ils existent, on les vide pour éviter les erreurs et les doublons (lancement plusieurs fois)
    dirs_to_clean = [train_img_dir, test_img_dir, valid_img_dir, 
                    train_label_dir, test_label_dir, valid_label_dir]
    clean_and_create_dirs(dirs_to_clean)
   
    # récup. fichiers annotations 
    txt_files = glob.glob(os.path.join(labels_dir,'*.txt'))

    # Si on veut limiter le nb de bâtiments/toits max par image (param nb_max_objets):
    if nb_max_objets == None:
        txt_files=txt_files
    else:
        filtered_txt_files = []
        for file_path in txt_files:
            with open(file_path, 'r') as file:
                num_lines = sum(1 for line in file)
                if num_lines <= nb_max_objets:
                    filtered_txt_files.append(file_path)
        txt_files = filtered_txt_files      

    # Obtenir la liste des fichiers .jpg qui correspondent aux labels générés)   
    jpg_files=[]
    for pathfile in txt_files:
        name = os.path.basename(pathfile)
        jpg_name = get_filename(name, 'ORTHOHR', '.jpg')
        jpg_file_path = f'{images_dir}{jpg_name}'
        jpg_files.append(jpg_file_path)
    
    
    #si on veut limiter à nb_images images pour restreindre le temps d'entrainement
    if not nb_images==None:
        jpg_files = jpg_files[:nb_images]
    else:
        jpg_files=jpg_files

   # Mélanger les fichiers pour une distribution aléatoire
    random.shuffle(jpg_files)

    # Séparer (1_testsize) % pour l'entraînement et test_size pour le test pour nb_images au total
    split_index = int((1-test_size)* len(jpg_files))
    train_files = jpg_files[:split_index]
    test_files = jpg_files[split_index:]

    # jeu de validation
    if valid_images_dir != None:
        valid_files = glob.glob(os.path.join(valid_images_dir,'*.jpg'))
    else:
        valid_files=[]   

    def copy_files(files,labels_dir,img_dest, label_dest):
        for img_file in files:
            # Nom de base du fichier sans l'extension
            base_name = os.path.basename(img_file).rsplit('.', 1)[0]
            
            # Chemin du fichier d'annotation correspondant
            txt_file = os.path.join(labels_dir, f'{base_name}.txt')
            
            # Copier le fichier image
            dest_img_file = os.path.join(img_dest, os.path.basename(img_file))
            if not os.path.exists(dest_img_file):
                shutil.copy(img_file, os.path.join(img_dest, os.path.basename(dest_img_file)))
        
            # Copier le fichier d'annotation s'il existe
            dest_txt_file = os.path.join(label_dest, os.path.basename(txt_file))
            if os.path.exists(txt_file) and not os.path.exists(dest_txt_file):
                shutil.copy(txt_file,os.path.join(label_dest, os.path.basename(dest_txt_file)))

    # Copier les fichiers d'entraînement
    copy_files(train_files,labels_dir, train_img_dir, train_label_dir)

    # Copier les fichiers de test
    copy_files(test_files,labels_dir, test_img_dir, test_label_dir)

    # Copier les fichiers de validation
    if valid_labels_dir!=None and len(valid_files)!=0 :
         copy_files(valid_files,valid_labels_dir,valid_img_dir,valid_label_dir)
    
    print(f'Total images train: {len(train_files)}')
    print(f'Total images test: {len(test_files)}')
    print(f'Total images valid: {len(valid_files)}')
    return os.path.basename(base_dest)


def yolo_prepare_yaml(yolo_absolute_dir_path,dataset_name,dic_labels, kaggle=False, dataset_kaggle ='lidar-dec23-tours-yolo'):
    """
    Fonction qui prépare le fichier .yaml de configuration attendu pour YOLO
    ce fichier est écrit au même endroit que le dataset concerné (path: yolo_absolute_dir_path )
    et prendra le nom du dataset et le dictionnaire des labels utilisé en paramètres.
    """
    data = {
    'path': f'{yolo_absolute_dir_path}{dataset_name}' if not kaggle else f'/kaggle/input/{dataset_kaggle}/',
    'train': 'images/train' , 
    'val': 'images/test' ,
    'test': 'images/valid',
    'nc': len(dic_labels),
    'names': ''
    }
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    with open(f'{yolo_absolute_dir_path}{dataset_name}/dataset_{dataset_name}_config.yaml', 'w') as file:
        for line in yaml_str.splitlines():
            if line.startswith('names:'):
                file.write('names:\n')
                for key, value in dic_labels.items():
                    file.write(f'  {value}: {key}\n')
            else:
                file.write(line + '\n')
            
                                          

def ENB0_image_preprocess(ortho, mask_img=None, alt_img=None, preprocess=None):
    """
    Fonction de preprocessing à appeler avant appel aux modèle NBATS EfficientNet B0
    Unifie les cas ORTHO et lidar (Lidar aura besoin des images mask et alt selon les cas (sur la même emprise)
    A appeler sur l'imge ou le jeu d'image à prédire par le modèle H5
    les cas P1 à P6 offrent des variantes basés sur de l'augmentation avec Image LIDAR
    :param ortho: Ortho image
    :param mask: Mask image
    :param alt: Altitude image
    :param preprocess :P0 à P7 #See code for description

    """

    if preprocess == 'P0':
        image = ortho # nos images ORTHO sont en BGR

    elif preprocess == 'P1':  # masque sur RGB ALT en alpha
        mask_img = mask_img // 255
        image = cv2.cvtColor(ortho, cv2.COLOR_BGR2BGRA)  # nos images ORTHO sont en BGR

        image = cv2.bitwise_and(image, image, mask=mask_img)
        alt_img = cv2.bitwise_and(alt_img, alt_img, mask=mask_img)
        image[:, :, 3] = 255 - alt_img  # ajout canal bat as alpha 255 et alt ensuite

    elif preprocess == 'P2':  # Add ALT on ALPHA
        image = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGBA)

        image[:, :, 3] = 255 - alt_img  # ajout canal bat as alpha 255 et alt ensuite

    elif preprocess == 'P3':  # Only MASK on RGB mask_img = mask_img // 255
        mask_img = mask_img // 255
        image = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)  # nos images ORTHO sont en BGR

        image = cv2.bitwise_and(image, image, mask=mask_img)

    elif preprocess == 'P4':  # ALT on MASK as ALPHA only
        mask_img = mask_img // 255
        alt_img = cv2.bitwise_and(alt_img, alt_img, mask=mask_img)
        image = cv2.cvtColor(ortho, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = alt_img  # ajout canal bat as alpha 255 et alt ensuite

    elif preprocess == 'P5':  # just decrease RGB intensity on lower alt
        # image = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)  # nos images ORTHO sont en BGR
        image = (image * (alt_img[:, :, np.newaxis] / 255.0)).astype(np.uint8)

    elif preprocess == 'P6':  # Only altitude within mask Gray image  # The MUST ?
        mask_img = mask_img // 255
        alt_img = cv2.bitwise_and(alt_img, alt_img, mask=mask_img)
        image = cv2.cvtColor(alt_img, cv2.COLOR_GRAY2RGB)

    elif preprocess == 'P7':  # Only altitude
        image = cv2.cvtColor(alt_img, cv2.COLOR_GRAY2RGB)

    elif preprocess == 'P8':  #Altitude as intensity and  mask
        # image = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)  # nos images ORTHO sont en BGR
        image = (ortho * (alt_img[:, :, np.newaxis] / 255.0)).astype(np.uint8)
        mask_img = mask_img // 255
        image = cv2.bitwise_and(image, image, mask=mask_img)

    else:
        image = ortho

    image = cv2.resize(image, (224, 224))  # EN B0 resize en 224x224

    return image


def YOLO_image_preprocess(ortho, mask_img=None, alt_img=None, preprocess=None):
    """
    Fonction de preprocessing à appeler avant appel aux modèle YOLO
    Chaque preprocess est identifié par un code qui spécifie les transformations à apporter
    P0 ou None => on ne prend que ortho
    :param ortho: Ortho image
    :param mask: Mask image
    :param alt: Altitude image
    :param preprocess :P0 à Pn #Voir dans le code
    Attention certaines images vont sortir sur 4 canaux - documenter pour adaptation entrée yolo

    """

    if preprocess == 'P0':
        image = ortho # nos images ORTHO sont en BGR

    elif preprocess == 'P1':  # masque sur RGB ALT en alpha
        mask_img = mask_img // 255
        image = cv2.cvtColor(ortho, cv2.COLOR_RGB2RGBA)  # nos images ORTHO sont en BGR
        image = cv2.bitwise_and(image, image, mask=mask_img)
        alt_img = cv2.bitwise_and(alt_img, alt_img, mask=mask_img)
        image[:, :, 3] = 255 - alt_img  # ajout canal bat as alpha 255 et alt ensuite

    elif preprocess == 'P2':  # Add ALT on ALPHA
        image = cv2.cvtColor(ortho, cv2.COLOR_RGB2RGBA)

        image[:, :, 3] = 255 - alt_img  # ajout canal bat as alpha 255 et alt ensuite

    elif preprocess == 'P3':  # Only MASK on RGB mask_img = mask_img // 255
        mask_img = mask_img // 255
        image = cv2.cvtColor(ortho, cv2.COLOR_RGB2RGBA)  # nos images ORTHO sont en BGR

        image = cv2.bitwise_and(image, image, mask=mask_img)

    elif preprocess == 'P4':  # ALT on MASK as ALPHA only
        mask_img = mask_img // 255
        alt_img = cv2.bitwise_and(alt_img, alt_img, mask=mask_img)
        image = cv2.cvtColor(ortho, cv2.COLOR_RGB2RGBA)
        image[:, :, 3] = alt_img  # ajout canal bat as alpha 255 et alt ensuite

    elif preprocess == 'P5':  # just decrease RGB intensity on lower alt
        # image = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)  # nos images ORTHO sont en BGR
        image = (image * (alt_img[:, :, np.newaxis] / 255.0)).astype(np.uint8)

    elif preprocess == 'P6':  # Only altitude within mask Gray image  # The MUST ?
        mask_img = mask_img // 255
        alt_img = cv2.bitwise_and(alt_img, alt_img, mask=mask_img)
        image = cv2.cvtColor(alt_img, cv2.COLOR_GRAY2RGB)

    elif preprocess == 'P7':  # Only altitude
        image = cv2.cvtColor(alt_img, cv2.COLOR_GRAY2RGB)

    elif preprocess == 'P8':  #Altitude as intensity and  mask
        # image = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)  # nos images ORTHO sont en BGR
        image = (ortho * (alt_img[:, :, np.newaxis] / 255.0)).astype(np.uint8)
        mask_img = mask_img // 255
        image = cv2.bitwise_and(image, image, mask=mask_img)

    else:
        image = ortho


    return image



class SpatialMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Classe qui implémente un scaler minmax sur toute les variable des données
    à l'exception de Z qui est standardisée localement (dans une fenêtre de
    10m x 10m par défaut) (1000cm)
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            columns = [f'var_{i}' for i in range(X.shape[1])]
            columns[:3] = ['X', 'Y', 'Z']
            df = pd.DataFrame(X, columns=columns)
        else:
            df = X.copy()

        # Create grid columns
        df['grid_x'] = (df['X'] // self.window_size).astype(int)
        df['grid_y'] = (df['Y'] // self.window_size).astype(int)

        # Calculate min and max for Z in each grid cell
        self.z_min_max_per_grid = df.groupby(['grid_x', 'grid_y'])['Z'].agg(['min', 'max']).reset_index()

        # Calculate global min and max for all columns
        self.global_min_max = df.drop(columns=['grid_x', 'grid_y']).agg(['min', 'max'])

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            columns = [f'var_{i}' for i in range(X.shape[1])]
            columns[:3] = ['X', 'Y', 'Z']
            df = pd.DataFrame(X, columns=columns)
        else:
            df = X.copy()

        # Create grid columns
        df['grid_x'] = (df['X'] // self.window_size).astype(int)
        df['grid_y'] = (df['Y'] // self.window_size).astype(int)

        # Merge with the min and max values per grid
        df = df.merge(self.z_min_max_per_grid, on=['grid_x', 'grid_y'], how='left', suffixes=('', '_grid'))

        # Normalize Z on the grid
        df['Z_normalized'] = (df['Z'] - df['min']) / (df['max'] - df['min'])
        df['Z_normalized'] = np.where(df['max'] != df['min'], df['Z_normalized'], 0)

        # Normalize all other columns and ensure compatibility with dtype
        for col in df.columns.drop(['X', 'Y', 'Z', 'grid_x', 'grid_y', 'min', 'max', 'Z_normalized']):
            df[col] = df[col].astype(float)  # Ensure compatibility
            df[col] = (df[col] - self.global_min_max.loc['min', col]) / (self.global_min_max.loc['max', col] - self.global_min_max.loc['min', col])

        # Drop grid columns and temporary min/max columns
        df.drop(columns=['grid_x', 'grid_y', 'min', 'max'], inplace=True)

        # Ensure the 'Z' column is replaced with the normalized 'Z' values
        df['Z'] = df['Z_normalized']
        df.drop(columns=['Z_normalized'], inplace=True)

        return df


if __name__ == '__main__':

    # exemple d'utilisation de roof label avec apply
    import src.IGN_API_utils as api

    bounds = ((520000, 6700000, 530000, 6705000))  # TOURS
    batiments = api.load_batiments(bounds)
    batiments['label'] = batiments.cleabs.apply(lambda x: get_roof_label(x))
    print(batiments.label.notna().sum() / batiments.shape[0])  # proportion étiquetée