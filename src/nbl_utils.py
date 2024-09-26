import os
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon
import numpy as np

def verif_completude_data(ville, rep_mask ='MASK'):
    """Cette fonction isdentifie, pour une Ville donnée, les boundaries pour lesquelles 
    nous n'avons pas une image ORTHO pour une image mask LIDAR. Elle retourne également la liste des boundaries pour
    lesquelles on a un fichier dans chaques répertoire (3 outputs)
    input:
        ville: strig. Correspond à la ville choisie (data organisées par ville)
    outputs:
        im_ortho_manquantes: liste des boundaries manquantes  (images ORTHO manquantes)
        m_mask_manquantes :liste des boundaries manquantes  (images MASK LIDAR manquantes)
        im_communes: liste des boundaries pour lesquelles on a une image ORTHO et un mask LIDAR associé
    """
    import re
    directory1 = f'../data/{ville}/ORTHO_WMS/'
    directory2 = f'../data/{ville}/IMG/{rep_mask}/'
    liste_fich_im_ortho = [f for f in os.listdir(directory1) if os.path.isfile(os.path.join(directory1, f)) and f.lower().endswith('.jpg', )]
    liste_fich_im_mask = [f for f in os.listdir(directory2) if os.path.isfile(os.path.join(directory2, f)) and f.lower().endswith('.png', )]
    im_ortho_count = len(liste_fich_im_ortho)
    im_mask_count = len( liste_fich_im_mask )
    print(f"Nombre d'images ortho pour {ville}: {im_ortho_count}")
    print(f"Nombre d'images mask pour {ville}: {im_mask_count}")
    liste_emprises_ortho = [re.search(r"(\d+-\d+)", f).group(1) for f in liste_fich_im_ortho]
    liste_emprises_mask =  [re.search(r"(\d+-\d+)", f).group(1) for f in liste_fich_im_mask]
    im_communes = [element for element in liste_emprises_mask if element in liste_emprises_ortho]
    if  im_ortho_count != im_mask_count:
       im_ortho_manquantes = [element for element in liste_emprises_mask if element not in liste_emprises_ortho]
       im_mask_manquantes = [element for element in liste_emprises_ortho if element not in liste_emprises_mask]  
       print(f'Il y a {len(im_ortho_manquantes)} images ortho manquantes et {len(im_mask_manquantes)} images mask manquantes')
       print ('')    
    return sorted(im_ortho_manquantes), sorted(im_mask_manquantes), sorted(im_communes)


def recup_df_chemins_ortho_topo_mask(image_paths,rep_mask = 'MASK_MIX',index_pos_ville_chemin = 2):
    """Cette fonction construit le df qui, à chaque chemin d'image ORTHO de image_paths, 
    associe son mask dans le fichier annotation_img_paths. On ne conserve que les paires completes dans le df
    input:
        image_paths: chemins vers les images ORTHO 
        index_pos_ville_chemin : index de la position du nom de ville dans le chemin .
            Il se trouve logiquement en 2ieme position si on splite avec les '/'
    outputs:
        df : dataframe contenant les chemins vers les images (BD_ORTHO),les csv (BD_TOPO) et les chemins vers les masques associés
    LIDAR_HD
    """
    df = pd.DataFrame()
    villes_train = []

    for path in image_paths:
        vil = path.split('/')[index_pos_ville_chemin] 
        villes_train.append(vil)
    for ville in villes_train:
        boundaries_to_use = verif_completude_data(ville)[2]
        IMG_PATH = f'../data/{ville}/ORTHO_WMS/'
        ANNOTATION_PATH =  f'../data/{ville}/IMG/{rep_mask}/'
        input_img_paths = sorted([
            os.path.join(IMG_PATH, fname)
            for fname in os.listdir(IMG_PATH)
            if (fname.endswith(".jpg")) and any(ref in fname for ref in boundaries_to_use)
                                ])
        imput_csv_paths = sorted([
            os.path.join(IMG_PATH, fname)
            for fname in os.listdir(IMG_PATH)
            if (fname.endswith(".csv")) and any(ref in fname for ref in boundaries_to_use)
                                ])
        nb_bat = [pd.read_csv(file).shape[0] for file in imput_csv_paths]           
        annotation_img_paths = sorted([
                os.path.join(ANNOTATION_PATH, fname)
                for fname in os.listdir(ANNOTATION_PATH)
                if fname.endswith(".png") and not fname.startswith(".") and any(ref in fname for ref in boundaries_to_use)
                                    ])
        df_ville = pd.DataFrame(zip(input_img_paths, imput_csv_paths,nb_bat,annotation_img_paths),
                                 columns=['image','bd_topo','nb_bat','masque'])
        df = pd.concat([df,df_ville])
        df = df.reset_index(drop=True)
    return df


