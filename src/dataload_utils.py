"""
script qui chaine l'exécution du downlaod de toutes la data pour des boundaries particulières
roothpath: sous répertoire de data correspondant à la ville de l'emprise (cf boundaries), à renseigner après la

"""
import os
import sys
sys.path.append('../')
import src.IGN_API_utils as api
import src.las_utils as las# pour la génération des images lidar


if __name__ == '__main__':

    from ast import literal_eval as make_tuple
    rootpath = ''  #valid for validation data
   
    if len(sys.argv) > 1:
        bds = make_tuple(sys.argv[1])
        if len(sys.argv) > 2:
            rootpath = sys.argv[2]+'/'
        # Si les chemins d'accès n'existent pas, les créer
        if not os.path.exists('../data/'+rootpath+'ORTHO_WMS/'):
            os.makedirs('../data/'+rootpath+'ORTHO_WMS/')
        if not os.path.exists('../data/'+rootpath+'LIDAR_HD/'):
            os.makedirs('../data/'+rootpath+'LIDAR_HD/')
        if not os.path.exists('../data/'+rootpath+'LIDAR_HD/LAZ/'):
            os.makedirs('../data/'+rootpath+'LIDAR_HD/LAZ/')
        api.batch_load_ortho_files(bds, filepath='../data/'+rootpath+'ORTHO_WMS/')
        api.batch_load_lidar_hd(bds, filepath='../data/'+rootpath+'LIDAR_HD/LAZ/')
        las.run_all_preprocesses(lidardir='../data/'+rootpath+'LIDAR_HD/')
        #las.batch_create_lidaralt_images('../data/'+rootpath+'LIDAR_HD/') # TODO reactivate after height modif
    else:
        print('usage ', "python dataload_utils.py (xmin,ymin,xmax,ymax) root_data_path\n"
                        "ex: python dataload_utils.py '(525000,6700000, 530000, 6705000)' TOURS" )


    # bds = (520000,6700000, 530000, 6705000) # attention mutiple de 200 requis (1000 c'est même mieux)
    # exemple alti
    # $python ./dataload_utils "(520000,6700000,530000,6705000)"


    # python dataload_utils.py '(845000,6425000,855000,6430000)' valence

    # EXEMPLE AIX POUR VALIDATION
    # python dataload_utils.py '(893000,6270000,901000,6275000)' AIX
