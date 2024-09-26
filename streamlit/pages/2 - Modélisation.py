import streamlit as st
import folium
from streamlit_folium import st_folium

import pandas as pd
import pyproj
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
rootdir = os.getcwd()

import src.topo_utils as tu
#import src.vis_utils as vu 
import src.las_utils as lu
import src.raster_utils as ru
import src.IGN_API_utils as api
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model    
from ultralytics import YOLO
# Ajout barre de recherche
from folium.plugins import Geocoder
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(
    page_title="LidarDec23 - Exploration",
    page_icon="🗺️",
    layout="wide")

st.title('Modélisation, comptage et détection des bâtiments')

@st.cache_data
def load_batiments(bounds):
    df = api.load_batiments(bounds)
    df['superficie'] = round(df.geometry.apply(lambda g: g.area),2)
    return df

@st.cache_data
def load_ortho_img(bounds):
    print(f'load_ortho_img {bounds}')
    img = api.load_bdortho(bounds)
    return img

@st.cache_resource
def load_models():
    modLN = tf.keras.models.load_model('./models/NBATS/NGA_ST_Lenet_simple_maetest_18_1.h5')
    modRN = tf.keras.models.load_model('./models/NBATS/NGA_ST_NBATS_ResNetV2_notransfer_mae_test_9-89.h5')
    modEN = tf.keras.models.load_model('./models/NBATS/BDA_ST_NBATS_ENB0_mae_7-36_finetune.h5')
    modelYO = YOLO('models/NBATS/NGA_ST_YOLO_v2_2500img_300ep.pt')
    return modLN, modRN, modEN, modelYO

def generate_map(gdf):
    # tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m = gdf.explore(column='usage_1', popup=["cleabs", "nature", "usage_1", "superficie", "hauteur"], # commande gdf.explore generates an interactive leaflet map based on GeoDataFrame
    tooltip=["cleabs", "nature", "usage_1", "superficie", "hauteur"], cmap="Blues")    
    return m

@st.cache_data
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@st.cache_data
def save_and_display_gradcam(img_array, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = img_array[0,:,:,:]

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    return superimposed_img


### Initialisation de la page 
starting_point = (45.7459, 4.8850)  # latitude longitude initiale
transformer = pyproj.Transformer.from_crs(4326, 2154) #transformer GPS en La93
la93 = transformer.transform(starting_point[0], starting_point[1])
# Tests pour les paramètres passés dans st.session_state : bounds, batiments et img 
if 'bounds' not in st.session_state:
    st.session_state.bounds = la93[0] - 100, la93[1] - 100, la93[0] + 100, la93[1] + 100
if 'batiments' not in st.session_state:
    batiments = load_batiments(st.session_state.bounds)
    st.session_state.batiments = batiments
if 'img' not in st.session_state:
    st.session_state.img= api.load_bdortho(st.session_state.bounds)

# Chargement des modèles 
if 'modLN' not in st.session_state:
    st.session_state.modLN, st.session_state.modRN, st.session_state.modEN, st.session_state.modelYO = load_models()


tab1, tab2, tab3  = st.tabs(['Choix de l\'emprise', 'Comptage des bâtiments', 'Segmentation Instances'])

# Onglet choix de l'emprise
with tab1:
    st.write("Sélectionner une zone sur la carte et cliquer sur Select pour sélectionner l'emprise à utiliser. On visualisera à droite l'image satellite correspondante")
    st.write("Attention, les modèles ont tous été entraînés sur des images 200m x 200m. Ne pas lancer de prédiction sans avoir cliqué sur Select")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(height=550, border=0):
            m = generate_map(st.session_state.batiments)
            invtransformer = pyproj.Transformer.from_crs(2154, 4326)
            SW = invtransformer.transform(st.session_state.bounds[0], st.session_state.bounds[1])
            NE = invtransformer.transform(st.session_state.bounds[2], st.session_state.bounds[3])
            # Ajout du rectangle à la carte Folium
            folium.Rectangle(bounds=[SW, NE], color="blue", fill=True, fill_color="blue", fill_opacity=0.2).add_to(m)
            # Ajout barre de recherche
            Geocoder().add_to(m)
            map_ = st_folium(m, width=500, height=500)

        with st.container(height=550, border=0):
            if st.button('Select'):
                # Recupération de la zone bounds à partir des coordonnées du centre de la carte
                center = transformer.transform(map_['center']['lat'], map_['center']['lng'])
                st.session_state.bounds = center[0] - 100, center[1] - 100, center[0] + 100, center[1] + 100
                sel_bounds= st.session_state.bounds      
                # Chargement des bâtiments dans la nouvelle zone
                st.session_state.batiments = load_batiments(st.session_state.bounds)
                m = generate_map(st.session_state.batiments)
                #Chargement de l'image ORTHO correspondant a la zones
                st.session_state.img = api.load_bdortho(st.session_state.bounds)
                # Transformation inverse des coordonnées pour SW et NE pour tracer la box sur la carte
                invtransformer = pyproj.Transformer.from_crs(2154, 4326)
                SW = invtransformer.transform(st.session_state.bounds[0], st.session_state.bounds[1])
                NE = invtransformer.transform(st.session_state.bounds[2], st.session_state.bounds[3])
                # Ajout du rectangle à la carte Folium
                folium.Rectangle(bounds=[SW, NE], color="blue", fill=True, fill_color="blue", fill_opacity=0.2).add_to(m)
                # Affichage de la carte avec le rectangle
                map_ = st_folium(m, width=500, height=500)
    
    with col2:
        with st.container(height=550, border=0):
            nb_batiments = st.session_state.batiments.shape[0]
            st.write('Nombre de bâtiments :', nb_batiments)
            # Chargement et affichage de l'image orthophoto correspondante
            fig = plt.figure(figsize=(3, 3))
            plt.clf()
            plt.imshow(st.session_state.img)
            plt.axis('off')
            plt.ylim([0, 1000])
            st.pyplot(fig, use_container_width=False)
            # affichage dans la sidebar pour garder entre les onglets
            st.sidebar.write('Nombre de bâtiments :', nb_batiments)
            st.sidebar.pyplot(fig) 

# Onglet comptage des bâtiments
with tab2:
    st.header("Choix des modèles CNN")
    section = st.radio('On compare les résultats de comptage de bâtiments pour 3 CNN', ['LeNet', 'ResNet', 'EfficientNet' ])

    # Problème de chargement des modèles lié à une version tensorflow trop vieille (2.12 vs 2.15 pour Kaggle)
    #st.write(keras.__version__) #3.5
    #st.write(tf.__version__)    #2.17
    #if st.button('Continue'): # Bouton supprimé parce que le gain de temps ne justifie pas la perte en ergonomie.
    if section == "LeNet":  
        model = st.session_state.modLN 
        MAE_val = 43.08
    if section == "ResNet":
        model = st.session_state.modRN 
        MAE_val = 9.89
    if section == "EfficientNet":
        model = st.session_state.modEN 
        MAE_val= 7.36
    
    IMG_SIZE = 224 # devrait être paramétrable selon le modèle ? 
    image = st.session_state.img
    nb_bat = nb_batiments
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    X=[]
    X.append(np.array(image))
    X = np.array(X)
    Y = int(nb_bat)
    Y_pred = model.predict(X).round()
    col1, col2 = st.columns(2)

    with col1:
        with st.container(height=500, border=0):
            st.write('Nombre total de paramètres du modèle (en millions):',model.count_params()/1000000)
            st.write('MAE moyenne Test (Aix en Provence):', MAE_val)
            model.summary(print_fn=lambda x: st.text(x))
    #Ajout gradcam pour la zone observée
    with col2:                    
        st.write('Nombre de bâtiments Réels', nb_batiments, ' / Prédits:', int(Y_pred[0][0]),  ', Mean Absolute Error', int(np.abs(Y_pred[0][0]-Y)))
        st.write("Gradcam pour l'image sélectionnée")
        img_array = X
        # Print what the top predicted class is
        # preds = model.predict(img_array).round()
        # Generate class activation heatmap, on prend la dernière couche de convolution de chaque modèle
        if section == "LeNet":
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_26")
        if section == "ResNet":
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out")
        if section == "EfficientNet":
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv")
        # Superpose heatmap and image
        result = save_and_display_gradcam(img_array, heatmap)
        fig=plt.figure(figsize=(3, 3))
        plt.imshow(result/255)
        plt.axis('off')
        plt.ylim([0, 224])
        st.pyplot(fig, use_container_width=False)

# Onglet YOLO
with tab3:
    st.header("Résultat avec modèle Yolo v8 Seg")    
    st.write("On utilise un modèle Yolo v8 pour détecter les bâtiments et leurs contours")
    # Ajout bouton Calculate pour éviter le calcul à chaque manip de la page
    if st.button('Calculate'):
        # Définir la taille d'image pour YOLO
        IMG_SIZE_YOLO = 640 

        # Charger l'image (supposons que 'img' est fourni)
        image = cv2.resize(st.session_state.img, (IMG_SIZE_YOLO, IMG_SIZE_YOLO))

        # Convertir l'image BGR en RGB pour l'afficher correctement avec matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Effectuer une prédiction avec le modèle
        results1 = st.session_state.modelYO.predict(image_rgb)

        # Pour chaque résultat, dessiner les boîtes sur l'image
        for result in results1:
            boxes = result.boxes  # Obtenir les boîtes de détection
            nb_box=0              # Compteur nombre de box donc batiment détecté
            # Dessiner les boîtes de détection sur l'image
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extraire les coordonnées
                confidence = box.conf[0]  # La confiance du modèle sur la détection
                label = f'{confidence:.2f}'  # Formatage du label avec confiance

                # Dessiner la boîte sur l'image
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Rectangle vert
                cv2.putText(image_rgb, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, 
                            cv2.LINE_AA,True) #Il faut flipper le texte puisque l'image est affichée flippée par la suite
                nb_box +=1
                
        # Afficher l'image modifiée avec les boîtes de détection
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Convertir l'image RGB en BGR pour l'afficher correctement avec matplotlib
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ax.imshow(image_rgb)
        ax.axis('off')  # Masquer les axes
        ax.set_ylim([0,640]) # flip image
        # Afficher la figure avec Streamlit
        st.pyplot(fig,use_container_width=False)
        st.write("Nombre de bâtiments détectés :", nb_box)