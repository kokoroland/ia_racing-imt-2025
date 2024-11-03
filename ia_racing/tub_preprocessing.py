__author__ = "KOKO"
__credits__ = ["KOKO", "MATHEO", "FATE"]
__date__ = "2024.11.03"
__version__ = "0.0.1"

'''
    Fichier pour prétraiter un tub pour entraîner un modèle avec différents preprocessings
    
    Structure de l'enregistrement :
        target_tub/
        |
        |-- prepro1/
        |   |-- images/
        |   |   |-- img1
        |   |   |-- …
        |   |-- catalog1
        |   |-- catalog…
        |   |-- manifest.json
    …
'''

############### IMPORTS ###############

import argparse # Arguments
import json # Utilisation du json
import math # Arrondis
import os # Chemins
import shutil # Copie de fichiers
import cv2 # Preprocessings
import numpy as np # La base

############### PARAMETRES ###############

TUBS_MASTER = '/home/ia-racing/ia_racing/data_all'
CROP = 40
PREPRO = ['lines', 'bnw'] # Prepro existant : lines|bnw
TARGET_DIR = os.path.join(os.getcwd(), "test_tub")

############### Classes utiles ###############

class Preprocess():
    
    ''' Classe pour prétraiter les images '''
    
    def __init__(self, dir, image, crop, method):
        self.img = cv2.imread(os.path.join(dir, image))
        self.img_name = image
        self.cropY(crop)
        if method == "lines":
            self.lines()
        elif method == "bnw":
            self.bnw()
        else:
            pass
    
    def cropY(self, crop):
        if len(np.shape(self.img)) == 3:
            self.img = self.img[crop:np.shape(self.img)[0], :, :]
    
    def gaussian(self, ksize=(3,3), sigmaX=0):
        self.img = cv2.GaussianBlur(self.img, ksize=ksize, sigmaX=sigmaX)
    
    def lines(self):
        self.gaussian()
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 150, 200, apertureSize=3)
        minLineLength = 20
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength, maxLineGap)
        
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")
        if lines is not None:  # Ajout d'une vérification pour les lignes
            for x in range(len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    pts = np.array([[x1, y1], [x2, y2]], np.int32)
                    cv2.polylines(img, [pts], True, (0, 0, 255), 3)
        img[54:, 33:128, :] = 0 # Masque pour le pare-chocs
        self.img = img
    
    def bnw(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    
    def save(self, path):
        cv2.imwrite(path, self.img)

class TubManager():
    
    ''' Classe pour faciliter la lecture '''
    
    def __init__(self, tub_dir):
        self.tub_dir = tub_dir
        self.images_dir = os.path.join(tub_dir, 'images')
        self.catalogs = []
        self.catalogs_manifests = []
        self.catalog_values = None
        self.line_lengths_values = None
        self.images = []
        self.manifest_file = os.path.join(tub_dir, 'manifest.json')
        self.load_catalogs(tub_dir)

    def load_catalogs(self, tub_dir):
        for f in os.scandir(tub_dir):
            if "catalog" in f.name:
                if "manifest" in f.name:
                    self.catalogs_manifests.append(os.path.join(tub_dir, f.name))
                else:
                    self.catalogs.append(os.path.join(tub_dir, f.name))
    
    def get_values(self):
        if len(self.catalogs) != 0:
            catalog_values = []
            for c in self.catalogs:
                with open(c, 'r') as f:
                    line = f.readline()
                    while line:
                        catalog_values.append(json.loads(line))
                        line = f.readline()
            self.catalog_values = catalog_values
            return catalog_values
        return None
    
    def get_line_lengths(self):
        if len(self.catalogs_manifests) != 0:
            line_lengths_values = []
            for c in self.catalogs_manifests:
                with open(c, 'r') as f:
                    line_lengths_values.append(json.loads(f.read())['line_lengths'])
            self.line_lengths_values = line_lengths_values
            return line_lengths_values
        return None
    
    def get_images(self):
        if self.catalog_values:
            images = []
            for c in self.catalog_values:
                img_name = c['cam/image_array']
                img_path_name = self.tub_dir
                images.append({"dir": img_path_name, "img": img_name})
            self.images = images
            return images
        return None

    def move_images(self, path, crop=40, preprocessing=None):
        if self.images:
            if not os.path.isdir(os.path.join(path, 'images')):
                os.mkdir(os.path.join(path, 'images'))
            for i in range(len(self.images)):
                new_image_name = str(i) + "_cam_image_array_.jpg"
                if preprocessing is None:
                    shutil.copy(os.path.join(self.images[i]['dir'], 'images', self.images[i]['img']), os.path.join(path, 'images', new_image_name))
                else:
                    (Preprocess(os.path.join(self.images[i]['dir'], 'images'), self.images[i]['img'], crop, preprocessing)).save(os.path.join(path, 'images', new_image_name))

        # Vérification et copie du fichier manifest.json
        if os.path.isfile(self.manifest_file):
            shutil.copy(self.manifest_file, path)
        else:
            print(f"Le fichier manifest.json n'a pas été trouvé à l'emplacement : {self.manifest_file}")

        # Copie des fichiers catalog et manifest
        for catalog in self.catalogs:
            shutil.copy(catalog, path)
        
        for manifest in self.catalogs_manifests:
            shutil.copy(manifest, path)

############### EXECUTION DU FICHIER ###############

if __name__ == "__main__":
    
    ## Récupération des arguments ##
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--from_dir", required=True, help="Dossier avec le tub à prétraiter")
    argParser.add_argument("-c", "--crop", help="Nombre de pixels à rogner par rapport au haut")
    argParser.add_argument("-p", "--preprocessing", nargs="*", help="Liste des prétraitements à utiliser")
    argParser.add_argument("-t", "--target", help="Chemin de destination")

    args = argParser.parse_args()

    tub_dir = args.from_dir
    cropPx = CROP if args.crop is None else int(args.crop)  # Convertir en entier
    prepros = PREPRO if args.preprocessing is None else args.preprocessing
    target_tub = TARGET_DIR if args.target is None else args.target

    ## Chargement du tub pour le prétraitement ##
    
    t = TubManager(tub_dir)
    catalog_values = t.get_values()
    line_lengths = t.get_line_lengths()
    images = t.get_images()

    ## Création du dossier de destination pour les images prétraitées ##
    
    if not os.path.isdir(target_tub):
        os.mkdir(target_tub)
        
    t.move_images(target_tub, crop=cropPx, preprocessing=prepros[0] if prepros else None)

    print("Prétraitement terminé.")
