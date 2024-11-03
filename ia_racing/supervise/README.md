# Apprentissage supervisé sur le simulateur 

## Récupération des données de la course
Lors de la course, la voiture enregistre l'image, la direction et la vitesse a une fréquence définie dans des répertoires nommés tubs ([découvrir ce qu'est un tub](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/mycar)), stocké dans le répertoire ```data``` de la voiture.

Pour faciliter la récupération des données de la dernière course, nous conseillons de créer un tub par exécution de la voiture (sinon toutes les données sont stockées dans le même tub et il devient difficile de faire le tri entre les anciennes et les nouvelles données). Pour cela, dans le fichier ```cars/mycar/myconfig.py```, il suffit de décommenter la constante ```AUTO_CREATE_NEW_TUB``` et de la passer à True.

## Preprocessing d'un seul tub

Les donnees que founisse la camera de la voiture peuvetres attachées de bruits (par exemple, des pixels parasites ou des variations de lumière) qui peut perturber l'entraînement du modèle. Des techniques de preprocessing sont cruciales pour ameliorer la qualité des données avant entrainement.
Nous avons utilisé les méthodes de preprocessing telles que:

    ### lines : pour la détection de lignes et le traitement des bords.
    ### bnw : pour la conversion en niveaux de gris simple.

Pour se faire, nous avons créé le script ```tub_preprocessing.py```.

Pour l'exécuter, il suffit de l'exécuter avec les paramètres suivants :

- -f le chemin du répertoire contenant tous les tubs qu'on souhaite fusionner
- -c le nombre de pixels qu'on souhaite rogner en haut de l'image
- -p les preprocessings qu'on souhaite utiliser
- -t le chemin du répertoire de destination dans lequel on veut que les tubs soient fusionnées et preprocesser

Un sous-répertoire par type de preprocessing indiqué en -p sera créé dans le répertoire indiqué en -t.

Par exemple, pour fusionner plusieurs tubs enregistrés dans le répertoire ```tub``` qu'on veut preprocesser avec les preprocessings 'bnw' et 'lines' rognés de 40px par rapport au haut de l'image dans le répertoire ```tub_preprocessed```, on peut taper :

```
python tub_preprocessing.py -f ./tub -c 40 -p bnw lines -t tub_preprocessed
```

On aura alors dans le dossier ```tub_preprocessed``` 3 sous-répertoires :

1. Les images brutes
2. Les images rognées en noir et blanc (bnw)
3. Les images rognées et traitées avec le preprocessing lines

En outre, dans chaque répertoire on retrouvera les ```catalog_X.catalog```, ```catalog_X.catalog_manifest``` et les ```manifest.json``` comme dans un tub standard.


## Fusion de plusieurs tubs + Preprocessing

Parfois, il peut être utile de fusionner plusieurs tubs entre eux (par exemple, on a très bien conduit sur plusieurs courses). Pour cela, nous avons  le script ```remaster_data.py```.

Pour l'exécuter, c'est la mememe chose que dans le preprocessing il suffit de l'exécuter avec les paramètres suivants :

- -f le chemin du répertoire contenant tous les tubs qu'on souhaite fusionner
- -c le nombre de pixels qu'on souhaite rogner en haut de l'image
- -p les preprocessings qu'on souhaite utiliser
- -t le chemin du répertoire de destination dans lequel on veut que les tubs soient fusionnées et preprocesser

Un sous-répertoire par type de preprocessing indiqué en -p sera créé dans le répertoire indiqué en -t.

Par exemple, pour fusionner plusieurs tubs enregistrés dans le répertoire ```tubs_to_merge``` qu'on veut preprocesser avec les preprocessings 'bnw' et 'lines' rognés de 40px par rapport au haut de l'image dans le répertoire ```tub_master```, on peut taper :

```
python remaster_data.py -f ./tubs_to_merge -c 40 -p bnw lines -t tub_master
```

### Création du modèle d'entrainément

Dans un premier temps, nous avons utiliser le model de reuseau interne de Donkey car( Boite noir) pour nous famialiser avant de commencer à creer notre prore modele de reseau de neurone

### Lancer un entraînement du model

> __Note__ : pour cette partie, il est vivement recommandé d'avoir un bon ordinateur car cela prend enormement du temps 
l'ideal c'est d'avoir un ordinateur qui a une carte graphique avec CUDA installé. Pour installer CUDA( à faire ulterieurement) 
Le temps d'exécution peut ainsi passer de quelques heures à quelques minutes.

Pour lancer un entraînement, il suffit de taper les commandes suivantes de se rendre dans le répertoire correspondant à la voiture qui contient les données.

```
$ cd ~/ia_racing/mycar/
```

Pour choisir le type de modèle qu'on souhaite entraîner, il faut modifier le fichier ```myconfig.py``` en tapant la commande suivante et modifier la constante ```DEFAULT_MODEL_TYPE``` (pour comprendre l'intérêt du fichier ```myconfig.py```, [voir ce document](https://github.com/Rom-1T/ia_racing_imt/tree/main/integration/mycar)). Il peut être nécessaire de modifier les constantes ```IMAGE_W```, ```IMAGE_H``` et ```IMAGE_DEPTH``` si les images ont été préprocessées. D'autres paramètres peuvent influer sur l'entraînement comme la taille des lots d'images (```BATCH_SIZE```), le learning rate (```LEARNING_RATE```) et le nombre d'epochs (```MAX_EPOCHS```).

> __Note__ : le nombre d'epochs renseigné correspond au nombre maximum d'epochs qui pourront être utilisées pour l'entraînement car, par défaut, lorsque la valeur de la fonction de coût n'évolue plus, l'entraînement s'arrête (cela peut être désactivé en passant la constante ```USE_EARLY_STOP``` à False). Cet arrêt est intéressant pour éviter le surapprentissage( Overfiting).

```
~/ia_racing/mycar/$ nano myconfig.py
```

Pour finir et lancer l'entraînement, il suffit de taper la ligne suivante.

```
~/ia_racing/mycar/$ donkey train --tub ./data --model ./models/mypilot.h5
````

Lorsque le modèle est entraîné, il y a 2 modèles créés : un .h5 et un .tflite (à condition que la constante ```CREATE_TF_LITE``` de ```myconfig.py``` ait la valeur Vrai). On conseille d’utiliser le tflite qui est plus léger et annoncé par Google être conçu pour les appareils embarqués. En outre, nous avons rencontré des difficultés à exécuter les .h5 issus des entraînements sur windows, mais pas pour les tflite.


