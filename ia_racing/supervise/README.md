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

### Création et inférence du modèle
Après avoir réfléchi au problème d'optimisation que l'on cherche à résoudre, on peut s'atteler à la création du modèle.

Le fichier permettant d'entraîner notre modèle (qui s'est révélé mauvais) est ```pilot_train.py```. Il peut réaliser un entraînement à partir des données présentes dans ```DATASET_DIR```. Le fichier ```pilot_test.py``` permet de faire l'inférence.
