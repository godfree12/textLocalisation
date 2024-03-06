## Localisation de texte sur une image avec Deep Learning

Ce projet Python vise à développer un système de localisation de texte sur des images en utilisant des techniques de Deep Learning. Il utilise un réseau de neurones convolutif (CNN) pour détecter et localiser efficacement les régions de texte dans une image.

## Introduction

La localisation de texte sur des images est une tâche essentielle dans de nombreux domaines tels que la reconnaissance optique de caractères (OCR), l'analyse d'images médicales, et la surveillance vidéo. Ce projet explore les techniques de Deep Learning pour résoudre ce problème de manière efficace et précise.

## Dataset
Le modèle est entraîné à l'aide d'un dataset construit spécifiquement pour ce projet. Le dataset contient une variété d'images annotées avec des régions de texte. 
Le dataset se compose de :

Images diverses avec une variété de polices, de tailles et d'orientations de texte.
Annotations précises des régions de texte dans chaque image.


## Modèle
Le modèle est implémenté en utilisant le framework TensorFlow/Keras. Il s'agit d'un modèle de réseau de neurones convolutif (CNN) qui prend une image en entrée et produit des coordonnées de boîtes englobantes (bounding boxes) autour des régions de texte détectées.

Le modèle s'est inspiré sur l'architecture YOLO (You Only Look Once) qui est connue pour sa rapidité et son efficacité dans la détection d'objets.


## Utilisation
Pour utiliser le modèle pour localiser du texte sur une image :

Assurez-vous d'avoir les dépendances requises installées.
Chargez le modèle entraîné à l'aide de model.load_model('/content/drive/MyDrive/Classroom/train_ckeck/model.pth'). Ce fichier se trouve dans le projet( copié le et mettez dans votre environnement locale pour une utlisation.
Le code inclut aussi la partie du test, donc pour une eventuelle execution a des fin de test, couper cette partie et mettez le dans un autre fichier que vous executerez.



## Conclusion
Ce projet démontre l'application des techniques de Deep Learning à la localisation de texte sur des images. Il offre une solution efficace et précise pour cette tâche importante dans de nombreux domaines d'application.

## Remarque
Ce projet est réalisé dans le cadre de l'intelligence artificielle, plus précisément du domaine du Deep Learning.
Le resultat final ainsi que es statistiques se trouves a la fin du projet se trouve dans le fichier (UE-INF3051L_DepotRapportIndiv_renduP_(8))


## Auteur

Ce projet a été développé par Godfree AKAKPO. N'hésitez pas à me contacter pour toute question ou suggestion concernant ce projet.
Profil Linkdin : https://www.linkedin.com/in/godfree-akakpo-3783a6198/

