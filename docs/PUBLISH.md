# Publier dans votre propre dépôt GitHub

Créez un dépôt vide sous AhmedZri, sans README initial, puis depuis le dossier du projet :

```bash
git init -b main
git add .
git commit -m "Initial ENSI academic implementation"
git remote add origin https://github.com/AhmedZri/REPLACE_WITH_REPOSITORY_NAME.git
git push -u origin main
```

Remplacez le nom de dépôt avant la commande. Authentifiez-vous avec votre méthode GitHub habituelle. Aucune publication n'a été effectuée automatiquement.

Le fichier .gitignore exclut les données, artefacts entraînés, environnements et secrets. Les métriques partageables et la provenance sont dans docs/results. Ne forcez pas l'ajout de données ou d'artefacts sans vérifier leurs droits et leur contenu.

La CI est manuelle, sans déploiement : lancez-la depuis Actions uniquement dans le quota inclus du compte. Les procédures locales fonctionnent sans GitHub Actions.
