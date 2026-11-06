# Revue technique

Revue indépendante des deux projets académiques, sans modification des fichiers par le relecteur.

Points traités avec reproduction puis tests : matrice temporelle irrégulière, valeurs API non finies et extrêmes, ouverture implicite du test régimes, téléchargement Yahoo tronqué. Les API renvoient des erreurs 422 explicites ; le test régimes est désormais opt-in ; les calendriers vérifient la couverture.

Contrôles scientifiques : features causales, purge des cibles futures et ajustement des transformations sur train. Limites conservées : stabilité des régimes, portée des pseudo-labels, ancienneté du dataset bancaire, qualification Docker/cloud différée. Voir STATUS.md.
