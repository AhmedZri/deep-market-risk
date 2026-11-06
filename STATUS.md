# État vérifié — projet académique ENSI

## Livré et exécuté

- Collecte Yahoo SPY/NASDAQ 2015–2024 : 2 516 séances par indice, calendriers contrôlés, hashes et Parquet/CSV.
- Features causales, cibles futures, split 70/15/15 purgé, scaler train-only.
- LSTM/CNN PyTorch entraînés sur trois seeds ; baselines 5/10/20/EWMA ; test SPY final figé.
- Autoencodeur, K-Means et LSTM régimes exécutés ; profils, transitions et bootstrap ARI dans latent fixé.
- Trois validations expanding-window LSTM/CNN/baseline, sans accès au bloc test.
- Huit figures EDA exportées dans un HTML local hors Git.
- API avec clé, tests de parité et erreurs 422 ; dashboard à trois pages parcouru par Streamlit AppTest.
- Notebook d'exploration : cellules exécutées ; package installable et CLI.
- 29 tests automatisés réussis lors de la recette initiale ; la preuve à jour figure dans docs/verification.json.

## Préparé, non déployé

- Dockerfile et Compose ; Docker absent de l'environnement de réalisation, build à vérifier localement.
- Workflow GitHub Actions manuel ; aucun dépôt distant créé et aucun workflow exécuté.
- Aucun hébergement cloud, abonnement, API payante ou ressource facturable.

## Qualification ultérieure

- Validation du sujet universitaire effectivement attribué.
- Stabilité des régimes faible (ARI ≈ 0,299) : révisions éventuelles sur train/validation d'un nouveau protocole, sans réutiliser le test actuel pour tuning.
- Le bootstrap livré mesure K-Means à encodeur fixé ; le réentraînement complet encodeur+clusters et des intervalles de stabilité complets restent une extension de recherche.
- Les cibles de gain, disponibilité, P95/RTO et niveaux L2/L3 du CDC ne sont pas déclarés atteints sans leurs campagnes de mesure.
- Les dépendances directes sont figées ; un lockfile transitif intégral et des builds multi-OS restent à qualifier.
