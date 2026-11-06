# Deep Financial Market Risk Intelligence

Projet académique personnel : prévision de volatilité sur **SPY** avec **PyTorch LSTM / CNN 1D**, et analyse séparée des régimes statistiques du **NASDAQ Composite** avec autoencodeur, K-Means et LSTM.

Le projet comprend la collecte Yahoo Finance, les contrôles de qualité, les fenêtres causales, les splits temporels purgés, l'entraînement reproductible, les baselines, les artefacts, une API FastAPI et un dashboard Streamlit. Aucun abonnement, GPU ou déploiement cloud n'est nécessaire.

> L’archive livrée contient déjà les artefacts entraînés de référence : vous pouvez lancer l’interface directement. Les commandes de training ci-dessous servent notamment après un clone Git sans artefacts. Conservez les scellés du test final ; ne les supprimez pas pour ajuster les modèles.

## Démarrage

Python **3.12** recommandé. Depuis ce dossier :

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

Les dépendances PyTorch utilisent les roues CPU gratuites. Les versions directes sont figées ; les dépendances transitives ne constituent pas encore un lockfile intégral.

### Données réelles et entraînement

```bash
python -m market_risk.cli download --ticker SPY --output data/SPY.csv
python -m market_risk.cli download --ticker '^IXIC' --output data/IXIC.csv

# Exploration sur validation : ne consulte pas le test final
python -m market_risk.cli volatility --csv data/SPY.csv --output artifacts/spy-validation --epochs 15
python -m market_risk.cli regimes --csv data/IXIC.csv --output artifacts/nasdaq-validation --epochs 15

# Évaluation finale : configuration figée, une seule ouverture par dossier
python -m market_risk.cli volatility --csv data/SPY.csv --output artifacts/spy --epochs 15 --evaluate-test
python -m market_risk.cli regimes --csv data/IXIC.csv --output artifacts/nasdaq --epochs 15 --evaluate-test
```

Ne réutilisez pas le test en changeant de dossier pour choisir des paramètres. Les fichiers `TEST_EVALUATED.json` / `REGIMES_TEST_EVALUATED.json` protègent une expérience locale, sans constituer un contrôle inviolable de toute recherche. Les téléchargements sont contrôlés par les calendriers NYSE/NASDAQ : séances manquantes, historique tronqué et empreinte modifiée provoquent une erreur, sans remplacement synthétique silencieux.

Pour tester le logiciel sans réseau :

```bash
python -m market_risk.cli demo --output artifacts/demo --epochs 2
```

Les résultats de cette commande portent explicitement la source `synthetic` et n'établissent aucune performance financière.

### Interface et API locales

Générez un token avec `python -c "import secrets; print(secrets.token_urlsafe(32))"`, puis affectez-le à la variable `API_TOKEN`.

```bash
# Linux / macOS
export API_TOKEN='<votre-token>'
export ARTIFACT_DIR=artifacts/spy
# PowerShell : $env:API_TOKEN='<votre-token>'; $env:ARTIFACT_DIR='artifacts/spy'
python -m uvicorn market_risk.api:app --host 127.0.0.1 --port 8000
# Autre terminal, même environnement et même token
python -m streamlit run src/market_risk/dashboard.py --server.address=127.0.0.1
```

Le dashboard demande ce token. `/health` vérifie la disponibilité du processus et indique si le modèle existe ; les routes `/metrics` et `/predict` exigent l'en-tête `X-API-Key`. Swagger est accessible sur `http://127.0.0.1:8000/docs`.

`POST /predict` attend `{"sequence": [[...], ...]}` avec **20 × 8** valeurs brutes de features, dans l'ordre : `return`, `vol5`, `vol10`, `vol20`, `momentum`, `range`, `drawdown`, `volume_change`. Le scaler entraîné est appliqué par l'API. La sortie est une volatilité annualisée à cinq séances. L'API sert le meilleur réseau sur validation ; le rapport compare également les baselines, qui peuvent être meilleures.

Le chemin optionnel Docker est prêt dans `Dockerfile` et `compose.yaml` : `docker compose up --build`. Il monte les artefacts existants, ne réentraîne aucun modèle et publie uniquement sur localhost. Le build Docker reste à vérifier sur une machine disposant de Docker.

## Protocole scientifique

- Yahoo Finance, données journalières du 01/01/2015 au 31/12/2024 ; OHLCV bruts et `Adj Close` conservés. Rendement : logarithme du rapport des clôtures ajustées.
- Cible au temps t : écart-type des rendements t+1 à t+5, `ddof=1`, annualisé par √252.
- Split chronologique 70/15/15 sur les séquences admissibles, puis purge des cibles qui franchissent une frontière ; aucune permutation temporelle.
- Scaler ajusté sur train ; early stopping et choix du modèle/seed sur validation ; test final ouvert explicitement.
- Baselines de volatilité historique 5/10/20 séances et EWMA ; seeds 42/7/123 ; comparaisons RMSE/MAE.
- Régimes : encodeur et clusters ajustés sur train ; LSTM comparé à leurs pseudo-labels, jamais à une vérité économique supposée. Les états restent numérotés, sans étiquettes économiques arbitraires.

`market_risk.research.walk_forward` fournit les diagnostics expanding-window LSTM/CNN/baseline, confinés à train+validation. `--mlflow` enregistre les résultats de la commande dans un suivi MLflow local. Aucun serveur externe n'est requis.

## Résultats de la première exécution

Snapshot réel : **2 516 séances par indice**, aucune séance manquante sur les calendriers retenus. SPY : LSTM sélectionné sur validation ; **RMSE test 0,04939**, **MAE test 0,03746**. Ces valeurs sont exprimées en unités de volatilité annualisée. Les résultats complets et paramètres sont dans [docs/results](docs/results).

Le bootstrap du clustering NASDAQ donne un **ARI moyen de 0,299** dans un espace latent fixé : stabilité faible. La concordance élevée du LSTM avec ces pseudo-labels ne prouve pas l'existence de régimes économiques fiables. Aucune performance de trading n'est revendiquée.

## Structure

```text
src/market_risk/  collecte, features, modèles, training, API, dashboard
configs/         protocole préenregistré
notebooks/       exploration reproductible des données
scripts/         diagnostics et analyse locale
 tests/          invariants temporels, modèles, API, reproductibilité
 docs/           CDC, résultats, model card, statut et procédures
```

## Statut et suite

Le code local et les entraînements de référence sont livrés. [STATUS.md](STATUS.md) distingue les vérifications réalisées des validations restantes. Le cloud reste différé ; aucune ressource n'est créée par la CI. Le workflow GitHub Actions est **manuel**, à lancer uniquement dans les minutes incluses de votre compte. La conformité au sujet universitaire effectivement attribué doit être confirmée : les branches SPY et NASDAQ restent indépendantes.

Les données Yahoo ne sont pas ajoutées au dépôt Git. Les prix ajustés sont un snapshot rétrospectif, sans garantie de disponibilité point-in-time des ajustements. Ce démonstrateur constitue un travail de recherche, sans conseil ni exécution automatique de trading.
