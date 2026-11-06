# Model card — SPY / NASDAQ

**Usage** : étude académique de volatilité et états statistiques. Aucun trading automatique.

**Données** : Yahoo Finance, 2015–2024, snapshot et hashes dans results. Les prix ajustés sont rétrospectifs ; leurs corrections ne sont pas des vintages point-in-time.

**SPY** : horizon futur cinq séances, annualisation √252, lookback 20, huit features. LSTM/CNN 16 unités, Adam 0,003, maximum 15 epochs pour la référence, early stopping patience 7. Trois seeds 42/7/123, sélection sur validation uniquement. Test final : RMSE 0,04939, MAE 0,03746. Les baselines test sont également rapportées, sans nouvelle sélection.

**NASDAQ** : autoencodeur latent 3, K-Means K=3 fit sur train, LSTM de concordance avec pseudo-labels. ARI bootstrap moyen ≈ 0,299 dans un encodeur fixé. La concordance pseudo-supervisée n'est pas une exactitude de véritables régimes.

**Incertitude** : distribution des erreurs par période, variance inter-seeds et quantile descriptif d'erreur validation. Ce quantile n'est pas une garantie de couverture prospective.

**Limites** : petit historique, dépendance au régime de marché, risque d'overfitting, absence de labels économiques indépendants, ajustements rétrospectifs. La sélection des hyperparamètres ne doit jamais utiliser le test final.
