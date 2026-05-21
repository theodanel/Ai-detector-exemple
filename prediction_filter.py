def filter_predictions(
    raw_predictions: list[dict],
    max_results: int = 3,
    min_confidence: float = 0.45,
    weak_confidence: float = 0.60,
) -> dict:
    """
    Nettoie les prédictions IA pour éviter d'afficher trop d'aliments absurdes.

    Le modèle actuel est encore entraîné sur un dataset minuscule.
    Ce filtre sert donc à rendre les résultats plus crédibles côté utilisateur.
    """

    if not raw_predictions:
        return {
            "status": "no_prediction",
            "message": "Aucun aliment détecté avec assez de confiance.",
            "global_confidence": 0,
            "predictions": []
        }

    sorted_predictions = sorted(
        raw_predictions,
        key=lambda item: item["confidence"],
        reverse=True
    )

    best_prediction = sorted_predictions[0]
    best_confidence = best_prediction["confidence"]

    filtered_predictions = []

    for prediction in sorted_predictions:
        confidence = prediction["confidence"]

        if confidence >= min_confidence:
            filtered_predictions.append(prediction)

    filtered_predictions = filtered_predictions[:max_results]

    if best_confidence < weak_confidence:
        status = "low_confidence"
        message = "Analyse incertaine : l'IA n'est pas encore assez sûre du résultat."
    else:
        status = "ok"
        message = "Analyse effectuée avec une confiance correcte."

    return {
        "status": status,
        "message": message,
        "global_confidence": round(best_confidence, 3),
        "predictions": filtered_predictions
    }