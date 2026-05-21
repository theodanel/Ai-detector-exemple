def generate_recommendation(
    total_calories: float,
    total_proteines: float,
    total_glucides: float,
    total_lipides: float,
    objective: str = "sante"
) -> dict:

    recommendations = []

    if total_proteines < 20:
        recommendations.append(
            "Le repas semble faible en protéines."
        )

    if total_legumes_missing(total_glucides, total_lipides):
        recommendations.append(
            "Pensez à ajouter davantage de légumes."
        )

    if total_calories > 900:
        recommendations.append(
            "Le repas est assez calorique."
        )

    if objective == "prise_de_muscle":
        if total_proteines >= 30:
            recommendations.append(
                "Bon apport en protéines pour la prise de muscle."
            )
        else:
            recommendations.append(
                "Ajoutez davantage de protéines pour favoriser la prise de muscle."
            )

    elif objective == "perte_de_poids":
        if total_calories < 600:
            recommendations.append(
                "Le repas reste raisonnable pour un objectif de perte de poids."
            )
        else:
            recommendations.append(
                "Le repas est peut-être un peu trop calorique pour une perte de poids."
            )

    elif objective == "maintien":
        recommendations.append(
            "Le repas semble adapté à un objectif de maintien."
        )

    else:
        recommendations.append(
            "Essayez de garder un bon équilibre entre protéines, glucides et lipides."
        )

    return {
        "objective": objective,
        "recommendations": recommendations
    }


def total_legumes_missing(glucides: float, lipides: float) -> bool:
    return glucides > 30 and lipides > 10