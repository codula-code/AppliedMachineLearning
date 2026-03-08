def score(text, model, threshold):
    propensity = float(model.predict_proba([text])[0][1])
    prediction = bool(propensity >= threshold)
    return prediction, propensity
