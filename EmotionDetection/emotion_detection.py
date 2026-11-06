import json
import requests


def emotion_detector(text_to_analyze):
    """Analiza el texto con Watson NLP y devuelve las emociones y la emoción dominante.

    Si el servicio responde con código de estado 400 (por ejemplo, texto en blanco),
    todos los valores devueltos son None.
    """
    url = ("https://sn-watson-emotion.labs.skills.network/v1/"
           "watson.runtime.nlp.v1/NlpService/EmotionPredict")
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, json=payload, headers=headers, timeout=15)

    # Manejo de errores: texto en blanco o inválido
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    # Convertir el texto de la respuesta en un diccionario
    formatted_response = json.loads(response.text)

    # Extraer las puntuaciones de cada emoción
    emotions = formatted_response["emotionPredictions"][0]["emotion"]
    anger_score = emotions["anger"]
    disgust_score = emotions["disgust"]
    fear_score = emotions["fear"]
    joy_score = emotions["joy"]
    sadness_score = emotions["sadness"]

    scores = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
    }

    # Encontrar la emoción con la puntuación más alta
    dominant_emotion = max(scores, key=scores.get)

    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion,
    }
