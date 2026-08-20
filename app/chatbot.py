def chatbot_response(user_input, recommendations):
    user_input = user_input.lower()

    for disease, recommendation in recommendations.items():

        if disease.lower() in user_input:
            return (
                f"Possible disease detected: {disease}\n\n"
                f"Recommended Fertilizer: {recommendation['fertilizer_name']}\n"
                f"Type: {recommendation['type']}\n"
                f"Dosage: {recommendation['dosage']}\n"
                f"Brand: {recommendation['brand']}"
            )

    return "Sorry, I could not identify the disease from your description."