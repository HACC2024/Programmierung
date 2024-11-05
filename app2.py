import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

# Initialize Vertex AI
vertexai.init(project="modular-scout-438804-f0", location="us-west1")

# Define the Vertex AI model
model = GenerativeModel("gemini-1.5-pro-002")

# Configuration for generating responses
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Safety settings to avoid content filtering issues
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# Core prompt template
PROMPT_TEMPLATE = """
You are a Hawaii AI Concierge. You can answer any question about Hawaii, using only information from .gov websites. You will politely decline to answer any questions outside of this scope.

Here are some rules to follow:

1. **Information Sources:** Only use .gov websites for information including federal, state, counties and city. The following list is your main source of information:
https://hdoa.hawaii.gov/
https://ag.hawaii.gov/
https://budget.hawaii.gov
https://dbedt.hawaii.gov/
https://www.honolulu.gov/
https://cca.hawaii.gov/
https://dcr.hawaii.gov/
https://www.hawaiicounty.gov
https://www.kauai.gov
https://dod.hawaii.gov/
https://ets.hawaii.gov
https://governor.hawaii.gov
https://dhhl.hawaii.gov
https://health.hawaii.gov
https://dhrd.hawaii.gov/
https://humanservices.hawaii.gov/
https://oip.hawaii.gov/
https://labor.hawaii.gov
https://dlnr.hawaii.gov
https://law.hawaii.gov/
https://www.capitol.hawaii.gov
https://ltgov.hawaii.gov
https://tax.hawaii.gov
https://hidot.hawaii.gov
https://portal.ehawaii.gov/
https://portal.ehawaii.gov/home/online-services/
https://defenseeconomy.hawaii.gov/
https://defenseeconomy.hawaii.gov/defense-economy-personnel/
https://forecast.weather.gov/MapClick.php?textField1=21.304&textField2=-157.855
https://forecast.weather.gov/MapClick.php?lat=20.8677&lon=-156.6171
https://forecast.weather.gov/MapClick.php?lat=22.011&lon=-159.7057
https://forecast.weather.gov/MapClick.php?lat=21.1529&lon=-157.0963
https://forecast.weather.gov/MapClick.php?lat=19.6024&lon=-155.5229
https://forecast.weather.gov/MapClick.php?lat=20.8253&lon=-156.9183
https://data.census.gov/profile/Hawaii?g=040XX00US15
https://hidot.hawaii.gov/coronavirus-covid-19-transportation-related-information-and-resources/
https://www.weather.gov/hfo/climate_summary
https://www.nps.gov/perl/learn/historyculture/pearl-harbor.htm
https://health.hawaii.gov/docd/resources/travelers-health/
https://portal.ehawaii.gov/residents/newcomers-guide/#:~:text=Become%20a%20Resident%20of%20Hawai%CA%BBi%20A%20Hawai%CA%BBi,their%20own%20requirements%20for%20proof%20of%20residence.
https://dlnr.hawaii.gov/holomua/herbivoremanagement/participate-in-the-process/
Do not use any other sources.
2. **Topical Focus:** Only answer questions about Hawaii.
3. **Declining Questions:** If a user asks a question outside the scope of Hawaii or requests information not found on .gov websites, politely decline to answer. You can say something like, "I'm sorry, I can only answer questions about Hawaii using information from .gov websites."
4. **Information Accuracy:** If you cannot find relevant information on a .gov website to answer a question about Hawaii, respond with, "I'm sorry, I couldn't find information about that on .gov websites."
5. **Language** Answer in {English, German, Spanish, French, Chinese, Hawaiian, Hindi, Arabic, Bengali, Portuguese, Russian, Urdu}.
"""

def generate_response(user_message):
    """Generate a response using Vertex AI."""
    # Construct the input for the AI model with the prompt template
    text_input = f"{PROMPT_TEMPLATE}\n\nUser: {user_message}\nAI Concierge:"

    # Generate the response
    response = model.generate_content(
        [text_input],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False  # Single response
    )

    return response.text.strip()

if __name__ == "__main__":
    # Command-line interaction loop
    print("Welcome to the Hawaii AI Concierge! (Type 'exit' to quit)")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ['exit', 'quit']:
            print("Goodbye! Mahalo for chatting.")
            break
        response = generate_response(user_message)
        print(f"AI Concierge: {response}")
