# FILE: weather.py
# PATH: AgroSaarthi_AI/backend_api/app/weather.py
# PURPOSE: Weather fetching and risk calculation logic

import requests

def get_weather_risk(lat: float, lon: float, api_key: str = "DUMMY_MODE"):
    """
    Fetches weather data and calculates crop disease risk.
    Using a dummy mode by default so development doesn't stop for API keys.
    """
    if api_key == "DUMMY_MODE":
        # Simulating Kanpur's hot and humid weather during monsoons
        temp = 29.5  # Celsius
        humidity = 85 # Percentage
    else:
        # Future Real API Call logic (OpenWeatherMap)
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            response = requests.get(url).json()
            temp = response.get("main", {}).get("temp", 25)
            humidity = response.get("main", {}).get("humidity", 60)
        except:
            temp = 25
            humidity = 50

    # Risk Calculation Logic
    # Fungi thrives in high humidity (>80%) and moderate-to-high temps
    risk_level = "Low"
    alert_message = "Weather is normal. No immediate risk."

    if humidity > 80 and temp > 25:
        risk_level = "High"
        alert_message = "High Humidity Detected! Extremely favorable conditions for Fungal Infections (e.g., Blight). Spray preventative fungicides."
    elif humidity > 70:
        risk_level = "Medium"
        alert_message = "Moderate risk. Keep an eye on the lower leaves for spots."

    return {
        "temperature_celsius": temp,
        "humidity_percent": humidity,
        "risk_level": risk_level,
        "alert_message": alert_message
    }