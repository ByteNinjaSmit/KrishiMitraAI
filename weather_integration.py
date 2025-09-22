# Weather Integration for Kerala Agricultural Advisory
# Provides weather-based agricultural recommendations

import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json

class WeatherAdvisory:
    def __init__(self):
        """Initialize weather advisory system"""
        # Using a free weather API - OpenWeatherMap (no API key needed for basic features)
        # For production, you would need to register for an API key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        # Kerala district coordinates for weather data
        self.kerala_locations = {
            "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
            "Kollam": {"lat": 8.8932, "lon": 76.6141},
            "Pathanamthitta": {"lat": 9.2648, "lon": 76.7873},
            "Alappuzha": {"lat": 9.4981, "lon": 76.3388},
            "Kottayam": {"lat": 9.5916, "lon": 76.5222},
            "Idukki": {"lat": 9.8475, "lon": 76.9717},
            "Ernakulam": {"lat": 9.9312, "lon": 76.2673},
            "Thrissur": {"lat": 10.5276, "lon": 76.2144},
            "Palakkad": {"lat": 10.7867, "lon": 76.6548},
            "Malappuram": {"lat": 11.0510, "lon": 76.0711},
            "Kozhikode": {"lat": 11.2588, "lon": 75.7804},
            "Wayanad": {"lat": 11.6854, "lon": 76.1320},
            "Kannur": {"lat": 11.8745, "lon": 75.3704},
            "Kasaragod": {"lat": 12.4996, "lon": 74.9869}
        }
        
        # Weather-based agricultural recommendations
        self.weather_advice = {
            "sunny": {
                "general": "Good weather for field operations and harvesting",
                "irrigation": "Increase irrigation frequency, especially for young plants",
                "crops": {
                    "rice": "Good for land preparation and transplanting",
                    "vegetables": "Provide shade for sensitive crops",
                    "coconut": "Monitor for water stress"
                },
                "activities": ["Harvesting", "Field preparation", "Fertilizer application"],
                "precautions": ["Protect young plants from direct sun", "Ensure adequate water supply"]
            },
            "rainy": {
                "general": "Monsoon conditions - focus on drainage and disease prevention",
                "irrigation": "Reduce or stop irrigation, ensure proper drainage",
                "crops": {
                    "rice": "Ideal for transplanting if not waterlogged",
                    "vegetables": "Watch for fungal diseases",
                    "spices": "Good growing conditions but monitor for diseases"
                },
                "activities": ["Drainage maintenance", "Disease monitoring", "Weed control"],
                "precautions": ["Prevent waterlogging", "Apply fungicides if needed", "Avoid field operations"]
            },
            "cloudy": {
                "general": "Moderate conditions suitable for most agricultural activities",
                "irrigation": "Normal irrigation schedule",
                "crops": {
                    "vegetables": "Good conditions for growth",
                    "coffee": "Ideal for coffee plants",
                    "tea": "Favorable for tea cultivation"
                },
                "activities": ["Planting", "Fertilizer application", "Pest monitoring"],
                "precautions": ["Monitor humidity levels", "Watch for pest buildup"]
            },
            "hot": {
                "general": "High temperature stress - focus on water management",
                "irrigation": "Increase frequency, preferably early morning or evening",
                "crops": {
                    "coconut": "May show stress symptoms",
                    "vegetables": "Provide shade and mulching",
                    "banana": "Ensure adequate water supply"
                },
                "activities": ["Early morning operations", "Mulching", "Shade provision"],
                "precautions": ["Avoid midday field work", "Increase water frequency", "Monitor for heat stress"]
            }
        }

    def get_mock_weather_data(self, district: str) -> Dict:
        """Generate realistic mock weather data for demonstration"""
        import random
        
        # Simulate realistic Kerala weather patterns
        current_month = datetime.now().month
        
        # Kerala weather characteristics by month
        if current_month in [12, 1, 2]:  # Winter
            temp_range = (22, 32)
            humidity_range = (60, 80)
            conditions = ["sunny", "cloudy"]
            rainfall_prob = 0.2
        elif current_month in [3, 4, 5]:  # Summer
            temp_range = (26, 38)
            humidity_range = (50, 75)
            conditions = ["sunny", "hot"]
            rainfall_prob = 0.1
        elif current_month in [6, 7, 8, 9]:  # Monsoon
            temp_range = (24, 30)
            humidity_range = (80, 95)
            conditions = ["rainy", "cloudy"]
            rainfall_prob = 0.8
        else:  # Post-monsoon
            temp_range = (25, 32)
            humidity_range = (70, 85)
            conditions = ["cloudy", "sunny"]
            rainfall_prob = 0.3
        
        # Generate mock data
        current_temp = random.randint(*temp_range)
        humidity = random.randint(*humidity_range)
        condition = random.choice(conditions)
        will_rain = random.random() < rainfall_prob
        
        # Generate 5-day forecast
        forecast = []
        for i in range(5):
            day_temp = random.randint(*temp_range)
            day_condition = random.choice(conditions)
            day_rain = random.random() < rainfall_prob
            
            forecast.append({
                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "temperature": {"min": day_temp-3, "max": day_temp+3},
                "condition": day_condition,
                "rainfall": random.randint(0, 50) if day_rain else 0,
                "humidity": random.randint(*humidity_range)
            })
        
        return {
            "location": district,
            "current": {
                "temperature": current_temp,
                "condition": condition,
                "humidity": humidity,
                "rainfall_today": random.randint(0, 30) if will_rain else 0,
                "wind_speed": random.randint(5, 15)
            },
            "forecast": forecast,
            "last_updated": datetime.now().isoformat()
        }

    def get_weather_recommendations(self, weather_data: Dict) -> str:
        """Generate agricultural recommendations based on weather data"""
        if not weather_data:
            return "Weather data unavailable. Please consult local weather services."
        
        current = weather_data.get("current", {})
        condition = current.get("condition", "").lower()
        temperature = current.get("temperature", 0)
        humidity = current.get("humidity", 0)
        rainfall = current.get("rainfall_today", 0)
        
        # Determine weather category
        if rainfall > 10:
            weather_category = "rainy"
        elif temperature > 35:
            weather_category = "hot"
        elif "sunny" in condition:
            weather_category = "sunny"
        else:
            weather_category = "cloudy"
        
        advice = self.weather_advice.get(weather_category, {})
        
        recommendations = f"""
🌤️ **Weather-Based Agricultural Advisory**

**📍 Location:** {weather_data.get('location', 'Kerala')}
**🌡️ Current Temperature:** {temperature}°C
**💧 Humidity:** {humidity}%
**🌧️ Rainfall Today:** {rainfall}mm
**☁️ Condition:** {condition.title()}

**📋 General Recommendation:**
{advice.get('general', 'Monitor weather conditions closely')}

**💧 Irrigation Advice:**
{advice.get('irrigation', 'Follow normal irrigation schedule')}

**🌱 Crop-Specific Advice:**
"""
        
        crop_advice = advice.get('crops', {})
        for crop, advice_text in crop_advice.items():
            recommendations += f"• **{crop.title()}:** {advice_text}\n"
        
        recommendations += f"""
**✅ Recommended Activities:**
{chr(10).join(['• ' + activity for activity in advice.get('activities', [])])}

**⚠️ Precautions:**
{chr(10).join(['• ' + precaution for precaution in advice.get('precautions', [])])}

**📅 5-Day Outlook:**
"""
        
        forecast = weather_data.get("forecast", [])
        for day in forecast[:3]:  # Show next 3 days
            date = day.get("date", "")
            temp_min = day.get("temperature", {}).get("min", 0)
            temp_max = day.get("temperature", {}).get("max", 0)
            day_condition = day.get("condition", "").title()
            day_rainfall = day.get("rainfall", 0)
            
            recommendations += f"• **{date}:** {temp_min}-{temp_max}°C, {day_condition}"
            if day_rainfall > 0:
                recommendations += f", Rain: {day_rainfall}mm"
            recommendations += "\n"
        
        recommendations += "\n*Weather data updates every 3 hours. For critical decisions, consult local meteorological services.*"
        
        return recommendations

    def get_seasonal_weather_advisory(self, month: int = None) -> str:
        """Get seasonal weather patterns and agricultural advice"""
        if month is None:
            month = datetime.now().month
        
        month_names = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        seasonal_advice = {
            1: {"season": "Winter", "weather": "Cool and dry", "advice": "Good for harvesting and field preparation"},
            2: {"season": "Winter", "weather": "Dry, temperature rising", "advice": "Focus on irrigation management"},
            3: {"season": "Pre-summer", "weather": "Hot and dry", "advice": "Water conservation critical"},
            4: {"season": "Summer", "weather": "Very hot", "advice": "Minimal field operations, focus on water"},
            5: {"season": "Pre-monsoon", "weather": "Hot with occasional showers", "advice": "Prepare for monsoon crops"},
            6: {"season": "Southwest Monsoon", "weather": "Heavy rainfall begins", "advice": "Plant monsoon crops, ensure drainage"},
            7: {"season": "Peak Monsoon", "weather": "Continuous heavy rains", "advice": "Monitor waterlogging and diseases"},
            8: {"season": "Monsoon", "weather": "Regular rainfall", "advice": "Continue monsoon crop care"},
            9: {"season": "Late Monsoon", "weather": "Reducing rainfall", "advice": "Prepare for post-monsoon activities"},
            10: {"season": "Post-monsoon", "weather": "Clear skies, pleasant", "advice": "Harvesting and land preparation"},
            11: {"season": "Northeast Monsoon", "weather": "Moderate rainfall", "advice": "Second crop planning"},
            12: {"season": "Winter onset", "weather": "Cool and pleasant", "advice": "Winter crop activities"}
        }
        
        month_info = seasonal_advice.get(month, {})
        
        return f"""
🗓️ **{month_names[month]} Weather Pattern for Kerala**

**Season:** {month_info.get('season', 'N/A')}
**Typical Weather:** {month_info.get('weather', 'N/A')}
**Agricultural Focus:** {month_info.get('advice', 'N/A')}

**General Recommendations for {month_names[month]}:**
• Plan agricultural activities based on typical weather patterns
• Monitor daily weather forecasts for variations
• Adjust irrigation schedules according to rainfall patterns
• Prepare for seasonal pest and disease management
"""

    def format_weather_alert(self, weather_data: Dict) -> Optional[str]:
        """Generate weather alerts for extreme conditions"""
        if not weather_data:
            return None
        
        current = weather_data.get("current", {})
        temperature = current.get("temperature", 0)
        rainfall = current.get("rainfall_today", 0)
        humidity = current.get("humidity", 0)
        
        alerts = []
        
        # Temperature alerts
        if temperature > 38:
            alerts.append("🌡️ **HEAT ALERT:** Extreme heat conditions. Avoid field work during midday hours.")
        elif temperature < 15:
            alerts.append("🌨️ **COLD ALERT:** Unusually low temperatures. Protect sensitive crops.")
        
        # Rainfall alerts
        if rainfall > 50:
            alerts.append("🌧️ **HEAVY RAIN ALERT:** Ensure proper drainage to prevent waterlogging.")
        elif rainfall > 100:
            alerts.append("⛈️ **FLOOD RISK:** Very heavy rainfall. Avoid field operations and ensure crop protection.")
        
        # Humidity alerts
        if humidity > 90:
            alerts.append("💨 **HIGH HUMIDITY ALERT:** Increased disease risk. Monitor crops closely and consider fungicide application.")
        
        return "\n".join(alerts) if alerts else None