# Kerala Agricultural Calendar and Seasonal Crop Recommendations
# Based on Kerala Agricultural Development Policy and local farming practices

import datetime
from typing import Dict, List, Tuple

class KeralaAgriculturalCalendar:
    def __init__(self):
        """Initialize Kerala agricultural calendar with crop recommendations"""
        
        # Kerala has three main seasons:
        # 1. Kharif (June-November) - Southwest monsoon
        # 2. Rabi (December-March) - Post-monsoon/Winter
        # 3. Summer (April-May) - Pre-monsoon
        
        self.crop_calendar = {
            1: {  # January
                "season": "Rabi",
                "main_crops": ["Rice (Puncha)", "Vegetables", "Coconut", "Pepper"],
                "planting": ["Bitter gourd", "Bottle gourd", "Snake gourd", "Okra", "Brinjal"],
                "harvesting": ["Rice (Virippu)", "Banana", "Tapioca"],
                "maintenance": ["Coconut (irrigation)", "Pepper (pruning)", "Coffee (harvesting)"],
                "weather": "Cool and dry, occasional showers",
                "irrigation": "Required for most crops",
                "diseases_watch": ["Leaf spot in banana", "Root rot in pepper"],
                "fertilizer": "Apply organic manure to vegetables"
            },
            2: {  # February
                "season": "Rabi",
                "main_crops": ["Rice (Puncha)", "Vegetables", "Coconut", "Cashew"],
                "planting": ["Cucumber", "Pumpkin", "Watermelon", "Muskmelon"],
                "harvesting": ["Cardamom", "Coffee", "Black pepper"],
                "maintenance": ["Coconut (fertilizer application)", "Rubber (tapping season begins)"],
                "weather": "Dry season, increasing temperature",
                "irrigation": "Essential for all crops",
                "diseases_watch": ["Powdery mildew in cucurbits", "Anthracnose in mango"],
                "fertilizer": "NPK application for fruit trees"
            },
            3: {  # March
                "season": "Rabi/Summer transition",
                "main_crops": ["Summer Rice", "Vegetables", "Mango", "Cashew"],
                "planting": ["Amaranthus", "Spinach", "Radish", "Beans"],
                "harvesting": ["Rice (Puncha)", "Mango (early varieties)", "Jackfruit"],
                "maintenance": ["Irrigation system preparation", "Mulching for moisture retention"],
                "weather": "Hot and dry, summer begins",
                "irrigation": "Critical - water conservation needed",
                "diseases_watch": ["Leaf blight in rice", "Fruit fly in mango"],
                "fertilizer": "Potash application for fruit development"
            },
            4: {  # April
                "season": "Summer",
                "main_crops": ["Coconut", "Cashew", "Mango", "Jackfruit"],
                "planting": ["Fodder crops", "Green manure crops"],
                "harvesting": ["Mango", "Jackfruit", "Cashew nuts"],
                "maintenance": ["Irrigation scheduling", "Shade management for young plants"],
                "weather": "Peak summer, very hot",
                "irrigation": "Maximum requirement - drip irrigation recommended",
                "diseases_watch": ["Sunburn in young plants", "Water stress symptoms"],
                "fertilizer": "Avoid heavy fertilization, focus on organic matter"
            },
            5: {  # May
                "season": "Summer/Pre-monsoon",
                "main_crops": ["Coconut", "Rubber", "Spices"],
                "planting": ["Land preparation for Kharif crops", "Nursery preparation"],
                "harvesting": ["Mango (late varieties)", "Cashew nuts"],
                "maintenance": ["Pre-monsoon preparations", "Drainage channel cleaning"],
                "weather": "Hot, pre-monsoon showers expected",
                "irrigation": "Reduce gradually as monsoon approaches",
                "diseases_watch": ["Heat stress", "Early blight preparations"],
                "fertilizer": "Organic manure application before monsoon"
            },
            6: {  # June
                "season": "Kharif - Monsoon begins",
                "main_crops": ["Rice (Virippu)", "Ginger", "Turmeric", "Banana"],
                "planting": ["Rice", "Ginger", "Turmeric", "Banana", "Vegetables"],
                "harvesting": ["Early coconut", "Rubber latex (peak season)"],
                "maintenance": ["Drainage management", "Disease monitoring"],
                "weather": "Southwest monsoon - heavy rainfall",
                "irrigation": "Natural rainfall sufficient, ensure drainage",
                "diseases_watch": ["Fungal diseases due to humidity", "Bacterial wilt"],
                "fertilizer": "Basal dose for newly planted crops"
            },
            7: {  # July
                "season": "Kharif - Peak monsoon",
                "main_crops": ["Rice", "Spices", "Plantation crops"],
                "planting": ["Vegetables (with drainage)", "Fodder crops"],
                "harvesting": ["Early vegetables", "Coconut"],
                "maintenance": ["Weed management", "Pest monitoring"],
                "weather": "Heavy monsoon rains",
                "irrigation": "Excess water management needed",
                "diseases_watch": ["Blast in rice", "Black pepper diseases"],
                "fertilizer": "Split application of nitrogen for rice"
            },
            8: {  # August
                "season": "Kharif - Monsoon continues",
                "main_crops": ["Rice", "Ginger", "Turmeric", "Vegetables"],
                "planting": ["Late vegetables", "Green manure crops"],
                "harvesting": ["Banana", "Vegetables"],
                "maintenance": ["Weeding", "Top dressing of fertilizers"],
                "weather": "Continued monsoon, high humidity",
                "irrigation": "Controlled irrigation, focus on drainage",
                "diseases_watch": ["Leaf spot diseases", "Root rot"],
                "fertilizer": "Second split dose for rice and vegetables"
            },
            9: {  # September
                "season": "Kharif - Late monsoon",
                "main_crops": ["Rice", "Vegetables", "Spices"],
                "planting": ["Rabi crop preparations", "Vegetable nurseries"],
                "harvesting": ["Ginger", "Turmeric", "Vegetables"],
                "maintenance": ["Crop protection", "Nutrient management"],
                "weather": "Monsoon weakening, intermittent rains",
                "irrigation": "Supplementary irrigation may be needed",
                "diseases_watch": ["Post-monsoon fungal diseases"],
                "fertilizer": "Final dose application for Kharif crops"
            },
            10: {  # October
                "season": "Post-monsoon",
                "main_crops": ["Rice (harvesting)", "Vegetables", "Coconut"],
                "planting": ["Rabi vegetables", "Pulses"],
                "harvesting": ["Rice (Virippu)", "Ginger", "Turmeric"],
                "maintenance": ["Post-harvest management", "Land preparation for Rabi"],
                "weather": "Post-monsoon, clear skies",
                "irrigation": "Resume regular irrigation",
                "diseases_watch": ["Storage pest management"],
                "fertilizer": "Soil testing and preparation for next season"
            },
            11: {  # November
                "season": "Post-monsoon/Early Rabi",
                "main_crops": ["Vegetables", "Pulses", "Coconut"],
                "planting": ["Rabi rice", "Vegetables", "Pulses"],
                "harvesting": ["Rice", "Late ginger", "Banana"],
                "maintenance": ["Irrigation system maintenance", "Organic matter incorporation"],
                "weather": "Pleasant weather, occasional showers",
                "irrigation": "Moderate irrigation required",
                "diseases_watch": ["Early winter diseases"],
                "fertilizer": "Organic manure and basal fertilizers"
            },
            12: {  # December
                "season": "Rabi - Winter",
                "main_crops": ["Rice (Puncha)", "Vegetables", "Spices"],
                "planting": ["Cool season vegetables", "Fodder crops"],
                "harvesting": ["Rice (Virippu late)", "Vegetables"],
                "maintenance": ["Mulching", "Protection from cold"],
                "weather": "Cool and dry winter",
                "irrigation": "Regular irrigation needed",
                "diseases_watch": ["Cold stress", "Aphid infestations"],
                "fertilizer": "Winter fertilization schedule"
            }
        }
        
        # Kerala-specific crop varieties and recommendations
        self.kerala_crops = {
            "rice": {
                "varieties": ["Jyothi", "Aiswarya", "Swetha", "Uma", "Kanchana"],
                "seasons": ["Virippu (June-Nov)", "Puncha (Dec-Mar)", "Summer (Apr-May)"],
                "duration": "90-150 days depending on variety"
            },
            "coconut": {
                "varieties": ["West Coast Tall", "Laccadive Ordinary", "Dwarf varieties"],
                "seasons": "Year-round with peak production",
                "duration": "Bearing starts from 6-7 years"
            },
            "pepper": {
                "varieties": ["Panniyur-1", "Karimunda", "Kalluvally"],
                "seasons": "June-July planting, harvesting in winter",
                "duration": "Bearing from 3rd year"
            },
            "cardamom": {
                "varieties": ["Mysore", "Malabar", "Vazhukka"],
                "seasons": "June-July planting, Oct-Feb harvesting",
                "duration": "Bearing from 3rd year"
            },
            "banana": {
                "varieties": ["Nendran", "Robusta", "Monthan", "Poovan"],
                "seasons": "Year-round planting possible",
                "duration": "12-15 months for fruiting"
            }
        }

    def get_monthly_recommendations(self, month: int) -> Dict:
        """Get recommendations for a specific month"""
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")
        
        return self.crop_calendar.get(month, {})

    def get_current_month_recommendations(self) -> Dict:
        """Get recommendations for current month"""
        current_month = datetime.datetime.now().month
        return self.get_monthly_recommendations(current_month)

    def get_seasonal_crops(self, season: str) -> List[str]:
        """Get crops suitable for a specific season"""
        season = season.lower()
        seasonal_crops = []
        
        for month_data in self.crop_calendar.values():
            if season in month_data.get("season", "").lower():
                seasonal_crops.extend(month_data.get("main_crops", []))
                seasonal_crops.extend(month_data.get("planting", []))
        
        return list(set(seasonal_crops))  # Remove duplicates

    def get_crop_info(self, crop_name: str) -> Dict:
        """Get detailed information about a specific crop"""
        crop_name = crop_name.lower()
        return self.kerala_crops.get(crop_name, {})

    def get_planting_calendar(self, crop_name: str) -> List[Tuple[int, str]]:
        """Get best planting months for a specific crop"""
        crop_name = crop_name.lower()
        planting_months = []
        
        for month, data in self.crop_calendar.items():
            planting_crops = [crop.lower() for crop in data.get("planting", [])]
            main_crops = [crop.lower() for crop in data.get("main_crops", [])]
            
            if any(crop_name in crop for crop in planting_crops + main_crops):
                planting_months.append((month, data.get("season", "")))
        
        return planting_months

    def format_monthly_advice(self, month: int = None) -> str:
        """Format monthly agricultural advice in a farmer-friendly way"""
        if month is None:
            month = datetime.datetime.now().month
        
        data = self.get_monthly_recommendations(month)
        if not data:
            return "No specific recommendations available for this month."
        
        month_names = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        advice = f"""
🗓️ **{month_names[month]} Agricultural Calendar for Kerala**

**Season:** {data.get('season', 'N/A')}
**Weather:** {data.get('weather', 'N/A')}

**🌱 Crops to Plant:**
{chr(10).join(['• ' + crop for crop in data.get('planting', [])])}

**🌾 Crops to Harvest:**
{chr(10).join(['• ' + crop for crop in data.get('harvesting', [])])}

**🔧 Maintenance Activities:**
{chr(10).join(['• ' + activity for activity in data.get('maintenance', [])])}

**💧 Irrigation:** {data.get('irrigation', 'N/A')}

**🌿 Fertilizer:** {data.get('fertilizer', 'N/A')}

**⚠️ Diseases to Watch:**
{chr(10).join(['• ' + disease for disease in data.get('diseases_watch', [])])}

*For specific queries about your crops, consult your local Krishi Bhavan officer.*
"""
        return advice