# Kerala Districts Agricultural Information and Location-based Advisory
# Based on district-specific agricultural practices, soil types, and government policies

from typing import Dict, List, Optional

class KeralaDistrictAdvisory:
    def __init__(self):
        """Initialize Kerala district-specific agricultural advisory system"""
        
        self.districts = {
            "Thiruvananthapuram": {
                "region": "Southern Kerala",
                "major_crops": ["Coconut", "Rice", "Tapioca", "Banana", "Vegetables"],
                "soil_types": ["Laterite", "Coastal sandy", "Alluvial"],
                "climate": "Tropical coastal",
                "rainfall": "1800-2000mm annually",
                "irrigation": "Wells, tanks, and lift irrigation schemes",
                "specialties": ["Coconut farming", "Coastal aquaculture", "Vegetable cultivation"],
                "krishi_bhavans": ["Neyyattinkara", "Thiruvananthapuram", "Nedumangad", "Varkala"],
                "schemes": ["Coconut Development Board schemes", "Coastal aquaculture support"],
                "soil_issues": ["Acidity in laterite soils", "Salt intrusion in coastal areas"],
                "recommendations": [
                    "Lime application for acidic soils",
                    "Drip irrigation for water conservation",
                    "Organic farming promotion"
                ]
            },
            "Kollam": {
                "region": "Southern Kerala",
                "major_crops": ["Coconut", "Rice", "Cashew", "Banana", "Spices"],
                "soil_types": ["Laterite", "Alluvial", "Coastal sandy"],
                "climate": "Tropical",
                "rainfall": "2000-2500mm annually",
                "irrigation": "Rivers, canals, and tanks",
                "specialties": ["Cashew cultivation", "Coconut processing", "Spice farming"],
                "krishi_bhavans": ["Kollam", "Karunagappally", "Kunnathur", "Pathanapuram"],
                "schemes": ["Cashew development schemes", "Spice Board initiatives"],
                "soil_issues": ["Erosion on slopes", "Waterlogging in low areas"],
                "recommendations": [
                    "Contour farming on slopes",
                    "Drainage improvement in low-lying areas",
                    "Cashew rejuvenation programs"
                ]
            },
            "Pathanamthitta": {
                "region": "Central Kerala",
                "major_crops": ["Rice", "Rubber", "Spices", "Tapioca", "Vegetables"],
                "soil_types": ["Laterite", "Forest soils", "Alluvial"],
                "climate": "Tropical highland",
                "rainfall": "2500-3000mm annually",
                "irrigation": "Rivers and streams",
                "specialties": ["Rubber plantation", "Spice cultivation", "Hill agriculture"],
                "krishi_bhavans": ["Pathanamthitta", "Adoor", "Ranni", "Mallappally"],
                "schemes": ["Rubber Board schemes", "Hill area development"],
                "soil_issues": ["Soil erosion", "Nutrient leaching"],
                "recommendations": [
                    "Terrace farming",
                    "Cover cropping",
                    "Integrated nutrient management"
                ]
            },
            "Alappuzha": {
                "region": "Central Kerala",
                "major_crops": ["Rice", "Coconut", "Fish", "Duck farming"],
                "soil_types": ["Alluvial", "Kuttanad clay", "Coastal sandy"],
                "climate": "Tropical coastal",
                "rainfall": "2500mm annually",
                "irrigation": "Extensive canal network, Kuttanad below sea level farming",
                "specialties": ["Kuttanad rice cultivation", "Backwater aquaculture", "Coconut farming"],
                "krishi_bhavans": ["Alappuzha", "Cherthala", "Kuttanad", "Haripad"],
                "schemes": ["Kuttanad development packages", "Backwater aquaculture schemes"],
                "soil_issues": ["Salinity intrusion", "Waterlogging", "Soil acidity"],
                "recommendations": [
                    "Salt-tolerant varieties",
                    "Improved drainage systems",
                    "Integrated farming systems"
                ]
            },
            "Kottayam": {
                "region": "Central Kerala",
                "major_crops": ["Rubber", "Rice", "Spices", "Coconut", "Banana"],
                "soil_types": ["Laterite", "Alluvial", "Hill soils"],
                "climate": "Tropical",
                "rainfall": "2500-3000mm annually",
                "irrigation": "Rivers, tanks, and wells",
                "specialties": ["Rubber cultivation", "Spice farming", "Mixed farming systems"],
                "krishi_bhavans": ["Kottayam", "Changanassery", "Vaikom", "Ettumanoor"],
                "schemes": ["Rubber development schemes", "Spice promotion programs"],
                "soil_issues": ["Acidity", "Erosion on slopes"],
                "recommendations": [
                    "Lime application",
                    "Soil conservation measures",
                    "Agroforestry systems"
                ]
            },
            "Idukki": {
                "region": "Eastern Kerala (High Ranges)",
                "major_crops": ["Tea", "Coffee", "Cardamom", "Pepper", "Vegetables"],
                "soil_types": ["Hill soils", "Forest soils", "Rocky soils"],
                "climate": "Tropical highland",
                "rainfall": "2000-4000mm annually",
                "irrigation": "Hill streams and springs",
                "specialties": ["Tea plantations", "Cardamom cultivation", "Coffee farming"],
                "krishi_bhavans": ["Thodupuzha", "Kumily", "Munnar", "Devikulam"],
                "schemes": ["Tea Board schemes", "Cardamom Board programs", "Hill area development"],
                "soil_issues": ["Soil erosion", "Nutrient depletion", "Landslides"],
                "recommendations": [
                    "Contour planting",
                    "Shade tree management",
                    "Soil conservation structures"
                ]
            },
            "Ernakulam": {
                "region": "Central Kerala",
                "major_crops": ["Coconut", "Rice", "Vegetables", "Spices", "Flowers"],
                "soil_types": ["Laterite", "Alluvial", "Coastal sandy"],
                "climate": "Tropical coastal",
                "rainfall": "2400mm annually",
                "irrigation": "Rivers, canals, and modern irrigation systems",
                "specialties": ["Coconut farming", "Vegetable cultivation", "Floriculture"],
                "krishi_bhavans": ["Ernakulam", "Aluva", "Muvattupuzha", "Kothamangalam"],
                "schemes": ["Coconut development", "Vegetable promotion", "Flower cultivation"],
                "soil_issues": ["Urban pressure", "Water scarcity", "Soil degradation"],
                "recommendations": [
                    "Urban agriculture promotion",
                    "Water harvesting",
                    "Organic farming methods"
                ]
            },
            "Thrissur": {
                "region": "Central Kerala",
                "major_crops": ["Rice", "Coconut", "Vegetables", "Banana", "Pepper"],
                "soil_types": ["Laterite", "Alluvial", "Black cotton"],
                "climate": "Tropical",
                "rainfall": "2800mm annually",
                "irrigation": "Rivers, tanks, and lift irrigation",
                "specialties": ["Rice cultivation", "Vegetable farming", "Coconut processing"],
                "krishi_bhavans": ["Thrissur", "Chalakudy", "Kodungallur", "Irinjalakuda"],
                "schemes": ["Rice development programs", "Vegetable mission", "Coconut Board schemes"],
                "soil_issues": ["Waterlogging", "Nutrient imbalance"],
                "recommendations": [
                    "Improved drainage",
                    "Balanced fertilization",
                    "Crop diversification"
                ]
            },
            "Palakkad": {
                "region": "Eastern Kerala",
                "major_crops": ["Rice", "Coconut", "Sugarcane", "Cotton", "Vegetables"],
                "soil_types": ["Alluvial", "Black cotton", "Red loam"],
                "climate": "Tropical dry",
                "rainfall": "1200-2000mm annually",
                "irrigation": "Bharathapuzha river system, tanks, and wells",
                "specialties": ["Rice bowl of Kerala", "Sugarcane cultivation", "Cotton farming"],
                "krishi_bhavans": ["Palakkad", "Ottappalam", "Mannarkkad", "Chittur"],
                "schemes": ["Rice mission", "Sugarcane development", "Irrigation improvement"],
                "soil_issues": ["Water scarcity", "Soil salinity", "Erosion"],
                "recommendations": [
                    "Water conservation techniques",
                    "Salt-tolerant varieties",
                    "Soil health management"
                ]
            },
            "Malappuram": {
                "region": "Northern Kerala",
                "major_crops": ["Coconut", "Rice", "Arecanut", "Pepper", "Ginger"],
                "soil_types": ["Laterite", "Alluvial", "Hill soils"],
                "climate": "Tropical",
                "rainfall": "2500-3000mm annually",
                "irrigation": "Rivers, streams, and wells",
                "specialties": ["Coconut cultivation", "Arecanut farming", "Spice production"],
                "krishi_bhavans": ["Malappuram", "Perinthalmanna", "Tirur", "Nilambur"],
                "schemes": ["Coconut development", "Arecanut Board schemes", "Spice promotion"],
                "soil_issues": ["Acidity", "Erosion", "Laterite hardpan"],
                "recommendations": [
                    "Lime application",
                    "Organic matter addition",
                    "Improved varieties"
                ]
            },
            "Kozhikode": {
                "region": "Northern Kerala",
                "major_crops": ["Coconut", "Rice", "Pepper", "Ginger", "Turmeric"],
                "soil_types": ["Laterite", "Coastal sandy", "Alluvial"],
                "climate": "Tropical coastal",
                "rainfall": "3000mm annually",
                "irrigation": "Rivers, wells, and traditional systems",
                "specialties": ["Spice cultivation", "Coconut farming", "Traditional agriculture"],
                "krishi_bhavans": ["Kozhikode", "Vadakara", "Koyilandy", "Balussery"],
                "schemes": ["Spice Board programs", "Coconut development", "Organic farming"],
                "soil_issues": ["Acidity", "Nutrient deficiency", "Waterlogging"],
                "recommendations": [
                    "Soil pH management",
                    "Micronutrient application",
                    "Drainage improvement"
                ]
            },
            "Wayanad": {
                "region": "Northern Kerala (High Ranges)",
                "major_crops": ["Coffee", "Tea", "Pepper", "Cardamom", "Vanilla"],
                "soil_types": ["Hill soils", "Forest soils", "Red loam"],
                "climate": "Tropical highland",
                "rainfall": "2000-4000mm annually",
                "irrigation": "Hill streams and natural springs",
                "specialties": ["Coffee plantations", "Spice cultivation", "Organic farming"],
                "krishi_bhavans": ["Kalpetta", "Mananthavady", "Sulthan Bathery"],
                "schemes": ["Coffee Board schemes", "Organic farming promotion", "Tribal area development"],
                "soil_issues": ["Soil erosion", "Organic matter depletion", "Landslides"],
                "recommendations": [
                    "Contour cultivation",
                    "Organic matter management",
                    "Agroforestry practices"
                ]
            },
            "Kannur": {
                "region": "Northern Kerala",
                "major_crops": ["Coconut", "Rice", "Arecanut", "Cashew", "Spices"],
                "soil_types": ["Laterite", "Coastal sandy", "Alluvial"],
                "climate": "Tropical coastal",
                "rainfall": "2500-3500mm annually",
                "irrigation": "Rivers, tanks, and wells",
                "specialties": ["Coconut cultivation", "Cashew farming", "Handloom crops"],
                "krishi_bhavans": ["Kannur", "Thalassery", "Payyanur", "Iritty"],
                "schemes": ["Coconut Board schemes", "Cashew development", "Handloom promotion"],
                "soil_issues": ["Acidity", "Coastal erosion", "Salt intrusion"],
                "recommendations": [
                    "Soil pH correction",
                    "Coastal protection measures",
                    "Salt-resistant varieties"
                ]
            },
            "Kasaragod": {
                "region": "Northern Kerala",
                "major_crops": ["Coconut", "Rice", "Arecanut", "Cashew", "Pepper"],
                "soil_types": ["Laterite", "Coastal sandy", "Red loam"],
                "climate": "Tropical coastal",
                "rainfall": "3000-4000mm annually",
                "irrigation": "Rivers, wells, and traditional systems",
                "specialties": ["Coconut cultivation", "Arecanut farming", "Traditional farming"],
                "krishi_bhavans": ["Kasaragod", "Kanhangad", "Hosdurg"],
                "schemes": ["Coconut development", "Arecanut Board programs", "Border area development"],
                "soil_issues": ["High acidity", "Laterite hardpan", "Coastal erosion"],
                "recommendations": [
                    "Intensive lime application",
                    "Deep ploughing",
                    "Coastal management"
                ]
            }
        }

    def get_district_info(self, district_name: str) -> Optional[Dict]:
        """Get comprehensive information about a specific district"""
        return self.districts.get(district_name, None)

    def get_all_districts(self) -> List[str]:
        """Get list of all Kerala districts"""
        return list(self.districts.keys())

    def get_districts_by_region(self, region: str) -> List[str]:
        """Get districts in a specific region"""
        return [district for district, info in self.districts.items() 
                if region.lower() in info.get("region", "").lower()]

    def get_district_recommendations(self, district_name: str, crop: str = None) -> str:
        """Get formatted recommendations for a district"""
        district_info = self.get_district_info(district_name)
        if not district_info:
            return f"Information not available for {district_name}. Please consult your local Krishi Bhavan."

        recommendations = f"""
🏞️ **{district_name} District Agricultural Advisory**

**📍 Region:** {district_info.get('region', 'N/A')}
**🌦️ Climate:** {district_info.get('climate', 'N/A')}
**🌧️ Rainfall:** {district_info.get('rainfall', 'N/A')}

**🌾 Major Crops:**
{chr(10).join(['• ' + crop for crop in district_info.get('major_crops', [])])}

**🏔️ Soil Types:**
{chr(10).join(['• ' + soil for soil in district_info.get('soil_types', [])])}

**💧 Irrigation Sources:**
{district_info.get('irrigation', 'N/A')}

**🌟 Agricultural Specialties:**
{chr(10).join(['• ' + specialty for specialty in district_info.get('specialties', [])])}

**⚠️ Common Soil Issues:**
{chr(10).join(['• ' + issue for issue in district_info.get('soil_issues', [])])}

**✅ Recommendations:**
{chr(10).join(['• ' + rec for rec in district_info.get('recommendations', [])])}

**🏢 Government Schemes:**
{chr(10).join(['• ' + scheme for scheme in district_info.get('schemes', [])])}

**📞 Krishi Bhavan Offices:**
{chr(10).join(['• ' + office for office in district_info.get('krishi_bhavans', [])])}

*For specific queries, contact your nearest Krishi Bhavan office.*
"""
        
        if crop:
            crop_specific = self._get_crop_specific_advice(district_name, crop)
            if crop_specific:
                recommendations += f"\n\n**🌱 {crop} Specific Advice for {district_name}:**\n{crop_specific}"
        
        return recommendations

    def _get_crop_specific_advice(self, district_name: str, crop: str) -> str:
        """Get crop-specific advice for a district"""
        district_info = self.get_district_info(district_name)
        if not district_info:
            return ""
        
        crop = crop.lower()
        advice = []
        
        # General crop advice based on district characteristics
        if crop in ["rice", "paddy"]:
            if "Kuttanad" in district_info.get('specialties', []):
                advice.append("Use salt-tolerant varieties suitable for below sea-level cultivation")
            if "waterlogging" in str(district_info.get('soil_issues', [])).lower():
                advice.append("Ensure proper drainage before sowing")
            if district_name == "Palakkad":
                advice.append("Take advantage of the fertile alluvial soils")
                
        elif crop in ["coconut"]:
            if "coastal" in district_info.get('climate', '').lower():
                advice.append("Monitor for salt intrusion and use appropriate varieties")
            if "acidity" in str(district_info.get('soil_issues', [])).lower():
                advice.append("Apply lime regularly to manage soil pH")
                
        elif crop in ["rubber"]:
            if district_name in ["Kottayam", "Pathanamthitta", "Idukki"]:
                advice.append("Follow Rubber Board recommendations for this traditional rubber growing area")
            if "erosion" in str(district_info.get('soil_issues', [])).lower():
                advice.append("Plant cover crops between rubber trees")
                
        elif crop in ["tea", "coffee"]:
            if "hill" in district_info.get('region', '').lower():
                advice.append("Maintain proper shade management")
                advice.append("Implement soil conservation measures on slopes")
                
        elif crop in ["pepper", "cardamom", "spices"]:
            if district_name in ["Idukki", "Wayanad", "Kozhikode"]:
                advice.append("This district is well-suited for spice cultivation")
            advice.append("Ensure proper drainage during monsoon season")
        
        return "\n".join([f"• {tip}" for tip in advice]) if advice else ""

    def find_suitable_districts(self, crop: str) -> List[str]:
        """Find districts suitable for a specific crop"""
        crop = crop.lower()
        suitable_districts = []
        
        for district, info in self.districts.items():
            major_crops = [c.lower() for c in info.get('major_crops', [])]
            if any(crop in mc for mc in major_crops):
                suitable_districts.append(district)
        
        return suitable_districts

    def get_region_overview(self, region: str) -> str:
        """Get overview of agricultural practices in a region"""
        districts = self.get_districts_by_region(region)
        if not districts:
            return f"No districts found for region: {region}"
        
        overview = f"""
🌏 **{region} Agricultural Overview**

**Districts:** {', '.join(districts)}

**Common Agricultural Features:**
"""
        
        # Aggregate common crops and practices
        all_crops = set()
        all_specialties = set()
        
        for district in districts:
            info = self.get_district_info(district)
            if info:
                all_crops.update(info.get('major_crops', []))
                all_specialties.update(info.get('specialties', []))
        
        overview += f"\n**Major Crops:** {', '.join(sorted(all_crops))}"
        overview += f"\n**Specialties:** {', '.join(sorted(all_specialties))}"
        
        return overview