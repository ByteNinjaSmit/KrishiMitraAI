import streamlit as st
import os
from rag_system import KrishiMitraRAG
from utils import detect_language, translate_to_english
from crop_calendar import KeralaAgriculturalCalendar
from kerala_districts import KeralaDistrictAdvisory
from weather_integration import WeatherAdvisory
from voice_assistant import VoiceAssistant, add_voice_capabilities_to_chat
import time

# Page configuration
st.set_page_config(
    page_title="Krishi Mitra - Agricultural Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agricultural theme
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E7D32;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #388E3C;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .farmer-message {
        background-color: #E8F5E8;
        border-left-color: #2E7D32;
    }
    .advisor-message {
        background-color: #F1F8E9;
        border-left-color: #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'agricultural_calendar' not in st.session_state:
    st.session_state.agricultural_calendar = KeralaAgriculturalCalendar()
if 'district_advisory' not in st.session_state:
    st.session_state.district_advisory = KeralaDistrictAdvisory()
if 'weather_advisory' not in st.session_state:
    st.session_state.weather_advisory = WeatherAdvisory()
if 'voice_assistant' not in st.session_state:
    st.session_state.voice_assistant = VoiceAssistant()

# Header
st.markdown('<h1 class="main-header">🌾 Krishi Mitra</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Kerala Farmers\' AI-Powered Agricultural Advisor | കേരള കർഷകരുടെ AI അധിഷ്ഠിത കാർഷിക ഉപദേശകൻ</p>', unsafe_allow_html=True)

# Helper to submit a question programmatically (used by sample questions)
def _submit_question(question: str):
    if not question:
        return
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    # Generate response immediately
    with st.spinner("Krishi Mitra is thinking..."):
        try:
            if st.session_state.rag_system is not None:
                response = st.session_state.rag_system.get_response(question)
            else:
                response = "System not initialized. Please refresh the page and try again."
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}. Please try again or consult your local Krishi Bhavan office."
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Sidebar for system status and information
with st.sidebar:
    st.header("📋 System Status")
    
    # Initialize RAG system
    if not st.session_state.system_initialized:
        with st.spinner("Initializing Krishi Mitra... Please wait..."):
            try:
                st.session_state.rag_system = KrishiMitraRAG()
                st.session_state.rag_system.initialize_system()
                st.session_state.system_initialized = True
                st.success("✅ System Ready!")
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                st.error("Please check your OpenAI API key and try again.")
    else:
        st.success("✅ System Ready!")
    
    # Show vector index build status
    if st.session_state.rag_system is not None:
        status = getattr(st.session_state.rag_system, 'build_status', None)
        if status:
            ready = status.get('vectorstore_ready', False)
            device = status.get('device', 'cpu')
            msg = status.get('last_message', '')
            if ready:
                st.info(f"Vector store: Ready on {device.upper()}")
            else:
                st.warning(f"Vector store building on {device.upper()} — {msg}")
    
    st.header("📚 Knowledge Base")
    if st.session_state.system_initialized and st.session_state.rag_system is not None:
        try:
            docs = getattr(st.session_state.rag_system, 'documents', [])
            stats = st.session_state.rag_system.document_processor.get_document_stats(docs) if docs else None
            if stats:
                st.markdown(f"**Total chunks indexed:** {stats.get('total_documents', 0)}")
                st.markdown("**By document type:**")
                for k, v in stats.get('document_types', {}).items():
                    st.markdown(f"- {k}: {v}")
                st.caption("Average chunk length: {:.0f} chars".format(stats.get('avg_content_length', 0)))
        except Exception:
            st.info("Knowledge base loaded: Policy PDF and FAQ CSV files")
    else:
        st.info("Knowledge base will load after initialization")
    
    st.header("🗣️ Language Support")
    st.info("Supports Malayalam & English queries")
    
    st.header("📞 Contact")
    st.info("For additional help, contact your local Krishi Bhavan office")

# Main interface tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat with Krishi Mitra", "📸 Crop Disease Detection", "🗓️ Seasonal Calendar", "📍 District Advisory", "🌤️ Weather Advisory"])

with tab1:
    st.header("💬 Ask Krishi Mitra")
    
    # Voice input interface
    voice_assistant = add_voice_capabilities_to_chat()

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message farmer-message">
                <strong>👨‍🌾 You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message advisor-message">
                <strong>🤖 Krishi Mitra:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Add voice output for assistant responses
            if len(message["content"]) > 20:
                voice_assistant.create_voice_output_for_response(message["content"])

    # Chat input
    if st.session_state.system_initialized:
        user_question = st.chat_input("Ask your agricultural question in Malayalam or English...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Display user message
            st.markdown(f"""
            <div class="chat-message farmer-message">
                <strong>👨‍🌾 You:</strong> {user_question}
            </div>
            """, unsafe_allow_html=True)
            
            # Generate response
            with st.spinner("Krishi Mitra is thinking..."):
                try:
                    # Detect language
                    detected_lang = detect_language(user_question)
                    
                    # Get response from RAG system
                    if st.session_state.rag_system is not None:
                        response = st.session_state.rag_system.get_response(user_question)
                    else:
                        response = "System not initialized. Please refresh the page and try again."
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display assistant response
                    st.markdown(f"""
                    <div class="chat-message advisor-message">
                        <strong>🤖 Krishi Mitra:</strong> {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-scroll to bottom
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again or consult your local Krishi Bhavan office."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    else:
        st.warning("⚠️ Please wait for the system to initialize before asking questions.")

# Crop Disease Detection Tab
with tab2:
    st.header("📸 Crop Disease Detection")
    st.write("Upload a photo of your crop to get disease identification and treatment recommendations")
    
    if st.session_state.system_initialized:
        uploaded_file = st.file_uploader(
            "Choose a crop image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo of the affected crop or plant"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image alongside actions (small, fixed width)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", width=320)
            
            with col2:
                if st.button("🔍 Analyze Crop", type="primary"):
                    with st.spinner("Analyzing your crop image..."):
                        try:
                            # Save uploaded file temporarily
                            import tempfile
                            import os
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Analyze image using RAG system
                            if st.session_state.rag_system is not None:
                                analysis = st.session_state.rag_system.analyze_crop_image(tmp_file_path)
                                
                                st.markdown(f"""
                                <div class="chat-message advisor-message">
                                    <strong>🤖 Crop Analysis:</strong><br>
                                    {analysis}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add to chat history
                                st.session_state.messages.append({
                                    "role": "user", 
                                    "content": "📸 Uploaded crop image for disease analysis"
                                })
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": f"🔍 **Crop Analysis:** {analysis}"
                                })
                            # After showing analysis result
                            try:
                                st.session_state.voice_assistant.create_voice_output_for_response(analysis)
                            except Exception:
                                pass

                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                            
                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")
            
            st.markdown("""
            **Tips for better analysis:**
            - Take photos in good lighting
            - Focus on affected areas of the plant
            - Include leaves, stems, or fruits showing symptoms
            - Avoid blurry or dark images
            """)
    else:
        st.warning("⚠️ Please wait for the system to initialize before using disease detection.")

# Seasonal Calendar Tab
with tab3:
    st.header("🗓️ Kerala Agricultural Calendar")
    st.write("Get month-wise crop recommendations and agricultural activities for Kerala")
    
    # Month selector
    col1, col2 = st.columns([1, 2])
    
    with col1:
        import datetime
        current_month = datetime.datetime.now().month
        month_names = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        selected_month = st.selectbox(
            "Select Month:",
            range(1, 13),
            index=current_month - 1,
            format_func=lambda x: f"{month_names[x-1]} ({x})",
            help="Choose a month to see agricultural recommendations"
        )
    
    with col2:
        if st.button("📅 Get Current Month Recommendations", type="primary"):
            selected_month = current_month
    
    # Display calendar recommendations
    calendar_advice = st.session_state.agricultural_calendar.format_monthly_advice(selected_month)
    st.markdown(calendar_advice)
    
    # Additional features
    st.subheader("🔍 Crop-Specific Information")
    
    kerala_crops = ["Rice", "Coconut", "Pepper", "Cardamom", "Banana", "Ginger", "Turmeric", "Coffee", "Tea", "Rubber"]
    selected_crop = st.selectbox("Select a crop for detailed information:", kerala_crops)
    
    if st.button(f"Get {selected_crop} Information"):
        crop_info = st.session_state.agricultural_calendar.get_crop_info(selected_crop.lower())
        planting_calendar = st.session_state.agricultural_calendar.get_planting_calendar(selected_crop.lower())
        
        if crop_info:
            st.info(f"""
            **{selected_crop} Information:**
            - **Varieties:** {crop_info.get('varieties', 'N/A')}
            - **Growing Seasons:** {crop_info.get('seasons', 'N/A')}
            - **Duration:** {crop_info.get('duration', 'N/A')}
            """)
        
        if planting_calendar:
            months = [f"{month_names[month-1]} ({season})" for month, season in planting_calendar]
            st.success(f"**Best planting months for {selected_crop}:** {', '.join(months)}")
    
    # Quick seasonal recommendations
    st.subheader("🌦️ Seasonal Overview")
    seasons = ["Kharif", "Rabi", "Summer"]
    selected_season = st.selectbox("Select season for crop recommendations:", seasons)
    
    if st.button(f"Show {selected_season} Crops"):
        seasonal_crops = st.session_state.agricultural_calendar.get_seasonal_crops(selected_season)
        if seasonal_crops:
            st.success(f"**{selected_season} Season Crops:** {', '.join(seasonal_crops)}")
        else:
            st.warning(f"No specific crops found for {selected_season} season.")

# District Advisory Tab
with tab4:
    st.header("📍 Kerala District Agricultural Advisory")
    st.write("Get location-specific agricultural recommendations for your district")
    
    # District selector
    col1, col2 = st.columns([1, 1])
    
    with col1:
        districts = st.session_state.district_advisory.get_all_districts()
        selected_district = st.selectbox(
            "Select your district:",
            districts,
            help="Choose your district to get location-specific agricultural advice"
        )
    
    with col2:
        # Optional crop filter
        crop_filter = st.selectbox(
            "Specific crop (optional):",
            ["All crops"] + ["Rice", "Coconut", "Pepper", "Rubber", "Tea", "Coffee", "Cardamom", "Banana", "Cashew"],
            help="Get crop-specific advice for your district"
        )
    
    # Display district recommendations
    if st.button("🏞️ Get District Advisory", type="primary"):
        crop = None if crop_filter == "All crops" else crop_filter
        district_advice = st.session_state.district_advisory.get_district_recommendations(selected_district, crop)
        st.markdown(district_advice)
    
    # Additional district features
    st.subheader("🔍 Explore by Region")
    
    regions = ["Southern Kerala", "Central Kerala", "Northern Kerala", "Eastern Kerala", "High Ranges"]
    selected_region = st.selectbox("Select region:", regions)
    
    if st.button(f"Show {selected_region} Overview"):
        region_overview = st.session_state.district_advisory.get_region_overview(selected_region)
        st.info(region_overview)
    
    # Crop suitability finder
    st.subheader("🌾 Find Suitable Districts")
    
    crop_to_find = st.selectbox(
        "Which crop are you interested in?",
        ["Rice", "Coconut", "Pepper", "Rubber", "Tea", "Coffee", "Cardamom", "Banana", "Cashew", "Spices"]
    )
    
    if st.button(f"Find Districts for {crop_to_find}"):
        suitable_districts = st.session_state.district_advisory.find_suitable_districts(crop_to_find)
        if suitable_districts:
            st.success(f"**Districts suitable for {crop_to_find} cultivation:** {', '.join(suitable_districts)}")
        else:
            st.warning(f"No specific districts found for {crop_to_find} in our database.")
    
    # Quick tips section
    st.subheader("💡 District-wise Quick Tips")
    st.info("""
    **How to use District Advisory:**
    - Select your district for location-specific recommendations
    - Choose a specific crop to get targeted advice
    - Explore regional overviews to understand your area's agriculture
    - Find which districts are best for specific crops
    - Contact your local Krishi Bhavan for personalized guidance
    """)

# Weather Advisory Tab
with tab5:
    st.header("🌤️ Weather-Based Agricultural Advisory")
    st.write("Get real-time weather information and agricultural recommendations")
    
    # Location selector
    col1, col2 = st.columns([1, 1])
    
    with col1:
        districts = list(st.session_state.weather_advisory.kerala_locations.keys())
        selected_location = st.selectbox(
            "Select your location:",
            districts,
            help="Choose your district to get weather-based agricultural advice"
        )
    
    with col2:
        if st.button("🌤️ Get Weather Advisory", type="primary"):
            with st.spinner("Fetching weather data..."):
                # Get weather data (using mock data for demo)
                weather_data = st.session_state.weather_advisory.get_mock_weather_data(selected_location)
                
                # Display weather recommendations
                weather_recommendations = st.session_state.weather_advisory.get_weather_recommendations(weather_data)
                st.markdown(weather_recommendations)
                
                # Check for weather alerts
                alerts = st.session_state.weather_advisory.format_weather_alert(weather_data)
                if alerts:
                    st.warning(f"⚠️ **Weather Alerts:**\n{alerts}")
    
    # Seasonal weather patterns
    st.subheader("📅 Seasonal Weather Patterns")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        import datetime
        current_month = datetime.datetime.now().month
        month_names = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        selected_month = st.selectbox(
            "Select Month for Seasonal Pattern:",
            range(1, 13),
            index=current_month - 1,
            format_func=lambda x: f"{month_names[x-1]} ({x})"
        )
    
    with col2:
        if st.button("📊 Show Seasonal Pattern"):
            seasonal_advice = st.session_state.weather_advisory.get_seasonal_weather_advisory(selected_month)
            st.info(seasonal_advice)
    
    # Weather tips
    st.subheader("💡 Weather-Smart Farming Tips")
    
    with st.expander("☀️ Sunny Weather Tips"):
        st.write("""
        - **Best for:** Harvesting, field preparation, fertilizer application
        - **Irrigation:** Increase frequency, early morning or evening
        - **Crops:** Provide shade for sensitive vegetables
        - **Precautions:** Protect young plants, ensure water supply
        """)
    
    with st.expander("🌧️ Rainy Weather Tips"):
        st.write("""
        - **Best for:** Monsoon crop planting, natural irrigation
        - **Drainage:** Ensure proper water drainage systems
        - **Disease:** Monitor for fungal diseases, apply preventive treatments
        - **Precautions:** Avoid field operations during heavy rain
        """)
    
    with st.expander("☁️ Cloudy Weather Tips"):
        st.write("""
        - **Best for:** Most agricultural activities, planting
        - **Conditions:** Moderate temperature and humidity
        - **Crops:** Ideal for coffee, tea, and vegetables
        - **Monitor:** Humidity levels and potential pest buildup
        """)
    
    with st.expander("🔥 Hot Weather Tips"):
        st.write("""
        - **Best for:** Early morning operations only
        - **Water:** Increase irrigation frequency significantly
        - **Protection:** Provide shade, use mulching
        - **Avoid:** Midday field work, transplanting
        """)

# Sample questions section
st.header("💡 Sample Questions")
col1, col2 = st.columns(2)

with col1:
    st.subheader("English")
    sample_questions_en = [
        "What is the best time to plant rice in Kerala?",
        "How can I control pests in my coconut farm?",
        "What fertilizer is good for pepper cultivation?",
        "How to apply for agricultural subsidies?",
        "What are the water management policies in Kerala?"
    ]
    
    for question in sample_questions_en:
        if st.button(question, key=f"en_{question}"):
            _submit_question(question)

with col2:
    st.subheader("Malayalam")
    sample_questions_ml = [
        "കേരളത്തിൽ നെല്ല് നടാനുള്ള ഏറ്റവും നല്ല സമയം എന്താണ്?",
        "തേങ്ങയിൽ കീടങ്ങളെ എങ്ങനെ നിയന്ത്രിക്കാം?",
        "കുരുമുളക് കൃഷിക്ക് ഏത് വളം നല്ലത്?",
        "കാർഷിക സബ്സിഡിക്ക് എങ്ങനെ അപേക്ഷിക്കാം?",
        "കേരളത്തിലെ ജല നയങ്ങൾ എന്തൊക്കെയാണ്?"
    ]
    
    for question in sample_questions_ml:
        if st.button(question, key=f"ml_{question}"):
            _submit_question(question)

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 Krishi Mitra - Empowering Kerala Farmers with AI | Powered by OpenAI GPT-5</p>
    <p>For technical support or feedback, contact your local agricultural extension office</p>
</div>
""", unsafe_allow_html=True)
