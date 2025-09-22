import re
from typing import Optional

def detect_language(text: str) -> str:
    """
    Simple language detection for Malayalam and English
    Returns 'malayalam' or 'english'
    """
    try:
        # Check for Malayalam Unicode characters
        malayalam_pattern = r'[\u0D00-\u0D7F]'
        
        # Count Malayalam characters
        malayalam_chars = len(re.findall(malayalam_pattern, text))
        total_chars = len(re.findall(r'[^\s\d\W]', text))
        
        # If more than 30% characters are Malayalam, consider it Malayalam
        if total_chars > 0 and (malayalam_chars / total_chars) > 0.3:
            return 'malayalam'
        else:
            return 'english'
            
    except Exception as e:
        print(f"Error in language detection: {str(e)}")
        return 'english'  # Default to English

def translate_to_english(text: str) -> str:
    """
    Simple placeholder for translation - in production, 
    you might want to use Google Translate API or similar
    """
    # For now, return the original text
    # In production, implement actual translation
    return text

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\u0D00-\u0D7F.,!?()-]', '', text)
    
    return text

def format_agricultural_response(response: str) -> str:
    """Format response for better readability in agricultural context"""
    if not response:
        return ""
    
    # Add bullet points for lists
    lines = response.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # If line starts with a dash or number, format as bullet point
            if re.match(r'^[-•*]\s*', line) or re.match(r'^\d+\.\s*', line):
                pattern = r'^[-•*\d.]\s*'
                formatted_lines.append(f"• {re.sub(pattern, '', line)}")
            else:
                formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def extract_keywords(text: str) -> list:
    """Extract important agricultural keywords from text"""
    # Common agricultural keywords
    agricultural_keywords = [
        'crop', 'farming', 'fertilizer', 'pesticide', 'irrigation', 'seed',
        'harvest', 'plantation', 'soil', 'water', 'pest', 'disease',
        'rice', 'coconut', 'pepper', 'cardamom', 'rubber', 'tea', 'coffee',
        'banana', 'mango', 'cashew', 'ginger', 'turmeric', 'vegetables',
        'subsidy', 'scheme', 'policy', 'krishi bhavan', 'kisan credit'
    ]
    
    # Malayalam agricultural terms (basic)
    malayalam_terms = [
        'കൃഷി', 'നെല്ല്', 'തേങ്ങ', 'കുരുമുളക്', 'ഏലം', 'റബ്ബര്',
        'വള', 'കീടനാശിനി', 'ജലസേചനം', 'വിത്ത്', 'കൊയ്ത്ത്'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    # Check for English keywords
    for keyword in agricultural_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Check for Malayalam keywords
    for keyword in malayalam_terms:
        if keyword in text:
            found_keywords.append(keyword)
    
    return list(set(found_keywords))

def validate_agricultural_question(question: str) -> bool:
    """Validate if the question is agriculture-related"""
    keywords = extract_keywords(question)
    
    # If we found agricultural keywords, it's likely agriculture-related
    return len(keywords) > 0

def format_policy_reference(policy_text: str, policy_number: Optional[str] = None) -> str:
    """Format policy references for better display"""
    if not policy_text:
        return ""
    
    if policy_number:
        return f"**Policy {policy_number}:** {policy_text}"
    else:
        return f"**Policy Reference:** {policy_text}"

def get_fallback_message(language: str = 'english') -> str:
    """Get fallback message when no relevant information is found"""
    if language == 'malayalam':
        return "ഈ വിഷയത്തെക്കുറിച്ച് എനിക്ക് ഔദ്യോഗിക വിവരങ്ങൾ ഇല്ല. ദയവായി നിങ്ങളുടെ പ്രാദേശിക കൃഷി ഭവൻ ഓഫീസറെ സമീപിക്കുക."
    else:
        return "I don't have official details about that. Please consult your local Krishi Bhavan officer."
