import os
from groq import Groq
from dotenv import load_dotenv
import json
import random
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv('.env.local')  # Specifically load .env.local

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# CLIP Functions (copied from main.py to avoid circular imports)
def load_and_preprocess_image(image_path):
    """Image loading and preprocessing"""
    image = Image.open(image_path).convert("RGB")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    return inputs, processor

def generate_image_embeddings(inputs):
    """Image understanding with CLIP"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features, model

def match_captions(image_features, captions, clip_model, processor):
    """Compare image features to text features of all possible captions"""
    # Get text embeddings for the captions
    text_inputs = processor(text=captions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)

    # Calculate cosine similarity between image and text features
    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()

    similarities = cosine_similarity(image_features, text_features)

    # Find the best matching captions
    best_indices = similarities.argsort(axis=1)[0][::-1]
    best_captions = [captions[i] for i in best_indices]

    return best_captions, similarities[0][best_indices].tolist()

# Default candidate captions
default_candidate_captions = [
    "Trees, Travel and Tea!",
    "A refreshing beverage.",
    "A moment of indulgence.",
    "The perfect thirst quencher.",
    "Your daily dose of delight.",
    "Taste the tradition.",
    "Savor the flavor.",
    "Refresh and rejuvenate.",
    "Unwind and enjoy.",
    "The taste of home.",
    "A treat for your senses.",
    "A taste of adventure.",
    "A moment of bliss.",
    "Your travel companion.",
    "Fuel for your journey.",
    "The essence of nature.",
    "The warmth of comfort.",
    "A sip of happiness.",
    "Pure indulgence.",
    "Quench your thirst, ignite your spirit."
]

def generate_captions_with_llm(prompt, max_tokens=500, temperature=0.8):
    """
    Base function to generate captions using Groq/Llama
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",  # or "llama3-70b-8192" for better quality
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating captions: {e}")
        return None
    
def generate_emotional_captions(emotion_type, count=10):
    """
    Generate emotion-focused captions
    emotion_type: 'happy', 'calm', 'energetic', 'nostalgic', 'mysterious'
    """
    prompt = f"""
    Generate {count} {emotion_type} and emotionally engaging captions for images.
    Focus on evoking {emotion_type} feelings.
    
    Requirements:
    - Each caption should be 5-20 words
    - Vary the style and approach
    - Make them suitable for social media or marketing
    - Return ONLY a Python list format: ["caption1", "caption2", ...]
    
    Emotion focus: {emotion_type}
    """
    
    response = generate_captions_with_llm(prompt)
    return parse_caption_response(response)

def generate_marketing_captions(focus_type, count=10):
    """
    Generate marketing-focused captions
    focus_type: 'product', 'lifestyle', 'experience', 'benefit'
    """
    # Similar structure to emotional captions
    # You'll implement the prompt based on marketing focus

def generate_length_specific_captions(length_type, count=10):
    """
    Generate captions of specific lengths
    length_type: 'short', 'medium', 'long'
    """
    # You'll implement this with length constraints

def generate_industry_captions(industry, count=10):
    """
    Generate industry-specific captions
    industry: 'food', 'travel', 'tech', 'fashion', 'health'
    """
    # You'll implement industry-specific prompts
    
    
    
def parse_caption_response(response):
    """
    Parse LLM response and extract captions
    """
    if not response:
        return []
    
    try:
        captions = []
        
        # Method 1: Try to find JSON-like list
        import re
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                # Clean and evaluate the JSON-like string
                json_str = json_match.group(0)
                captions = eval(json_str)
                if isinstance(captions, list):
                    return [str(cap).strip().strip('"').strip("'") for cap in captions if cap and len(str(cap).strip()) > 3][:20]
            except:
                pass
        
        # Method 2: Split by lines and clean
        lines = response.strip().split('\n')
        for line in lines:
            # Skip metadata lines
            if any(skip in line.lower() for skip in ['here are', 'let me know', 'requirements', 'generate']):
                continue
                
            # Clean the line
            cleaned = line.strip().strip('"').strip("'").strip('- ').strip('`').strip()
            
            # Check if it's a valid caption
            if (cleaned and 
                len(cleaned) > 3 and 
                len(cleaned) < 200 and
                not cleaned.startswith(('http', 'www', '#')) and
                not cleaned.endswith((':', '?', 'caption')):
                captions.append(cleaned)
        
        return captions[:20]  # Limit to 20 captions
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []
    
    
class CaptionPoolManager:
    def __init__(self):
        self.caption_pools = {
            'emotional': {},
            'marketing': {},
            'length': {},
            'industry': {}
        }
        self.default_captions = default_candidate_captions
    
    def generate_diverse_pool(self, total_captions=100):
        """
        Generate a diverse pool of captions from all categories
        """
        all_captions = []
        
        # Generate emotional captions (for now, just implement this one)
        try:
            emotions = ['happy', 'calm', 'energetic', 'nostalgic', 'mysterious']
            for emotion in emotions:
                emotional_caps = generate_emotional_captions(emotion, 8)
                all_captions.extend(emotional_caps)
        except Exception as e:
            print(f"Error generating emotional captions: {e}")
        
        # Add default captions as backup
        all_captions.extend(self.default_captions)
        
        # Remove duplicates and limit
        unique_captions = list(set(all_captions))
        return unique_captions[:total_captions]
    
    def get_balanced_caption_set(self, image_context="", count=50):
        """
        Get a balanced set of captions based on image context
        """
        try:
            diverse_pool = self.generate_diverse_pool(count * 2)  # Generate more to select from
            if len(diverse_pool) >= count:
                return random.sample(diverse_pool, count)
            else:
                return diverse_pool
        except Exception as e:
            print(f"Error generating balanced caption set: {e}")
            return self.default_captions[:count]
    
    
def image_captioning_with_llm_categories(image_path, use_llm=True, caption_context=""):
    """
    Enhanced image captioning with LLM-generated categories
    """
    if use_llm:
        print("🤖 Generating LLM captions...")
        caption_manager = CaptionPoolManager()
        candidate_captions = caption_manager.get_balanced_caption_set(
            image_context=caption_context, count=50
        )
        
        # Fallback to default if LLM fails
        if not candidate_captions:
            print("⚠️ LLM failed, using default captions")
            candidate_captions = default_candidate_captions
        else:
            print(f"✅ Generated {len(candidate_captions)} captions using LLM")
    else:
        candidate_captions = default_candidate_captions
        print(f"📝 Using {len(candidate_captions)} default captions")
    
    # Your existing image processing code
    inputs, processor = load_and_preprocess_image(image_path)
    image_features, clip_model = generate_image_embeddings(inputs)
    best_captions, similarities = match_captions(image_features, candidate_captions, clip_model, processor)
    
    return best_captions, similarities, candidate_captions

def test_llm_integration():
    """Test the LLM integration"""
    print("🧪 Testing Groq connection...")
    try:
        test_captions = generate_emotional_captions("happy", 3)
        print(f"✅ Generated {len(test_captions)} test captions:")
        for i, caption in enumerate(test_captions, 1):
            print(f"   {i}. {caption}")
        return len(test_captions) > 0
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LLM-ENHANCED IMAGE CAPTIONING SYSTEM")
    print("=" * 60)
    
    # Test LLM connection first
    if test_llm_integration():
        print("\n" + "=" * 60)
        print("📸 RUNNING IMAGE CAPTIONING WITH LLM")
        print("=" * 60)
        
        # Test image path - update this to your actual image
        image_path = "content/musab_talal.jpg"  # Change this to your image path
        
        try:
            # Run LLM-enhanced captioning
            llm_best, llm_similarities, llm_captions = image_captioning_with_llm_categories(
                image_path, 
                use_llm=True, 
                caption_context="casual photo"
            )
            
            print(f"\n🎯 Top 5 LLM-Enhanced Captions:")
            for i, (caption, similarity) in enumerate(zip(llm_best[:5], llm_similarities[:5])):
                print(f"   {i+1}. {caption} (Similarity: {similarity:.4f})")
            
            print(f"\n📊 Total captions generated: {len(llm_captions)}")
            
        except Exception as e:
            print(f"❌ Error running image captioning: {e}")
            print("💡 Make sure your image path is correct and the image exists")
    
    else:
        print("\n❌ LLM integration failed. Please check:")
        print("   1. Your .env.local file exists")
        print("   2. GROQ_API_KEY is set correctly")
        print("   3. You have internet connection")