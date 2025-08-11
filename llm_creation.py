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

def call_llm_api(prompt, max_tokens=500, temperature=0.8, system_prompt=None, count=50, caption_context="general image"):
    """
    Core LLM API interface function - handles communication with Groq/Llama model
    
    Args:
        prompt: User prompt to send to the LLM
        max_tokens: Maximum tokens for LLM response
        temperature: Creativity level (0.0-1.0)
        system_prompt: System instructions (auto-generated if None)
        count: Number of captions to generate (used in system prompt)
        caption_context: Image context (used in system prompt)
        
    Returns:
        str: Raw LLM response text
    """
    try:
        # Default system prompt if none provided - now uses variables
        if system_prompt is None:
            system_prompt = f"""You are an AI captioning engine using a multimodal model. Your task is to generate {count} short, catchy, and shareable captions for Instagram images, tailored to a Gen Z Pakistani audience.

Context: {caption_context}

Your expertise:
- Create captions with mixed emotions: happy, energetic, calm, nostalgic, mysterious
- Vary caption lengths (5-15 words each)
- Use casual, relatable, and trending Pakistani Gen Z slang and humor
- Blend English, Urdu, and Roman Urdu naturally, reflecting local digital culture
- Include pop culture references commonly used in Pakistan
- Prioritize captions that are witty, memeable, and have high social media engagement potential
- Make them shareable and engaging for Instagram

Requirements:
- Keep captions under 15 words
- Return as Python list format: ["caption1", "caption2", ...]
- Avoid offensive, political, or NSFW content
- Generate diverse, creative captions suitable for Instagram
- Output only the final list without any extra commentary or text"""

        from groq.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating captions: {e}")
        return None

    
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
                not cleaned.endswith((':', '?', 'caption'))):
                captions.append(cleaned)
        
        return captions[:20]  # Limit to 20 captions
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []
    
    
def generate_and_match_captions(image_path, use_llm=True, caption_context="", count=50):
    """
    Complete caption generation and matching pipeline
    
    1. Generates captions using LLM (or uses defaults)
    2. Processes image with CLIP 
    3. Matches captions to image using similarity scoring
    
    Args:
        image_path: Path to the image file
        use_llm: Whether to use LLM for caption generation
        caption_context: Context description for the image
        count: Number of captions to generate
        
    Returns:
        tuple: (best_captions, similarities, all_candidate_captions)
    """
    if use_llm:
        print("🤖 Generating LLM captions...")
        
        # Generate captions directly using LLM
        prompt = f"""Generate {count} creative, engaging captions for this image.

Context: {caption_context if caption_context else "general image"}

Focus on creating diverse captions that are:
- Shareable and engaging for Instagram
- Mix of different emotions and vibes
- Perfect for Pakistani Gen Z audience
- Naturally blending English, Urdu, and Roman Urdu

Return as Python list: ["caption1", "caption2", ...]"""
        
        response = call_llm_api(
            prompt, 
            max_tokens=800, 
            temperature=0.9,
            count=count,
            caption_context=caption_context
        )
        candidate_captions = parse_caption_response(response)
        
        # Fallback to default if LLM fails
        if not candidate_captions or len(candidate_captions) < 10:
            print("⚠️ LLM failed or generated insufficient captions, using default captions")
            candidate_captions = default_candidate_captions
        else:
            print(f"✅ Generated {len(candidate_captions)} captions using LLM")
    else:
        candidate_captions = default_candidate_captions
        print(f"📝 Using {len(candidate_captions)} default captions")

    # Image processing with CLIP
    inputs, processor = load_and_preprocess_image(image_path)
    image_features, clip_model = generate_image_embeddings(inputs)
    best_captions, similarities = match_captions(image_features, candidate_captions, clip_model, processor)
    
    return best_captions, similarities, candidate_captions

def test_llm_integration():
    """Test the LLM integration"""
    print("🧪 Testing Groq connection...")
    try:
        # Test with a simple caption generation
        test_prompt = "Generate 3 happy captions for an image. Return as list: ['caption1', 'caption2', 'caption3']"
        response = call_llm_api(
            test_prompt, 
            max_tokens=200, 
            temperature=0.7,
            count=3,
            caption_context="test image"
        )
        test_captions = parse_caption_response(response)
        
        if test_captions and len(test_captions) > 0:
            print(f"✅ Generated {len(test_captions)} test captions:")
            for i, caption in enumerate(test_captions[:3], 1):  # Show max 3
                print(f"   {i}. {caption}")
            return True
        else:
            print("❌ No captions generated")
            return False
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    
    # Test LLM connection first
    if test_llm_integration():
        print("\n" + "=" * 60)
        print("📸 RUNNING IMAGE CAPTIONING WITH LLM")
        print("=" * 60)
        
        # Test image path - update this to your actual image
        image_path = "content/musab_talal.jpg"  # Change this to your image path
        
        try:
            # Run LLM-enhanced captioning
            llm_best, llm_similarities, llm_captions = generate_and_match_captions(
                image_path, 
                use_llm=True, 
                caption_context="casual photo",
                count=50
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