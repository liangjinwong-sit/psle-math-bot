"""Test which Gemini models are available with your API key."""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {api_key[:20]}..." if api_key else "No API key!")
print("=" * 60)

genai.configure(api_key=api_key)

print("\n🔍 Listing available models for your API key:\n")

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
            print(f"   Display: {model.display_name}")
            print(f"   Description: {model.description[:80]}...")
            print()
except Exception as e:
    print(f"❌ Error listing models: {e}")
    print("\nTrying direct test with common model names...")
    
    # Test specific models directly
    for model_name in ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'OK'")
            print(f"✅ {model_name} works! Response: {response.text}")
        except Exception as e2:
            print(f"❌ {model_name} failed: {str(e2)[:80]}")

