import os
import google.generativeai as genai

# Configure the Gemini API
GEMINI_API_KEY = "AIzaSyBhtyDbmxsZ14hqlkjdyQRmKIRFqz1vlfs"
genai.configure(api_key=GEMINI_API_KEY)

def test_gemini():
    """Test the Gemini API directly."""
    try:
        # Get the Gemini Pro Experimental model
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
        
        # Generate content
        print("Making request to Gemini API...")
        response = model.generate_content("Tell me a joke about programming.")
        
        print("\nResponse from Gemini model:")
        print(response.text)
            
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_gemini() 