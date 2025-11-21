import os
import string
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=API_KEY)

def preprocess_input(text):
    """
    Basic preprocessing: lowercase and remove punctuation.
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_llm_response(question):
    """
    Send the question to the LLM and get the response.
    """
    try:
        # Using gemini-pro as a stable default
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "**[API Quota Exceeded]**\n\nThis is a mock response because the API rate limit was hit.\n\n*Mock Answer:* The capital of France is Paris. (This is a placeholder answer for demonstration purposes)."
        return f"Error communicating with LLM: {e}"

def main():
    print("NLP Question-Answering System (CLI)")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nEnter your question: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break

            if not user_input.strip():
                continue

            processed_input = preprocess_input(user_input)
            print(f"Processed Question: {processed_input}")
            
            print("Fetching answer...")
            answer = get_llm_response(processed_input)
            print(f"\nAnswer:\n{answer}")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
