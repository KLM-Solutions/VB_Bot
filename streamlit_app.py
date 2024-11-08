import os
import re
import numpy as np
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, BaseRetriever
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import psycopg2
from psycopg2.extras import RealDictCursor
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import Field
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEON_DB_URL = os.getenv("NEON_DB_URL")

# Streamlit secrets
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
if not NEON_DB_URL:
    NEON_DB_URL = st.secrets["NEON_DB_URL"]

# Categories for content filtering
CONTENT_CATEGORIES = {
    "all": "All Recipes",
    "vegetarian": "Vegetarian Dishes",
    "non-vegetarian": "Non-Vegetarian Dishes",
    "desserts": "Desserts & Sweets",
    "techniques": "Cooking Techniques"
}

# System instructions for the AI
SYSTEM_INSTRUCTIONS = """You are an AI assistant for Chef Venkatesh Bhat's cooking channel, specializing in Tamil cuisine and restaurant-style cooking.
Key responsibilities:
1. Recipe Information:
   - ONLY provide recipes and information that are explicitly found in the database
   - If a recipe is not found in the database, clearly state that this specific recipe is not available
   - Never create or generate recipes that aren't in the database
   - Don't give related videos for unavailable recipes
   - For available recipes:
     * Provide detailed ingredient lists with exact measurements from database
     * Explain cooking steps clearly and precisely
     * Include relevant timestamps using {{timestamp:MM:SS}} format, for each step or important point
     * Include traditional Tamil cooking techniques
     * Mention special ingredients and possible substitutes
     * Share Chef VB's special tips only from the database
     * Include the video mini player in the response

2. Response Formatting:
   - Mark ingredients with quantities clearly
   - Highlight important tips and warnings
   - Use Tamil terms with English explanations when relevant
  
3. Special Notes:
   - Refer to the chef as "Chef VB" or "Chef Venkatesh Bhat"
   - Include festival-specific details when relevant
   - Note restaurant-style adaptations for home cooking
   - Always integrate timestamps within the instruction steps, not separately

Always maintain the authenticity of Chef VB's teaching style while making information accessible to home cooks."""

class CustomRetriever(BaseRetriever):
    """Custom retriever for vector similarity search"""
    
    db_url: str = Field(description="Database URL for connection")
    embeddings: OpenAIEmbeddings = Field(default_factory=lambda: OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    category: str = Field(default="all", description="Content category for filtering")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db_url: str, category: str = "all"):
        super().__init__(db_url=db_url, category=category)
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def format_vector_for_postgres(self, embedding):
        """Format embeddings for PostgreSQL"""
        return f"[{','.join(map(str, embedding))}]"

    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # 1. Convert query to embeddings
            query_embedding = self.embeddings.embed_query(query)
            vector_string = self.format_vector_for_postgres(query_embedding)
            
            # 2. Search database for top 3 similar results
            with psycopg2.connect(self.db_url).cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT id, chunk_id, text, url, ingredients,
                           1 - (vector <=> %s::vector(1536)) as similarity
                    FROM transcripts
                    WHERE text IS NOT NULL
                    ORDER BY vector <=> %s::vector(1536)
                    LIMIT 3
                """
                cur.execute(sql, (vector_string, vector_string))
                results = cur.fetchall()
            
            # 3. Convert results to JSON-like format for LLM
            documents = []
            for idx, result in enumerate(results, 1):
                formatted_text = (
                    f"Response {idx}:\n"
                    f"Content: {result['text']}\n"
                    f"URL: {result['url']}\n"
                    f"Ingredients: {result['ingredients']}\n"
                    f"Relevance Score: {result['similarity']:.2f}"
                )
                
                documents.append(Document(
                    page_content=formatted_text,
                    metadata={
                        'chunk_id': result['chunk_id'],
                        'url': result['url'],
                        'ingredients': result['ingredients'],
                        'similarity': result['similarity']
                    }
                ))
            
            return documents
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return []

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

def extract_youtube_video_id(url: str) -> str:
    """Extract YouTube video ID from URL"""
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:]
    return None

def process_ai_response(response: str, video_url: str = None) -> str:
    """Process and format AI response with emojis and make timestamps clickable"""
    # Extract video ID if URL is provided
    video_id = extract_youtube_video_id(video_url) if video_url else None
    
    # Convert timestamps to clickable links if video_id is available
    if video_id:
        def timestamp_to_seconds(match):
            timestamp = match.group(1)
            parts = timestamp.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        
        # Simply show MM:SS with clock emoji, but make it clickable
        response = re.sub(
            r'\{timestamp:([0-9:]+)\}',
            lambda m: f'[🕒 {m.group(1)}](https://youtube.com/watch?v={video_id}&t={timestamp_to_seconds(m)}s)',
            response
        )
    else:
        # If no video URL, just show the timestamp with clock emoji
        response = re.sub(r'\{timestamp:([^\}]+)\}', r'🕒 \1', response)

    # Add emojis to section headers
    replacements = {
        r'Ingredients:': '📝 Ingredients:',
        r'Instructions:': '👨‍🍳 Instructions:',
        r'Tips?:': '💡 Tip:',
        r'Warning:': '⚠️ Warning:',
        r'Temperature:': '🌡️ Temperature:',
        r'Cooking Time:': '⏲️ Cooking Time:'
    }
    
    for pattern, replacement in replacements.items():
        response = re.sub(pattern, replacement, response)

    return response

def combine_ingredients(db_ingredients: str, content_text: str) -> str:
    """Combine ingredients from database and content"""
    all_ingredients = set()
    
    # Add database ingredients
    if db_ingredients:
        # Split and clean database ingredients
        db_items = [item.strip() for item in db_ingredients.split(',')]
        all_ingredients.update(db_items)
    
    # Extract additional ingredients from content
    # Look for common measurement patterns
    measurements = r'\d+(?:\s*(?:grams?|g|kilos?|kg|ml|liters?|l|cups?|tbsp|tsp|pieces?|nos|numbers?))?'
    ingredient_pattern = rf'({measurements}\s+[\w\s]+)'
    
    content_ingredients = re.findall(ingredient_pattern, content_text, re.IGNORECASE)
    if content_ingredients:
        all_ingredients.update(content_ingredients)
    
    return '\n'.join(f"• {ingredient}" for ingredient in sorted(all_ingredients))

def create_qa_chain(retriever):
    """Create the question-answering chain"""
    try:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Answer using the provided context. Include all ingredients mentioned in both the ingredients list and the content. "
                "Format ingredients clearly with measurements when available.\n\n"
                "At the end of your response, always include:\n"
                "VIDEO_URL: [url from the context]\n"
                "INGREDIENTS: [ingredients from the context]\n\n" + 
                SYSTEM_INSTRUCTIONS
            ),
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\n\nChat History: {chat_history}\n\nQuestion: {question}"
            )
        ])

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        raise

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "all"
    if 'retriever' not in st.session_state:
        st.session_state.retriever = CustomRetriever(NEON_DB_URL)
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None

def relevance_check_prompt(user_query: str, formatted_history: List = None) -> str:
    """
    Create a prompt to check the relevance of user queries in the context of 
    Chef Venkatesh Bhat's cooking assistant.
    """
    return f"""
        Given the following question or message and the chat history, determine if it is:
        1. A greeting or general conversation starter
        2. Related to cooking, Tamil cuisine, recipes, or Chef Venkatesh Bhat's cooking techniques
        3. Related to ingredients, kitchen tips, or cooking methods
        4. A continuation or follow-up question to the previous conversation
        5. Related to harmful activities, inappropriate content, or non-cooking topics
        6. Completely unrelated to cooking or Chef VB's content
        7. Specific questions about Chef Venkatesh Bhat or his restaurant experience

        If it falls under category 1, respond with 'GREETING'.
        If it falls under categories 2, 3, 4 or 7 respond with 'RELEVANT'.
        If it falls under category 5, respond with 'INAPPROPRIATE'.
        If it falls under category 6, respond with 'NOT RELEVANT'.

        Chat History:
        {formatted_history[-3:] if formatted_history else "No previous context"}

        Current Question: {user_query}
        
        Response (GREETING, RELEVANT, INAPPROPRIATE, or NOT RELEVANT):
        """

def handle_relevance_response(relevance_response: str, llm) -> dict:
    """Handle different types of relevance responses and return appropriate responses"""
    
    if "GREETING" in relevance_response.upper():
        greeting_response = llm.predict(
            "Generate a warm, friendly greeting as Chef VB's cooking assistant. "
            "Include a brief invitation to ask about Tamil cuisine, recipes, or cooking techniques."
        )
        return {
            'answer': greeting_response,
            'source_documents': []
        }
        
    elif "INAPPROPRIATE" in relevance_response.upper():
        return {
            'answer': ("I apologize, but I can only assist with cooking-related questions and Chef VB's recipes. "
                    "Please feel free to ask about Tamil cuisine, cooking techniques, or specific recipes from "
                    "Chef Venkatesh Bhat's collection."),
            'source_documents': []
        }
        
    elif "NOT RELEVANT" in relevance_response.upper():
        return {
            'answer': ("I'm specialized in Chef Venkatesh Bhat's cooking techniques, "
                      "Tamil cuisine, and general cooking advice. Would you like to know about:\n"
                      "• Traditional Tamil recipes\n"
                      "• Restaurant-style cooking techniques\n"
                      "• Ingredient substitutions\n"
                      "• Kitchen tips and tricks"),
            'source_documents': []
        }
        
    # For RELEVANT responses, continue with the normal QA chain
    return None

def parse_llm_response(response: str) -> tuple:
    """Parse the LLM response to extract URL and ingredients"""
    # Default values
    url = None
    ingredients = None
    main_response = response

    # Extract URL and remove from main response
    url_match = re.search(r'VIDEO_URL:\s*(.+?)(?:\n|$)', response)
    if url_match:
        url = url_match.group(1).strip()
        main_response = main_response.replace(url_match.group(0), '')

    # Extract ingredients and remove from main response
    ingredients_match = re.search(r'INGREDIENTS:\s*(.+?)(?:\n|$)', response)
    if ingredients_match:
        ingredients = ingredients_match.group(1).strip()
        main_response = main_response.replace(ingredients_match.group(0), '')

    # Clean up any trailing whitespace or newlines
    main_response = main_response.strip()

    # Add video player section at the end if URL exists
    if url:
        video_id = extract_youtube_video_id(url)
        if video_id:
            main_response = (
                f"{main_response}\n\n"
                f"📺 **Watch the full recipe video here:**\n"
                f"<iframe width='400' height='225' "
                f"src='https://www.youtube.com/embed/{video_id}' "
                f"frameborder='0' allowfullscreen></iframe>"
            )

    return main_response, url, ingredients

def main():
    # Page configuration
    st.set_page_config(
        page_title="Chef VB's Cooking Assistant",
        page_icon="👨‍🍳",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("👨‍🍳 Chef Venkatesh Bhat's Cooking Assistant")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        selected_category = st.selectbox(
            "Choose Category",
            options=list(CONTENT_CATEGORIES.keys()),
            format_func=lambda x: CONTENT_CATEGORIES[x]
        )

        if selected_category != st.session_state.selected_category:
            st.session_state.selected_category = selected_category
            st.session_state.retriever = CustomRetriever(NEON_DB_URL, selected_category)
        
        st.markdown("---")
        st.markdown("""
        ### About Chef VB Assistant
        
        This AI assistant helps you explore:
        - 🍲 Traditional Tamil Recipes
        - 👨‍🍳 Professional Cooking Techniques
        - 💡 Kitchen Tips & Tricks
        - 🏪 Restaurant Style Secrets
        
        Ask questions about:
        - Specific recipes
        - Cooking methods
        - Ingredient substitutions
        - Traditional techniques
        """)

    # Main chat interface
    st.markdown("### Ask about recipes, techniques, or kitchen tips! 💬")
    user_input = st.text_input("Your question:", key="user_input")

    # Display chat history with markdown enabled for clickable links
    for speaker, message in st.session_state.chat_history:
        with st.container():
            if speaker == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Chef VB Assistant:** {message}", unsafe_allow_html=True)

    # Handle user input
    if user_input and user_input != st.session_state.get('last_input', ''):
        st.session_state.last_input = user_input
        try:
            with st.spinner("Searching for information..."):
                # Create LLM instance for relevance check
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    model="gpt-4o-mini",
                    temperature=0
                )
                
                # Check relevance first
                relevance_prompt = relevance_check_prompt(
                    user_input, 
                    [(q, a) for q, a in st.session_state.chat_history]
                )
                relevance_result = llm.predict(relevance_prompt)
                
                # Handle relevance response
                relevance_response = handle_relevance_response(relevance_result, llm)
                if relevance_response:
                    # Process the response similar to regular responses
                    main_response, _, _ = parse_llm_response(relevance_response['answer'])
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", main_response))
                    st.experimental_rerun()
                    return

                # Continue with normal QA chain if relevant
                qa_chain = create_qa_chain(st.session_state.retriever)
                result = qa_chain({
                    "question": user_input,
                    "chat_history": [(q, a) for q, a in st.session_state.chat_history[:-1]]
                })

                # Process response components
                main_response, video_url, ingredients = parse_llm_response(result['answer'])
                
                # Add ingredients to the response if they exist and are relevant
                if ingredients and "ingredient" in user_input.lower():
                    main_response = f"📝 Ingredients:\n{ingredients}\n\n{main_response}"
                
                # Format final response with enhanced timestamp handling
                processed_response = process_ai_response(main_response, video_url)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", processed_response))

                # Force a rerun to update the chat display
                st.experimental_rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(f"Full error details: {str(e)}")

if __name__ == "__main__":
    main()
