#upto second point

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

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "vb-assistant"

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEON_DB_URL = os.getenv("NEON_DB_URL")

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
     * Share Chef VB's special tips only from the database
     * Include the video mini player in the response

2. Response Formatting:
   - Mark ingredients with quantities clearly
   - Highlight important tips and warnings
   - Use Tamil terms with English explanations when relevant
   
3. Health Considerations:
   - For ANY recipe, when health-related questions or concerns are mentioned:
     * First provide the original recipe as is from the database with miniplayer
     * Then add a "Health Adaptations" section that:
       - Analyzes ALL ingredients in the recipe for potential health impacts
       - Provides multiple adaptation options based on different health needs:
         • Diabetes-friendly modifications
         • Heart-healthy alternatives
         • Low-sodium versions
         • Gluten-free options (if applicable)
         • Low-fat adaptations
       - For EACH ingredient that could impact health:
         • List the original measurement
         • Provide specific alternatives with exact proportions
         • Explain the health benefit of the substitution
         • Suggest cooking method modifications if needed
       - Include any necessary changes to cooking techniques
       - Note how modifications might affect taste/texture
   - Always include a detailed health impact analysis for:
     * Oils and fats
     * Sweeteners
     * Refined ingredients
     * High-sodium components
     * Allergen-containing ingredients
   - Include a comprehensive disclaimer about consulting healthcare providers

4. Special Notes:
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
    # Existing timestamp and emoji processing
    processed_response = response
    
    # Extract video ID if URL is provided
    video_id = extract_youtube_video_id(video_url) if video_url else None
    
    # Process timestamps (existing code)
    if video_id:
        def timestamp_to_seconds(match):
            timestamp = match.group(1)
            parts = timestamp.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        
        processed_response = re.sub(
            r'\{timestamp:([0-9:]+)\}',
            lambda m: f'[🕒 {m.group(1)}](https://youtube.com/watch?v={video_id}&t={timestamp_to_seconds(m)}s)',
            processed_response
        )
    else:
        processed_response = re.sub(r'\{timestamp:([^\}]+)\}', r'🕒 \1', processed_response)

    # Add emojis to section headers (existing code plus new health-related headers)
    replacements = {
        r'Ingredients:': '📝 Ingredients:',
        r'Instructions:': '👨‍🍳 Instructions:',
        r'Tips?:': '💡 Tip:',
        r'Warning:': '⚠️ Warning:',
        r'Temperature:': '🌡️ Temperature:',
        r'Cooking Time:': '⏲️ Cooking Time:',
        r'Health Adaptations:': '🌿 Health Adaptations:',
        r'Health Note:': '⚕️ Health Note:',
        r'Dietary Alternatives:': '🥗 Dietary Alternatives:'
    }
    
    for pattern, replacement in replacements.items():
        processed_response = re.sub(pattern, replacement, processed_response)

    return processed_response

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
            return_source_documents=True
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

    main_response = main_response.strip()

    # Extract all health-related content and move it to the end
    health_pattern = r'(?:🌿 Health Adaptations:|⚕️ Health Note:|🥗 Dietary Alternatives:).*?(?=\n\n(?:[^🌿⚕️🥗]|$)|$)'
    health_sections = re.findall(health_pattern, main_response, re.DOTALL)
    
    # Remove health sections from main content
    recipe_section = main_response
    for section in health_sections:
        recipe_section = recipe_section.replace(section, '').strip()
    
    # Process timestamps in recipe section
    if url:
        recipe_section = process_ai_response(recipe_section, url)
        video_id = extract_youtube_video_id(url)
        if video_id:
            video_section = (
                f"\n\n📺 **Watch the recipe video:**\n"
                f'<div class="stVideo" style="margin: 20px 0;">'
                f'<iframe src="https://www.youtube.com/embed/{video_id}" '
                f'width="400" height="225" frameborder="0" allowfullscreen></iframe>'
                f'</div>\n\n'
            )
            recipe_section = video_section + recipe_section

    # Combine all health sections at the end
    if health_sections:
        health_content = "\n\n".join(health_sections)
        if url:
            health_content = process_ai_response(health_content, url)
        main_response = f"{recipe_section}\n\n{health_content}"
    else:
        main_response = recipe_section

    return main_response, url, ingredients

def format_chat_history(chat_history):
    """Format chat history for LLM consumption"""
    if not chat_history:
        return []
    
    # Convert to list of tuples format that LangChain expects
    formatted_history = []
    for role, content in chat_history[-6:]:  # Get last 3 exchanges
        # Create tuple pairs that LangChain expects (human message, ai message)
        if role == "user":
            current_exchange = [content]
        elif role == "assistant" and current_exchange:
            current_exchange.append(content)
            formatted_history.append(tuple(current_exchange))
            current_exchange = []
    
    return formatted_history

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
    user_input = st.chat_input("Ask about recipes")

    if user_input:
        # Check if this is a new input
        if user_input != st.session_state.get('last_input'):
            st.session_state.last_input = user_input
            
            try:
                # Format chat history for LLM context
                formatted_history = format_chat_history(st.session_state.chat_history)
                
                # Get response from LLM
                qa_chain = create_qa_chain(st.session_state.retriever)
                result = qa_chain({
                    "question": user_input,
                    "chat_history": formatted_history
                })
                
                # Store raw response and query in chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", result['answer']))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                print(f"Full error details: {str(e)}")

    # Display chat history (outside the if condition)
    for role, content in st.session_state.chat_history:
        with st.container():
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                # Process the raw response for display
                main_response, video_url, ingredients = parse_llm_response(content)
                processed_response = process_ai_response(main_response, video_url)
                st.markdown(f"**Chef VB Assistant:** {processed_response}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
