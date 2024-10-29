import re
import numpy as np
import streamlit as st
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, BaseRetriever
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import psycopg2
from psycopg2.extras import RealDictCursor
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import Field

# Configuration
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
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
   - Provide detailed ingredient lists with exact measurements from database
   - Explain cooking steps clearly and precisely
   - Include traditional Tamil cooking techniques
   - Mention special ingredients and possible substitutes
   - Share Chef VB's special

2. Response Formatting:
   - Include relevant timestamps using {{timestamp:MM:SS}} format
   - Mark ingredients with quantities clearly
   - Highlight important tips and warnings
   - Use Tamil terms with English explanations when relevant

3. Special Notes:
   - Refer to the chef as "Chef VB" or "Chef Venkatesh Bhat"
   - Include festival-specific details when relevant
   - Note restaurant-style adaptations for home cooking

Always maintain the authenticity of Chef VB's teaching style while making information accessible to home cooks."""

class CustomRetriever(BaseRetriever):
    """Custom retriever for vector similarity search"""
    
    db_url: str = Field(description="Database URL for connection")
    embeddings: OpenAIEmbeddings = Field(default_factory=lambda: OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    category: str = Field(default="all", description="Content category for filtering")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_params(cls, db_url: str, category: str = "all") -> "CustomRetriever":
        """Factory method to create CustomRetriever instance"""
        return cls(
            db_url=db_url,
            category=category,
            embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        )

    def __init__(self, db_url: str, category: str = "all", embeddings: Optional[OpenAIEmbeddings] = None):
        super().__init__()
        self.db_url = db_url
        self.category = category
        self.embeddings = embeddings or OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def format_vector_for_postgres(self, embedding):
        """Format embeddings for PostgreSQL"""
        return f"[{','.join(map(str, embedding))}]"

    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            vector_string = self.format_vector_for_postgres(query_embedding)
            
            conn = psycopg2.connect(self.db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Base SQL query
                base_sql = """
                    SELECT id, chunk_id, text, url, ingredients,
                    1 - (vector <=> %s::vector(1536)) as similarity
                    FROM transcripts
                    WHERE text IS NOT NULL
                """
                
                if self.category != "all":
                    sql = base_sql + " AND ingredients ILIKE %s ORDER BY vector <=> %s::vector(1536) LIMIT 3"
                    cur.execute(sql, (vector_string, f"%{self.category}%", vector_string))
                else:
                    sql = base_sql + " ORDER BY vector <=> %s::vector(1536) LIMIT 3"
                    cur.execute(sql, (vector_string, vector_string))
                
                results = cur.fetchall()
            conn.close()
            
            # Convert to LangChain documents
            return [
                Document(
                    page_content=result['text'],
                    metadata={
                        'chunk_id': result['chunk_id'],
                        'url': result['url'],
                        'ingredients': result['ingredients'],
                        'similarity': result['similarity']
                    }
                ) for result in results
            ]
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            print(f"Full error: {str(e)}")
            return []

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

def process_ai_response(response: str) -> str:
    """Process and format AI response with emojis and combine ingredients"""
    # Extract ingredients from response if present
    ingredients_section = re.search(r'📝 Ingredients:(.*?)(?=👨‍🍳|$)', response, re.DOTALL)
    if ingredients_section:
        # Remove the existing ingredients section
        response = response.replace(ingredients_section.group(0), '')
    
    replacements = {
        r'Ingredients:': '📝 Ingredients:',
        r'Instructions:': '👨‍🍳 Instructions:',
        r'Tips?:': '💡 Tip:',
        r'Warning:': '⚠️ Warning:',
        r'Temperature:': '🌡️ Temperature:',
        r'Cooking Time:': '⏲️ Cooking Time:',
        r'\{timestamp:([^\}]+)\}': r'🕒 \1'
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
            model="gpt-4o-mini",  # Updated to latest model
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Answer using the provided context. Include all ingredients mentioned in both the ingredients list and the content. "
                "Format ingredients clearly with measurements when available.\n\n" + 
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
        st.session_state.retriever = CustomRetriever.from_params(NEON_DB_URL)

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
            st.session_state.retriever = CustomRetriever.from_params(NEON_DB_URL, selected_category)
        
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

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", key="ask_button")
    with col2:
        clear_button = st.button("Clear Chat")

    # Handle user input
    if ask_button and user_input:
        try:
            with st.spinner("Searching for information..."):
                # Add user message to chat history
                st.session_state.chat_history.append(("user", user_input))
                
                # Create QA chain and get response
                qa_chain = create_qa_chain(st.session_state.retriever)
                result = qa_chain({
                    "question": user_input,
                    "chat_history": [(q, a) for q, a in st.session_state.chat_history[:-1]]
                })

                # Process ingredients and response
                combined_ingredients = ""
                if result['source_documents']:
                    doc = result['source_documents'][0]  # Using primary source document
                    combined_ingredients = combine_ingredients(
                        doc.metadata['ingredients'],
                        doc.page_content
                    )
                
                # Add combined ingredients to the response if relevant
                if combined_ingredients and "ingredient" in user_input.lower():
                    processed_response = f"📝 Ingredients:\n{combined_ingredients}\n\n" + process_ai_response(result['answer'])
                else:
                    processed_response = process_ai_response(result['answer'])

                # Process and add AI response
                st.session_state.chat_history.append(("assistant", processed_response))

            # Display chat history
            for speaker, message in st.session_state.chat_history:
                if speaker == "user":
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**Chef VB Assistant:** {message}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(f"Full error details: {str(e)}")

    # Handle clear chat
    if clear_button:
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
