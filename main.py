import streamlit as st
import pandas as pd
import google.generativeai as genai
from typing import List, Dict
import json

# Configure the page settings
st.set_page_config(
    page_title="SEO Content Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS to improve UI
st.markdown("""
    <style>
    .stApp {
        max-width: none !important;
        padding: 1rem;
    }
    .main {
        padding: 2rem;
    }
    .highlight {
        background-color: #ffd700;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .suggestion {
        color: #1e88e5;
        font-style: italic;
    }
    .meta-panel {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini API
def initialize_gemini():
    try:
        genai.configure(api_key='AIzaSyAXLzqx3cJbwyxdKqPgNFY_FyDMxnQB6p4')
        model = genai.GenerativeModel('models/gemini-2.0-flash')  # Updated to Flash 2.0
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        return None

def process_keywords_file(uploaded_file) -> pd.DataFrame:
    """Process the uploaded keywords CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['keyword', 'search_volume', 'difficulty']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain columns: keyword, search_volume, difficulty")
            return None
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None

def optimize_content(model, text: str, title: str, keywords_df: pd.DataFrame) -> Dict:
    """Use Gemini to optimize the content and generate meta information"""
    content_guidelines = '''
    Voice and Tone
    - Write in a helpful and explanatory voice that supports users while showcasing capabilities
    - Use a conversational, friendly tone as if speaking directly to the reader
    - Employ contractions (it's, you'll, we're) to maintain natural flow
    - Sound trustworthy by providing relevant examples and consistent language
    - Write for readers with varying levels of technical expertise

    Writing Style
    - Lead with the most important information
    - Start sentences with action verbs
    - Remove unnecessary phrases like "you can" and "there is/are"
    - Use US English spelling exclusively
    - Keep content concise and minimal
    - Write like you speak, avoiding unnecessary jargon
    - Explain technical terms specific to the product, frameworks, or languages
    - Maintain consistent terminology throughout the document

    Formatting Rules
    - Use sentence case for all headings (capitalize only first word and proper nouns)
    - Capitalize product-specific terms (e.g., Prompts, Workflows, Datasets)
    - Omit end punctuation (periods, colons, exclamation marks, question marks) in:
    - Titles
    - Headings
    - Subheadings
    - UI titles
    - List items of three or fewer words
    - Use periods in paragraphs and body copy

    ## SEO Requirements
    - Create descriptive page titles with focus keywords
    - Include metadata (meta title and meta description) for each page
    - Use descriptive anchor text for internal links (avoid "click here" or "learn more")
    - Keep URL slugs descriptive (2-4 words)
    - Add captions and alt text for all media

    ## Example Transformations
    Before: "If you're ready to start integrating Maxim SDK to log your AI application, start by following the steps given in our tutorial"
    After: "Ready to start logging? Integrate Maxim SDK."

    Before: "Node level evaluation"
    After: "Evaluate every step in your AI workflow"
    Before: "The HTTP Workflow is designed to facilitate the integration of your existing Application APIs with the Maxim platform without requiring any code integrations at your end."
    After: "Run tests on your AI application via a simple HTTP endpoint, without needing any code integrations."

    ## Content Review Checklist
    - Is the content helpful and explanatory?
    - Does it lead with the most important information?
    - Are all sentences action-oriented?
    - Is technical jargon explained where necessary?
    - Are headings in sentence case?
    - Is formatting consistent throughout?
    - Are all images properly captioned with alt text?
    - Are internal links using descriptive anchor text?
    - Is the document structure following heading hierarchy (H1 > H2 > H3)?
    - Is the language consistent with US English spelling?

    '''

    prompt = f"""
    You are an SEO expert. Analyze this blog text and title:
    
    Title: {title}
    Content: {text}
    
    Using this keyword bank:
    {keywords_df.to_json(orient='records')}
    
    Follow these content guidelines:
    {content_guidelines}
    
    Provide an analysis in the exact JSON format below. Do not include any other text:
    {{
        "optimizations": [
            {{"original": "phrase from the text", "suggested": "optimized phrase, grammar fix, or keyword usage"}}
        ],
        "meta_titles": [
            "title 1",
            "title 2",
            "title 3",
            "title 4",
            "title 5"
        ],
        "meta_descriptions": [
            "description 1",
            "description 2",
            "description 3",
            "description 4",
            "description 5"
        ],
        "targeted_keywords": [
            {{"keyword": "found keyword", "search_volume": "SV", "difficulty": "score", "context": "how it's used"}}
        ],
        "content_evaluation": {{
            "score": "1-5",
            "grammatical_errors": ["error1", "error2"],
            "guideline_violations": [
                {{
                    "error": "description",
                    "reason": "why it's wrong",
                    "fix": "how to fix"
                }}
            ]
        }}
    }}
    
    Rules:
    1. For optimizations, find actual phrases from the original text that could be improved
    2. Suggest replacements using keywords with higher search volume and lower difficulty
    3. Keep meta titles under 60 characters
    4. Keep meta descriptions under 160 characters
    5. Include all keywords found in the content that match the keyword bank
    6. Evaluate content against the provided guidelines
    7. Return ONLY the JSON, no other text
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        # Remove any markdown code block indicators if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing Gemini response: {str(e)}")
        st.error(f"Raw response: {response.text}")
        return None
    except Exception as e:
        st.error(f"Error in content optimization: {str(e)}")
        return None

def main():
    st.title("🎯 SEO Content Optimizer")
    
    # Initialize Gemini model
    model = initialize_gemini()
    if not model:
        st.stop()
    
    # File upload for keywords
    st.sidebar.header("Upload Keywords Bank")
    keywords_file = st.sidebar.file_uploader(
        "Upload your SEO keywords CSV",
        type=["csv"],
        help="CSV should contain columns: keyword, search_volume, difficulty"
    )
    
    if keywords_file is not None:
        keywords_df = process_keywords_file(keywords_file)
        if keywords_df is None:
            st.stop()
            
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Content Input")
            blog_title = st.text_input("Blog Title", placeholder="Enter your blog title")
            blog_content = st.text_area(
                "Blog Content",
                height=300,
                placeholder="Paste your blog content here..."
            )
            
            if st.button("Optimize Content", type="primary"):
                if blog_content and blog_title:
                    with st.spinner("Optimizing your content..."):
                        results = optimize_content(model, blog_content, blog_title, keywords_df)
                        
                        if results:
                            # Display Content Evaluation
                            st.subheader("📝 Content Evaluation")
                            eval_score = results["content_evaluation"]["score"]
                            st.markdown(f"**Overall Score**: {eval_score}/5")
                            
                            if results["content_evaluation"]["grammatical_errors"]:
                                st.markdown("**Grammatical Errors:**")
                                for error in results["content_evaluation"]["grammatical_errors"]:
                                    st.markdown(f"- {error}")
                            
                            if results["content_evaluation"]["guideline_violations"]:
                                st.markdown("**Guideline Violations:**")
                                for violation in results["content_evaluation"]["guideline_violations"]:
                                    with st.expander(f"🔍 {violation['error']}"):
                                        st.markdown(f"**Reason**: {violation['reason']}")
                                        st.markdown(f"**Fix**: {violation['fix']}")
                                        
                            # Display Keyword Analysis
                            st.subheader("📊 Targeted Keywords Analysis")
                            if results["targeted_keywords"]:
                                keywords_data = pd.DataFrame(results["targeted_keywords"])
                                st.dataframe(
                                    keywords_data,
                                    column_config={
                                        "keyword": "Keyword",
                                        "search_volume": "Search Volume",
                                        "difficulty": "Difficulty",
                                        "context": "Usage Context"
                                    },
                                    hide_index=True
                                )
                            
                            # Display optimized content
                            st.subheader("Optimization Suggestions")
                            optimized_text = blog_content
                            for opt in results["optimizations"]:
                                optimized_text = optimized_text.replace(
                                    opt["original"],
                                    f'<span class="highlight">{opt["original"]}</span> ' +
                                    f'<span class="suggestion">(Suggested: {opt["suggested"]})</span>'
                                )
                            st.markdown(optimized_text, unsafe_allow_html=True)

                            # Display meta information in the second column
                            with col2:
                                st.markdown(
                                    '<div class="meta-panel">'
                                    '<h3>📋 Meta Information</h3>', 
                                    unsafe_allow_html=True
                                )
                                
                                st.subheader("Meta Titles")
                                for i, title in enumerate(results["meta_titles"], 1):
                                    st.markdown(f"{i}. {title}")
                                
                                st.subheader("Meta Descriptions")
                                for i, desc in enumerate(results["meta_descriptions"], 1):
                                    st.markdown(f"{i}. {desc}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Please enter both title and content before optimizing.")

if __name__ == "__main__":
    main()