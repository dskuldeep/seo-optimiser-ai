import streamlit as st
import pandas as pd
import google.generativeai as genai
from typing import List, Dict
import json
from io import BytesIO
from xhtml2pdf import pisa
import base64 # Kept import, potentially useful though not strictly used now
import html # Import the html module for escaping

# --------------------------------------------------------------------------
# Page Configuration and Styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="SEO Content Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better UI and PDF basic styling hints
st.markdown("""
    <style>
    /* General App Styles */
    .stApp {
        max-width: none !important; /* Use full width */
        padding: 1rem;
    }
    .main > div { /* Target the main container within stApp */
         padding-left: 2rem;
         padding-right: 2rem;
    }
    .stButton>button { /* Style Streamlit buttons */
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }

    /* Specific Element Styles */
    .highlight {
        background-color: #ffd700; /* Yellow highlight */
        padding: 2px 4px;
        border-radius: 3px;
        display: inline; /* Keep highlight inline */
    }
    .suggestion {
        color: #1e88e5; /* Blue for suggestions */
        font-style: italic;
        margin-left: 5px;
        display: inline; /* Keep suggestion inline */
    }
    .meta-panel {
        background-color: #f8f9fa; /* Light grey background */
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6; /* Subtle border */
        margin-top: 20px; /* Add space above */
        height: auto; /* Adjust height automatically */
    }
    .meta-panel h3, .meta-panel h4 {
        margin-top: 0;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }

    /* Basic styles for PDF rendering (interpreted by xhtml2pdf) */
    @media print { /* Styles specifically for PDF generation */
        body { font-family: sans-serif; margin: 20px; line-height: 1.4; }
        h1, h2, h3 { color: #333; margin-bottom: 0.5em; }
        h1 { border-bottom: 2px solid #eee; padding-bottom: 5px; font-size: 24pt; }
        h2 { border-bottom: 1px solid #eee; padding-bottom: 3px; font-size: 18pt; }
        h3 { font-size: 14pt; }
        .section { margin-bottom: 25px; padding-bottom: 10px; border-bottom: 1px dashed #ccc; }
        .section:last-child { border-bottom: none; }
        .meta-panel-pdf { background-color: #f8f9fa; padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-top: 15px; }
        .highlight-pdf { background-color: #ffffdd; padding: 1px 3px; } /* Lighter highlight for PDF */
        .suggestion-pdf { color: #0056b3; font-style: italic; }
        /* Style for inline suggestions in PDF */
        .inline-suggestions-pdf-container {
            border: 1px solid #eee;
            padding: 10px;
            background-color: #fdfdfd;
            margin-top: 10px;
            white-space: pre-wrap; /* Preserve whitespace and line breaks */
        }
        ul { list-style-type: disc; margin-left: 20px; padding-left: 0; }
        li { margin-bottom: 8px; }
        table.pdf-table { border-collapse: collapse; width: 100%; margin-bottom: 1em; border: 1px solid #ccc; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .violation-pdf { border-left: 4px solid #dc3545; padding-left: 10px; margin-bottom: 10px; background-color: #fdf7f7; padding: 10px; }
        .violation-pdf strong { color: #a94442; } /* Darker red for text */
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# API Initialization
# --------------------------------------------------------------------------

def initialize_gemini():
    """Initializes the Gemini API model."""
    try:
        # --- IMPORTANT ---
        # Replace 'YOUR_API_KEY' with your actual key.
        # Best practice: Use Streamlit Secrets (`st.secrets["GEMINI_API_KEY"]`)
        # or environment variables instead of hardcoding.
        # Example using secrets:
        # api_key = st.secrets.get("GEMINI_API_KEY")
        api_key = 'AIzaSyAXLzqx3cJbwyxdKqPgNFY_FyDMxnQB6p4' # Replace with your key or use secrets
        # --- --- --- ---

        if not api_key or api_key == 'YOUR_API_KEY':
             st.warning("Please replace 'YOUR_API_KEY' in the code with your actual Gemini API key, or set it up using Streamlit Secrets (e.g., `st.secrets['GEMINI_API_KEY']`).", icon="⚠️")
             return None # Stop initialization if key is missing

        genai.configure(api_key=api_key)
        # Use a valid and available model name. 'gemini-1.5-flash' is a common choice.
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        st.error("Please ensure your API key is correct and you have the necessary permissions.")
        return None

# --------------------------------------------------------------------------
# File Processing
# --------------------------------------------------------------------------

def process_keywords_file(uploaded_file) -> pd.DataFrame | None:
    """
    Processes the uploaded keywords CSV file.
    Expects columns: 'keyword', 'search_volume', 'difficulty'.
    Returns a DataFrame or None if an error occurs.
    """
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['keyword', 'search_volume', 'difficulty']

        # Check for required columns (case-insensitive check is more robust)
        df.columns = df.columns.str.lower().str.strip() # Normalize column names
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV file must contain columns: {', '.join(required_columns)}. Found: {', '.join(df.columns)}")
            return None

        # Select and potentially rename columns to ensure consistency
        df = df[required_columns]

        # Basic data cleaning
        df['keyword'] = df['keyword'].astype(str).str.strip()
        # Optional: Convert numeric columns if they aren't already, handle errors
        for col in ['search_volume', 'difficulty']:
             try:
                 # Attempt conversion, coercing errors to NaN, then fill NaN if needed
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 # Decide how to handle non-numeric values (e.g., fill with 0 or drop row)
                 # df[col] = df[col].fillna(0)
             except Exception as conv_e:
                 st.warning(f"Could not reliably convert column '{col}' to numeric. Please check data. Error: {conv_e}", icon="⚠️")
                 # Keep the column as object type if conversion fails broadly


        df = df.dropna(subset=['keyword']) # Remove rows with no keyword
        if df.empty:
            st.error("No valid keyword data found after processing the CSV.")
            return None

        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
        return None
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        st.error("Please ensure the file is a valid CSV and has the correct structure.")
        return None

# --------------------------------------------------------------------------
# Content Optimization Logic (Gemini API Call)
# --------------------------------------------------------------------------

def optimize_content(model, text: str, title: str, keywords_df: pd.DataFrame) -> Dict | None:
    """
    Uses the Gemini model to analyze content based on keywords and guidelines.
    Returns a dictionary with optimization results or None on failure.
    """
    # --- Content Guidelines (Keep as is or refine further) ---
    content_guidelines = '''
    Voice and Tone:
    - Helpful, explanatory, conversational, friendly, trustworthy.
    - Write for varying technical expertise.

    Writing Style:
    - Lead with important info.
    - Concise, minimal, remove fluff ("you can", "there is/are").
    - US English spelling.
    - Explain technical terms.
    - Consistent terminology.

    Formatting Rules:
    - Sentence case for headings.
    - Capitalize specific product terms (if applicable, provide examples).
    - Omit end punctuation in titles, headings, short list items. Use periods in body copy.

    SEO Requirements:
    - Descriptive page titles with focus keywords.
    - Include meta title & description (generate suggestions).
    - Descriptive anchor text for links (avoid "click here").
    - Descriptive URL slugs (2-4 words) - (Suggest based on title).
    - Captions and alt text for media (mention importance).

    Example Transformations:
    Before: "If you're ready to start integrating Maxim SDK..." -> After: "Ready to start logging? Integrate Maxim SDK."
    Before: "Node level evaluation" -> After: "Evaluate every step in your AI workflow"

    Content Review Checklist (For Evaluation):
    - Helpful? Leads with important info? Jargon explained? Headings sentence case?
    - Consistent formatting? Images captioned/alt text? Descriptive links? Heading hierarchy? US English?
    '''

    # --- Prompt Engineering ---
    prompt = f"""
    Analyze the following blog post based on the provided guidelines and keyword bank.

    **Blog Title:**
    {title}

    **Blog Content:**
    ```
    {text}
    ```

    **Keyword Bank (Keywords to prioritize if relevant):**
    ```json
    {keywords_df.to_json(orient='records', indent=2)}
    ```

    **Content & SEO Guidelines:**
    ```
    {content_guidelines}
    ```

    **Task:**
    Generate a JSON object containing your analysis. The JSON object must strictly adhere to the following structure and rules. Do NOT include any text outside this JSON object (no introductions, explanations, or summaries before or after the JSON).

    **JSON Output Structure:**
    ```json
    {{
      "optimizations": [
        {{
          "original": "Exact phrase found in the original text that can be improved.",
          "suggested": "Improved phrase incorporating keywords, better grammar, or conciseness, following guidelines."
        }}
      ],
      "meta_titles": [
        "Suggested Meta Title 1 (under 60 chars, includes keywords)",
        "Suggested Meta Title 2",
        "Suggested Meta Title 3",
        "Suggested Meta Title 4",
        "Suggested Meta Title 5"
      ],
      "meta_descriptions": [
        "Suggested Meta Description 1 (under 160 chars, includes keywords, compelling)",
        "Suggested Meta Description 2",
        "Suggested Meta Description 3",
        "Suggested Meta Description 4",
        "Suggested Meta Description 5"
      ],
      "targeted_keywords": [
        {{
          "keyword": "Keyword from the bank found in the text",
          "search_volume": "Its search volume from the bank",
          "difficulty": "Its difficulty score from the bank",
          "context": "Brief example sentence showing how the keyword is used in the text."
        }}
      ],
      "content_evaluation": {{
        "score": "Overall score (1-5) based on adherence to guidelines.",
        "grammatical_errors": [
          "List any specific grammatical errors found.",
          "Example: 'Incorrect subject-verb agreement in sentence X.'"
        ],
        "guideline_violations": [
          {{
            "error": "Description of the guideline violation (e.g., 'Heading not in sentence case').",
            "reason": "Why this violates the guidelines.",
            "fix": "How to correct the violation.",
            "example": "The specific text violating the rule (e.g., 'The Original Heading')."
          }}
        ]
      }}
    }}
    ```

    **Rules for JSON Generation:**
    1.  **Strict JSON:** Output ONLY the JSON object starting with `{{` and ending with `}}`.
    2.  **Optimizations:** Find *actual phrases* from the original text. Suggest improvements based on guidelines and relevant, high-value keywords from the bank. If no improvements, provide an empty list `[]`.
    3.  **Meta Info:** Generate 5 distinct titles (<60 chars) and 5 distinct descriptions (<160 chars), incorporating the main topic and relevant keywords naturally.
    4.  **Targeted Keywords:** List *only* keywords from the provided bank that are present in the content. Include their SV, difficulty, and context. If none found, use an empty list `[]`.
    5.  **Evaluation:** Provide an honest score (1-5). List specific grammar errors and guideline violations with details as per the structure. If no errors/violations, use empty lists `[]`. Ensure violations reference specific guidelines.
    6.  **Escaping:** Ensure all strings within the JSON are properly escaped (e.g., quotes within strings).
    """

    try:
        # Configure safety settings if needed (e.g., to allow discussion of certain topics if relevant)
        # safety_settings = [...]
        # response = model.generate_content(prompt, safety_settings=safety_settings)

        response = model.generate_content(prompt)

        # -- Robust JSON Extraction --
        response_text = response.text.strip()

        # Try to find the JSON block even if there's leading/trailing text/markdown
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            potential_json = response_text[json_start:json_end+1]
            try:
                # Validate and parse the extracted JSON
                parsed_json = json.loads(potential_json)
                # Basic validation of top-level keys (optional but good)
                required_keys = {"optimizations", "meta_titles", "meta_descriptions", "targeted_keywords", "content_evaluation"}
                if required_keys.issubset(parsed_json.keys()):
                     return parsed_json
                else:
                     st.error("API response JSON is missing required keys.")
                     st.text_area("Received JSON structure:", json.dumps(parsed_json, indent=2), height=150)
                     return None

            except json.JSONDecodeError as json_e:
                st.error(f"Error parsing extracted JSON from API response: {json_e}")
                st.text_area("Problematic Text Block:", potential_json, height=200)
                return None
        else:
            # JSON block not found
            st.error("Could not find a valid JSON object in the API response.")
            st.text_area("Raw Response from API:", response_text, height=200)
            return None

    except Exception as e:
        st.error(f"An error occurred during content optimization API call: {str(e)}")
        # Consider logging the full error trace for debugging
        # import traceback
        # st.error(traceback.format_exc())
        return None

# --------------------------------------------------------------------------
# PDF Generation Functions
# --------------------------------------------------------------------------

# MODIFICATION: Added original_content parameter
def generate_report_html(title: str, original_content: str, results: Dict) -> str:
    """Generates an HTML string for the PDF report content."""

    # Helper to safely get data from results dictionary
    def get_data(key_path, default=None):
        data = results
        try:
            for key in key_path.split('.'):
                # Handle potential list indices if needed in the future
                if isinstance(data, list) and key.isdigit():
                    data = data[int(key)]
                elif isinstance(data, dict):
                     data = data[key]
                else: # Not a dict or list index access, return default
                     return default if default is not None else []
            return data
        except (KeyError, TypeError, IndexError):
            return default if default is not None else [] # Default to empty list for iterables

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SEO Optimization Report: {html.escape(title)}</title>
        </head>
    <body>
        <h1>SEO Content Optimization Report</h1>
        <h2>Blog Title: {html.escape(title)}</h2>

        <div class="section">
            <h2>Content Evaluation</h2>
            <p><strong>Overall Score:</strong> {html.escape(str(get_data('content_evaluation.score', 'N/A')))}/5</p>
    """

    errors = get_data('content_evaluation.grammatical_errors', [])
    if errors:
        html_content += "<h3>Grammatical Errors:</h3><ul>"
        for error in errors:
            html_content += f"<li>{html.escape(error)}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No grammatical errors identified.</p>"

    violations = get_data('content_evaluation.guideline_violations', [])
    if violations:
        html_content += "<h3>Guideline Violations:</h3>"
        for v in violations:
            html_content += f"""
            <div class="violation-pdf">
                <strong>Violation:</strong> {html.escape(v.get('error', 'N/A'))}<br/>
                <em>Reason:</em> {html.escape(v.get('reason', 'N/A'))}<br/>
                <em>Fix:</em> {html.escape(v.get('fix', 'N/A'))}<br/>
                <em>Example:</em> {html.escape(v.get('example', '(Not provided)'))}
            </div>
            """
    else:
         html_content += "<p>No guideline violations identified.</p>"
    html_content += "</div>" # End section Content Evaluation

    # Section: Targeted Keywords
    keywords = get_data('targeted_keywords', [])
    html_content += '<div class="section"><h2>Targeted Keywords Analysis</h2>'
    if keywords:
        try:
            keywords_df_report = pd.DataFrame(keywords)
            # Specify columns for the table to ensure order
            cols_to_show = ['keyword', 'search_volume', 'difficulty', 'context']
            # Filter df to only include existing columns from cols_to_show
            keywords_df_report = keywords_df_report[[col for col in cols_to_show if col in keywords_df_report.columns]]
            html_content += keywords_df_report.to_html(escape=True, index=False, border=1, classes="pdf-table", justify='left')
        except Exception as df_e:
            html_content += f"<p>Error creating keywords table: {html.escape(str(df_e))}</p>"
            html_content += f"<p>Data: {html.escape(str(keywords))}</p>" # Show raw data for debugging
    else:
         html_content += "<p>No targeted keywords from the bank were identified in the content.</p>"
    html_content += '</div>' # End section Targeted Keywords

    # --- *** MODIFIED SECTION: Optimization Suggestions (Inline for PDF) *** ---
    optimizations = get_data('optimizations', [])
    html_content += '<div class="section"><h2>Optimization Suggestions (Inline)</h2>'

    if optimizations and original_content:
        # Start with the original content
        optimized_text_html_for_pdf = original_content

        # Apply replacements iteratively
        for opt in optimizations:
            original = opt.get("original")
            suggested = opt.get("suggested")
            if original and suggested: # Ensure both exist
                # Create the HTML snippet for replacement
                # Escape original and suggested text to prevent HTML injection issues
                escaped_original = html.escape(original)
                escaped_suggested = html.escape(suggested)
                replacement_html = (
                    f'<span class="highlight-pdf">{escaped_original}</span> '
                    f'<span class="suggestion-pdf">(Suggested: {escaped_suggested})</span>'
                )
                # Replace the *plain text* original phrase with the *HTML snippet*
                # Note: This simple replace might have issues if 'original' strings overlap
                # or contain special regex characters if a regex replace was used.
                # For complex cases, a more robust replacement strategy might be needed.
                optimized_text_html_for_pdf = optimized_text_html_for_pdf.replace(original, replacement_html)

        # Convert newlines to <br> tags for HTML display and wrap in a container
        optimized_text_html_for_pdf = optimized_text_html_for_pdf.replace('\n', '<br/>\n')
        html_content += f'<div class="inline-suggestions-pdf-container">{optimized_text_html_for_pdf}</div>'

    elif original_content: # Content exists, but no suggestions
        html_content += "<p>No specific optimization suggestions were generated.</p>"
        # Optionally display the original content escaped:
        # html_content += f"<h3>Original Content:</h3><div class='inline-suggestions-pdf-container'>{html.escape(original_content).replace('\n', '<br/>\n')}</div>"
    else: # No suggestions and no original content provided
        html_content += "<p>No specific optimization suggestions were generated or original content was not available for inline display.</p>"

    html_content += '</div>' # End section Optimization Suggestions
    # --- *** END OF MODIFIED SECTION *** ---

    # Section: Meta Information (within a styled panel)
    html_content += '<div class="section meta-panel-pdf">'
    html_content += '<h2>Meta Information</h2>'
    meta_titles = get_data('meta_titles', [])
    if meta_titles:
        html_content += '<h3>Suggested Meta Titles:</h3><ul>'
        for mt in meta_titles:
            html_content += f'<li>{html.escape(mt)}</li>'
        html_content += '</ul>'
    else:
        html_content += "<p>No meta titles suggested.</p>"

    meta_descriptions = get_data('meta_descriptions', [])
    if meta_descriptions:
        html_content += '<h3>Suggested Meta Descriptions:</h3><ul>'
        for md in meta_descriptions:
            html_content += f'<li>{html.escape(md)}</li>'
        html_content += '</ul>'
    else:
        html_content += "<p>No meta descriptions suggested.</p>"
    html_content += '</div>' # End meta-panel-pdf

    html_content += """
    </body>
    </html>
    """
    return html_content

def create_pdf(html_content: str) -> BytesIO | None:
    """Converts HTML string to a PDF file in memory using xhtml2pdf."""
    pdf_buffer = BytesIO()
    try:
        pisa_status = pisa.CreatePDF(
            src=html_content,  # The HTML content string
            dest=pdf_buffer,  # Write PDF to memory buffer
            encoding='UTF-8' # Ensure UTF-8 encoding
        )

        if pisa_status.err:
            st.error(f"Error generating PDF using pisa: {pisa_status.err}")
            # Log the HTML content that caused the error for debugging
            # print("------ Problematic HTML for PDF ------")
            # print(html_content)
            # print("--------------------------------------")
            return None
        else:
            pdf_buffer.seek(0) # Reset buffer position to the beginning for reading
            return pdf_buffer
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF creation: {e}")
        # import traceback
        # st.error(traceback.format_exc()) # Uncomment for detailed debug info
        # print("------ Problematic HTML for PDF ------")
        # print(html_content)
        # print("--------------------------------------")
        return None

# --------------------------------------------------------------------------
# Main Streamlit Application Logic
# --------------------------------------------------------------------------

def main():
    st.title("🎯 SEO Content Optimizer")
    model = initialize_gemini()
    if not model:
        st.error("Application cannot proceed without a working API connection.")
        st.stop()

    # Initialize session state variables
    st.session_state.setdefault('optimization_results', None)
    # st.session_state.setdefault('blog_title', "") # Removed, rely on input key directly
    st.session_state.setdefault('keywords_df', None)
    st.session_state.setdefault('last_keyword_filename', None)
    st.session_state.setdefault('last_keyword_filesize', None)
    # Use unique keys for input widgets to better manage state
    st.session_state.setdefault('blog_title_input_key', "")
    st.session_state.setdefault('blog_content_input_key', "")

    # --- Sidebar for keyword upload ---
    st.sidebar.header("1. Upload Keywords Bank")
    keywords_file = st.sidebar.file_uploader(
        "Upload SEO keywords CSV", type=["csv"], help="CSV must contain columns: keyword, search_volume, difficulty"
    )
    if keywords_file is not None:
        current_filename = keywords_file.name
        current_filesize = keywords_file.size
        # Re-process only if the file is different or keywords_df is not set
        if (st.session_state.keywords_df is None or
            current_filename != st.session_state.last_keyword_filename or
            current_filesize != st.session_state.last_keyword_filesize):
            st.sidebar.info(f"Processing '{current_filename}'...")
            processed_df = process_keywords_file(keywords_file)
            if processed_df is not None:
                st.session_state.keywords_df = processed_df
                st.session_state.last_keyword_filename = current_filename
                st.session_state.last_keyword_filesize = current_filesize
                st.sidebar.success(f"Keywords loaded ({len(processed_df)} valid rows).")
                # Clear previous results when new keywords are loaded
                st.session_state.optimization_results = None
                # Force rerun to update the main page state
                st.rerun()
            else:
                 st.sidebar.error("Keyword loading failed.")
                 # Clear potentially invalid state
                 st.session_state.last_keyword_filename = None
                 st.session_state.last_keyword_filesize = None
                 st.session_state.keywords_df = None
                 st.session_state.optimization_results = None


    # --- Main Content Area Check ---
    if st.session_state.keywords_df is None:
        st.info("⬅️ Please upload a valid keywords CSV file using the sidebar to begin.")
        st.stop()

    # --- Layout ---
    col1, col2 = st.columns([0.65, 0.35]) # Main content and Meta/Export side panel

    with col1:
        st.header("2. Input Content")
        # Use session state keys directly for input widgets
        blog_title_input = st.text_input(
            "Blog Title",
            placeholder="Enter the title of your content",
            key="blog_title_input_key" # Use the session state key
        )
        blog_content_input = st.text_area(
            "Blog Content",
            height=400,
            placeholder="Paste your blog post content here...",
            key="blog_content_input_key" # Use the session state key
        )
        submit_button = st.button("🚀 Optimize Content", type="primary", use_container_width=True)

        # --- Optimization Button Logic ---
        if submit_button:
            # Access input values directly from session state using their keys
            current_title = st.session_state.blog_title_input_key
            current_content = st.session_state.blog_content_input_key

            if not current_content or not current_title:
                st.warning("⚠️ Please enter both a title and content before optimizing.")
                st.session_state.optimization_results = None
            else:
                with st.spinner("🧠 Analyzing and optimizing content..."):
                    results = optimize_content(model, current_content, current_title, st.session_state.keywords_df)
                    st.session_state.optimization_results = results
                # No need to explicitly set 'blog_title' in session state if using input key

        # --- Display Optimization Results (if available) ---
        results = st.session_state.get('optimization_results')
        if results:
            st.divider()
            st.subheader("📊 Optimization Analysis")

            # --- Content Evaluation Display ---
            st.markdown("#### Content Evaluation Score & Issues")
            eval_data = results.get("content_evaluation", {})
            score = eval_data.get("score", "N/A")
            st.metric(label="Overall Score (Guideline Adherence)", value=f"{score}/5")

            grammatical_errors = eval_data.get("grammatical_errors", [])
            if grammatical_errors:
                with st.expander("Grammatical Errors Found", expanded=False):
                    for error in grammatical_errors:
                        st.markdown(f"- {error}")

            guideline_violations = eval_data.get("guideline_violations", [])
            if guideline_violations:
                with st.expander("Guideline Violations Found", expanded=True):
                    for violation in guideline_violations:
                        st.error(f"**Violation:** {violation.get('error', 'N/A')}", icon="❗")
                        st.markdown(f"  - **Reason:** {violation.get('reason', 'N/A')}")
                        st.markdown(f"  - **Fix:** {violation.get('fix', 'N/A')}")
                        if 'example' in violation and violation['example']:
                             st.markdown(f"  - **Example:** `{violation.get('example')}`")
                             st.divider() # Add divider after each violation detail
            elif submit_button: # Only show success if button was just pressed and no violations
                 st.success("✅ No specific guideline violations identified.")

            # --- Targeted Keywords Display ---
            st.markdown("#### Targeted Keywords Analysis")
            targeted_keywords_data = results.get("targeted_keywords", [])
            if targeted_keywords_data:
                try:
                    keywords_df_display = pd.DataFrame(targeted_keywords_data)
                    st.dataframe(
                        keywords_df_display,
                        column_config={
                            "keyword": st.column_config.TextColumn("Keyword", help="Keyword from your bank found in the text"),
                            "search_volume": st.column_config.NumberColumn("Search Volume", help="Search volume from your bank"),
                            "difficulty": st.column_config.NumberColumn("Difficulty", help="Difficulty score from your bank"),
                            "context": st.column_config.TextColumn("Usage Context", help="Example of how the keyword is used", width="large")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                except Exception as df_e:
                    st.error(f"Error displaying keywords table: {df_e}")
                    st.write(targeted_keywords_data) # Show raw data if dataframe fails
            else:
                 st.info("ℹ️ No keywords from your bank were found in the provided content.")

            # --- Optimization Suggestions (Inline in Streamlit App) ---
            st.markdown("#### Optimization Suggestions (Inline)")
            optimizations_list = results.get("optimizations", [])
            if optimizations_list:
                # Start with the current content from the input area (session state)
                optimized_text = st.session_state.blog_content_input_key
                # Apply replacements iteratively
                for opt in optimizations_list:
                    original = opt.get("original")
                    suggested = opt.get("suggested")
                    if original and suggested: # Ensure both exist
                        # Replace ALL occurrences of the original phrase using HTML for display
                        # Escape original and suggested text before inserting into HTML
                        escaped_original = html.escape(original)
                        escaped_suggested = html.escape(suggested)
                        replacement_html = (
                            f'<span class="highlight">{escaped_original}</span> '
                            f'<span class="suggestion">(Suggested: {escaped_suggested})</span>'
                        )
                        # Replace plain text `original` with the generated `replacement_html`
                        optimized_text = optimized_text.replace(original, replacement_html)

                # Display the final text with all replacements applied
                # Convert newlines to <br> for proper display in markdown
                optimized_text_display = optimized_text.replace('\n', '<br/>\n')
                st.markdown(f'<div style="border: 1px solid #eee; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">{optimized_text_display}</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ No specific inline optimization suggestions were generated.")

        elif submit_button: # If button was pressed but results are None/empty after processing
             st.error("❌ Optimization failed or returned no results. Please check the API key, input content, connection, and try again.")

    # --- Right Column: Meta Info and Export ---
    with col2:
        st.header("3. Review & Export")
        results = st.session_state.get('optimization_results') # Get results again
        st.markdown("#### Suggested Meta Information")

        if results:
            # --- MODIFICATION: Display Meta Titles as List ---
            st.markdown("##### Meta Titles (< 60 chars)")
            meta_titles_list = results.get("meta_titles", [])
            if meta_titles_list:
                for title in meta_titles_list:
                    st.markdown(f"- {title}") # Display each title as a markdown list item
            else:
                st.markdown("_No titles suggested._")

            st.markdown("---") # Separator

            # --- MODIFICATION: Display Meta Descriptions as List ---
            st.markdown("##### Meta Descriptions (< 160 chars)")
            meta_descriptions_list = results.get("meta_descriptions", [])
            if meta_descriptions_list:
                 for desc in meta_descriptions_list:
                    st.markdown(f"- {desc}") # Display each description as a markdown list item
            else:
                st.markdown("_No descriptions suggested._")
            # --- END OF MODIFICATIONS ---

            st.divider()
            st.markdown("#### Export Full Report")
            try:
                # MODIFICATION: Pass original content to generate_report_html
                # Get title and content from session state where they were stored by input widgets
                report_title = st.session_state.blog_title_input_key
                report_content = st.session_state.blog_content_input_key

                if not report_title: # Handle case where title might be empty if optimization failed early
                    report_title = "Untitled Content"

                # Generate HTML using title, original content, and results
                report_html = generate_report_html(report_title, report_content, results)
                pdf_buffer = create_pdf(report_html)

                if pdf_buffer:
                    # Sanitize title for filename
                    safe_title = report_title.replace(" ", "_").replace("/", "-")
                    safe_title = "".join(c for c in safe_title if c.isalnum() or c in ['_', '-']).strip()[:40]
                    pdf_filename = f"SEO_Report_{safe_title}.pdf" if safe_title else "SEO_Optimization_Report.pdf"
                    st.download_button(
                        label="📥 Download Report as PDF",
                        data=pdf_buffer,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("PDF generation failed. Cannot create download link.")
            except Exception as e:
                st.error(f"An error occurred during PDF preparation: {e}")
                # import traceback # For detailed debugging
                # st.error(traceback.format_exc())

        else:
            st.info("Optimize content first (using the button on the left) to see suggestions and export options.")

        st.markdown('</div>', unsafe_allow_html=True) # Close meta-panel div

# --- Run the App ---
if __name__ == "__main__":
    main()