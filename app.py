import streamlit as st
import requests
from datetime import datetime
from openai import OpenAI

# ----------------------------------------------------
# 1. Environment Setup and Session Initialization
# ----------------------------------------------------
st.set_page_config(layout="wide")

# Load API Keys (from secrets.toml)
try:
    GUARDIAN_API_KEY = st.secrets["guardian_api_key"]
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    st.error("API keys not found. Please check your '.streamlit/secrets.toml' file.")
    st.stop()

# Session State Initialization: Stores search results and analysis reports
if 'reports' not in st.session_state:
    st.session_state.reports = {}

# ----------------------------------------------------
# 2. Core Functions
# ----------------------------------------------------

def guardian_search(keyword, page_size=10):
    """Calls the Guardian API to fetch the latest articles."""
    url = "https://content.guardianapis.com/search"
    params = {
        "q": keyword,
        "api-key": GUARDIAN_API_KEY,
        "show-fields": "headline,trailText,webPublicationDate",
        "page-size": page_size,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        data = response.json()
        articles = data.get("response", {}).get("results", [])
        return articles
    except Exception as e:
        st.error(f"Guardian API call error while searching for '{keyword}': {e}")
        return []

@st.cache_data(show_spinner="AI is generating the summary reports...")
def generate_analysis(keyword, articles, previous_articles=None):
    """Uses OpenAI API to generate summaries and trend change analysis in English."""
    
    # Compile article snippets for the prompt
    article_texts = "\n".join([
        f"- {a.get('webTitle', '')} ({a.get('webPublicationDate', '')[:10]}) - {a.get('fields', {}).get('trailText', '')}"
        for a in articles
    ])
    
    # 1. Current Trend Summary Prompt
    summary_prompt = f"""
    The following is a list of recent news articles for the keyword '{keyword}'.
    Based on these articles, write a **one-page summary report** on the overall media trend in **English**.
    (Include the general sentiment, key issues, and major related figures/companies. Format cleanly using Markdown.)
    ---
    Article List:
    {article_texts}
    ---
    """
    
    # 2. Trend Change Comparison Prompt
    trend_change_text = "No previous search data available to compare trend changes."
    
    if previous_articles:
        previous_article_texts = "\n".join([
            f"- {a.get('webTitle', '')}" for a in previous_articles
        ])
        comparison_prompt = f"""
        Compare the previous list of articles with the latest list for the keyword '{keyword}' and analyze the **key shifts in media trends**.
        Please structure the result using the following format:
        ## Major Trend Change Analysis
        ### 1. Key Shifts Summary
        [Analysis content]
        ### 2. Newly Emerging or Rising Issues
        * [Issue 1]
        * [Issue 2]
        ### 3. Issues Decreasing in Importance
        * [Issue 1]
        
        * Previous Article List: {previous_article_texts}
        * Latest Article List: {article_texts}
        """
        
        # LLM Call: Trend Change Analysis
        try:
            trend_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": comparison_prompt}]
            )
            trend_change_text = trend_response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI Trend Analysis Error: {e}")
    
    # LLM Call: Current Summary Report
    try:
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        current_summary = summary_response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI Summary Report Generation Error: {e}")
        current_summary = "Failed to generate the summary report."

    return current_summary, trend_change_text

def run_agent(keywords):
    """Executes the search and analysis based on keywords and updates the session state."""
    
    st.info("Searching for new data and generating analysis reports. Please wait...")
    new_reports = {}
    
    for keyword in keywords:
        with st.spinner(f"Searching and analyzing articles for '{keyword}'..."):
            articles = guardian_search(keyword)
            
            if not articles:
                st.warning(f"Could not find recent articles for keyword '{keyword}'.")
                continue
            
            # Load previous data
            previous_data = st.session_state.reports.get(keyword, {})
            previous_articles = previous_data.get('current')
            
            # Execute LLM analysis
            current_summary, trend_change = generate_analysis(keyword, articles, previous_articles)
            
            # Update data: move current to previous, store new data as current
            new_reports[keyword] = {
                'current': articles,
                'previous': previous_articles if previous_articles else [],
                'current_summary': current_summary,
                'trend_change': trend_change,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    # Update session state
    st.session_state.reports.update(new_reports)
    st.success("Data search and analysis completed!")


# ----------------------------------------------------
# 3. Streamlit UI (Dashboard)
# ----------------------------------------------------

st.title("📰 AI News Trend Analysis Dashboard")
st.markdown("Search for the latest news articles from The Guardian for keywords of interest and receive trend reports.")

# Keyword Input Field
default_keywords = "Space Industry, Artificial Intelligence, Quantum Computing, Climate Change"
keyword_input = st.text_input(
    "Enter keywords to analyze, separated by commas (,).",
    value=default_keywords
)
keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]

# Search and Update Buttons
col1, col2 = st.columns(2)

if col1.button("🔍 Run Search & Analysis", help="Starts a new search and analysis for the entered keywords."):
    if keywords:
        run_agent(keywords)
    else:
        st.warning("Please enter keywords to analyze.")
        
if col2.button("🔄 Refresh & Compare Trends", help="Compares the latest results with previous data to analyze trend changes."):
    if keywords and st.session_state.reports:
        run_agent(keywords)
    elif keywords:
        st.warning("No previous data available. Please run 'Search & Analysis' first.")
    else:
        st.warning("Please enter keywords to analyze.")
        
st.markdown("---")


# ----------------------------------------------------
# 4. Display Analysis Results
# ----------------------------------------------------

if st.session_state.reports:
    st.header("📈 Analysis Results")
    
    # Create tabs for each keyword
    tab_titles = list(st.session_state.reports.keys())
    tabs = st.tabs(tab_titles)
    
    for i, keyword in enumerate(tab_titles):
        with tabs[i]:
            report_data = st.session_state.reports[keyword]
            st.caption(f"Last Updated: **{report_data['last_updated']}**")
            
            # --- Trend Change Comparison ---
            st.subheader("📊 Trend Change Analysis")
            st.markdown(report_data['trend_change'])
            
            st.divider()

            # --- Summary Report ---
            st.subheader("📄 One-Page Summary Report")
            st.markdown(report_data['current_summary'])
            
            st.divider()

            # --- Latest Article List ---
            st.subheader("📜 Latest Articles Found")
            
            for article in report_data['current']:
                date = article.get('webPublicationDate', '')[:10]
                headline = article.get('webTitle', 'No Title')
                url = article.get('webUrl', '#')
                trail_text = article.get('fields', {}).get('trailText', 'No summary available')
                
                st.markdown(
                    f"* **[{headline}](<{url}>)** (_{date}_)  \n"
                    f"  {trail_text}"
                )