import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from deep_translator import GoogleTranslator
from langdetect import detect_langs
import re
import sweetviz as sv
from streamlit.components.v1 import html
import io
import numpy as np
import time
import warnings
from fancyimpute import KNN  # For AI-based imputation
from scipy import stats  # For outlier detection

# Monkey patches (unchanged from your original code)
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items
if not hasattr(pd.Series, 'iteritems'):
    pd.Series.iteritems = pd.Series.items
if not hasattr(pd.Series, 'mad'):
    def mad(self, axis=None, skipna=True, level=None):
        if axis is not None or level is not None:
            raise NotImplementedError("Only default axis and level=None are supported.")
        return (self - self.mean(skipna=skipna)).abs().mean(skipna=skipna)
    pd.Series.mad = mad
if not hasattr(np, 'warnings'):
    np.warnings = warnings
if not hasattr(np, 'VisibleDeprecationWarning'):
    try:
        from numpy.exceptions import VisibleDeprecationWarning
        np.VisibleDeprecationWarning = VisibleDeprecationWarning
    except ImportError:
        np.VisibleDeprecationWarning = warnings.Warning

translator = GoogleTranslator(source='auto', target='en')

# Custom CSS (unchanged, but added a class for cleaning feedback)
st.markdown("""
<style>
    .stChatMessage { padding: 10px; border-radius: 10px; margin: 5px 0; width: 90%; max-width: 800px; }
    .stChatMessage.user { background-color: #f0f2f6; margin-left: auto; }
    .stChatMessage.assistant { background-color: #e6f3ff; margin-right: auto; }
    .stButton button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 8px 16px; width: 100%; max-width: 200px; }
    .center-plotly { display: flex; justify-content: center; width: 100%; }
    .cleaning-feedback { color: #4CAF50; font-weight: bold; }
    @media (max-width: 768px) {
        .stChatMessage { width: 95%; font-size: 14px; }
        .stButton button { max-width: 100%; font-size: 14px; }
    }
</style>
""", unsafe_allow_html=True)

# Authentication (unchanged)
def authenticate(username, password):
    return username == st.secrets["USERNAME"] and password == st.secrets["PASSWORD"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title(" Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

# AI-Powered Data Cleaning Function
def clean_data_with_ai(df, impute_method="knn", outlier_threshold=3):
    """
    Cleans data using AI-powered imputation and outlier removal.
    - impute_method: "knn" (default) or "mean"
    - outlier_threshold: Z-score threshold for outlier detection
    """
    cleaned_df = df.copy()
    
    # Step 1: Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    if cleaned_df[numeric_cols].isnull().sum().sum() > 0:
        if impute_method == "knn":
            # Use KNN imputation for numeric columns
            knn_imputer = KNN()
            cleaned_df[numeric_cols] = knn_imputer.fit_transform(cleaned_df[numeric_cols])
        else:
            # Fallback to mean imputation
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    
    # Step 2: Remove outliers (only for numeric columns)
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
        cleaned_df = cleaned_df[cleaned_df[col].isin(cleaned_df[col][z_scores < outlier_threshold]) | cleaned_df[col].isna()]
    
    # Step 3: Drop duplicates
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df, {
        "missing_imputed": df[numeric_cols].isnull().sum().sum() - cleaned_df[numeric_cols].isnull().sum().sum(),
        "outliers_removed": len(df) - len(cleaned_df) + (initial_rows - len(cleaned_df)),
        "duplicates_removed": initial_rows - len(cleaned_df)
    }

# Main app
st.title(" Data Insights Chatbot")
st.sidebar.title("Options")
st.sidebar.write("Upload a file and start chatting with the bot!")

# File upload and session state initialization (unchanged)
uploaded_file = st.file_uploader("Upload a file", type=["csv", "json", "xlsx"])
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data" not in st.session_state:
    st.session_state.data = None
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = "{data}\n\n{question}"
if "show_visualization" not in st.session_state:
    st.session_state.show_visualization = False
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Histogram"
if "column_to_visualize" not in st.session_state:
    st.session_state.column_to_visualize = None
if "show_eda_report" not in st.session_state:
    st.session_state.show_eda_report = False
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None

# Customizable Prompt in Sidebar (unchanged)
st.sidebar.write("### Customize Prompt")
custom_prompt_input = st.sidebar.text_area(
    "Prompt Template",
    value=st.session_state.custom_prompt,
    height=100,
    help="Example: 'Analyze this data: {data}\nAnswer this: {question}'"
)
if st.sidebar.button("Save Prompt"):
    st.session_state.custom_prompt = custom_prompt_input
    st.sidebar.success("Prompt saved!")

# Load data function (unchanged)
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endsiwth(".xlsx"):
            return pd.read_excel(uploaded_file, sheet_name=None)
    return None

if uploaded_file and st.session_state.data is None:
    st.session_state.data = load_data(uploaded_file)

# Display file preview (unchanged)
if st.session_state.data is not None:
    with st.expander("📂 File Preview"):
        if isinstance(st.session_state.data, dict):
            sheet_name = st.selectbox("Select a sheet", list(st.session_state.data.keys()))
            st.write(st.session_state.data[sheet_name].head())
        elif isinstance(st.session_state.data, pd.DataFrame):
            st.write(st.session_state.data.head())

# Interactive Data Exploration with AI Cleaning
if isinstance(st.session_state.data, pd.DataFrame):
    with st.expander("🔍 Interactive Data Exploration", expanded=False):
        st.write("### Filter, Explore, and Clean Data")
        
        df = st.session_state.data.copy()
        
        # AI Data Cleaning Section
        st.write("#### AI-Powered Data Cleaning")
        impute_method = st.selectbox("Imputation Method", ["KNN (AI)", "Mean"], key="impute_method")
        outlier_threshold = st.slider("Outlier Removal Threshold (Z-score)", 1.0, 5.0, 3.0, key="outlier_threshold")
        if st.button("Clean Data with AI"):
            with st.spinner("Cleaning data with AI..."):
                cleaned_df, cleaning_stats = clean_data_with_ai(
                    df,
                    impute_method="knn" if impute_method == "KNN (AI)" else "mean",
                    outlier_threshold=outlier_threshold
                )
                st.session_state.filtered_data = cleaned_df
                st.success("Data cleaned successfully!")
                st.markdown(
                    f'<p class="cleaning-feedback">'
                    f'Missing values imputed: {cleaning_stats["missing_imputed"]}<br>'
                    f'Outliers removed: {cleaning_stats["outliers_removed"]}<br>'
                    f'Duplicates removed: {cleaning_stats["duplicates_removed"]}</p>',
                    unsafe_allow_html=True
                )
        
        # Existing Filtering Logic (unchanged)
        filter_columns = st.multiselect("Select columns to filter", options=df.columns.tolist(), default=[])
        filtered_df = df.copy()
        for col in filter_columns:
            col_type = df[col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                if min_val != max_val:
                    range_val = st.slider(f"Filter {col} (range)", min_val, max_val, (min_val, max_val), key=f"slider_{col}")
                    filtered_df = filtered_df[(filtered_df[col] >= range_val[0]) & (filtered_df[col] <= range_val[1])]
            else:
                unique_vals = df[col].unique().tolist()
                selected_vals = st.multiselect(f"Filter {col} (values)", options=unique_vals, default=unique_vals, key=f"multiselect_{col}")
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
        
        st.write(f"Filtered Data ({len(filtered_df)} rows):")
        st.write(filtered_df.head())
        if st.button("Use Filtered Data in Chatbot"):
            st.session_state.filtered_data = filtered_df
            st.success("Filtered data is now active for chatbot analysis!")

# Use filtered or cleaned data
active_data = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data

# Process prompt function (unchanged)
def process_prompt(prompt, data):
    if data is None:
        return "Please upload a valid CSV, JSON or XLSX."
    lang_probs = detect_langs(prompt)
    detected_lang = lang_probs[0].lang
    confidence = lang_probs[0].prob
    is_english_input = detected_lang == 'en' or (len(prompt.split()) < 5 and confidence < 0.9)
    translated_prompt = prompt if is_english_input else GoogleTranslator(source=detected_lang, target='en').translate(prompt)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.secrets["GEMINI_API_KEY"], temperature=0.3, max_output_tokens=1000)
    
    if isinstance(data, pd.DataFrame):
        summary_stats = data.describe().to_string()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        cat_summary = {col: data[col].value_counts().head(1000).to_string() for col in categorical_cols}
        data_summary = (
            f"Dataset Overview:\n- Rows: {len(data)}\n- Columns: {len(data.columns)}\nNumeric Summary:\n{summary_stats}\n"
            f"Categorical Summary:\n{'\n'.join(f'{col}:\n{counts}' for col, counts in cat_summary.items()) or 'None'}"
        )
    else:
        data_summary = str(data)
    
    prompt_template = ChatPromptTemplate.from_template(st.session_state.custom_prompt)
    chain = prompt_template | llm | RunnablePassthrough()
    response = chain.invoke({"data": data_summary, "question": translated_prompt})
    response_text = response.content if hasattr(response, "content") else str(response)
    final_response = response_text if is_english_input else GoogleTranslator(source='en', target=detected_lang).translate(response_text)
    return final_response or "No meaningful response generated."

# Chat interface (unchanged)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"{message['content']} *({message['timestamp']})*")
if prompt := st.chat_input("Ask a question about the data"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user"):
        st.markdown(f"{prompt} *({timestamp})*")
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_prompt(prompt, active_data)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"{response} *({timestamp})*")
            st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})

# Sidebar options (added AI cleaning button)
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
if st.sidebar.button("Download Chat History"):
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    st.sidebar.download_button(label="Download Chat", data=chat_history, file_name="chat_history.txt", mime="text/plain")
if st.sidebar.button("Reset Data Filters"):
    st.session_state.filtered_data = None
    st.sidebar.success("Data filters reset to original dataset!")
if st.sidebar.button("Clean Data with AI (Quick)"):
    if isinstance(st.session_state.data, pd.DataFrame):
        with st.spinner("Cleaning data with AI..."):
            cleaned_df, cleaning_stats = clean_data_with_ai(st.session_state.data)
            st.session_state.filtered_data = cleaned_df
            st.sidebar.success(f"Data cleaned! Missing: {cleaning_stats['missing_imputed']}, Outliers: {cleaning_stats['outliers_removed']}")
    else:
        st.sidebar.error("Upload a DataFrame to clean.")

# Predefined Queries, Visualization, EDA, and Export (unchanged)
if st.sidebar.button("Show Summary Statistics"):
    if isinstance(active_data, pd.DataFrame):
        st.write(active_data.describe())
if st.sidebar.button("Detect Missing Values"):
    if isinstance(active_data, pd.DataFrame):
        st.write(active_data.isnull().sum())
if isinstance(active_data, pd.DataFrame):
    st.sidebar.write("#### Generate Visualization")
    chart_type = st.sidebar.selectbox("Select chart type", ["Histogram", "Bar", "Scatter"])
    numeric_columns = active_data.select_dtypes(include=['number']).columns
    if not numeric_columns.empty:
        column_to_visualize = st.sidebar.selectbox("Select a column", numeric_columns, key="column_select")
        if st.sidebar.button("Generate Visualization"):
            st.session_state.show_visualization = True
            st.session_state.chart_type = chart_type
            st.session_state.column_to_visualize = column_to_visualize
if isinstance(active_data, pd.DataFrame):
    st.sidebar.write("#### Advanced Data Analysis")
    if st.sidebar.button("Generate EDA Report"):
        with st.spinner("Generating EDA report..."):
            report = sv.analyze(active_data)
            report.show_html("eda_report.html", open_browser=False)
            st.session_state.show_eda_report = True
if st.session_state.show_visualization and isinstance(active_data, pd.DataFrame):
    st.write("### Data Visualization")
    data = active_data.reset_index()
    chart_type = st.session_state.chart_type
    column = st.session_state.column_to_visualize
    if chart_type == "Histogram":
        fig = px.histogram(data, x=column, title=f"Histogram of {column}")
    elif chart_type == "Bar":
        fig = px.bar(data, x='index', y=column, title=f"Bar Chart of {column}")
    elif chart_type == "Scatter":
        fig = px.scatter(data, x='index', y=column, title=f"Scatter Plot of {column}")
    st.markdown('<div class="center-plotly">', unsafe_allow_html=True)
    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)
if st.session_state.show_eda_report and isinstance(active_data, pd.DataFrame):
    st.write("### Exploratory Data Analysis Report")
    with open("eda_report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    html(html_content, height=800, scrolling=True)
if active_data is not None and isinstance(active_data, pd.DataFrame):
    st.sidebar.write("### Export Data")
    export_format = st.sidebar.selectbox("Select format", ["CSV", "Excel"])
    if export_format == "CSV":
        csv_buffer = io.StringIO()
        active_data.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(label="Download as CSV", data=csv_buffer.getvalue(), file_name="exported_data.csv", mime="text/csv")
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            active_data.to_excel(writer, index=False)
        st.sidebar.download_button(label="Download as Excel", data=excel_buffer.getvalue(), file_name="exported_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")