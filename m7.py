import streamlit as st
import pandas as pd
#import PyPDF2
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from deep_translator import GoogleTranslator
from langdetect import detect
import re
import sweetviz as sv
from streamlit.components.v1 import html
import io
import numpy as np

if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

# Monkey patch iteritems for Pandas 2.0+ (Series)
if not hasattr(pd.Series, 'iteritems'):
    pd.Series.iteritems = pd.Series.items

# Monkey patch mad for Pandas 2.1+
if not hasattr(pd.Series, 'mad'):
    def mad(self, axis=None, skipna=True, level=None):
        """Calculate the mean absolute deviation."""
        if axis is not None or level is not None:
            raise NotImplementedError("Only default axis and level=None are supported in this patch.")
        return (self - self.mean(skipna=skipna)).abs().mean(skipna=skipna)
    pd.Series.mad = mad

# Monkey patch np.warnings for newer NumPy versions
import warnings
if not hasattr(np, 'warnings'):
    np.warnings = warnings
if not hasattr(np, 'VisibleDeprecationWarning'):
    # Use NumPy's internal exception if available, otherwise fallback to a generic Warning
    try:
        from numpy.exceptions import VisibleDeprecationWarning
        np.VisibleDeprecationWarning = VisibleDeprecationWarning
    except ImportError:
        np.VisibleDeprecationWarning = warnings.Warning

translator = GoogleTranslator(source='auto', target='en')

# Custom CSS for responsive design
st.markdown("""
<style>
    /* General styling */
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        width: 90%;
        max-width: 800px;
        box-sizing: border-box;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
        margin-left: auto;
    }
    .stChatMessage.assistant {
        background-color: #e6f3ff;
        margin-right: auto;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        width: 100%;
        max-width: 200px;
    }
    .center-plotly {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    /* Sidebar styling */
    .css-1d391kg {  /* Streamlit sidebar class */
        width: 300px;
        max-width: 80%;
    }

    /* Main content adjustments */
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stChatMessage {
            width: 95%;
            font-size: 14px;
        }
        .stButton button {
            max-width: 100%;
            font-size: 14px;
        }
        .css-1d391kg {
            width: 100%;
            position: relative;
            padding: 1rem;
        }
        .main .block-container {
            padding: 0.5rem;
        }
        .stTextInput, .stTextArea, .stFileUploader {
            width: 100% !important;
        }
        .stSelectbox, .stExpander {
            width: 100% !important;
        }
        .plotly-chart {
            width: 100% !important;
            height: auto !important;
        }
    }

    @media (max-width: 480px) {
        .stChatMessage {
            font-size: 12px;
            padding: 8px;
        }
        .stButton button {
            font-size: 12px;
            padding: 6px 12px;
        }
        h1, h2, h3 {
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Authentication
def authenticate(username, password):
    """Simple authentication using Streamlit secrets."""
    if username == st.secrets["USERNAME"] and password == st.secrets["PASSWORD"]:
        return True
    return False

# Login screen
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

# Main app
st.title(" Data Insights Chatbot")
st.sidebar.title("Options")
st.sidebar.write("Upload a file and start chatting with the bot!")

# File upload
uploaded_file = st.file_uploader("Upload a file", type=["csv", "json", "xlsx"])

# Initialize session state for chat history and data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data" not in st.session_state:
    st.session_state.data = None
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = "{data}\n\n{question}"  # Default prompt
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

# Customizable Prompt in Sidebar
st.sidebar.write("### Customize Prompt")
st.sidebar.write("Edit the prompt template below. Use `{data}` for the dataset and `{question}` for your input.")
custom_prompt_input = st.sidebar.text_area(
    "Prompt Template",
    value=st.session_state.custom_prompt,
    height=100,
    help="Example: 'Analyze this data: {data}\nAnswer this: {question}'"
)
if st.sidebar.button("Save Prompt"):
    st.session_state.custom_prompt = custom_prompt_input
    st.sidebar.success("Prompt saved!")

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, sheet_name=None)  # Support for multiple sheets
        '''elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text'''
    return None

# Load data and store in session state
if uploaded_file and st.session_state.data is None:
    st.session_state.data = load_data(uploaded_file)

# Display file preview
if st.session_state.data is not None:
    with st.expander("📂 File Preview"):
        if isinstance(st.session_state.data, dict):  # For Excel files with multiple sheets
            sheet_name = st.selectbox("Select a sheet", list(st.session_state.data.keys()))
            st.write(st.session_state.data[sheet_name].head())
        elif isinstance(st.session_state.data, pd.DataFrame):
            st.write(st.session_state.data.head())
        '''elif isinstance(st.session_state.data, str):  # For PDF files
            st.write(st.session_state.data[:1000] + "...")'''  # Show first 1000 characters

# New Interactive Data Exploration Section
if isinstance(st.session_state.data, pd.DataFrame):
    with st.expander("🔍 Interactive Data Exploration", expanded=False):
        st.write("### Filter and Explore Data")
        
        # Create a copy of the data for filtering
        df = st.session_state.data.copy()
        
        # Select columns for filtering
        filter_columns = st.multiselect(
            "Select columns to filter",
            options=df.columns.tolist(),
            default=[],
            key="filter_columns"
        )
        
        # Dynamic filters based on selected columns
        filtered_df = df.copy()
        for col in filter_columns:
            col_type = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                # Slider for numeric columns
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                if min_val != max_val:
                    range_val = st.slider(
                        f"Filter {col} (range)",
                        min_val,
                        max_val,
                        (min_val, max_val),
                        key=f"slider_{col}"
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col] >= range_val[0]) & 
                        (filtered_df[col] <= range_val[1])
                    ]
            
            else:
                # Multiselect for categorical columns
                unique_vals = df[col].unique().tolist()
                selected_vals = st.multiselect(
                    f"Filter {col} (values)",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"multiselect_{col}"
                )
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
        
        # Display filtered data
        st.write(f"Filtered Data ({len(filtered_df)} rows):")
        st.write(filtered_df.head())
        
        # Option to use filtered data in chatbot
        if st.button("Use Filtered Data in Chatbot"):
            st.session_state.filtered_data = filtered_df
            st.success("Filtered data is now active for chatbot analysis!")

# Use filtered data if available, otherwise use original data
active_data = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data

# Function to process user prompts
def process_prompt(prompt, data):
    if data is None:
        return "Please upload a valid CSV, JSON or XLSX."
    # Detect language
    detected_lang = detect(prompt)
    
    # Translate prompt to English if necessary
    try:
        # Translate prompt to English for processing (since the model works in English)
        translated_prompt = GoogleTranslator(source=detected_lang, target='en').translate(prompt) if detected_lang != 'en' else prompt

        # Initialize Gemini API
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.secrets["GEMINI_API_KEY"])

        # Create a ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_template(
            "You are a data analyst. Analyze the following data and answer the user's question:\n\n"
            "Data:\n{data}\n\n"
            "Question: {question}"
        )

        # Use the custom prompt from session state
        prompt_template = ChatPromptTemplate.from_template(st.session_state.custom_prompt)

        # Create an LLMChain
        chain = prompt_template | llm | RunnablePassthrough()
        response = chain.invoke({"data": data.head().to_string() if isinstance(data, pd.DataFrame) else data, "question": translated_prompt})
        response_text = response.content if hasattr(response, "content") else str(response)  # Extract text
        #return response_text

        cleaned_response = re.sub(r'[^0-9a-zA-Z\s.,]', '', response_text)
        # Translate response back to original language
        #translated_response = translator.translate(response_text, src='en', dest=detected_lang).text
        #return translated_response
        final_response = (GoogleTranslator(source='en', target=detected_lang).translate(response_text) 
                         if detected_lang != 'en' else response_text)
        return final_response
        #final_response = ''.join(c for c in translated_response if ord(c) >= 0x0B80 and ord(c) <= 0x0BFF or c.isspace() or c in '.,0123456789')

        # Translate back only if the original input was not in English
        #final_response = response_text if detected_lang == "en" else translator.translate(response_text, src='en', dest=detected_lang).text

        #return final_response if final_response.strip() else translated_response  # Fallback if filtering removes too much

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the prompt and generate a response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_prompt(prompt, st.session_state.data)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar options
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

if st.sidebar.button("Download Chat History"):
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    st.sidebar.download_button(
        label="Download Chat",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain"
    )
if st.sidebar.button("Reset Data Filters"):
    st.session_state.filtered_data = None
    st.sidebar.success("Data filters reset to original dataset!")

# Predefined queries
st.sidebar.write("### Predefined Queries")
if st.sidebar.button("Show Summary Statistics"):
    if isinstance(st.session_state.data, pd.DataFrame):
        st.write(st.session_state.data.describe())
    else:
        st.write("Summary statistics are only available for tabular data (CSV, Excel).")

if st.sidebar.button("Detect Missing Values"):
    if isinstance(st.session_state.data, pd.DataFrame):
        st.write(st.session_state.data.isnull().sum())
    else:
        st.write("Missing value detection is only available for tabular data (CSV, Excel).")

# Data Visualization in Sidebar under Predefined Queries
if isinstance(st.session_state.data, pd.DataFrame):
    st.sidebar.write("#### Generate Visualization")
    chart_type = st.sidebar.selectbox("Select chart type", ["Histogram", "Bar", "Scatter"])
    numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns
    if numeric_columns.empty:
        st.sidebar.write("No numeric columns available for visualization.")
    else:
        column_to_visualize = st.sidebar.selectbox("Select a column to visualize", numeric_columns, key="column_select")
        if st.sidebar.button("Generate Visualization"):
            st.session_state.show_visualization = True
            st.session_state.chart_type = chart_type
            st.session_state.column_to_visualize = column_to_visualize

# Advanced Data Analysis (EDA) under Predefined Queries
if isinstance(st.session_state.data, pd.DataFrame):
    st.sidebar.write("#### Advanced Data Analysis")
    if st.sidebar.button("Generate EDA Report"):
        with st.spinner("Generating EDA report..."):
            try:
                # Preprocess data: reset index and ensure 'index' column
                data_for_report = st.session_state.data.copy()  # Avoid modifying original data
                if data_for_report.index.name is not None:
                    data_for_report = data_for_report.reset_index().rename(columns={data_for_report.index.name: 'index'})
                else:
                    data_for_report = data_for_report.reset_index()
                
                # Ensure all columns are properly named and index is set
                if 'index' not in data_for_report.columns:
                    data_for_report = data_for_report.rename(columns={data_for_report.columns[0]: 'index'})
                
                # Generate the report
                report = sv.analyze(data_for_report)
                report_file = "eda_report.html"
                report.show_html(report_file, open_browser=False)
                st.session_state.show_eda_report = True
            except Exception as e:
                st.error(f"Failed to generate EDA report: {str(e)}")

# Display visualization in main body if triggered
if st.session_state.show_visualization and isinstance(st.session_state.data, pd.DataFrame):
    st.write("### Data Visualization")
    data = st.session_state.data.reset_index()
    chart_type = st.session_state.chart_type
    column = st.session_state.column_to_visualize
    
    if chart_type == "Histogram":
        fig = px.histogram(data, x=column, title=f"Histogram of {column}")
    elif chart_type == "Bar":
        fig = px.bar(data, x=data.index, y=column, title=f"Bar Chart of {column}")
    elif chart_type == "Scatter":
        fig = px.scatter(data, x=data.index, y=column, title=f"Scatter Plot of {column}")
    
    # Center the chart using CSS
    st.markdown('<div class="center-plotly">', unsafe_allow_html=True)
    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Display EDA report in main body if triggered
if st.session_state.show_eda_report and isinstance(st.session_state.data, pd.DataFrame):
    st.write("### Exploratory Data Analysis Report")
    with open("eda_report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    html(html_content, height=800, scrolling=True)

# File export (updated section)
if active_data is not None and isinstance(active_data, pd.DataFrame):
    st.sidebar.write("### Export Data")
    export_format = st.sidebar.selectbox("Select format", ["CSV", "Excel"])
    
    if export_format == "CSV":
        csv_buffer = io.StringIO()
        active_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        st.sidebar.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="exported_data.csv",
            mime="text/csv",
            key="download_csv"
        )
    
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            active_data.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
        st.sidebar.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="exported_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )