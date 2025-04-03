import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from deep_translator import GoogleTranslator
from langdetect import detect_langs
import io
import numpy as np
import time
import warnings
import re
from fancyimpute import KNN, IterativeImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import sweetviz as sv
from streamlit.components.v1 import html

# Monkey patches
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items
if not hasattr(pd.Series, 'iteritems'):
    pd.Series.iteritems = pd.Series.items
if not hasattr(pd.Series, 'mad'):
    def mad(self, axis=None, skipna=True, level=None):
        if axis is not None or level is not None:
            raise NotImplementedError("Only default axis and level=None supported.")
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

def process_prompt(prompt, data):
    if data is None:
        return "Please upload a valid file."
    lang_probs = detect_langs(prompt)
    detected_lang = lang_probs[0].lang
    is_english_input = detected_lang == 'en' or lang_probs[0].prob < 0.9
    translated_prompt = prompt if is_english_input else translator.translate(prompt)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.secrets["GEMINI_API_KEY"], temperature=0.3)
    if isinstance(data, pd.DataFrame):
        data_summary = f"Rows: {len(data)}\nColumns: {data.columns.tolist()}\n{data.describe().to_string()}"
    else:
        data_summary = str(data)
    
    chat_history = ""
    if st.session_state.messages:
        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']} ({msg['timestamp']})" 
                                 for msg in st.session_state.messages])
    else:
        chat_history = "No previous conversation available."
    
    prompt_template = ChatPromptTemplate.from_template(st.session_state.custom_prompt)
    chain = prompt_template | llm | RunnablePassthrough()
    response = chain.invoke({"history": chat_history, "data": data_summary, "question": translated_prompt})
    return response.content if is_english_input else translator.translate(response.content, target=detected_lang)

st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Roboto', sans-serif; }
    .stChatMessage { padding: 15px; border-radius: 8px; margin: 10px 0; width: 100%; max-width: 1400px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stChatMessage.user { background-color: #d4edda; margin-left: auto; }
    .stChatMessage.assistant { background-color: #cce5ff; margin-right: auto; }
    .stButton button { background-color: #007bff; color: white; border-radius: 6px; padding: 10px 20px; }
    .stButton button:hover { background-color: #0056b3; }
    .cleaning-feedback { color: #28a745; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #f8f9fa; transition: width 0.3s; }
    .material-icons { font-size: 24px; vertical-align: middle; margin-right: 8px; color: #333; }
    .material-icons:hover { animation: pulse 1s infinite; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
    .eda-card { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .centered-image { display: flex; justify-content: center; align-items: center; margin: 20px 0; }
    .collapsed .sidebar .sidebar-content { width: 60px; overflow: hidden; }
    .collapsed .sidebar .sidebar-content h2, .collapsed .sidebar .sidebar-content h3 { display: none; }
</style>
""", unsafe_allow_html=True)


if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

def toggle_sidebar():
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
    st.markdown(f'<body class="{"collapsed" if st.session_state.sidebar_collapsed else ""}">', unsafe_allow_html=True)


def authenticate(username, password):
    return username == st.secrets["USERNAME"] and password == st.secrets["PASSWORD"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.container():
        st.markdown('<h1><span class="material-icons">lock</span> Login to Data Insights</h1>', unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    st.stop()


def clean_data_with_ai(df, impute_method="auto", outlier_method="hybrid", z_threshold=3.0, iso_contamination=0.1, normalize=False):
    cleaned_df = df.copy()
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=['number']).columns
    progress_bar = st.progress(0)
    stats_dict = {"missing_before": 0, "missing_after": 0, "outliers_removed": 0, "duplicates_removed": 0, "normalized": False}

    stats_dict["missing_before"] = cleaned_df[numeric_cols].isnull().sum().sum()
    if stats_dict["missing_before"] > 0:
        if impute_method == "auto":
            missing_ratio = stats_dict["missing_before"] / (len(cleaned_df) * len(numeric_cols))
            if missing_ratio < 0.1 and len(cleaned_df) < 10000:
                impute_method = "knn"
            elif missing_ratio < 0.3:
                impute_method = "iterative"
            else:
                impute_method = "mean"
        
        if impute_method == "knn":
            imputer = KNN()
            cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])
        elif impute_method == "iterative":
            imputer = IterativeImputer(random_state=42)
            cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])
        elif impute_method == "median":
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        else: 
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        for col in categorical_cols:
            if cleaned_df[col].isnull().sum() > 0:
                mode_val = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                stats_dict["missing_before"] += cleaned_df[col].isnull().sum()
        
        stats_dict["missing_after"] = cleaned_df.isnull().sum().sum()
    progress_bar.progress(25)

    if normalize and len(numeric_cols) > 0:
        cleaned_df[numeric_cols] = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std()
        stats_dict["normalized"] = True
    progress_bar.progress(50)

    initial_rows = len(cleaned_df)
    if outlier_method in ["zscore", "hybrid"]:
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
            mask = cleaned_df[col].index.isin(cleaned_df[col][z_scores < z_threshold].index)
            cleaned_df = cleaned_df.loc[mask]
    
    if outlier_method in ["isolation", "hybrid"] and len(numeric_cols) > 0:
        iso_forest = IsolationForest(contamination=iso_contamination, random_state=42)
        outliers = iso_forest.fit_predict(cleaned_df[numeric_cols].dropna())
        cleaned_df = cleaned_df.iloc[outliers == 1]
    
    stats_dict["outliers_removed"] = initial_rows - len(cleaned_df)
    progress_bar.progress(75)

    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    stats_dict["duplicates_removed"] = initial_rows - len(cleaned_df)
    progress_bar.progress(100)

    return cleaned_df, stats_dict


def load_data(uploaded_file):
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                return pd.read_excel(uploaded_file, sheet_name=None)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None


def generate_visualization(df, chart_type, x_col=None, y_col=None, color=None, marker_size=10, grid=False, custom_title=None):
    plt.figure(figsize=(10, 6))
    
    if chart_type == "Bar":
        if x_col and y_col:
            df.groupby(x_col)[y_col].mean().plot(kind='bar', color=color)
            plt.title(custom_title if custom_title else f"Bar Plot of {y_col} by {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
    elif chart_type == "Histogram":
        if x_col:
            df[x_col].hist(bins=20, color=color)
            plt.title(custom_title if custom_title else f"Histogram of {x_col}")
            plt.xlabel(x_col)
            plt.ylabel("Frequency")
    elif chart_type == "Scatter":
        if x_col and y_col:
            plt.scatter(df[x_col], df[y_col], s=marker_size, color=color)
            plt.title(custom_title if custom_title else f"Scatter Plot of {x_col} vs {y_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
    elif chart_type == "Line":
        if x_col and y_col:
            plt.plot(df[x_col], df[y_col], color=color)
            plt.title(custom_title if custom_title else f"Line Plot of {y_col} over {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
    elif chart_type == "Boxplot":
        if x_col:
            sns.boxplot(data=df, y=x_col, color=color)
            plt.title(custom_title if custom_title else f"Boxplot of {x_col}")
            plt.ylabel(x_col)
    elif chart_type == "Violin":
        if x_col:
            sns.violinplot(data=df, y=x_col, color=color)
            plt.title(custom_title if custom_title else f"Violin Plot of {x_col}")
            plt.ylabel(x_col)
    elif chart_type == "Heatmap":
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap=color if color else "coolwarm")
        plt.title(custom_title if custom_title else "Correlation Heatmap")
    elif chart_type == "Pairplot":
        sns.pairplot(df.select_dtypes(include=['number']), palette=color)
        plt.suptitle(custom_title if custom_title else "Pairplot of Numeric Columns", y=1.02)
    
    if grid and chart_type not in ["Heatmap", "Pairplot"]:
        plt.grid(True)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


st.markdown('<h1><span class="material-icons">bar_chart</span> Data Insights Chatbot</h1>', unsafe_allow_html=True)
st.markdown("**Unlock insights from your data with AI-powered analysis and visualization.**")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "data" not in st.session_state:
    st.session_state.data = None
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = "Chat History:\n{history}\n\nDataset Summary:\n{data}\n\nUser Question:\n{question}"
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="file_uploader")


if uploaded_file is None:
    st.session_state.data = None
    st.session_state.filtered_data = None
    st.session_state.pop("chart_image", None) 
elif uploaded_file and (st.session_state.data is None or uploaded_file.name != st.session_state.get("last_uploaded_file", "")):
    st.session_state.data = load_data(uploaded_file)
    st.session_state.filtered_data = None 
    st.session_state["last_uploaded_file"] = uploaded_file.name 

active_data = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data


with st.sidebar:
    st.markdown('<h3><span class="material-icons">edit</span> Prompt Customization</h3>', unsafe_allow_html=True)
    custom_prompt_input = st.text_area("Edit Prompt", value=st.session_state.custom_prompt, height=100)
    if st.button("Save Prompt", help="Save the custom prompt template"):
        st.session_state.custom_prompt = custom_prompt_input
        st.success("Prompt updated!")

    st.markdown('<h3><span class="material-icons">build</span> Tools</h3>', unsafe_allow_html=True)
    if st.button("Clear Chat", help="Clear all chat messages"):
        st.session_state.messages = []
        st.success("Chat cleared!")
    
    if st.button("Export Chat as PDF", help="Download chat history as PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        for msg in st.session_state.messages:
            style = styles["Heading3"] if msg["role"] == "user" else styles["BodyText"]
            story.append(Paragraph(f"{msg['role'].capitalize()}: {msg['content']} ({msg['timestamp']})", style))
            story.append(Spacer(1, 12))
        doc.build(story)
        st.download_button("Download PDF", buffer.getvalue(), "chat_history.pdf", "application/pdf")

    if st.button("Reset Data", help="Revert to original dataset"):
        st.session_state.filtered_data = None
        st.success("Data reset!")
    
    if st.button("Clear Prompt", help="Reset to default prompt"):
        st.session_state.custom_prompt = "Chat History:\n{history}\n\nDataset Summary:\n{data}\n\nUser Question:\n{question}"
        st.success("Prompt reset to default!")

    st.markdown('<h3><span class="material-icons">search</span> Predefined Queries</h3>', unsafe_allow_html=True)
    if st.button("Show Summary Stats", help="Display summary statistics"):
        if isinstance(active_data, pd.DataFrame):
            st.write(active_data.describe())
    if st.button("Detect Missing Values", help="Show count of missing values"):
        if isinstance(active_data, pd.DataFrame):
            st.write(active_data.isnull().sum())

    if isinstance(active_data, pd.DataFrame):
        st.markdown('<h3><span class="material-icons">insert_chart</span> Visualization</h3>', unsafe_allow_html=True)
        chart_type = st.selectbox("Chart Type", ["Bar", "Histogram", "Scatter", "Line", "Boxplot", "Violin", "Heatmap", "Pairplot"])
        columns = active_data.columns.tolist()
        
        if chart_type in ["Bar", "Scatter", "Line"]:
            x_col = st.selectbox("X-Axis Column", columns, key="x_col")
            y_col = st.selectbox("Y-Axis Column", columns, key="y_col")
        else:
            x_col = st.selectbox("Column", columns, key="x_col") if chart_type not in ["Heatmap", "Pairplot"] else None
            y_col = None
        
        color = st.selectbox("Color Palette", [None, "blue", "green", "red", "purple", "orange", "gray"], help="Select a color for the chart")
        if chart_type == "Scatter":
            marker_size = st.slider("Marker Size", 5, 50, 10, help="Adjust the size of scatter points")
        else:
            marker_size = 10
        grid = st.checkbox("Show Grid", value=False, help="Toggle grid lines on the chart")
        custom_title = st.text_input("Custom Title", "", help="Enter a custom title for the chart (optional)")
        
        if st.button("Show Visualization", help="Display the selected chart with customizations"):
            with st.spinner("Generating visualization..."):
                img_buffer = generate_visualization(active_data, chart_type, x_col, y_col, color, marker_size, grid, custom_title)
                st.session_state["chart_image"] = img_buffer

    if isinstance(active_data, pd.DataFrame):
        st.markdown('<h3><span class="material-icons">analytics</span> EDA Report</h3>', unsafe_allow_html=True)
        if st.button("Show Full Report", help="Generate and download full EDA report"):
            with st.spinner("Generating EDA report..."):
                report = sv.analyze(active_data)
                report.show_html("eda_report.html", open_browser=False)
                with open("eda_report.html", "rb") as f:
                    st.download_button(
                        label="Download EDA Report",
                        data=f.read(),
                        file_name="eda_report.html",
                        mime="text/html",
                        key="download_eda_report",
                        help="Click to download the generated EDA report"
                    )
                st.success("EDA report generated and ready for download!")

    if active_data is not None and isinstance(active_data, pd.DataFrame):
        st.markdown('<h3><span class="material-icons">file_download</span> Export Data</h3>', unsafe_allow_html=True)
        export_format = st.selectbox("Format", ["CSV", "Excel"])
        if export_format == "CSV":
            csv_buffer = io.StringIO()
            active_data.to_csv(csv_buffer, index=False)
            st.download_button("Download CSV", csv_buffer.getvalue(), "exported_data.csv", "text/csv", help="Download data as CSV")
        elif export_format == "Excel":
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                active_data.to_excel(writer, index=False)
            st.download_button("Download Excel", excel_buffer.getvalue(), "exported_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", help="Download data as Excel")


with st.container():
    st.markdown('<h2><span class="material-icons">chat</span> Chat with Your Data</h2>', unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f"{message['content']} *({message['timestamp']})*")
        
        if prompt := st.chat_input("Ask a question about your data"):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
            with st.chat_message("user"):
                st.markdown(f"{prompt} *({timestamp})*")
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    if active_data is None:
                        response = "Please upload a dataset first using the sidebar."
                    else:
                        prompt_lower = prompt.lower()
                        
                        if "top 5 rows" in prompt_lower or "head" in prompt_lower:
                            response = st.write(active_data.head(5))
                            response = "Here are the top 5 rows of the dataset:"
                        elif "summary statistics" in prompt_lower or "describe" in prompt_lower:
                            response = st.write(active_data.describe())
                            response = "Here are the summary statistics of the dataset:"
                        elif "missing values" in prompt_lower:
                            response = st.write(active_data.isnull().sum())
                            response = "Here are the missing value counts per column:"
                        elif "plot" in prompt_lower or "chart" in prompt_lower:
                            chart_match = re.search(r"(bar|histogram|scatter|line|boxplot|violin|heatmap|pairplot)", prompt_lower)
                            chart_type = chart_match.group(0) if chart_match else "Bar"
                            col_match = re.findall(r"(\w+)\s*(?:vs|and|\sand\s)?\s*(\w+)?", prompt_lower)
                            x_col = col_match[0][0] if col_match and col_match[0][0] in active_data.columns else active_data.columns[0]
                            y_col = col_match[0][1] if col_match and len(col_match[0]) > 1 and col_match[0][1] in active_data.columns else None
                            if chart_type in ["Heatmap", "Pairplot"]:
                                x_col, y_col = None, None
                            img_buffer = generate_visualization(active_data, chart_type.capitalize(), x_col, y_col)
                            st.image(img_buffer, use_container_width=True)
                            response = f"Here is a {chart_type} chart for your data:"
                        else:
                            response = process_prompt(prompt, active_data)
                    
                    st.markdown(f"{response} *({timestamp})*")
                    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})

    if "chart_image" in st.session_state and isinstance(active_data, pd.DataFrame):
        st.markdown('<h2><span class="material-icons">insert_chart</span> Visualization</h2>', unsafe_allow_html=True)
        st.markdown('<div class="centered-image">', unsafe_allow_html=True)
        st.image(st.session_state["chart_image"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("Download Chart", st.session_state["chart_image"], "chart.png", "image/png")

    if active_data is not None:
        with st.expander("Data Overview"):
            st.markdown('<h5><span class="material-icons">table_view</span> Data Overview</h5>', unsafe_allow_html=True)
            st.markdown("##### Preview and Stats")
            if isinstance(active_data, dict):
                sheet_name = st.selectbox("Select Sheet", list(active_data.keys()))
                st.write(active_data[sheet_name].head())
            elif isinstance(active_data, pd.DataFrame):
                st.write(active_data.head())
            if isinstance(active_data, pd.DataFrame):
                st.markdown("###### Quick Stats")
                st.write(f"Rows: {len(active_data)}")
                st.write(f"Columns: {len(active_data.columns)}")
                st.write(f"Missing Values: {active_data.isnull().sum().sum()}")
    else:
        with st.expander("Data Overview"):
            st.markdown('<h5><span class="material-icons">table_view</span> Data Overview</h5>', unsafe_allow_html=True)
            st.write("No dataset uploaded. Please upload a CSV or Excel file using the sidebar.")

    if isinstance(active_data, pd.DataFrame):
        with st.expander("Explore and Clean Data"):
            st.markdown('<h5><span class="material-icons">filter_alt</span> Explore and Clean Data</h5>', unsafe_allow_html=True)
            st.markdown("##### Filter and Clean")
            df = active_data.copy()
            
            st.markdown("###### AI-Powered Cleaning")
            impute_method = st.selectbox("Imputation Method", ["auto", "knn", "iterative", "mean", "median"], key="impute_method")
            outlier_method = st.selectbox("Outlier Detection", ["hybrid", "zscore", "isolation"], key="outlier_method")
            z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key="z_threshold")
            iso_contamination = st.slider("Isolation Forest Contamination", 0.01, 0.5, 0.1, key="iso_contamination")
            normalize = st.checkbox("Normalize Numeric Data", value=False, key="normalize")
            
            if st.button("Clean Data with AI", help="Clean data using advanced AI methods"):
                with st.spinner("Cleaning in progress..."):
                    cleaned_df, stats = clean_data_with_ai(
                        df,
                        impute_method=impute_method,
                        outlier_method=outlier_method,
                        z_threshold=z_threshold,
                        iso_contamination=iso_contamination,
                        normalize=normalize
                    )
                    st.session_state.filtered_data = cleaned_df
                    st.success("Data cleaned successfully!")
                    st.markdown(
                        f'<p class="cleaning-feedback">'
                        f'Missing Values Before: {stats["missing_before"]}<br>'
                        f'Missing Values After: {stats["missing_after"]}<br>'
                        f'Outliers Removed: {stats["outliers_removed"]}<br>'
                        f'Duplicates Removed: {stats["duplicates_removed"]}<br>'
                        f'Normalized: {stats["normalized"]}</p>',
                        unsafe_allow_html=True
                    )
            
            filter_columns = st.multiselect("Filter Columns", options=df.columns.tolist(), default=[])
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
                    selected_vals = st.multiselect(f"Filter {col}", options=unique_vals, default=unique_vals, key=f"multiselect_{col}")
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            st.write(f"Filtered Data ({len(filtered_df)} rows):")
            st.write(filtered_df.head())
            if st.button("Use Filtered Data", help="Apply filtered data to analysis"):
                st.session_state.filtered_data = filtered_df
                st.success("Filtered data activated!")

    if isinstance(active_data, pd.DataFrame):
        st.markdown('<h2><span class="material-icons">analytics</span> Correlation Heatmap</h2>', unsafe_allow_html=True)
        st.markdown('<div class="eda-card">', unsafe_allow_html=True)
        if st.checkbox("Show Correlation Heatmap"):
            corr = active_data.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)