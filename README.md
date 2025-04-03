# Chatbot-using-custom-dataset

 
*A powerful AI-driven Streamlit application for data analysis, visualization, and exploration.*

## Overview

The **Data Insights Chatbot** is an interactive web application built with Streamlit, designed to empower users to upload, clean, analyze, and visualize tabular data (CSV or Excel) effortlessly. Leveraging AI-powered tools like Google’s Gemini model, it provides a conversational interface to query datasets, generate visualizations, and perform advanced data cleaning and exploratory data analysis (EDA). Whether you're a data analyst, researcher, or enthusiast, this tool simplifies the process of unlocking insights from your data.

### Key Features
- **Data Upload**: Supports CSV and Excel files with multi-sheet Excel handling.
- **AI-Powered Chat**: Ask questions about your data in natural language and get intelligent responses.
- **Data Cleaning**: Advanced AI-driven cleaning with options for imputation, outlier detection, and normalization.
- **Visualization**: Generate customizable charts (Bar, Histogram, Scatter, Line, Boxplot, Violin, Heatmap, Pairplot) with Matplotlib, Seaborn, and Plotly.
- **EDA Reports**: Create comprehensive exploratory data analysis reports using Sweetviz.
- **Authentication**: Secure login system to protect access.
- **Export Options**: Download chat history, charts, and cleaned datasets in various formats.
- **Customizable UI**: Material Icons, custom CSS, and a collapsible sidebar for an enhanced user experience.

---
## Streamlit Deployed Application link

Application:[Chatbot](https://langchat-2.streamlit.app/)

```bash
Username - sbsra
Password - blueyous
```

## Installation

### Prerequisites
- Python 3.8 or higher
- A Google API key for the Gemini model (stored in Streamlit secrets)
- Basic familiarity with running Python applications

### Dependencies
The project relies on several Python libraries. Install them using the following command:

```bash
pip install -r requirements.txt
```

```bash
git clone https://github.com/yourusername/Workcohol-Chatbot-using-custom-dataset.git
cd Workcohol-Chatbot-using-custom-dataset
```
### ADD-ONS
Create a toml file for your own usage purpose to add these details.(users can customize their own username and password and add their own api key)

```bash
GEMINI_API_KEY = "your-google-api-key"
USERNAME = "username"
PASSWORD = "password"
```
### Execution
To start streamlit server

```bash
streamlit run Chatbot.py
```

## Usage

### 1. Login
Upon launching, you’ll be prompted to enter a username and password (configured in secrets.toml).
Successful authentication grants access to the main interface.

### 2. Upload Data
Use the sidebar’s File Uploader to upload a .csv or .xlsx file.
For Excel files with multiple sheets, select a sheet from the "Data Overview" expander.

### 3. Chat with Your Data
Enter questions in the chat input (e.g., "Show top 5 rows," "Plot a bar chart of sales by region").
The chatbot supports:
Predefined queries (e.g., "summary statistics," "missing values").
Visualization generation based on keywords like "plot" or "chart."
General questions processed by the Gemini AI model.

### 4. Data Cleaning
Open the "Explore and Clean Data" expander:
AI-Powered Cleaning: Choose imputation methods (auto, KNN, iterative, mean, median), outlier detection (hybrid, z-score, isolation forest), and normalization.
Filtering: Apply filters to numeric or categorical columns using sliders or multiselect options.
Results are saved to filtered_data for further analysis.

### 5. Visualization
In the sidebar under Visualization:
Select a chart type and configure axes, colors, grid, and titles.
Click "Show Visualization" to display and download the chart.
Charts are also generated via chat queries (e.g., "scatter plot of age vs salary").

### 6. EDA and Exports
Generate a full EDA report with Sweetviz under EDA Report.
Export options:
Chat history as PDF.
Cleaned data as CSV or Excel.

### 7. Customization
Edit the prompt template in the sidebar under Prompt Customization to tailor AI responses.
Use tools like "Clear Chat," "Reset Data," or "Export Chat as PDF" for workflow management.

## Project Structure

```bash
Workcohol-Chatbot-using-custom-dataset/
├── chatbot.py              # Main Streamlit application
├── .streamlit/
│   └── secrets.toml    # Configuration file for API keys and credentials
└── README.md           # Project documentation
```

## Key Functions

- **load_data(uploaded_file)**: Loads CSV or Excel files into a DataFrame.
- **clean_data_with_ai(df, ...)**: Performs AI-driven cleaning with progress feedback.
- **generate_visualization(df, chart_type, ...)**: Creates and returns chart images.
- **process_prompt(prompt, data)**: Handles chat queries with AI and chat history.

## Examples

### Chat Queries
- **"What are the top 5 rows?"**
- **"Show summary statistics."**
- **"Plot a bar chart of revenue by category."**
- **"How many missing values are in the dataset?"**

### Cleaning Example
- **Upload a dataset with missing values and outliers.**
- **Select "auto" imputation and "hybrid" outlier detection.**
- **Apply cleaning and review the feedback (e.g., "Missing Values Before: 50, After: 0").**

### Visualization Example
- **Choose "Scatter" chart type.**
- **Set X-axis to "Age," Y-axis to "Salary," color to "blue," and marker size to 20.**

## Technical Details

### Libraries Used
- **Streamlit: Web app framework.**
- **Pandas: Data manipulation.**
- **Matplotlib/Seaborn/Plotly: Visualization.**
- **LangChain/Google Gemini: AI chatbot functionality.**
- **FancyImpute/Sklearn: Data cleaning and outlier detection.**
- **Sweetviz: EDA reports.**
- **ReportLab: PDF generation.**

### AI Integration
- **The Gemini 1.5 Pro model processes natural language queries, leveraging chat history and dataset summaries.**
- **Language detection and translation ensure multilingual support.**

### UI Enhancements
- **Custom CSS with Material Icons for a modern look.**
- **Collapsible sidebar (not fully implemented in current code but initialized).**

## Contributing

### Contributions are welcome! To contribute:

- **Fork the repository.**
- **Create a feature branch (git checkout -b feature/new-feature).**
- **Commit changes (git commit -m "Add new feature").**
- **Push to the branch (git push origin feature/new-feature).**
- **Open a pull request.**

Built with  by MA.A.Harinesh | Last Updated: April 2025