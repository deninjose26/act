import streamlit as st
import pandas as pd
import re
import json
import io
import os
import requests
import importlib
from together import Together

# Check for required dependencies
def check_dependencies():
    missing_deps = []
    
    # Check for openpyxl (required for Excel reading/writing)
    try:
        importlib.import_module('openpyxl')
    except ImportError:
        missing_deps.append('openpyxl')
    
    return missing_deps

# Page configuration
st.set_page_config(page_title="Family Relationship Analyzer", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .output-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .reasoning-section {
        margin-bottom: 20px;
    }
    .dependency-warning {
        background-color: #ffebee;
        color: #c62828;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .dependency-command {
        background-color: #f5f5f5;
        padding: 8px;
        font-family: monospace;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get("TOGETHER_API_KEY", "")

# Check dependencies
missing_dependencies = check_dependencies()
excel_support_available = 'openpyxl' not in missing_dependencies

# Call Together API directly
def call_together_api(api_key, model, prompt, temperature=0.3, max_tokens=2000):
    if not api_key:
        st.error("API key is required")
        return None
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

# Load examples from JSON file
@st.cache_data
def load_examples():
    try:
        with open('examples.json', 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return examples
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning("Warning: Could not load examples.json. Proceeding without examples.")
        return []

# Format examples string
def format_examples(examples):
    examples_str = ""
    for i, example in enumerate(examples, 1):
        examples_str += f"### Example {i}\n**Input CSV:**\n{example['input']}\n\n**Expected Output:**\n{example['output']}\n\n"
    return examples_str

# Define the Chain-of-Thought prompt with placeholders for examples and input data
def get_cot_prompt(examples_str, input_data):
    return """**Family Data Processing - Chain of Thought**

I'll help you transform this prompt into a chain-of-thought approach that guides the reasoning process step by step for converting family relationship data from CSV to a structured hierarchical table. I'll break down the thinking process explicitly.

# Chain-of-Thought Family Relationship Data Conversion Prompt

## Examples
{examples}

## Step 1: Understand the Input Data
First, carefully analyze the CSV data to identify:
- All unique individuals mentioned
- The explicit relationships between them
- The generations they belong to
- Any missing relationships that need placeholder entries

Think: "Who are all the people mentioned? What exact relationships are stated in the data? Which generations do they belong to?"

## Step 2: Create Individual IDs
Assign a unique numeric identifier to each person mentioned in the CSV data, starting with 1.
- Include placeholder individuals (UK#) needed to explain relationships
- Resolve duplicates with numbered suffixes

Think: "Who needs a unique ID? Are there any implied individuals that need placeholder IDs?"

## Step 3: Construct Names
Combine Given name + Surname (when provided) to create the Name field.
- For placeholder individuals, use UK1, UK2, etc.
- For duplicates, add numbered suffixes like (1), (2)

Think: "What's the full name of each person? Do I need any placeholder names?"

## Step 4: Map Relationships
Directly transcribe the relationship information from the CSV's "Relation" field.
- For placeholder individuals, infer their relationship based on context
- Example: If there's a "भतीजा" (nephew), create a placeholder for their parent with relation "ओमप्रकाश का भाई" (brother of ओमप्रकाश)

Think: "What is each person's relationship to others? What relationships must I infer for placeholder individuals?"

## Step 5: Determine Generational Structure
Analyze the family structure to identify:
- Grandparents/great-grandparents (1P)
- Parents/uncles/aunts (1C or 2P)
- Nuclear family members (2P, 2C)
- Children/grandchildren (2C, 3C)

Think: "What's the oldest generation present? How many generations exist? Who belongs to each generation?"

## Step 6: Assign Family Group IDs
Apply the hierarchical coding system:
- Format: [Generation Number][P/C]
- P = Parent role, C = Child role
- Individuals with dual roles get comma-separated IDs (e.g., 1C,2P)

Think: "What role does each person play in the family structure? Who serves dual roles across generations?"

## Step 7: Validate Relationships
Check for consistency in the family structure:
- Every child should have at least one parent (real or placeholder)
- Relationships like भतीजा (nephew) require an uncle/aunt relationship
- Cousins require siblings in the parent generation

Think: "Are all relationships logically consistent? Are there any relationships that require additional placeholder individuals?"

## Step 8: Final Table Construction
Organize all information into the required table format:
- Individual ID | Name | Relation | Family Group ID | Actions

Think: "Is my table complete? Have I correctly represented all relationships and followed all the specified rules?"

## Example Thought Process
For the input:
```
Caste,Subcaste,Given name,Surname,Relation,Gender,Place,Date
साहू,नगरिया,सुरेन्द्र,-,ओमप्रकाश के भतीजा,Male,रिछरा फाटक,२०५१
```

Reasoning steps:
1. I see सुरेन्द्र is "ओमप्रकाश के भतीजा" (nephew of ओमप्रकाश)
2. A nephew requires an uncle/aunt relationship, but no parent for सुरेन्द्र is mentioned
3. I need to create a placeholder (UK1) for an unmentioned sibling of ओमप्रकाश
4. सुरेन्द्र would be in the child generation (2C) relative to ओमप्रकाश's generation
5. The placeholder UK1 would be in the same generation as ओमप्रकाश (1C) and also a parent (2P)
6. Final table should include both सुरेन्द्र and the placeholder UK1

This chain-of-thought approach ensures methodical reasoning through complex family relationships and accurate application of the hierarchical coding system.
**CSV Data:**
{input_data}

**Required Output Format:**
## Reasoning Steps
[Detailed step-by-step analysis in markdown]

## Final Output Table
| Individual ID | Name | Relation | Family Group ID | Actions |
|---------------|------|----------|-----------------|---------|
[...table data...]""".format(examples=examples_str, input_data=input_data)

# Function to parse CSV text input
def parse_csv_text(text_input):
    # Clean up and normalize input
    text_input = text_input.strip()
    lines = text_input.split('\n')
    
    # Try to detect if the text is already structured as CSV
    if ',' in lines[0] and len(lines) > 1:
        # It may already be a CSV format
        try:
            return pd.read_csv(io.StringIO(text_input))
        except Exception:
            pass
    
    # If not properly formatted as CSV, try to parse as space-separated values
    # First, assume the standard columns
    columns = ["Caste", "Subcaste", "Given name", "Surname", "Relation", "Gender", "Place", "Date"]
    
    # Extract data using a more robust approach
    data = []
    current_line = ""
    words = text_input.split()
    
    # Group words into 8-column rows
    row = []
    for word in words:
        row.append(word)
        if len(row) == 8:
            data.append(row)
            row = []
    
    # Add any remaining data (partial row)
    if row:
        while len(row) < 8:
            row.append("")
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

# Function to parse the LLM response
def parse_response(result):
    reasoning = []
    table_data = []
    current_section = None
    
    for line in result.split('\n'):
        if re.match(r'^##\s*Reasoning\s*Steps', line, re.I):
            current_section = 'reasoning'
            continue
        elif re.match(r'^##\s*Final\s*Output\s*Table', line, re.I):
            current_section = 'table'
            continue
            
        if current_section == 'reasoning':
            if line.strip() and not line.startswith('---'):
                reasoning.append(line.strip())
        elif current_section == 'table' and line.startswith('|'):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == 5 and not all('---' in cell for cell in cells):
                table_data.append(cells)
    
    return reasoning, table_data

# Function to read Excel file if openpyxl is available
def read_excel_file(file, sheet_name=None):
    if not excel_support_available:
        st.error("Cannot read Excel file: missing openpyxl dependency")
        st.markdown(
            '<div class="dependency-warning">'
            'Excel support requires the openpyxl package. Please install it using:<br>'
            '<span class="dependency-command">pip install openpyxl</span><br>'
            'Then restart the application.'
            '</div>',
            unsafe_allow_html=True
        )
        return None
    
    try:
        if sheet_name:
            return pd.read_excel(file, sheet_name=sheet_name)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

# Main app function
def main():
    st.markdown('<h1 class="title">Family Relationship Analyzer</h1>', unsafe_allow_html=True)
    
    # Display dependency warnings at the top if needed
    if missing_dependencies:
        st.markdown(
            '<div class="dependency-warning">'
            f'Missing dependencies: {", ".join(missing_dependencies)}<br>'
            'To enable full functionality, please install the missing packages:<br>'
            f'<span class="dependency-command">pip install {" ".join(missing_dependencies)}</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("API Settings")
        api_key = st.text_input(
            "Together API Key", 
            value=st.session_state.api_key,
            type="password",
            help="Enter your Together API key"
        )
        st.session_state.api_key = api_key
        
        st.header("Model Settings")
        model = st.selectbox(
            "Select Model", 
            ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "deepseek-ai/deepseek-coder-6.7b-instruct"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max Tokens", 500, 4000, 2000, 100)
    
    examples = load_examples()
    examples_str = format_examples(examples)
    
    # Create tabs for different input methods
    tabs = []
    if excel_support_available:
        tabs = st.tabs(["Excel File Upload", "CSV File Upload", "Text Input"])
    else:
        tabs = st.tabs(["CSV File Upload", "Text Input"])
    
    # Excel tab - only show if openpyxl is available
    if excel_support_available:
        with tabs[0]:
            st.markdown('<h2 class="subheader">Upload Family Data Excel File</h2>', unsafe_allow_html=True)
            uploaded_excel = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], key="excel_uploader")
            
            if uploaded_excel is not None:
                try:
                    # Check if the file contains multiple sheets
                    xl = pd.ExcelFile(uploaded_excel)
                    sheet_names = xl.sheet_names
                    
                    if len(sheet_names) > 1:
                        selected_sheet = st.selectbox("Select a sheet:", sheet_names)
                        excel_data = read_excel_file(uploaded_excel, sheet_name=selected_sheet)
                    else:
                        excel_data = read_excel_file(uploaded_excel)
                    
                    if excel_data is not None:
                        st.success("Excel file successfully loaded!")
                        st.dataframe(excel_data)
                        
                        if st.button("Process Excel File", key="process_excel"):
                            if not st.session_state.api_key:
                                st.error("Please enter a Together API key in the sidebar")
                            else:
                                with st.spinner("Processing family relationships..."):
                                    try:
                                        # Convert to CSV string for processing
                                        csv_string = excel_data.to_csv(index=False)
                                        
                                        # Create full prompt
                                        full_prompt = get_cot_prompt(examples_str, csv_string)
                                        
                                        # Call API directly
                                        result = call_together_api(
                                            st.session_state.api_key,
                                            model,
                                            full_prompt,
                                            temperature,
                                            max_tokens
                                        )
                                        
                                        if result:
                                            # Parse response
                                            reasoning, table_data = parse_response(result)
                                            
                                            # Display results
                                            st.markdown('<div class="output-container">', unsafe_allow_html=True)
                                            
                                            st.markdown("### Reasoning Steps")
                                            st.markdown('<div class="reasoning-section">', unsafe_allow_html=True)
                                            for line in reasoning:
                                                st.markdown(line)
                                            st.markdown('</div>', unsafe_allow_html=True)
                                            
                                            st.markdown("### Final Output Table")
                                            if len(table_data) > 1:  # Check if we have data beyond the header
                                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                                st.table(df)
                                                
                                                # Add a download button for the processed data as CSV
                                                csv_buffer = io.StringIO()
                                                df.to_csv(csv_buffer, index=False)
                                                st.download_button(
                                                    label="Download processed data as CSV",
                                                    data=csv_buffer.getvalue(),
                                                    file_name="processed_family_data.csv",
                                                    mime="text/csv"
                                                )
                                                
                                                # Also export to Excel if openpyxl is available
                                                if excel_support_available:
                                                    try:
                                                        excel_buffer = io.BytesIO()
                                                        df.to_excel(excel_buffer, index=False)
                                                        excel_buffer.seek(0)  # Reset the buffer position
                                                        
                                                        st.download_button(
                                                            label="Download processed data as Excel",
                                                            data=excel_buffer.getvalue(),
                                                            file_name="processed_family_data.xlsx",
                                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                                        )
                                                    except Exception as e:
                                                        st.warning(f"Excel export error: {str(e)}")
                                            else:
                                                st.warning("No table data was generated")
                                            
                                            st.markdown('</div>', unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing Excel file: {str(e)}")
    
    # CSV tab
    with tabs[0 if not excel_support_available else 1]:
        st.markdown('<h2 class="subheader">Upload Family Data CSV</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
        
        if uploaded_file is not None:
            try:
                csv_data = pd.read_csv(uploaded_file, encoding='utf-8')
                st.success("CSV file successfully loaded!")
                st.dataframe(csv_data)
                
                if st.button("Process CSV File", key="process_csv"):
                    if not st.session_state.api_key:
                        st.error("Please enter a Together API key in the sidebar")
                    else:
                        with st.spinner("Processing family relationships..."):
                            try:
                                # Convert to CSV string
                                csv_string = csv_data.to_csv(index=False)
                                
                                # Create full prompt
                                full_prompt = get_cot_prompt(examples_str, csv_string)
                                
                                # Call API directly
                                result = call_together_api(
                                    st.session_state.api_key,
                                    model,
                                    full_prompt,
                                    temperature,
                                    max_tokens
                                )
                                
                                if result:
                                    # Parse response
                                    reasoning, table_data = parse_response(result)
                                    
                                    # Display results
                                    st.markdown('<div class="output-container">', unsafe_allow_html=True)
                                    
                                    st.markdown("### Reasoning Steps")
                                    st.markdown('<div class="reasoning-section">', unsafe_allow_html=True)
                                    for line in reasoning:
                                        st.markdown(line)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    st.markdown("### Final Output Table")
                                    if len(table_data) > 1:  # Check if we have data beyond the header
                                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                        st.table(df)
                                        
                                        # Add download button for CSV
                                        csv_buffer = io.StringIO()
                                        df.to_csv(csv_buffer, index=False)
                                        st.download_button(
                                            label="Download processed data as CSV",
                                            data=csv_buffer.getvalue(),
                                            file_name="processed_family_data.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Try to export as Excel if openpyxl is available
                                        if excel_support_available:
                                            try:
                                                excel_buffer = io.BytesIO()
                                                df.to_excel(excel_buffer, index=False)
                                                excel_buffer.seek(0)  # Reset the buffer position
                                                
                                                st.download_button(
                                                    label="Download processed data as Excel",
                                                    data=excel_buffer.getvalue(),
                                                    file_name="processed_family_data.xlsx",
                                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                                )
                                            except Exception as e:
                                                st.warning(f"Excel export error: {str(e)}")
                                    else:
                                        st.warning("No table data was generated")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"API Error: {str(e)}")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # Text input tab
    with tabs[1 if not excel_support_available else 2]:
        st.markdown('<h2 class="subheader">Enter Family Data as Text</h2>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Enter family relationship data (in either CSV format or as text)",
            height=200,
            help="You can paste CSV data or unstructured text with family relationships"
        )
        
        if text_input:
            if st.button("Process Text Input", key="process_text"):
                if not st.session_state.api_key:
                    st.error("Please enter a Together API key in the sidebar")
                else:
                    with st.spinner("Processing family relationships..."):
                        try:
                            # Try to parse the text input as CSV
                            csv_data = parse_csv_text(text_input)
                            st.success("Text data successfully parsed!")
                            st.dataframe(csv_data)
                            
                            # Convert to CSV string
                            csv_string = csv_data.to_csv(index=False)
                            
                            # Create full prompt
                            full_prompt = get_cot_prompt(examples_str, csv_string)
                            
                            # Call API directly
                            result = call_together_api(
                                st.session_state.api_key,
                                model,
                                full_prompt,
                                temperature,
                                max_tokens
                            )
                            
                            if result:
                                # Parse response
                                reasoning, table_data = parse_response(result)
                                
                                # Display results
                                st.markdown('<div class="output-container">', unsafe_allow_html=True)
                                
                                st.markdown("### Reasoning Steps")
                                st.markdown('<div class="reasoning-section">', unsafe_allow_html=True)
                                for line in reasoning:
                                    st.markdown(line)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.markdown("### Final Output Table")
                                if len(table_data) > 1:  # Check if we have data beyond the header
                                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                    st.table(df)
                                    
                                    # Add download button for CSV
                                    csv_buffer = io.StringIO()
                                    df.to_csv(csv_buffer, index=False)
                                    st.download_button(
                                        label="Download processed data as CSV",
                                        data=csv_buffer.getvalue(),
                                        file_name="processed_family_data.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Try to export as Excel if openpyxl is available
                                    if excel_support_available:
                                        try:
                                            excel_buffer = io.BytesIO()
                                            df.to_excel(excel_buffer, index=False)
                                            excel_buffer.seek(0)  # Reset the buffer position
                                            
                                            st.download_button(
                                                label="Download processed data as Excel",
                                                data=excel_buffer.getvalue(),
                                                file_name="processed_family_data.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                            )
                                        except Exception as e:
                                            st.warning(f"Excel export error: {str(e)}")
                                else:
                                    st.warning("No table data was generated")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
