import streamlit as st
import pandas as pd
import re
import json
import io
from together import Together

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
</style>
""", unsafe_allow_html=True)

# Load examples from JSON file
@st.cache_data
def load_examples():
    try:
        with open('examples.json', 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return examples
    except Exception:
        return []

# Format examples string
def format_examples(examples):
    examples_str = ""
    for i, example in enumerate(examples, 1):
        examples_str += f"### Example {i}\n**Input CSV:**\n{example['input']}\n\n**Expected Output:**\n{example['output']}\n\n"
    return examples_str

# Get Together client
def get_together_client(api_key):
    if not api_key:
        st.warning("Please enter a Together API key.")
        return None
    return Together(api_key=api_key)

# Chain-of-thought prompt
def get_cot_prompt(examples_str, input_data):
    return f"""**Family Data Processing - Chain of Thought**

# Chain-of-Thought Family Relationship Data Conversion Prompt

## Examples
{examples_str}

...

**CSV Data:**
{input_data}

**Required Output Format:**
## Reasoning Steps
[Detailed step-by-step analysis in markdown]

## Final Output Table
| Individual ID | Name | Relation | Family Group ID | Actions |
|---------------|------|----------|-----------------|---------|
[...table data...]"""

# CSV text parser
def parse_csv_text(text_input):
    try:
        return pd.read_csv(io.StringIO(text_input))
    except Exception:
        # Fallback parsing
        data = []
        columns = ["Caste", "Subcaste", "Given name", "Surname", "Relation", "Gender", "Place", "Date"]
        tokens = text_input.split()
        for i in range(0, len(tokens), 8):
            row = tokens[i:i+8]
            if len(row) < 8:
                row += [""] * (8 - len(row))
            data.append(row)
        return pd.DataFrame(data, columns=columns)

# LLM response parser
def parse_response(result):
    reasoning = []
    table_data = []
    current_section = None

    for line in result.splitlines():
        if line.strip().startswith("## Reasoning Steps"):
            current_section = "reasoning"
            continue
        elif line.strip().startswith("## Final Output Table"):
            current_section = "table"
            continue
        if current_section == "reasoning" and line.strip():
            reasoning.append(line.strip())
        elif current_section == "table" and line.startswith("|") and "---" not in line:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == 5:
                table_data.append(cells)

    return reasoning, table_data

# Main app function
def main():
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    st.markdown('<h1 class="title">Family Relationship Analyzer</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.text_input("Together API Key", type="password", key="api_key")
        st.header("Model Settings")
        model = st.selectbox("Select Model", [
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "deepseek-ai/DeepSeek-V3"
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max Tokens", 500, 4000, 2000, 100)

    client = get_together_client(st.session_state.api_key)
    examples = load_examples()
    examples_str = format_examples(examples)

    tab1, tab2 = st.tabs(["CSV File Upload", "Text Input"])

    with tab1:
        st.markdown('<h2 class="subheader">Upload Family Data CSV</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            try:
                csv_data = pd.read_csv(uploaded_file)
                st.dataframe(csv_data)

                if st.button("Process CSV File"):
                    if not client:
                        return
                    with st.spinner("Processing..."):
                        csv_str = csv_data.to_csv(index=False)
                        full_prompt = get_cot_prompt(examples_str, csv_str)

                        try:
                            response = client.chat.completions.create(
                                model=model,
                                messages=[{
                                    "role": "user",
                                    "content": [{"type": "text", "text": full_prompt}]
                                }],
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            result = response.choices[0].message.content
                            reasoning, table_data = parse_response(result)

                            st.markdown('<div class="output-container">', unsafe_allow_html=True)
                            st.markdown("### Reasoning Steps")
                            for r in reasoning:
                                st.markdown(r)

                            st.markdown("### Final Output Table")
                            if table_data:
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                st.table(df)
                            else:
                                st.warning("No structured table generated.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"LLM Error: {e}")
            except Exception as e:
                st.error(f"CSV Error: {e}")

    with tab2:
        st.markdown('<h2 class="subheader">Enter Family Data as Text</h2>', unsafe_allow_html=True)
        text_input = st.text_area("Enter CSV-style or free-text family data", height=200)
        if text_input and st.button("Process Text Input"):
            if not client:
                return
            with st.spinner("Processing..."):
                try:
                    df = parse_csv_text(text_input)
                    st.dataframe(df)
                    csv_str = df.to_csv(index=False)
                    prompt = get_cot_prompt(examples_str, csv_str)

                    response = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    result = response.choices[0].message.content
                    reasoning, table_data = parse_response(result)

                    st.markdown('<div class="output-container">', unsafe_allow_html=True)
                    st.markdown("### Reasoning Steps")
                    for r in reasoning:
                        st.markdown(r)

                    st.markdown("### Final Output Table")
                    if table_data:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        st.table(df)
                    else:
                        st.warning("No structured table generated.")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
