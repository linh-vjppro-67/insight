import streamlit as st
import json
import requests
import pdfplumber
import io

# Azure OpenAI API endpoint and key from secrets
endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
api_key = st.secrets["AZURE_OPENAI_API_KEY"]

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_data):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to process the response from the OpenAI API
def process_response(response_data):
    if not response_data:
        st.warning("No response data. Skipping parsing.")
        return None

    try:
        generated_text = response_data['choices'][0]['message']['content']
        extracted_info = json.loads(generated_text)
        return extracted_info
    except (KeyError, IndexError):
        st.error("Unexpected API response format.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error parsing generated text into JSON: {e}")
        return None

# Function to process the uploaded file and call the API
def process_file(file_path, schema_json, prompt_text):
    extracted_text = ""

    # Handle PDF file
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            extracted_text = extract_text_from_pdf(pdf_data)
    else:
        return {
            'statusCode': 400,
            'message': 'Unsupported file type. Only .pdf is supported.'
        }

    if not extracted_text.strip():
        return {
            'statusCode': 400,
            'message': 'No text extracted from the file. Please check the content of the PDF.'
        }

    # Prepare data to send to API
    system_message = {
        "role": "system",
        "content": "You are an AI assistant that helps extract information from resumes (CVs)."
    }

    user_message = {
        "role": "user",
        "content": prompt_text.replace("{extracted_text}", extracted_text)
    }

    data = {
        "messages": [system_message, user_message],
        "max_tokens": 16000,
        "temperature": 1,
        "top_p": 0.25
    }

    # API request to Azure OpenAI
    try:
        response = requests.post(
            endpoint,
            headers={'Content-Type': 'application/json', 'api-key': api_key},
            data=json.dumps(data)
        )
    except requests.RequestException as e:
        return {
            'statusCode': 500,
            'message': f"Error calling OpenAI API: {e}"
        }

    if response.status_code == 200:
        response_data = response.json()
        extracted_info = process_response(response_data)

        if extracted_info is None:
            return {
                'statusCode': 500,
                'message': 'Error processing OpenAI response'
            }

        return {
            'statusCode': 200,
            'data': extracted_info
        }
    else:
        return {
            'statusCode': response.status_code,
            'message': 'Error calling OpenAI API',
            'error': response.text
        }

# Streamlit UI
def app():
    st.title("Resume Insights and Career Recommendations")

    # Section 1: Upload File and Schema Setup
    st.header("Section 1: Upload Resume PDF and Enter Prompt")

    st.sidebar.header("File Input")
    schema_path = './schema.json'  # Path to schema file
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])

    # Load the schema JSON
    try:
        with open(schema_path, 'r') as file:
            schema_json = json.load(file)
            schema_string = json.dumps(schema_json, indent=4)  # Convert schema to a pretty JSON string
    except FileNotFoundError:
        st.error("Schema file not found.")
        return
    except json.JSONDecodeError as e:
        st.error(f"Error reading schema JSON: {e}")
        return

    # Default Prompt
    default_prompt = """
        Section 1: Extract Information
        You are an AI assistant that helps extract information from resumes (CVs).
        Keep the language of the CV unchanged.
        Use the following schema to structure the extracted information: {schema_string}
        Only return valid JSON with the extracted information, without any additional explanations.
        Export object format to store json file.
        List all skills.
        Please list all positions held at the same company along with their corresponding time periods, company name, and detailed duties and responsibilities for each role. If the same position is held at different times or in different teams within the same company, include each occurrence separately with its unique time period and team information. Ensure that all distinct roles, teams, and time periods are captured in a **separate array item** for each specific instance.
        Remove special characters to properly format it as an object before saving it to a JSON file.
        Remove ```json, remove $schema
        Text extracted from PDF (with coordinates). Keep the language of the CV unchanged:
        Analyze file content: {extracted_text}

        Section 2: Analyze Candidate Profile
        Analyze the candidate's CV data and provide insights based on the following criteria:
        1. Work Experience Analysis:
           - For each company listed, extract and summarize the job title, tenure, and level of expertise 
             (categorized as beginner, intermediate, or expert) in relevant fields. 
           - Organize this information by company in a structured format.
        
        2. Job Trends and Stability:
           - Analyze the candidate’s career progression by evaluating the time spent in each role. 
           - Identify patterns such as frequent job changes, promotions, extended tenures, or gaps between roles. 
           - Assess the likelihood of long-term job stability versus a tendency for frequent transitions.
        
        3. Suggested Job Titles:
           - Based on the candidate’s skills, years of experience, and educational background, recommend potential 
             job titles or career paths. 
           - Ensure suggestions align with their demonstrated expertise, industry trends, and career growth opportunities.
        
        4. Job Resignation Prediction:
           - Predict the likelihood of the candidate changing jobs at this time (output as Yes or No) based on factors such as:
             * Duration in the current role relative to past roles.
             * Alignment of current role with skills, career goals, and industry trends.
             * Patterns of frequent transitions, gaps, or promotions in job history.
             * Indicators of dissatisfaction, stagnation, or misalignment with expertise.
           - Predict the exact timeframe (e.g., "3 months," "12 months," or "2 years") in which the candidate is likely to change jobs.
    """

    # Prompt Input Area
    prompt_text = st.text_area("Prompt Editor", default_prompt, height=600)

    # Add a "Generate" button
    if st.button("Generate"):
        if uploaded_file is not None:
            file_path = uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Call process_file
            result = process_file(file_path, None, prompt_text)

            if result['statusCode'] == 200:
                # Hiển thị toàn bộ result JSON
                st.header("Result JSON")
                st.json(result)

            else:
                # Handle lỗi
                st.error(result['message'])
                if 'error' in result:
                    st.error(f"Details: {result['error']}")
        else:
            st.warning("Please upload a PDF file before generating.")

if __name__ == "__main__":
    app()
