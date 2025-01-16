import streamlit as st
import json
import requests
import pdfplumber
import io

# Azure OpenAI API endpoint and key from secrets
endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
api_key = st.secrets["AZURE_OPENAI_API_KEY"]

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf_plumber(pdf_data):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
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
def process_file(file_path, schema_path):
    # Read schema from JSON file
    try:
        with open(schema_path, 'r') as file:
            schema_json = json.load(file)
    except FileNotFoundError:
        return {'statusCode': 500, 'message': "Schema file not found."}
    except json.JSONDecodeError as e:
        return {'statusCode': 500, 'message': f"Error reading schema JSON: {e}"}

    extracted_text = ""

    # Handle PDF file
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            extracted_text = extract_text_from_pdf_plumber(pdf_data)
    else:
        return {
            'statusCode': 400,
            'message': 'Unsupported file type. Only .pdf is supported.'
        }

    if not extracted_text.strip():
        return {
            'statusCode': 400,
            'message': 'No text extracted from the file.'
        }

    # Prepare data to send to API
    system_message_content = """
        You are an AI assistant that helps extract information from resumes (CVs).
        Keep the language of the CV unchanged.
    """
    system_message = {
        "role": "system",
        "content": system_message_content
    }

    user_message = {
        "role": "user",
        "content": f"""
        Use the following schema to structure the extracted information: {json.dumps(schema_json)}
        Only return valid JSON with the extracted information, without any additional explanations.
        Export object format to store json file.
        List all skills.
        Please list all positions held at the same company along with their corresponding time periods, company name, and detailed duties and responsibilities for each role. If the same position is held at different times or in different teams within the same company, include each occurrence separately with its unique time period and team information. Ensure that all distinct roles, teams, and time periods are captured in a **separate array item** for each specific instance.
        Remove special characters to properly format it as an object before saving it to a JSON file.
        Remove ```json, remove $schema
        Text extracted from PDF (with coordinates). Keep the language of the CV unchanged:
        Analyze file content: {extracted_text}
 
        After extracting the basic candidate information, perform the following analysis:
            1. Work experience in each company:
            - Define the level of commitment by analyzing the duration and responsibilities within each company.
            2. Work experience in each job title:
            - Define the career trend by analyzing transitions and movement between job titles over time.
            3. Responsibilities and outcomes in each job title:
            - Highlight specific responsibilities and outcomes associated with each job title, and assess how well outcomes align with responsibilities.
            4. Job relocation trends:
            - Compare the extracted work location from the CV with the candidate’s basic location to identify potential job relocation trends (relocation.trends).
            5. Suitability for different job types:
            - Determine whether the candidate is more suited for local, remote, or international work based on their experiences and locations.
            6. Job titles and level of expertise:
            - Identify different job titles and assess the level of expertise in various fields (beginner, intermediate, expert).
            7. Job trends and stability:
            - Analyze job trends by examining the time spent in each job title to assess the likelihood of long-term job stability versus frequent changes.
            8. Career progression:
            - Explore the career progression from entry-level positions to more senior job titles (e.g., Software Developer -> Team Leader -> Manager).
            9. Gaps between jobs:
            - Detect gaps between jobs to understand the reasons behind these gaps (e.g., education, rest, or other factors).
            10. Suggested job titles:
                - Based on the candidate’s skills, years of experience, and educational background, suggest potential job titles they could pursue.
            11. Career growth potential:
                - Predict the career growth potential by analyzing time spent in each job title and career development trends.
            12. Missing skills and improvements:
                - Identify missing skills compared to the FSoft skill taxonomy and suggest additional skills the candidate should learn.
            13. Optional – Publications Evaluation:
                - Evaluate any publications mentioned in the CV to assess the prestige and impact of the conferences where papers were published.
            14. Job resignation prediction:
                - Analyze job history (role durations, gaps, transitions) to predict the likelihood of resignation. Consider patterns like frequent changes, alignment with skills, and relocation trends to provide retention insights.
 
            Return a JSON object with the following keys:
            - `basic_info`: Basic candidate information.
            - `insights`: Analysis results based on the points listed above.
            - `recommendations`: Suggested career moves, skill improvements, or other insights.
        """
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

    st.sidebar.header("File Input")
    schema_path = './schema.json'  # Path to schema file
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])

    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Call process_file
        result = process_file(file_path, schema_path)

        if result['statusCode'] == 200:
            data = result['data']

            st.header("Basic Candidate Information")
            st.json(data.get('basic_info', "No basic information found."))

            st.header("Insights")
            st.json(data.get('insights', "No insights found."))

            st.header("Recommendations")
            st.json(data.get('recommendations', "No recommendations found."))
        else:
            st.error(result['message'])
            if 'error' in result:
                st.error(f"Details: {result['error']}")

if __name__ == "__main__":
    app()
