import streamlit as st
import json
import requests
import pdfplumber
import io

# Azure OpenAI API endpoint and key từ secrets
endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
api_key = st.secrets["AZURE_OPENAI_API_KEY"]

# Hàm đọc văn bản từ PDF
def extract_text_from_pdf(pdf_data):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Hàm xử lý phản hồi từ API
def process_response(response_data):
    if not response_data:
        st.warning("No response data. Skipping parsing.")
        return None

    try:
        return response_data['choices'][0]['message']['content']
    except (KeyError, IndexError):
        st.error("Unexpected API response format.")
        return None

# Hàm xử lý tệp và gọi API
def process_file(file_path, schema_string, prompt_text):
    extracted_text = ""

    # Đọc PDF
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

    # Chuẩn bị dữ liệu gửi API
    system_message = {
        "role": "system",
        "content": "You are an AI assistant that helps extract information from resumes (CVs)."
    }

    # Thay thế {schema_string} trong prompt bằng nội dung thực tế
    user_message = {
        "role": "user",
        "content": prompt_text.replace("{schema_string}", schema_string).replace("{extracted_text}", extracted_text)
    }

    data = {
        "messages": [system_message, user_message],
        "max_tokens": 16000,
        "temperature": 1,
        "top_p": 0.25
    }

    # Gửi yêu cầu API
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

    # Upload file PDF
    st.sidebar.header("File Input")
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])

    # Đọc schema JSON
    schema_path = './schema.json'  # Đường dẫn tới schema.json
    try:
        with open(schema_path, 'r') as file:
            schema_json = json.load(file)  # Đọc JSON từ tệp
            schema_string = json.dumps(schema_json, indent=4)  # Định dạng JSON thành chuỗi
    except FileNotFoundError:
        st.error("Schema file not found.")
        return
    except json.JSONDecodeError as e:
        st.error(f"Error reading schema JSON: {e}")
        return

    # Prompt mặc định
    default_prompt = """
        You are an AI assistant that helps extract information from resumes (CVs).

        Analyze the candidate's CV {extracted_text} and provide insights based on the following criteria:
        0. Name and Contact Information.
        1. Work Experience Analysis:
        For each company listed, extract and summarize the job title, tenure, and level of expertise (categorized as beginner, intermediate, or expert) in relevant fields. Organize this information by company in a structured format.
        2. Job Trends and Stability:
        Analyze the candidate’s career progression by evaluating the time spent in each role. Identify patterns such as frequent job changes, promotions, extended tenures, or gaps between roles. Assess the likelihood of long-term job stability versus a tendency for frequent transitions.
        3. Suggested Job Titles:
        Based on the candidate’s skills, years of experience, and educational background, recommend potential job titles or career paths. Ensure suggestions align with their demonstrated expertise, industry trends, and career growth opportunities.
        4. Job Resignation Prediction:
        Predict the likelihood of the candidate changing jobs at this time (output as Yes or No) based on factors such as:
        Duration in the current role relative to past roles.
        Alignment of current role with skills, career goals, and industry trends.
        Patterns of frequent transitions, gaps, or promotions in job history.
        Indicators of dissatisfaction, stagnation, or misalignment with expertise.
        predict the exact timeframe (e.g., "3 months," "12 months," or "2 years") in which the candidate is likely to change jobs
    """

    # Prompt Editor
    prompt_text = st.text_area("Prompt Editor", default_prompt, height=600)

    # Nút "Generate"
    if st.button("Generate"):
        if uploaded_file is not None:
            file_path = uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Gọi hàm process_file
            result = process_file(file_path, schema_string, prompt_text)

            if result['statusCode'] == 200:
                # Hiển thị kết quả thô từ API
                st.header("Result Output")
                st.write("Below is the raw result:")
                st.markdown(result['data'])  # Hiển thị kết quả dạng văn bản thô
            else:
                # Hiển thị lỗi
                st.error(result['message'])
                if 'error' in result:
                    st.error(f"Details: {result['error']}")
        else:
            st.warning("Please upload a PDF file before generating.")

if __name__ == "__main__":
    app()
