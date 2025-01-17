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
                st.text(result['data'])  # Hiển thị kết quả dạng văn bản thô
            else:
                # Hiển thị lỗi
                st.error(result['message'])
                if 'error' in result:
                    st.error(f"Details: {result['error']}")
        else:
            st.warning("Please upload a PDF file before generating.")

if __name__ == "__main__":
    app()
