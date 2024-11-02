
### Unstructured Data Analysis and Chatbot

#### Overview
This application processes unstructured text data from PDFs and provides a conversational interface using a chatbot. It utilizes `Streamlit` for a user-friendly web interface and integrates several AI tools for processing, storing, and retrieving text data, including the use of AWS SageMaker for large language model (LLM) deployment.

#### Features
- **PDF Text Extraction**: Reads and extracts text from uploaded PDF documents.
- **Text Chunking**: Breaks down extracted text into smaller chunks for efficient processing.
- **Embedding and Vector Storage**: Uses Hugging Face embeddings and FAISS for efficient storage and retrieval of text embeddings.
- **Conversational Interface**: Allows users to query the extracted data through a chatbot interface powered by LangChain and SageMaker.

#### Requirements
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) (for AWS services)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [LangChain](https://github.com/hwchase17/langchain)
- Other packages listed in `requirements.txt`

#### Setup
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Jencheng1/RAG-SageMakerJumpstart.git
    cd RAG-SageMakerJumpstart
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure AWS Credentials** (for SageMaker integration):
    Make sure to set up your AWS credentials via the AWS CLI or directly in the code as per your environment’s requirements.

4. **Run the Application**:
    ```bash
    streamlit run unstructure_data_analysis_and_chatbot.py
    ```

#### Usage
1. **Upload PDFs**: Drag and drop your PDF files into the interface.
2. **Interact**: Use the chatbot to ask questions about the content in your PDFs.

#### Project Structure
- **Functions**:
  - `get_pdf_text`: Extracts text from PDF files.
  - `get_text_chunks`: Splits text into manageable chunks.
  - `get_vectorstore`: Creates a vector store for efficient retrieval.
- **Classes**:
  - `ContentHandler`: Manages input/output transformations for the LLM.
- **Main Application**: Hosted with Streamlit for a user-friendly interface.

#### License
This project is licensed under the MIT License.

