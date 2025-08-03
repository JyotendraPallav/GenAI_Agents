import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI

load_dotenv()

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai_api_version = os.getenv("OPENAI_API_VERSION")
Conf_api_key = os.getenv("CONFLUENCES_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME_4o_MINI")

# llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
# Check that all required environment variables are set
if not all([openai_api_key, openai_api_version, azure_endpoint, openai_api_type]):
    raise ValueError("One or more required OpenAI/Azure environment variables are not set.")

llm = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_version=openai_api_version,
    azure_endpoint=azure_endpoint,
    openai_api_type=openai_api_type,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    temperature=0.2,
    verbose=False,
    max_tokens=1000,
)