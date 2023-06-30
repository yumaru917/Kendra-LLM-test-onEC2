from aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.prompts import PromptTemplate
import sys
import json
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5

def build_chain():
  region = os.environ["AWS_REGION"]
  kendra_index_id = os.environ["KENDRA_INDEX_ID"]
  endpoint_name = os.environ["OPEN_CALM_ENDPOINT"]

  class ContentHandler(ContentHandlerBase):
      content_type = "application/json"
      accepts = "application/json"

      def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
          input_str = json.dumps({"input": prompt, **model_kwargs})
          print(input_str)
          return input_str.encode('utf-8')
      
      def transform_output(self, output: bytes) -> str:
          response_json = json.loads(output.read().decode("utf-8"))
          print(response_json)
          return str(response_json)

  content_handler = ContentHandler()

  llm=SagemakerEndpoint(
          endpoint_name=endpoint_name, 
          region_name=region, 
          model_kwargs={"temperature":0.1, "max_new_tokens": 128, "do_sample": True, "pad_token_id": 1},
          content_handler=content_handler
      )
      
  retriever = KendraIndexRetriever(kendraindex=kendra_index_id, 
      awsregion=region, 
      return_source_documents=True)

  prompt_template = """
  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
  ### Instruction: {question} 
  ### Input: {context}
  ### Response: 
  """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, qa_prompt=PROMPT, return_source_documents=True)
  return qa

def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt, "chat_history": history})


if __name__ == "__main__":
  chat_history = []
  qa = build_chain()
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
  print(">", end=" ", flush=True)
  for query in sys.stdin:
    if (query.strip().lower().startswith("new search:")):
      query = query.strip().lower().replace("new search:","")
      chat_history = []
    elif (len(chat_history) == MAX_HISTORY_LENGTH):
      chat_history.pop(0)
    result = run_chain(qa, query, chat_history)
    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
