from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import sys
import json
import os

from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer, JSONDeserializer

endpoint_name = "Rinna-Inference"

predictor_client=Predictor(
    endpoint_name=endpoint_name,
    # sagemaker_session=sess,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

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
  # endpoint_name = os.environ["FLAN_XL_ENDPOINT"]
  endpoint_name = "Rinna-Inference"


  class ContentHandler(LLMContentHandler):
      content_type = "application/json"
      accepts = "application/json"
      response_list = []

      def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
          input_str = json.dumps({"text_inputs": prompt, **model_kwargs})

          # prompt="こんにちは"

          data = {
              "instruction": prompt.replace("\n", "<NL>"),  # システム
              # "instruction": prompt.replace("\n", "。"),  # システム
              "input": "",  # ユーザー
              "max_new_tokens": 128,
              "temperature": 0.7,
              "do_sample": True,
              "pad_token_id": 0,
              "bos_token_id": 2,
              "eos_token_id": 3,
              # "stop_ids": [50278, 50279, 50277, 1, 0],
          }
          response = predictor_client.predict(
              data=data
          )
          # print(response.replace("<NL>", "\n"))
          self.response_list.append(response.replace("<NL>", "\n"))
          # self.response_list.append(response.replace("。", "\n"))

          print(prompt)

          return input_str.encode('utf-8')
      
      def transform_output(self, output: bytes) -> str:
          response_json = json.loads(output.read().decode("utf-8"))
          # return response_json["generated_texts"][0]
          # return str(response_json)
          return str(self.response_list.pop(0))

  content_handler = ContentHandler()

  llm=SagemakerEndpoint(
          endpoint_name=endpoint_name, 
          region_name=region, 
          model_kwargs={"temperature":1e-10, "max_length": 500},
          content_handler=content_handler
      )
      
  retriever = AmazonKendraRetriever(
     index_id=kendra_index_id,
     attribute_filter={
        'AndAllFilters': [
              {
                'EqualsTo': {
                  'Key': '_language_code',
                  'Value': {
                    'StringValue': 'ja'
                  }
                }
              },
            ]
     }
  )

  # prompt_template = """
  # The following is a friendly conversation between a human and an AI. 
  # The AI is talkative and provides lots of specific details from its context.
  # If the AI does not know the answer to a question, it truthfully says it 
  # does not know.
  # {context}
  # Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  # if not present in the document. 
  # Solution:"""


  prompt_template = """
  以下は、人間とAIの友好的な会話です。 
  AIはおしゃべり好きで、その文脈から具体的な詳細をたくさん提供します。
  AIが質問の答えを知らなければ、「知らない」と答えます。
  質問に対応する文章はこちらです。
  「{context}」
  指示:上記の条件をもとに、「{question}」の回答を詳しく答えてください。 
  文書がない場合、「わからない」と答えてください。
  """



  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  # condense_qa_template = """
  # Given the following conversation and a follow up question, rephrase the follow up question 
  # to be a standalone question.

  # Chat History:
  # {chat_history}
  # Follow Up Input: {question}
  # Standalone question:"""

  condense_qa_template = """
  次の会話とフォローアップの質問を踏まえて、フォローアップの質問を独立した質問と表現し直してください。

  チャット履歴:
  {chat_history}
  フォローアップ入力:{question}
  スタンドアロンの質問:"""



  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

  qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT})
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
