from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os

OPENAI_API_KEY="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"

def build_chain():
  region = os.environ["AWS_REGION"]
  kendra_index_id = os.environ["KENDRA_INDEX_ID"]

  llm = OpenAI(batch_size=5, temperature=0, max_tokens=300)

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

  prompt_template = """
  You need to speak Japanese.
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:"""
#   prompt_template = """
#   以下は、人間とAIの友好的な会話です。 
#   AIはおしゃべり好きで、その文脈から具体的な詳細をたくさん提供します。
#   AIが質問の答えを知らなければ、「知らない」と答えます。
#   質問に対応する文章はこちらです。
#   「{context}」
#   指示:上記の条件をもとに、「{question}」の回答を詳しく答えてください。 
#   文書がない場合、「わからない」と答えてください。
#   """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  chain_type_kwargs = {"prompt": PROMPT}
    
  return RetrievalQA.from_chain_type(
      llm, 
      chain_type="stuff", 
      retriever=retriever, 
      chain_type_kwargs=chain_type_kwargs, 
      return_source_documents=True
  )

def run_chain(chain, prompt: str, history=[]):
    print(prompt)
    result = chain(prompt)
    # To make it compatible with chat samples
    return {
        "answer": result['result'],
        "source_documents": result['source_documents']
    }

if __name__ == "__main__":
    chain = build_chain()
    result = run_chain(chain, "What's SageMaker?")
    print(result['answer'])
    if 'source_documents' in result:
        print('Sources:')
        for d in result['source_documents']:
          print(d.metadata['source'])
