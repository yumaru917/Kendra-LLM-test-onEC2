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
# predictor_client = AsyncPredictor(
#     predictor=predictor_client,
#     name=endpoint_name
# )
data = {
    "instruction": """こんにちは""".replace("\n", "<NL>"),  # システム
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
print(response.replace("<NL>", "\n"))