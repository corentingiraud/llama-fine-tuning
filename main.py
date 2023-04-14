from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

llm = LlamaCpp(
    model_path="models/llama/7B/ggml-model-q4_0.bin",
    n_threads=8,
    n_ctx=100000,
)


template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="history")
chain = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
)

while True:
    question = input("? ")
    print(chain.predict(input=question))
