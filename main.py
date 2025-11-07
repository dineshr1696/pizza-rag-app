import os
from huggingface_hub import InferenceClient
from vector import retriever

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    if question.lower() == "q":
        break

    reviews = retriever.invoke(question)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are an expert in answering questions about a pizza restaurant."},
            {"role": "user", "content": f"Reviews:\n{reviews}\n\nQuestion:\n{question}"}
        ],
        max_tokens=200
    )

    answer = response.choices[0].message["content"]
    print("\nAnswer:\n", answer)
