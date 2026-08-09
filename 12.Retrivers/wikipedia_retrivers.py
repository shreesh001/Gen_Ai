from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")


# Define your query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print the retrieved documents
for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("\n---\n")    