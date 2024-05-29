#Importing All Necessary Librqaries
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain.chains import GraphCypherQAChain

from langchain_core.documents import Document
from langchain_community.chat_models.openai import ChatOpenAI

from langchain.graphs import Neo4jGraph
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument
)
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter

from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
#from langchain.chat_models import AzureChatOpenAI
#from langchain_community.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from tqdm import tqdm

from KnowledgeGraphForm import *

os.environ["OPENAI_API_KEY"] = 'fe77aff6afd640f78539d8fcad008a4c'

def neo4jConnction():
  
    #url = "neo4j+s://databases.neo4j.io"
    from langchain_community.graphs import Neo4jGraph

    url=os.getenv("NEO4J_URI","bolt://localhost:7687")
    username=os.getenv("NEO4J_USER","neo4j")
    password=os.getenv("NEO4J_PASSWORD","Pk@12345678")
    #url=os.getenv("NEO4J_URI","neo4j+s://5292e0f3.databases.neo4j.io")
    #username=os.getenv("NEO4J_USER","neo4j")
    #password=os.getenv("NEO4J_PASSWORD","uT0_PcdAy6VlQSfXROAFOi34yVO5h_0aoO3Sm9w0TPM")

    graph = Neo4jGraph(url=url, username=username, password=password)
    #print(graph)
    return graph

# Define mapping functions
def map_to_base_node(node: Node) -> BaseNode:
    return BaseNode(
        id=node.id,
        type=node.type,
        properties={prop.key: prop.value for prop in node.properties} if node.properties else {}
    )

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    return BaseRelationship(
        source=BaseNode(
            id=rel.source.id,
            type=rel.source.type,
            properties={prop.key: prop.value for prop in rel.source.properties} if rel.source.properties else {}
        ),
        target=BaseNode(
            id=rel.target.id,
            type=rel.target.type,
            properties={prop.key: prop.value for prop in rel.target.properties} if rel.target.properties else {}
        ),
        type=rel.type,
        properties={prop.key: prop.value for prop in rel.properties} if rel.properties else {}
    )

def get_extraction_chain(llm,
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    prompt = ChatPromptTemplate.from_messages(
    [(
      "system",
      f"""# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
## 5. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)

def extract_and_store_graph(graph,
    llm,
    document: Document,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(llm,nodes, rels)
    data = extract_chain.run(document.page_content)
    print(data.nodes)
    print(data.rels)
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = [map_to_base_relationship(rel) for rel in data.rels],
      source = document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])


def retrieval(graph,llm,question):
    #Query the knowledge graph in a RAG application
    graph.refresh_schema()

    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=llm,
        qa_llm=llm,
        validate_cypher=True, # Validate relationship directions
        verbose=True
    )

    answer=cypher_chain.run(question)
    #cypher_chain.run("Where ICC T20 world cup held ?")
    return answer

def main():
    # Read the wikipedia article
    #Fecting data related to Virat Kohli from Wikipedia
    raw_documents = WikipediaLoader(query="Virat Kohli").load()
    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=24)

    # Only take the first the raw_documents
    documents = text_splitter.split_documents(raw_documents[:])
    print(len(documents))
    ##Used Azure Deployed LLM model
    llm=AzureChatOpenAI(azure_deployment=os.environ['EmbeddingModelDeploymentName'],model="gpt-35-turbo-16k",azure_endpoint=os.environ['OpenAIEndpoint'],api_key=os.environ['OpenAIKey'],api_version=os.environ['OpenAIVersion'])
    ##Connect Graph Database Neo4j
    graph=neo4jConnction()

    for i, d in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(graph,llm,d)

    # Call Retrieval and Ask your question
    question="Who is the father of Virat Kohli"
    answer=retrieval(graph, llm,question)
    print(answer)

if __name__=="__main__":
    main()


