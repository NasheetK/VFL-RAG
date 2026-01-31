# -----------------------------
# Minimal RAG Integration for VFL SHAP Results
# Add this after Section 11 or 13
# -----------------------------

# Install required packages:
# pip install langchain openai faiss-cpu sentence-transformers

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
import pandas as pd

# -----------------------------
# Step 1: Load SHAP Results as Knowledge Base
# -----------------------------
print("Loading SHAP results as knowledge base...")

# Load all CSV results
global_df = pd.read_csv("vfl_shap_global_summary.csv")
attack_df = pd.read_csv("vfl_shap_ddos_summary.csv")
benign_df = pd.read_csv("vfl_shap_benign_summary.csv")
policy_df = pd.read_csv("vfl_shap_mitigation_policies.csv")

# Convert DataFrames to text documents
knowledge_base = []

# Add global summary
knowledge_base.append(f"""
GLOBAL SHAP ANALYSIS:
Party 1 ({global_df.iloc[0]['Domain']}): 
  - Mean contribution: {global_df.iloc[0]['Mean_contrib_All']*100:.2f}%
  - Feature group: {global_df.iloc[0]['Feature_Group']}

Party 2 ({global_df.iloc[1]['Domain']}): 
  - Mean contribution: {global_df.iloc[1]['Mean_contrib_All']*100:.2f}%
  - Feature group: {global_df.iloc[1]['Feature_Group']}

Party 3 ({global_df.iloc[2]['Domain']}): 
  - Mean contribution: {global_df.iloc[2]['Mean_contrib_All']*100:.2f}%
  - Feature group: {global_df.iloc[2]['Feature_Group']}
""")

# Add attack-specific analysis
knowledge_base.append(f"""
ATTACK DETECTION ANALYSIS:
For DDoS/attack flows:
Party 1: {attack_df.iloc[0]['Mean_contrib_Attack']*100:.2f}% contribution
Party 2: {attack_df.iloc[1]['Mean_contrib_Attack']*100:.2f}% contribution
Party 3: {attack_df.iloc[2]['Mean_contrib_Attack']*100:.2f}% contribution

Party 3 ({attack_df.iloc[2]['Domain']}) is the dominant party for attack detection.
""")

# Add policy comparison
knowledge_base.append(f"""
POLICY COMPARISON:
Global threshold policy:
  - Blocks {policy_df.iloc[0]['Attack_blocked_pct']:.2f}% of attacks
  - Blocks {policy_df.iloc[0]['Benign_blocked_pct']:.2f}% of benign flows (false positives)

SHAP-guided policy:
  - Blocks {policy_df.iloc[1]['Attack_blocked_pct']:.2f}% of attacks
  - Blocks {policy_df.iloc[1]['Benign_blocked_pct']:.2f}% of benign flows (false positives)

SHAP-guided policy reduces false positives significantly.
""")

# -----------------------------
# Step 2: Create Vector Store (RAG)
# -----------------------------
print("Creating vector store...")

# Use free embeddings (no API key needed)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create documents
from langchain.schema import Document
documents = [Document(page_content=text) for text in knowledge_base]

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# -----------------------------
# Step 3: Create RAG Chain
# -----------------------------
print("Setting up RAG system...")

# Option 1: Use OpenAI (requires API key)
# llm = OpenAI(temperature=0)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
# )

# Option 2: Use free local LLM (if available)
# Or use simple retrieval + template
def simple_rag_query(question):
    """Simple RAG without LLM - just retrieval + template"""
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Simple template-based response
    response = f"""
Based on the VFL SHAP analysis:

{context}

Answer: {question}

Key insights:
- Party 3 (Enterprise/edge gateway) dominates attack detection with {attack_df.iloc[2]['Mean_contrib_Attack']*100:.2f}% contribution
- SHAP-guided policy reduces false positives from {policy_df.iloc[0]['Benign_blocked_pct']:.2f}% to {policy_df.iloc[1]['Benign_blocked_pct']:.2f}%
- Recommended action: Focus mitigation on Party 3's domain ({attack_df.iloc[2]['Domain']})
"""
    return response

# -----------------------------
# Step 4: Query Examples
# -----------------------------
print("\n=== RAG Q&A System ===\n")

queries = [
    "Which party should focus on DDoS mitigation?",
    "Why is Party 3 more important for attacks?",
    "Compare the two blocking policies",
    "What are the key findings from the SHAP analysis?"
]

for query in queries:
    print(f"Q: {query}")
    answer = simple_rag_query(query)
    print(f"A: {answer}\n")
    print("-" * 80)

# -----------------------------
# Step 5: Interactive Query (Optional)
# -----------------------------
# Uncomment to enable interactive queries:
# while True:
#     question = input("\nAsk a question about the SHAP results (or 'quit'): ")
#     if question.lower() == 'quit':
#         break
#     answer = simple_rag_query(question)
#     print(f"\n{answer}\n")

print("\nRAG system ready! Use simple_rag_query('your question') to query.")
