# -----------------------------
# Minimal RAG Integration - Add as new cell after Section 11
# -----------------------------
# Install: pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# -----------------------------
# Step 1: Load SHAP Results
# -----------------------------
global_df = pd.read_csv("vfl_shap_global_summary.csv")
attack_df = pd.read_csv("vfl_shap_ddos_summary.csv")
policy_df = pd.read_csv("vfl_shap_mitigation_policies.csv")

# -----------------------------
# Step 2: Create Knowledge Base
# -----------------------------
knowledge_base = [
    f"Party 1 ({global_df.iloc[0]['Domain']}) contributes {global_df.iloc[0]['Mean_contrib_All']*100:.2f}% globally. "
    f"For attacks, Party 1 contributes {attack_df.iloc[0]['Mean_contrib_Attack']*100:.2f}%.",
    
    f"Party 2 ({global_df.iloc[1]['Domain']}) contributes {global_df.iloc[1]['Mean_contrib_All']*100:.2f}% globally. "
    f"For attacks, Party 2 contributes {attack_df.iloc[1]['Mean_contrib_Attack']*100:.2f}%.",
    
    f"Party 3 ({global_df.iloc[2]['Domain']}) contributes {global_df.iloc[2]['Mean_contrib_All']*100:.2f}% globally. "
    f"For attacks, Party 3 contributes {attack_df.iloc[2]['Mean_contrib_Attack']*100:.2f}% and is the dominant party.",
    
    f"Global threshold policy blocks {policy_df.iloc[0]['Attack_blocked_pct']:.2f}% of attacks "
    f"but also {policy_df.iloc[0]['Benign_blocked_pct']:.2f}% of benign flows (false positives).",
    
    f"SHAP-guided policy blocks {policy_df.iloc[1]['Attack_blocked_pct']:.2f}% of attacks "
    f"and only {policy_df.iloc[1]['Benign_blocked_pct']:.2f}% of benign flows, reducing false positives significantly."
]

# -----------------------------
# Step 3: Create Embeddings
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, no API key needed

# Encode knowledge base
embeddings = model.encode(knowledge_base)
embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# Step 4: RAG Query Function
# -----------------------------
def rag_query(question, k=2):
    """Simple RAG: retrieve relevant context and generate answer"""
    # Encode question
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search for similar documents
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve relevant context
    context = "\n".join([knowledge_base[idx] for idx in indices[0]])
    
    # Generate answer (simple template-based, or use LLM)
    answer = f"""
Based on VFL SHAP analysis:

{context}

Summary: Party 3 ({attack_df.iloc[2]['Domain']}) is the dominant party for attack detection 
with {attack_df.iloc[2]['Mean_contrib_Attack']*100:.2f}% contribution. 
SHAP-guided mitigation reduces false positives significantly.
"""
    return answer

# -----------------------------
# Step 5: Example Queries
# -----------------------------
print("\n=== Minimal RAG Q&A ===\n")

queries = [
    "Which party should focus on DDoS mitigation?",
    "Compare the two blocking policies",
    "What are the key findings?"
]

for query in queries:
    print(f"Q: {query}")
    answer = rag_query(query)
    print(f"A: {answer}\n")
    print("-" * 60)

print("\nRAG system ready! Use rag_query('your question') to query.")
