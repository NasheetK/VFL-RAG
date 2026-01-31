# ============================================
# MINIMAL RAG INTEGRATION - Add as new cell after Section 11
# ============================================
# Install first: pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load SHAP results
global_df = pd.read_csv("vfl_shap_global_summary.csv")
attack_df = pd.read_csv("vfl_shap_ddos_summary.csv")
policy_df = pd.read_csv("vfl_shap_mitigation_policies.csv")

# Create knowledge base from results
kb = [
    f"Party 1 contributes {global_df.iloc[0]['Mean_contrib_All']*100:.2f}% globally, {attack_df.iloc[0]['Mean_contrib_Attack']*100:.2f}% for attacks.",
    f"Party 2 contributes {global_df.iloc[1]['Mean_contrib_All']*100:.2f}% globally, {attack_df.iloc[1]['Mean_contrib_Attack']*100:.2f}% for attacks.",
    f"Party 3 contributes {global_df.iloc[2]['Mean_contrib_All']*100:.2f}% globally, {attack_df.iloc[2]['Mean_contrib_Attack']*100:.2f}% for attacks. Party 3 is dominant.",
    f"Global policy: {policy_df.iloc[0]['Attack_blocked_pct']:.2f}% attacks, {policy_df.iloc[0]['Benign_blocked_pct']:.2f}% false positives.",
    f"SHAP policy: {policy_df.iloc[1]['Attack_blocked_pct']:.2f}% attacks, {policy_df.iloc[1]['Benign_blocked_pct']:.2f}% false positives."
]

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.array(model.encode(kb)).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# RAG query function
def rag_query(q, k=2):
    q_emb = np.array(model.encode([q])).astype('float32')
    _, idxs = index.search(q_emb, k)
    context = "\n".join([kb[i] for i in idxs[0]])
    return f"Context: {context}\n\nAnswer: Based on analysis, {context.split('.')[0]}."

# Example usage
print(rag_query("Which party should focus on mitigation?"))
print(rag_query("Compare the policies"))
