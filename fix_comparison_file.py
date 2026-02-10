"""
Script to add all_actions comparison fields to save_comparison_file function.
Run this script to update the notebook.
"""

import json
import re

# Read the notebook
with open('RAG_LLM_action_plan.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with save_comparison_file function
for cell_idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def save_comparison_file' in source:
            # Add all_actions to with_rag section
            source = source.replace(
                '"execution_priority": (\n                    llm_response_with_rag.get("execution_priority")\n                    if llm_response_with_rag\n                    else None\n                ),\n                "primary_actions": (',
                '"execution_priority": (\n                    llm_response_with_rag.get("execution_priority")\n                    if llm_response_with_rag\n                    else None\n                ),\n                "all_actions": (\n                    llm_response_with_rag.get("all_actions", [])\n                    if llm_response_with_rag\n                    else []\n                ),\n                "primary_actions": ('
            )
            
            # Add all_actions to without_rag section
            source = source.replace(
                '"execution_priority": (\n                    llm_response_without_rag.get("execution_priority")\n                    if llm_response_without_rag\n                    else None\n                ),\n                "primary_actions": (',
                '"execution_priority": (\n                    llm_response_without_rag.get("execution_priority")\n                    if llm_response_without_rag\n                    else None\n                ),\n                "all_actions": (\n                    llm_response_without_rag.get("all_actions", [])\n                    if llm_response_without_rag\n                    else []\n                ),\n                "primary_actions": ('
            )
            
            # Add all_actions comparison fields before the closing of differences
            all_actions_comparison = '''            "all_actions_different": (
                set(llm_response_with_rag.get("all_actions", []))
                if llm_response_with_rag
                else set()
            )
            != (
                set(llm_response_without_rag.get("all_actions", []))
                if llm_response_without_rag
                else set()
            ),
            "num_all_actions_with_rag": (
                len(llm_response_with_rag.get("all_actions", []))
                if llm_response_with_rag
                else 0
            ),
            "num_all_actions_without_rag": (
                len(llm_response_without_rag.get("all_actions", []))
                if llm_response_without_rag
                else 0
            ),
            "all_actions_with_rag": (
                llm_response_with_rag.get("all_actions", [])
                if llm_response_with_rag
                else []
            ),
            "all_actions_without_rag": (
                llm_response_without_rag.get("all_actions", [])
                if llm_response_without_rag
                else []
            ),'''
            
            source = source.replace(
                '            "num_primary_actions_without_rag": (\n                len(llm_response_without_rag.get("primary_actions", []))\n                if llm_response_without_rag\n                else 0\n            ),\n        },',
                f'            "num_primary_actions_without_rag": (\n                len(llm_response_without_rag.get("primary_actions", []))\n                if llm_response_without_rag\n                else 0\n            ),\n{all_actions_comparison}\n        }},'
            )
            
            # Update the cell source
            notebook['cells'][cell_idx]['source'] = source.splitlines(keepends=True)
            print(f"✓ Updated cell {cell_idx} with all_actions comparison fields")
            break

# Save the notebook
with open('RAG_LLM_action_plan.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Notebook updated successfully!")
