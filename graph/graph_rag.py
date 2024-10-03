from dotenv import load_dotenv

from graph.workflow import build_workflow

load_dotenv()

# Create first basic c-rag
workflow_crag = build_workflow()

crag_app = workflow_crag.compile()
# crag_app.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Create self C-Rag (add self checking)
workflow_self_rag = build_workflow(add_grade_generation=True)

self_rag_app = workflow_self_rag.compile()
# self_rag_app.get_graph().draw_mermaid_png(output_file_path="graph_self_rag.png")

# Create adaptive self RAG with checking option
workflow_adaptive_rag = build_workflow(
    add_grade_generation=True, conditional_entry_point=True
)

adaptive_app = workflow_adaptive_rag.compile()
# adaptive_app.get_graph().draw_mermaid_png(output_file_path="graph_adaptive_rag.png")
