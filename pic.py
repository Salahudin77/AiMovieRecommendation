import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle

# Function to create the Workflow Diagram
def create_workflow_diagram():
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Define nodes for the workflow
    workflow_nodes = {
        "User Input": (0.5, 8),
        "Data Preprocessing": (0.5, 6),
        "Genre-Based Filtering": (-2, 4),
        "Collaborative Filtering (NMF/SVD)": (0.5, 4),
        "Neural Collaborative Filtering (NCF)": (3, 4),
        "Recommendation Output": (0.5, 2),
    }

    # Draw nodes as text boxes
    for label, (x, y) in workflow_nodes.items():
        ax.text(
            x, y, label, fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightgray")
        )

    # Define edges (arrows)
    workflow_edges = [
        ("User Input", "Data Preprocessing"),
        ("Data Preprocessing", "Genre-Based Filtering"),
        ("Data Preprocessing", "Collaborative Filtering (NMF/SVD)"),
        ("Data Preprocessing", "Neural Collaborative Filtering (NCF)"),
        ("Genre-Based Filtering", "Recommendation Output"),
        ("Collaborative Filtering (NMF/SVD)", "Recommendation Output"),
        ("Neural Collaborative Filtering (NCF)", "Recommendation Output"),
    ]

    # Draw arrows for the workflow
    for start, end in workflow_edges:
        start_x, start_y = workflow_nodes[start]
        end_x, end_y = workflow_nodes[end]
        arrow = FancyArrowPatch(
            (start_x, start_y - 0.2), (end_x, end_y + 0.2),
            connectionstyle="arc3,rad=0.3", arrowstyle='-|>', mutation_scale=10, color='gray'
        )
        ax.add_patch(arrow)

    # Finalize the workflow diagram
    plt.title("Workflow Diagram of Movie Recommendation System", fontsize=14, fontweight="bold")
    plt.xlim(-3, 3)
    plt.ylim(1, 9)
    plt.axis("off")
    plt.show()


