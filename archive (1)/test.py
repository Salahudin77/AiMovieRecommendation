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


# Function to create the Architecture Diagram
def create_architecture_diagram():
    plt.figure(figsize=(14, 10))

    # Define positions for architecture components
    positions = {
        "User Interface (Streamlit)": (0, 5),
        "Dataset (Movies/Ratings)": (2, 4.25),
        "Data Preprocessing": (4, 4.25),
        "Collaborative Filtering (NMF/SVD)": (6, 5.5),
        "Neural Collaborative Filtering (NCF)": (6, 4.25),
        "Genre-Based Filtering": (6, 3),
        "Recommendation Engine": (8, 4.25),
        "Recommendation Output": (10, 4.25),
    }

    # Draw components
    for name, (x, y) in positions.items():
        shape = "ellipse" if "User Interface" in name or "Output" in name else "box"
        color = (
            "lightblue" if "User Interface" in name or "Output" in name else
            "orange" if "Dataset" in name else
            "yellow" if "Preprocessing" in name else
            "lightpink" if "Filtering" in name else
            "violet"
        )
        if shape == "ellipse":
            plt.gca().add_patch(Ellipse((x, y), 1.8, 1, color=color, alpha=0.9))
        else:
            plt.gca().add_patch(Rectangle((x - 0.9, y - 0.5), 1.8, 1, color=color, alpha=0.9))

        plt.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold", color="black")

    # Define arrows
    arrows = [
        ("User Interface (Streamlit)", "Dataset (Movies/Ratings)"),
        ("Dataset (Movies/Ratings)", "Data Preprocessing"),
        ("Data Preprocessing", "Collaborative Filtering (NMF/SVD)"),
        ("Data Preprocessing", "Neural Collaborative Filtering (NCF)"),
        ("Data Preprocessing", "Genre-Based Filtering"),
        ("Collaborative Filtering (NMF/SVD)", "Recommendation Engine"),
        ("Neural Collaborative Filtering (NCF)", "Recommendation Engine"),
        ("Genre-Based Filtering", "Recommendation Engine"),
        ("Recommendation Engine", "Recommendation Output"),
        ("User Interface (Streamlit)", "Recommendation Output"),
    ]

    # Draw arrows
    for start, end in arrows:
        start_x, start_y = positions[start]
        end_x, end_y = positions[end]
        plt.arrow(
            start_x, start_y,
            end_x - start_x, end_y - start_y,
            head_width=0.2, head_length=0.25, fc="gray", ec="gray", alpha=0.8, length_includes_head=True
        )

    # Finalize the architecture diagram
    plt.title("Architecture Diagram of Movie Recommendation System", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()


# Run both diagrams
create_workflow_diagram()
create_architecture_diagram()
