import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import umap


def plot_scatter_cid(X, ids, file_path):
    r"""Plots extracted features colored by their class ids."""
    pl = go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=ids, colorscale='Viridis'))
    data = [pl]
    print(f"Saving plot to: {file_path}")
    py.plot({"data": data, "layout": go.Layout(title="Feature Dimensionality Reduction Visualization")},
                     image='jpeg', image_filename='plot', filename=file_path, auto_open=False)


def dimreduct_umap(features):
    r"""Dimensionality reduction with UMAP."""
    print("Running UMAP on features ...")
    umap_embedding = umap.UMAP(n_neighbors=5, init="spectral").fit_transform(features)
    
    return umap_embedding


def execute(args, model):
    os.makedirs(args["output_path"], exist_ok=True)

    features = np.load(args["features_path"])
    labels = np.load(args["labels_path"])

    # Dimensionality reduction
    if args["dim_reduct_method"] == "umap":
        feat_red = dimreduct_umap(features)
    else:
        raise RuntimeError(f"Dimensionality reduction method not defined.")
    
    # Visualization
    plot_scatter_cid(feat_red, labels, os.path.join(args["output_path"], "plot.html"))


