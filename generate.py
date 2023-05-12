import pathlib
import re
import pandas as pd
import plotly.express as px
import torchvision.models as M

plots = pathlib.Path(__file__).parent / "plots"


def generate_weights_table(
    module,
    metrics,
    dataset,
    include_patterns=None,
    exclude_patterns=None,
):
    """Code from pytorch/vision/docs/source/conf.py"""
    weights_endswith = (
        "_QuantizedWeights"
        if module.__name__.split(".")[-1] == "quantization"
        else "_Weights"
    )
    weight_enums = [
        getattr(module, name) for name in dir(module) if name.endswith(weights_endswith)
    ]
    weights = [w for weight_enum in weight_enums for w in weight_enum]

    if include_patterns is not None:
        weights = [w for w in weights if any(p in str(w) for p in include_patterns)]
    if exclude_patterns is not None:
        weights = [w for w in weights if all(p not in str(w) for p in exclude_patterns)]

    ops_name = "GIPS" if "QuantizedWeights" in weights_endswith else "GFLOPS"

    metrics_keys, metrics_names = zip(*metrics)
    column_names = ["Family", "Weight", *list(metrics_names), "Params", ops_name]

    df = pd.DataFrame(columns=column_names)

    for w in weights:
        row = [
            re.search(r"([A-Za-z]+)", str(w)).group(1),
            str(w),
            *(w.meta["_metrics"][dataset][metric] for metric in metrics_keys),
            w.meta["num_params"],
            w.meta["_ops"],
        ]

        df.loc[len(df)] = row

    return df


df = generate_weights_table(
    module=M,
    metrics=[("acc@1", "Acc@1"), ("acc@5", "Acc@5")],
    dataset="ImageNet-1K",
)
px.scatter(
    df,
    title="Classification",
    x="Params",
    y="Acc@1",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "classification.html")

df = generate_weights_table(
    module=M,
    metrics=[("acc@1", "Acc@1"), ("acc@5", "Acc@5")],
    dataset="ImageNet-1K",
)
px.scatter(
    df,
    title="Classification Quantized Weights",
    x="Params",
    y="Acc@1",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "classification_quant.html")

df = generate_weights_table(
    module=M.detection,
    metrics=[("box_map", "Box MAP")],
    exclude_patterns=["Mask", "Keypoint"],
    dataset="COCO-val2017",
)
px.scatter(
    df,
    title="Detection",
    x="Params",
    y="Box MAP",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "detection.html")


df = generate_weights_table(
    module=M.detection,
    metrics=[("box_map", "Box MAP"), ("mask_map", "Mask MAP")],
    dataset="COCO-val2017",
    include_patterns=["Mask"],
)
px.scatter(
    df,
    title="Instance Segmentation",
    x="Params",
    y="Box MAP",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "instance_segmentation.html")


df = generate_weights_table(
    module=M.detection,
    metrics=[("box_map", "Box MAP"), ("kp_map", "Keypoint MAP")],
    dataset="COCO-val2017",
    include_patterns=["Keypoint"],
)
px.scatter(
    df,
    title="Detection Keypoint",
    x="Params",
    y="Box MAP",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "detection_keypoint.html")


df = generate_weights_table(
    module=M.segmentation,
    metrics=[("miou", "Mean IoU"), ("pixel_acc", "pixelwise Acc")],
    dataset="COCO-val2017-VOC-labels",
)
px.scatter(
    df,
    title="Segmentation",
    x="Params",
    y="Mean IoU",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "segmentation.html")


df = generate_weights_table(
    module=M.video,
    metrics=[("acc@1", "Acc@1"), ("acc@5", "Acc@5")],
    dataset="Kinetics-400",
)
px.scatter(
    df,
    title="Video",
    x="Params",
    y="Acc@1",
    color="Family",
    hover_data=df.columns,
).write_html(plots / "video.html")
