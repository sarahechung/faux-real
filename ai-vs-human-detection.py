"""
Faux Real: Best-Performing Model — CNN with Transfer Learning

This script trains and evaluates a binary classifier to detect AI-generated images vs real images
using transfer learning (EfficientNetB0). It reports Accuracy, Precision, Recall, F1, Confusion
Matrix, and ROC-AUC. It saves the best model weights and a classification report.

Directory layout expected (change paths with CLI args):

DATA_ROOT/
  train/
    real/
    ai/
  val/
    real/
    ai/
  test/
    real/
    ai/

Usage (example):
  python faux_real_transfer_learning.py \
      --data_root /path/to/DATA_ROOT \
      --img_size 224 \
      --batch_size 32 \
      --epochs 20 \
      --model_out ./best_faux_real.keras

Requirements: tensorflow>=2.12, scikit-learn, matplotlib, numpy
"""

import argparse
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    p = argparse.ArgumentParser(
        description="CNN with transfer learning to detect AI-generated images"
    )
    p.add_argument("--data_root", type=str, required=True, help="Root folder with train/val/test")
    p.add_argument("--img_size", type=int, default=224, help="Square image size")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--base", type=str, default="efficientnetb0", choices=["efficientnetb0", "resnet50"],
        help="Backbone architecture"
    )
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    p.add_argument("--fine_tune_at", type=int, default=None, help="Layer index to start fine-tuning (after warmup)")
    p.add_argument("--warmup_epochs", type=int, default=3, help="Train top layers first for these epochs")
    p.add_argument("--model_out", type=str, default="best_faux_real.keras")
    p.add_argument("--plots_dir", type=str, default="plots")
    return p.parse_args()


def load_datasets(data_root: str, img_size: int, batch_size: int):
    data_root = Path(data_root)
    def make_ds(split):
        return tf.keras.utils.image_dataset_from_directory(
            data_root / split,
            labels="inferred",
            label_mode="binary",
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True if split != "test" else False,
        )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    class_names = train_ds.class_names  # e.g., ['ai', 'real']

    # Prefetch + cache
    def prepare(ds, augment=False):
        ds = ds.cache()
        if augment:
            ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return (
        prepare(train_ds, augment=True),
        prepare(val_ds, augment=False),
        prepare(test_ds, augment=False),
        class_names,
    )


def augment_fn(image, label):
    # Lightweight augmentations to improve generalization
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def preprocess_fn(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def build_model(img_size: int, base: str = "efficientnetb0"):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    if base == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, input_tensor=inputs, weights="imagenet"
        )
    else:  # resnet50
        preprocess = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(
            include_top=False, input_tensor=inputs, weights="imagenet"
        )

    # Use the selected preprocess inside the graph
    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=f"faux_real_{base}")
    return model, base_model


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", out_path=None):
    """Plots confusion matrix using matplotlib (no seaborn)."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            verticalalignment="center",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    print("\nLoading datasets…")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_root, args.img_size, args.batch_size
    )
    print("Classes:", class_names)

    print("\nBuilding model…")
    model, base_model = build_model(args.img_size, args.base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    print("\nWarmup training (top layers)…")
    base_model.trainable = False
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        verbose=1,
        callbacks=callbacks,
    )

    if args.fine_tune_at is not None:
        print("\nFine-tuning backbone from layer", args.fine_tune_at)
        base_model.trainable = True
        for layer in base_model.layers[: args.fine_tune_at]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr * 0.2),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    print("\nFull training…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    print("\nEvaluating on test set…")
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    print("Test metrics:", test_metrics)

    # Predictions for detailed metrics
    y_true = []
    y_prob = []
    for batch_imgs, batch_labels in test_ds:
        y_true.append(batch_labels.numpy().ravel())
        y_prob.append(model.predict(batch_imgs, verbose=0).ravel())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\nDetailed metrics -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(args.plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes=class_names, title="Confusion Matrix", out_path=cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    roc_path = os.path.join(args.plots_dir, "roc_curve.png")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    print("\nDone. Best weights saved to:", args.model_out)


if __name__ == "__main__":
    main()
