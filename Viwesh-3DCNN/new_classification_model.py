
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, regularizers
import seaborn as sns
import einops
import cv2
import os
import pandas as pd
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, average_precision_score,
    confusion_matrix, classification_report
)
import random


DATA_ROOT  = 'COSC_574/final_project/sim_dataset/data'
HEIGHT     = 224
WIDTH      = 224
N_FRAMES   = 16
BATCH_SIZE = 4
EPOCHS     = 30
LR         = 1e-4

CLASS_MAP = {
    'head-on':   0,
    'rear-end':  1,
    'sideswipe': 2,
    'single':    3,
    't-bone':    4,
    #'non-accident': 5,
}
CLASS_NAMES = ['head-on', 'rear-end', 'sideswipe', 'single', 't-bone']
N_CLASSES   = len(CLASS_MAP)

def parse_crash_type_from_filename(stem: str) -> str | None:
    video_id_part = stem.split('_frame_')[0].replace('.mp4', '')
    #print(f"DEBUG: stem={stem!r}  video_id_part={video_id_part!r}  CLASS_NAMES={CLASS_NAMES}")

    for class_name in CLASS_NAMES:
        if video_id_part.endswith(f'_{class_name}'):
            return class_name

    for class_name in CLASS_NAMES:
        if f'_{class_name}_' in f'_{video_id_part}_':
            return class_name

    return None

def build_clip_paths(split_dir, n_frames=N_FRAMES, stride=None):
    stride = stride or n_frames
    root = pathlib.Path(split_dir)
    all_clip_paths = []
    all_labels     = []
    video_frames   = collections.defaultdict(list)
    skipped        = 0

    for img_path in sorted(root.glob('**/*.jpg')):
        current_dir_name = img_path.parent.name

        if current_dir_name == 'Non_Accident':
            #class_name = 'non-accident'
            continue
        elif current_dir_name in CLASS_MAP:
            # For frames in the 'Accident' directory, parse the specific crash type from the filename
            class_name = parse_crash_type_from_filename(img_path.stem)
        else:
            class_name = None

        if class_name is None:
            skipped += 1
            if skipped <= 5:
                print(f'WARNING: could not determine crash type for {img_path.name} — skipping.')
            continue

        parts = img_path.stem.split('_frame_')
        if len(parts) != 2:
            skipped += 1
            continue

        video_id = parts[0]
        video_frames[(class_name, video_id)].append(str(img_path))

    if skipped > 5:
        print(f'WARNING: {skipped} total files skipped (showing first 5 above).')

    non_accident_clips = []

    for (class_name, video_id), frames in video_frames.items():
        if class_name not in CLASS_MAP:
            print(f"WARNING: '{class_name}' not in CLASS_MAP, skipping video_id {video_id}")
            continue
        label  = CLASS_MAP[class_name]
        frames = sorted(frames)
        clips  = [frames[start : start + n_frames]
                  for start in range(0, len(frames) - n_frames + 1, stride)]

        #if class_name == 'non-accident':
        #    non_accident_clips.extend([(c, label) for c in clips])
        #else:
        all_clip_paths.extend(clips)
        all_labels.extend([label] * len(clips))

    # Cap and shuffle non-accident clips
    #random.shuffle(non_accident_clips)
    #non_accident_clips = non_accident_clips[:1000]
    #for clip, label in non_accident_clips:
    #    all_clip_paths.append(clip)
    #    all_labels.append(label)

    return all_clip_paths, all_labels


def oversample_minority(clip_paths, labels, min_count=400):
    import random
    counts  = collections.Counter(labels)
    out_paths, out_labels = list(clip_paths), list(labels)
    for class_idx, count in counts.items():
        if count < min_count:
            indices = [i for i, l in enumerate(labels) if l == class_idx]
            extras  = random.choices(indices, k=min_count - count)
            out_paths  += [clip_paths[i] for i in extras]
            out_labels += [class_idx] * len(extras)
    combined = list(zip(out_paths, out_labels))
    random.shuffle(combined)
    paths, lbls = zip(*combined)
    return list(paths), list(lbls)



train_paths, train_labels = build_clip_paths(f'{DATA_ROOT}/train', stride=N_FRAMES // 2)
val_paths,   val_labels   = build_clip_paths(f'{DATA_ROOT}/val')
test_paths,  test_labels  = build_clip_paths(f'{DATA_ROOT}/test')

train_paths, train_labels = oversample_minority(train_paths, train_labels, min_count=400)

for split_name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
    print(f'{split_name} clips: {len(labels)}')
    # Use CLASS_NAMES to print counts for each class correctly
    class_counts = collections.Counter(labels)
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name}: {class_counts[i]}') 

MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def load_frame(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [HEIGHT, WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - MEAN) / STD
    return img


def load_clip(frame_paths: tf.Tensor, label: tf.Tensor):
    frames = tf.map_fn(load_frame, frame_paths, fn_output_signature=tf.float32)
    return frames, label


def augment_clip(frames, label):
    # Consistent horizontal flip
    flip = tf.random.uniform(()) > 0.5
    frames = tf.cond(flip, lambda: frames[:, :, ::-1, :], lambda: frames)

    # Stronger colour jitter
    frames = tf.image.random_brightness(frames, max_delta=0.2)
    frames = tf.image.random_contrast(frames, lower=0.8, upper=1.2)
    frames = tf.image.random_saturation(frames, lower=0.8, upper=1.2)
    frames = tf.image.random_hue(frames, max_delta=0.05)

    # Random temporal crop — forces model not to rely on fixed window
    start  = tf.random.uniform((), 0, 3, dtype=tf.int32)
    frames = frames[start : start + N_FRAMES]
    # Pad if needed
    pad = N_FRAMES - tf.shape(frames)[0]
    frames = tf.concat([frames, tf.repeat(frames[-1:], pad, axis=0)], axis=0)
    frames.set_shape([N_FRAMES, HEIGHT, WIDTH, 3])

    frames = tf.clip_by_value(frames, -3.0, 3.0)
    return frames, label

def make_dataset(clip_paths, labels, training=False, batch_size=BATCH_SIZE):
    path_tensor  = tf.ragged.constant(clip_paths).to_tensor()
    label_tensor = tf.constant(labels, dtype=tf.int32)
    # One-hot encode the labels
    label_tensor = tf.one_hot(label_tensor, depth=N_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))
    if training:
        ds = ds.shuffle(buffer_size=min(len(labels), 1000), reshuffle_each_iteration=True)
    ds = ds.map(load_clip, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment_clip, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds   = make_dataset(val_paths,   val_labels,   training=False)
test_ds  = make_dataset(test_paths,  test_labels,  training=False)

class_counts = collections.Counter(train_labels)
n_total      = len(train_labels)

max_weight = 6.0
class_weight = {
    i: min(n_total / (N_CLASSES * class_counts[i]), max_weight) if class_counts[i] > 0 else 1.0
    for i in range(N_CLASSES)
}

print('Class weights:')
for i, name in enumerate(CLASS_NAMES):
    print(f'  {name}: {class_weight[i]:.3f}')

class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', l2=1e-4):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        reg = regularizers.l2(l2)
        self.seq = keras.Sequential([
            layers.Conv3D(filters, kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding, kernel_regularizer=reg),
            layers.Conv3D(filters, kernel_size=(kernel_size[0], 1, 1),
                          padding=padding),
        ])

    def call(self, x):
        return self.seq(x)


class ResidualMain(keras.layers.Layer):
    def __init__(self, filters, kernel_size, spatial_dropout=0.1, l2=1e-4):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters, kernel_size, padding='same', l2=l2),
            layers.LayerNormalization(),
            layers.ReLU(),
            layers.SpatialDropout3D(spatial_dropout),
            Conv2Plus1D(filters, kernel_size, padding='same', l2=l2),
            layers.LayerNormalization(),
        ])

    def call(self, x):
        return self.seq(x)


class Project(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization(),
        ])

    def call(self, x):
        return self.seq(x)


def add_residual_block(x, filters, kernel_size, spatial_dropout=0.1, l2=1e-4):
    out = ResidualMain(filters, kernel_size, spatial_dropout=spatial_dropout, l2=l2)(x)
    res = x if out.shape[-1] == x.shape[-1] else Project(out.shape[-1])(x)
    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.resize = layers.Resizing(height, width)

    def call(self, video):
        shape  = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resize(images)
        return einops.rearrange(images, '(b t) h w c -> b t h w c', t=shape['t'])


def build_model(
    dropout_rate    = 0.3,
    spatial_dropout = 0.1,
    l2_reg          = 1e-4,
    base_filters    = 16,      
    learning_rate   = 1e-4,
    n_frames        = N_FRAMES,
    height          = HEIGHT,
    width           = WIDTH,
    num_classes     = N_CLASSES,
):
    f = base_filters
    inp = layers.Input(shape=(n_frames, height, width, 3))
    x   = inp

    x = Conv2Plus1D(f, (3, 7, 7), padding='same', l2=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    x = add_residual_block(x, f,      (3, 3, 3), spatial_dropout, l2=0.0)
    x = ResizeVideo(height // 4, width // 4)(x)

    x = add_residual_block(x, f * 2,  (3, 3, 3), spatial_dropout, l2=0.0)
    x = ResizeVideo(height // 8, width // 8)(x)

    x = add_residual_block(x, f * 4,  (3, 3, 3), spatial_dropout, l2_reg)
    x = ResizeVideo(height // 16, width // 16)(x)

    x = add_residual_block(x, f * 8,  (3, 3, 3), spatial_dropout, l2_reg)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes)(x)

    model = keras.Model(inp, x)
    model.compile(
        loss      = keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        metrics   = [keras.metrics.CategoricalAccuracy(name='accuracy')],
    )
    return model
model = build_model(base_filters=8, dropout_rate=0.4, spatial_dropout=0.2)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
]

history = model.fit(
    train_ds,
    epochs          = EPOCHS,
    validation_data = val_ds,
    class_weight    = class_weight,
    callbacks       = callbacks,
)


def plot_history(history):
    metrics = ['loss', 'accuracy']
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4))
    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric],          label='train')
        ax.plot(history.history[f'val_{metric}'], label='val')
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_xlabel('Epoch')
        ax.legend()
    plt.tight_layout()
    plt.savefig('accuracy_log.png',dpi=150)
    plt.show()

plot_history(history)

model.evaluate(test_ds, return_dict=True)


y_true_int, y_pred_int, y_pred_prob = [], [], []

for clips, labels in val_ds:
    probs = model(clips, training=False)
    probs = tf.nn.softmax(probs, axis=-1).numpy()
    y_pred_prob.append(probs)
    y_true_int.append(tf.argmax(labels, axis=1).numpy())
    y_pred_int.append(np.argmax(probs, axis=1))

y_true_int  = np.concatenate(y_true_int)
y_pred_int  = np.concatenate(y_pred_int)
y_pred_prob = np.concatenate(y_pred_prob)
y_true_oh   = np.eye(N_CLASSES)[y_true_int]

precision = precision_score(y_true_int, y_pred_int, average=None, zero_division=0)
recall    = recall_score(y_true_int, y_pred_int, average=None, zero_division=0)
avg_prec  = average_precision_score(y_true_oh, y_pred_prob, average=None)

print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'Avg Precision':>15}")
print("-" * 58)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<20} {precision[i]:>10.3f} {recall[i]:>10.3f} {avg_prec[i]:>15.3f}")

macro_ap = average_precision_score(y_true_oh, y_pred_prob, average='macro')
print(f"\n{'Mean AP (macro)':<20} {'':>10} {'':>10} {macro_ap:>15.3f}")
print("\n" + classification_report(y_true_int, y_pred_int, target_names=CLASS_NAMES, zero_division=0))

cm      = confusion_matrix(y_true_int, y_pred_int)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm], ['d', '.2f'],
    ['Confusion Matrix (counts)', 'Confusion Matrix (normalised)']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()


for layer in model.layers:
    print(layer.name, layer.output.shape)

TARGET_LAYER = 'add_3'

SAMPLE_CLIP, TRUE_LABEL = None, None
for paths, label in zip(val_paths, val_labels):
    frames_t = tf.stack([load_frame(tf.constant(p)) for p in paths])
    clip_t   = tf.expand_dims(frames_t, 0)
    pred     = tf.argmax(tf.nn.softmax(model(clip_t, training=False)[0])).numpy()
    if pred == label:
        SAMPLE_CLIP, TRUE_LABEL = paths, label
        break

grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer(TARGET_LAYER).output, model.output]
)

frames = tf.stack([load_frame(tf.constant(p)) for p in SAMPLE_CLIP])
clip   = tf.expand_dims(frames, 0)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(clip, training=False)
    probs     = tf.nn.softmax(predictions[0])
    class_idx = tf.argmax(probs).numpy()
    loss      = predictions[:, class_idx]

grads        = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2,3))

conv_out = conv_outputs[0]
heatmap  = conv_out @ pooled_grads[..., tf.newaxis]
heatmap  = tf.squeeze(heatmap)
heatmap  = tf.nn.relu(heatmap).numpy()
heatmap  = heatmap / (heatmap.max() + 1e-8)

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 6))
for t in range(len(frames)):
    rgb      = (frames[t].numpy() * STD + MEAN).clip(0, 1)
    heat_up  = cv2.resize(heatmap[t], (rgb.shape[1], rgb.shape[0]))
    map_t    = np.uint8(255 * (1 - heat_up))
    map_t    = cv2.applyColorMap(map_t, cv2.COLORMAP_JET)
    map_t    = cv2.cvtColor(map_t, cv2.COLOR_BGR2RGB) / 255.0
    feat_2d  = np.expand_dims(cv2.resize(heatmap[t], (rgb.shape[1], rgb.shape[0])), axis=2)
    overlay  = np.multiply(0.3 * rgb + 0.7 * map_t, feat_2d) + np.multiply(rgb, 1 - feat_2d)
    overlay  = overlay.clip(0, 1)

    axes[0, t].imshow(rgb);       axes[0, t].axis('off'); axes[0, t].set_title(f'Frame {t}')
    axes[1, t].imshow(overlay);   axes[1, t].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=11)
axes[1, 0].set_ylabel('CAM',      fontsize=11)
plt.suptitle(f'Predicted: {CLASS_NAMES[class_idx]}  |  True: {CLASS_NAMES[TRUE_LABEL]}')
plt.tight_layout()
plt.savefig('cam_output.png', dpi=150)
plt.show()

print(f"Predicted: {CLASS_NAMES[class_idx]} ({probs[class_idx]:.1%})")
print(f"True:      {CLASS_NAMES[TRUE_LABEL]}")
