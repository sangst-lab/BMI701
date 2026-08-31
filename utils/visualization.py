from pathlib import Path
import random

from PIL import Image, ImageDraw
import torch


def _line_points(values, box, y_min, y_max):
    left, top, right, bottom = box

    if len(values) == 1:
        x = (left + right) // 2
        y = (top + bottom) // 2
        return [(x, y)]

    y_range = y_max - y_min
    points = []
    for i, value in enumerate(values):
        x = left + int(i * (right - left) / (len(values) - 1))
        if y_range == 0:
            y = (top + bottom) // 2
        else:
            y = bottom - int((value - y_min) * (bottom - top) / y_range)
        points.append((x, y))
    return points


def save_training_progress(
    epochs,
    train_losses,
    validation_losses,
    validation_accuracies,
    output_path,
    show_panel=False,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 980
    height = 560
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)

    draw.text((32, 24), "Training Progress", fill="black")
    draw.text((32, 52), f"Epochs finished: {len(epochs)}", fill="black")

    loss_box = (70, 120, 610, 450)
    acc_box = (700, 120, 930, 450)

    for box, title in [(loss_box, "Loss"), (acc_box, "Validation Accuracy")]:
        left, top, right, bottom = box
        draw.rectangle(box, outline="black")
        draw.text((left, top - 28), title, fill="black")
        for step in range(1, 4):
            y = top + step * (bottom - top) // 4
            draw.line((left, y, right, y), fill=(225, 225, 225))

    loss_values = train_losses + validation_losses
    loss_min = min(loss_values)
    loss_max = max(loss_values)
    loss_margin = max((loss_max - loss_min) * 0.1, 0.05)
    loss_min -= loss_margin
    loss_max += loss_margin

    train_points = _line_points(train_losses, loss_box, loss_min, loss_max)
    valid_points = _line_points(validation_losses, loss_box, loss_min, loss_max)

    if len(train_points) > 1:
        draw.line(train_points, fill="blue", width=3)
        draw.line(valid_points, fill="red", width=3)
    for point in train_points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="blue")
    for point in valid_points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="red")

    acc_points = _line_points(validation_accuracies, acc_box, 0, 1)
    if len(acc_points) > 1:
        draw.line(acc_points, fill="green", width=3)
    for point in acc_points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="green")

    draw.text((70, 470), f"Epoch 1", fill="black")
    draw.text((560, 470), f"Epoch {epochs[-1]}", fill="black")
    draw.text((70, 500), "blue: train loss", fill="blue")
    draw.text((230, 500), "red: validation loss", fill="red")
    draw.text((420, 500), "green: validation accuracy", fill="green")

    draw.text((700, 470), "0%", fill="black")
    draw.text((700, 100), "100%", fill="black")

    draw.text(
        (700, 500),
        f"Last acc: {100 * validation_accuracies[-1]:.1f}%",
        fill="green",
    )
    draw.text(
        (700, 522),
        f"Last loss: train {train_losses[-1]:.4f}, valid {validation_losses[-1]:.4f}",
        fill="black",
    )

    panel.save(output_path)
    if show_panel:
        panel.show()
    print(f"Updated training progress panel: {output_path}")


def tensor_to_pil_image(image):
    image = image.detach().cpu().clamp(0, 1)

    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    image = (image * 255).to(torch.uint8)
    height, width = image.shape
    return Image.frombytes("L", (width, height), bytes(image.flatten().tolist()))


def show_random_predictions(
    model,
    dataset,
    classes,
    device,
    output_path,
    n_images=20,
    n_cols=5,
):
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), k=min(n_images, len(dataset)))
    n_rows = (len(indices) + n_cols - 1) // n_cols

    cell_width = 170
    cell_height = 150
    image_size = 96
    grid = Image.new("RGB", (n_cols * cell_width, n_rows * cell_height), "white")
    draw = ImageDraw.Draw(grid)

    with torch.no_grad():
        for position, dataset_index in enumerate(indices):
            image, true_label = dataset[dataset_index]
            pred = model(image.unsqueeze(0).to(device))
            pred_label = pred.argmax(1).item()

            image_pil = tensor_to_pil_image(image).resize((image_size, image_size))

            row = position // n_cols
            col = position % n_cols
            x = col * cell_width + 35
            y = row * cell_height + 8

            grid.paste(image_pil.convert("RGB"), (x, y))
            draw.text(
                (col * cell_width + 8, y + image_size + 8),
                f"True: {classes[true_label]}",
                fill="black",
            )
            draw.text(
                (col * cell_width + 8, y + image_size + 24),
                f"Pred: {classes[pred_label]}",
                fill="blue" if pred_label == true_label else "red",
            )

    grid.save(output_path)
    grid.show()
    print(f"Saved prediction examples to: {output_path}")
