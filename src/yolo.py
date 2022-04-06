import torch

yolo_model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="models/yolov5_best.pt"
)


def get_target_point(image_filename):
    results = yolo_model(image_filename)
    xyxy = results.xyxy
    if len(xyxy) < 1:
        raise ValueError("Target not found")

    el = results.xyxy[0]
    xmin, ymin = el[0, :2]
    xmax, ymax = el[0, 2:4]
    x = torch.tensor([xmin, xmax]).mean()
    y = torch.tensor([ymin, ymax]).mean()
    # target_point = (x.tolist()), y.tolist())
    return torch.tensor([x, y]).to(torch.int32).numpy()


# image_filename = 'images/hand3.jpg'
# results = model(img)
