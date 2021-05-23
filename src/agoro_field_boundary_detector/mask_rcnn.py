"""TODO: Documentation."""
import os
from pathlib import Path

import numpy as np

from src.agoro_field_boundary_detector.field_detection.dataset import Dataset
from src.agoro_field_boundary_detector.field_detection.model import FieldBoundaryDetector

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

model = FieldBoundaryDetector(model_path=Path(__file__).parent / "../../models/mask_rcnn")
dataset_train = Dataset(path=Path(__file__).parent / "../../data/augmented")

# Train
# model.train(
#         dataset=dataset_train,
# )

# Test
model.test(dataset=dataset_train, n_show=10, write_path=Path.home() / "Downloads")

with open(
    Path(__file__).parent / "../../data/augmented/fields/41.481412--88.8422360.npy", "rb"
) as f:
    field = np.load(f)
result = model(field)
print(result)

# model = get_instance_segmentation_model(2)
# dataset_train = Dataset(path=Path(__file__).parent / "../../data/augmented")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
# model.to(device)
# data_loader = torch.utils.data.DataLoader(
#     dataset_train,
#     batch_size=1,
#     shuffle=True,
#     num_workers=0,
#     collate_fn=lambda x: tuple(zip(*x)),
# )
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=0.005,
#     momentum=0.9,
#     weight_decay=0.0005,
# )
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=3,
#     gamma=0.1,
# )
# # let's train
# print()
# prev = 0
# for epoch in range(10):
#     # train for one epoch, printing every 10 iterations
#     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#
#     # update the learning rate
#     lr_scheduler.step()
#
#     # evaluate on the test dataset
#     f1 = evaluate(model, data_loader, device=device)
#     print(f"F1: {f1}")
#
#     # Stop if validation F1 starts to decrease
#     if prev > f1:
#         break
#     prev = f1
