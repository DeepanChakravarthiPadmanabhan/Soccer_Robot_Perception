import torch

def seg_label_preprocessor(label_mask):
    label_mask = torch.sum(torch.tensor(label_mask, dtype=torch.float), dim=2)
    label_mask[label_mask == 0] = 0  # Background
    label_mask[label_mask == 128.] = 1  # Field
    label_mask[label_mask == 256.] = 2  # Lines
    label_mask.unsqueeze_(2)
    label_mask = label_mask.expand(label_mask.shape[0], label_mask.shape[1], 3)

    return label_mask