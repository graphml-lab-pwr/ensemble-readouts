from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class OneHotEncoder(BaseTransform):
    """One-hot encodes ALL features of the given data."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, data: Data) -> Data:
        data.x = one_hot(data.x.flatten(), num_classes=self.num_classes).float()
        return data


class LabelShapeAdjust(BaseTransform):
    """Adjusts label shape."""

    def __init__(self, is_output_binary: bool):
        self.is_output_binary = is_output_binary

    def __call__(self, data: Data) -> Data:
        if self.is_output_binary:
            data.y = data.y.reshape(-1, 1)  # reshape binary/regression labels
        else:
            data.y = data.y.flatten()  # flat multiclass labels

        return data
