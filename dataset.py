import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MNIST(Dataset):
    def __init__(self, X, Y):

        # Defining normalisation transform,
        self.norm_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])

        # Converting Y to tensors,
        self.Y = torch.tensor(Y, dtype = torch.long)
        del Y

        # Converting X to tensors and normalising,
        self.X = self.normalise(X)
        del X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def normalise(self, X):
        normalised_X = []
        for image in X:
            normalised_image = self.norm_trans(image)
            normalised_X.append(normalised_image)
            del image
        return torch.stack(normalised_X)