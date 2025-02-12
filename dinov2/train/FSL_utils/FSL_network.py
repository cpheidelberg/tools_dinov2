

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.z_proto = None
    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        if self.z_proto is None:
            raise ValueError("Support Prototypes need to be initialized")

        z_proto = self.z_proto

#        with torch.inference_mode():
#            z_query = self.backbone.encode_image(query_images, proj_contrast=False, normalize=False)
        z_query  = self.backbone(query_images)
       # print(z_query.shape)
        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query.to(dtype=z_proto.dtype), z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        # Compute Gaussian probabilities
        probs = F.softmax(scores, dim=1)

        return scores, probs



    def return_prototypes(self):
        return self.z_proto

    def change_prototypes_device(self,device):
        self.z_proto = self.z_proto.to(device)

    def return_features(
        self,
        support_images: torch.Tensor,
    ) -> torch.Tensor:

        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
     #   with torch.inference_mode():
     #       z_support = self.backbone.encode_image(support_images, proj_contrast=False, normalize=False)

        return z_support

    def save_model(self, filepath: str):
        """
        Save the model to a file.

        Parameters:
            filepath (str): Path to save the model.
        """
        torch.save(self.backbone.state_dict(), filepath)

    def predict_labels(self, query_images: torch.Tensor) -> torch.Tensor:
        """
        Predict the labels for the query images based on the nearest prototype.

        Parameters:
            query_images (torch.Tensor): Tensor of query images.

        Returns:
            torch.Tensor: Tensor of predicted labels for the query images.
        """
        # Get the scores and probabilities from the forward pass
        scores, probs = self.forward(query_images)

        # Get the index of the max probability for each query image (this is the predicted label)
        _, predicted_labels=  torch.max(scores.data, 1)

        return predicted_labels, probs

    def make_prediction(self, loader, device):
        # Initialize an array to store features, predicted labels, and probabilities
        predictions_output = np.zeros((len(loader.dataset), 1026))  # Adjust size as per dataset

        # Counter to track the current position in the output array
        n = 0

        # Iterate through the DataLoader
        with torch.no_grad():
            for imgs, labels in loader:
                # Move images to the device (e.g., GPU) and ensure correct dtype
                imgs_gpu = imgs.to(torch.float32).to(device)

                # Get predictions and probabilities from the model
                pred_labels, probs = self.predict_labels(imgs_gpu)

                # Get feature vectors from the model
                features = self.return_features(imgs_gpu)

                # Convert tensors to NumPy arrays
                pred_labels = pred_labels.cpu().detach().numpy()
                probs = probs.cpu().detach().numpy()
                features = features.cpu().detach().numpy()

                # Calculate the batch size
                batch_size = len(imgs)

                # Store features in the first 387 columns
                predictions_output[n:n + batch_size, :-2] = features

                # Store predicted labels in the second-to-last column
                predictions_output[n:n + batch_size, -2] = pred_labels.astype(np.int32)

                # Store probabilities in the last column
                predictions_output[n:n + batch_size, -1] = probs[:,0]

                # Update the position index
                n += batch_size
                torch.cuda.empty_cache()

        return predictions_output

    def calculate_prototypes(self, support_loader,device):

        # Initialize an array to store features, predicted labels, and probabilities
        predictions_output = np.zeros((len(support_loader.dataset), 1025))  # Adjust size as per dataset

        # Counter to track the current position in the output array
        n = 0
        # Iterate through the DataLoader
        with torch.no_grad():
            for imgs, labels in support_loader:
                # Move images to the device (e.g., GPU) and ensure correct dtype
                imgs_gpu = imgs.to(torch.float32).to(device)

                # Get feature vectors from the model
                features = self.return_features(imgs_gpu)


                features = features.cpu().detach().numpy()

                # Calculate the batch size
                batch_size = len(imgs)

                # Store features in the first 387 columns
                predictions_output[n:n + batch_size,:-1] = features
                predictions_output[n:n + batch_size,-1] = labels.cpu().detach().numpy()

                # Update the position index
                n += batch_size
                torch.cuda.empty_cache()



        # Encode the support images using the backbone network
     #   with torch.inference_mode():
     #       z_support = self.backbone.encode_image(support_images, proj_contrast=False, normalize=False)
        z_support = torch.from_numpy(predictions_output[:,:-1])
        support_labels = torch.from_numpy(predictions_output[:,-1])

        # Get the number of classes (n_way)
        n_way = len(torch.unique(support_labels))

        # Calculate prototypes using the median
        self.z_proto = torch.cat(
            [
                torch.median(z_support[torch.nonzero(support_labels == label)], dim=0)[0]
                for label in range(n_way)
            ]
        ).to(torch.float32).to(device)
        

def evaluate_on_one_task(
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    model,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """

   # model.calculate_prototypes(support_images, support_labels)    

    query_labels_scores, _ = model( query_images)

    correct = (
        torch.max(
            model( query_images)[0]
            .detach()
            .data,
            1,
        )[1]
        == query_labels
    ).sum().item()

    _, query_labels_predicted = torch.max(query_labels_scores.data, 1)
    return correct, len(query_labels), query_labels_predicted




