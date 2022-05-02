import torch
import torch.nn.functional as F


def custom_objective(y_pred, y_true, lambda_smoothness, lambda_sparsity):
    # y_pred (batch_size, 32, 1)
    # y_true (batch_size)

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    # (batch/2, 32, 1)
    normal_segments_scores = y_pred[normal_vids_indices].squeeze(-1)
    # (batch/2, 32, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices].squeeze(-1)

    if normal_segments_scores.dim() == 3:
        # aggregate over the crops
        normal_segments_scores = normal_segments_scores.mean(dim=-1)
        anomal_segments_scores = anomal_segments_scores.mean(dim=-1)

    # get the max score for each video
    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

    hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    """
    Smoothness of anomalous video
    """
    smoothed_scores = anomal_segments_scores[:,
                                             1:] - anomal_segments_scores[:, :-1]
    smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (hinge_loss + lambda_smoothness *
                  smoothed_scores_sum_squared + lambda_sparsity * sparsity_loss).mean()
    return final_loss


class RegularizedMIL(torch.nn.Module):
    def __init__(self, model, original_objective=custom_objective, lambda_regularization=0.001,
                 lambda_smoothness=8e-5, lambda_sparsity=0.0):
        super(RegularizedMIL, self).__init__()
        self.lambdas = lambda_regularization
        self.model = model
        self.objective = original_objective
        self.lambda_smoothness = lambda_smoothness
        self.lambda_sparsity = lambda_sparsity

    def forward(self, y_pred, y_true):
        fc_params = []
        for layer in self.model.classifier:
            if isinstance(layer, torch.nn.Linear):
                fc_params.append([x.view(-1) for x in layer.parameters()])

        fc1_params = torch.cat(tuple(fc_params[0]))
        fc2_params = torch.cat(tuple(fc_params[1]))

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = 0.0
        if len(fc_params) == 3:
            fc3_params = torch.cat(tuple(fc_params[2]))
            l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

        return self.objective(y_pred, y_true, self.lambda_smoothness, self.lambda_sparsity) \
            + l1_regularization + l2_regularization + l3_regularization


class MIL_loss(torch.nn.Module):
    def __init__(self, lambda_smoothness=8e-5, lambda_sparsity=.0):
        super(MIL_loss, self).__init__()
        self.lambda_smoothness = lambda_smoothness
        self.lambda_sparsity = lambda_sparsity

    def forward(self, y_pred, y_true):

        normal_vids_indices = torch.where(y_true == 0)
        anomal_vids_indices = torch.where(y_true == 1)

        normal_segments_scores = y_pred[normal_vids_indices].squeeze(-1)
        anomal_segments_scores = y_pred[anomal_vids_indices].squeeze(-1)

        normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
        anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

        hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
        hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

        """
        Smoothness of anomalous video
        """
        smoothed_scores = anomal_segments_scores[:, 1:] - \
            anomal_segments_scores[:, :-1]
        smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

        """
        Sparsity of anomalous video
        """
        sparsity_loss = anomal_segments_scores.sum(dim=-1)

        final_loss = (hinge_loss + self.lambda_smoothness * smoothed_scores_sum_squared +
                      self.lambda_sparsity * sparsity_loss).mean()
        return final_loss
