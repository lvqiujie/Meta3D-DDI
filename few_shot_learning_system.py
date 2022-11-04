import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.model import *
from utils.dataset_tools import do_compute_metrics
from inner_loop_optimizers import LSLRGradientDescentLearningRule


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)
    random.seed(torch_seed)
    np.random.seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, device, args):
        """
        Initializes a MAML few shot learning system
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)

        if 'drugbank' in self.args.dataset_name or 'twoside' in self.args.dataset_name :
            self.classifier = myModel_graph_sch_cnn(args= args,
                                                    num_class=args.num_class,
                                    cutoff=args.cutoff,
                                    num_layers=args.num_layers, hidden_channels=args.hidden_channels,
                                    num_filters=args.num_filters, num_gaussians=args.num_gaussians,
                                    g_out_channels=args.g_out_channels,
                                    device=self.device).to(device=self.device)
        # else:
        #     self.classifier = myModel_text_graph_pos_cnn_sch(model_name=args.model_name,
        #                                num_class=args.num_class,
        #                                cutoff=args.cutoff,
        #                                num_layers=args.num_layers, hidden_channels=args.hidden_channels,
        #                                num_filters=args.num_filters, num_gaussians=args.num_gaussians,
        #                                g_out_channels=args.g_out_channels,
        #                                device=self.device
        #                                )

        self.task_learning_rate = args.init_inner_loop_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.classifier.named_parameters()))

        # print("Inner Loop parameters")
        # for key, value in self.inner_loop_optimizer.named_parameters():
        #     print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        # print("Outer Loop parameters")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape, param.device, param.requires_grad)


        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param.to(device=self.device)
            for name, param in params
            if param.requires_grad
            and (
                not self.args.enable_inner_loop_optimizable_bn_params
                and "norm_layer" not in name
                or self.args.enable_inner_loop_optimizable_bn_params
            )
        }

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)


        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}


        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies, total_auc, total_f1,
                                                            total_precision, total_recall, total_ap):
        losses = {'loss': torch.mean(torch.stack(total_losses))}

        losses['accuracy'] = np.mean(total_accuracies)
        losses['auc'] = np.mean(total_auc)
        losses['f1'] = np.mean(total_f1)
        losses['precision'] = np.mean(total_precision)
        losses['recall'] = np.mean(total_recall)
        losses['ap'] = np.mean(total_ap)

        return losses

    def get_across_task_loss_metrics2(self, total_losses, total_accuracies):
        losses = {'loss': torch.mean(torch.stack(total_losses))}

        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        s_head_list, s_tail_list, s_label_list, s_rel_list, \
        q_head_list, q_tail_list, q_label_list, q_rel_list = data_batch

        num_classes = len(s_head_list)

        self.num_classes_per_set = num_classes

        total_losses = []
        total_accuracies = []
        total_auc = []
        total_f1 = []
        total_precision = []
        total_recall = []
        total_ap = []
        per_task_target_preds = [[] for i in range(num_classes)]
        self.classifier.zero_grad()
        task_accuracies = []
        for task_id, (s_head_list_task, s_tail_list_task, s_label_list_task, s_rel_lis_taskt,
                              q_head_list_task, q_tail_list_task, q_label_list_task, q_rel_list_task) in enumerate(
                                        zip(s_head_list, s_tail_list, s_label_list, s_rel_list,
                                            q_head_list, q_tail_list, q_label_list, q_rel_list)):
            # print("task_id :   ",task_id)
            x_support_set_task = (s_head_list_task, s_tail_list_task, s_rel_lis_taskt)
            y_support_set_task = s_label_list_task.float().to(device=self.device)
            x_target_set_task = (q_head_list_task, q_tail_list_task, q_rel_list_task)
            y_target_set_task = q_label_list_task.float().to(device=self.device)
            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            # n, s, c, h, w = x_target_set_task.shape
            #
            # x_support_set_task = x_support_set_task.view(-1, c, h, w)
            # y_support_set_task = y_support_set_task.view(-1)
            # x_target_set_task = x_target_set_task.view(-1, c, h, w)
            # y_target_set_task = y_target_set_task.view(-1)

            for num_step in range(num_steps):

                support_loss, support_preds = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=num_step == 0,
                    training=True,
                    num_step=num_step,
                )


                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)
                    task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            # _, predicted = torch.max(target_preds.data, 1)
            predicted = (torch.sigmoid(target_preds.detach()) > 0.5).float()
            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()

            _, auroc, f1_score, precision, recall, ap = do_compute_metrics(torch.sigmoid(target_preds).detach().cpu().numpy(),
                                                                                  y_target_set_task.cpu().numpy())
            # if auroc == 0 or f1_score == 0 or precision == 0 or recall == 0 or ap == 0:
            #     continue

            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)


            total_accuracies.extend(accuracy)



            total_auc.append(auroc)
            total_f1.append(f1_score)
            total_precision.append(precision)
            total_recall.append(recall)
            total_ap.append(ap)

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies,
                                                   total_auc=total_auc,
                                                   total_f1=total_f1,
                                                   total_precision=total_precision,
                                                   total_recall=total_recall,
                                                   total_ap=total_ap,
                                                   )


        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)

        # loss = F.cross_entropy(input=preds, target=y)
        loss = F.binary_cross_entropy_with_logits(input=preds, target=y)
        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds


    def zero_forward_prop(self, data_batch, epoch):
        s_head_list, s_tail_list, s_label_list, s_rel_list, \
        q_head_list, q_tail_list, q_label_list, q_rel_list = data_batch

        num_classes = self.args.num_classes_per_set
        num_steps = self.args.number_of_training_steps_per_iter
        total_losses = []
        total_accuracies = []
        total_auc = []
        total_f1 = []
        total_precision = []
        total_recall = []
        total_ap = []
        per_task_target_preds = [[] for i in range(num_classes)]
        task_accuracies = []
        for task_id, (q_head_list_task, q_tail_list_task, q_label_list_task, q_rel_list_task) in enumerate(
                                        zip(q_head_list, q_tail_list, q_label_list, q_rel_list)):

            x_target_set_task = (q_head_list_task, q_tail_list_task, q_rel_list_task)
            y_target_set_task = q_label_list_task.float().to(device=self.device)
            task_losses = []

            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            for num_step in range(num_steps):
                target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                             y=y_target_set_task, weights=names_weights_copy,
                                                             backup_running_statistics=False, training=True,
                                                             num_step=num_step)
                task_losses.append(target_loss)


            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            # _, predicted = torch.max(target_preds.data, 1)

            predicted = torch.sigmoid(target_preds)
            accuracy, auroc, f1_score, precision, recall, ap = do_compute_metrics(predicted.detach().cpu().numpy(),
                                                                                  y_target_set_task.data.float().cpu().numpy())

            # predicted1 = (torch.sigmoid(target_preds.detach()) > 0.5).float()
            # accuracy1 = predicted1.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.append(accuracy)
            total_auc.append(auroc)
            total_f1.append(f1_score)
            total_precision.append(precision)
            total_recall.append(recall)
            total_ap.append(ap)

            # if not training_phase:
            #     self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies,
                                                   total_auc=total_auc,
                                                   total_f1=total_f1,
                                                   total_precision=total_precision,
                                                   total_recall=total_recall,
                                                   total_ap=total_ap,
                                                   )

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

    # def run_train_iter(self, data_batch, epoch):
    #     """
    #     Runs an outer loop update step on the meta-model's parameters.
    #     :param data_batch: input data batch containing the support set and target set input, output pairs
    #     :param epoch: the index of the current epoch
    #     :return: The losses of the ran iteration.
    #     """
    #     epoch = int(epoch)
    #     self.scheduler.step(epoch=epoch)
    #     if self.current_epoch != epoch:
    #         self.current_epoch = epoch
    #
    #     if not self.training:
    #         self.train()
    #
    #     s_head_list, s_tail_list, s_label_list, s_rel_list, \
    #     q_head_list, q_tail_list, q_label_list, q_rel_list = data_batch
    #
    #     # x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
    #     # x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
    #     # y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
    #     # y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)
    #     #
    #     # data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)
    #
    #     losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)
    #
    #     self.meta_update(loss=losses['loss'])
    #     losses['learning_rate'] = self.scheduler.get_lr()[0]
    #     self.optimizer.zero_grad()
    #     self.zero_grad()
    #
    #     return losses, per_task_target_preds

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        s_head_list, s_tail_list, s_label_list, s_rel_list, \
        q_head_list, q_tail_list, q_label_list, q_rel_list = data_batch

        stacked_loss = None
        stacked_acc = None
        stacked_auc = None
        stacked_f1 = None
        stacked_precision = None
        stacked_recall = None
        stacked_ap = None
        per_task_target_preds = None
        self.optimizer.zero_grad()

        for nt in range(self.args.batch_size):
            # print("batch_size  :", nt)
            s_head_list_t = s_head_list[nt]
            s_tail_list_t = s_tail_list[nt]
            s_label_list_t = s_label_list[nt]
            s_rel_list_t = s_rel_list[nt]

            q_head_list_t = q_head_list[nt]
            q_tail_list_t = q_tail_list[nt]
            q_label_list_t = q_label_list[nt]
            q_rel_list_t = q_rel_list[nt]

            data_batch = (s_head_list_t, s_tail_list_t, s_label_list_t, s_rel_list_t,
                          q_head_list_t, q_tail_list_t, q_label_list_t, q_rel_list_t)

            losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)
            self.meta_update(loss=losses['loss'])
            # print("loss ", losses['loss'])
            if stacked_loss is None:
                stacked_loss = losses['loss'].detach()
                stacked_acc = losses['accuracy']
                stacked_auc = losses['auc']
                stacked_f1 = losses['f1']
                stacked_precision = losses['precision']
                stacked_recall = losses['recall']
                stacked_ap = losses['ap']
            else:
                stacked_loss = torch.cat((stacked_loss.view(1, -1), losses['loss'].detach().view(1, -1)), 0)
                stacked_acc = np.concatenate((stacked_acc.reshape(1, -1), losses['accuracy'].reshape(1, -1)), 0)
                stacked_auc = np.concatenate((stacked_auc.reshape(1, -1), losses['auc'].reshape(1, -1)), 0)
                stacked_f1 = np.concatenate((stacked_f1.reshape(1, -1), losses['f1'].reshape(1, -1)), 0)
                stacked_precision = np.concatenate((stacked_precision.reshape(1, -1), losses['precision'].reshape(1, -1)), 0)
                stacked_recall = np.concatenate((stacked_recall.reshape(1, -1), losses['recall'].reshape(1, -1)), 0)
                stacked_ap = np.concatenate((stacked_ap.reshape(1, -1), losses['ap'].reshape(1, -1)), 0)

        # self.meta_update(loss=losses['loss'])
        losses['loss'] = torch.mean(stacked_loss).item()
        losses['accuracy'] = np.mean(stacked_acc)
        losses['auc'] = np.mean(stacked_auc)
        losses['f1'] = np.mean(stacked_f1)
        losses['precision'] = np.mean(stacked_precision)
        losses['recall'] = np.mean(stacked_recall)
        losses['ap'] = np.mean(stacked_ap)
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        s_head_list, s_tail_list, s_label_list, s_rel_list, \
        q_head_list, q_tail_list, q_label_list, q_rel_list = data_batch

        target_preds = []
        stacked_loss = None
        stacked_acc = None
        for nt in range(self.args.batch_size):
            # print("batch_size  :", nt)
            s_head_list_t = s_head_list[nt]
            s_tail_list_t = s_tail_list[nt]
            s_label_list_t = s_label_list[nt]
            s_rel_list_t = s_rel_list[nt]

            q_head_list_t = q_head_list[nt]
            q_tail_list_t = q_tail_list[nt]
            q_label_list_t = q_label_list[nt]
            q_rel_list_t = q_rel_list[nt]

            data_batch = (s_head_list_t, s_tail_list_t, s_label_list_t, s_rel_list_t,
                          q_head_list_t, q_tail_list_t, q_label_list_t, q_rel_list_t)

            if len(s_head_list_t) == 0:
                print("!!!!!!!!!!!!!!!  zero  shot !!!!!!!!!!!!!!!!!!!!!! ")
                losses, per_task_target_preds = self.zero_forward_prop(data_batch=data_batch, epoch=self.current_epoch)
            else:
                losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

            target_preds.extend(np.array(per_task_target_preds).reshape(-1))
            if stacked_loss is None:
                stacked_loss = losses['loss'].detach()
                stacked_acc = losses['accuracy']
            else:
                stacked_loss = torch.cat((stacked_loss.view(1, -1), losses['loss'].detach().view(1, -1)), 0)
                stacked_acc = np.concatenate((stacked_acc.reshape(1, -1), losses['accuracy'].reshape(1, -1)), 0)

        losses['loss'] = torch.mean(stacked_loss).item()
        losses['accuracy'] = np.mean(stacked_acc)


        # x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        # x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        # y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        # y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        # data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        # losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.optimizer.load_state_dict(state['optimizer'])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
