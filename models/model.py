from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn.functional import selu
from models.SchNet import MetaSchNet
from models.meta_model import *
from torch import nn
import random
from torch.nn import Embedding, Sequential, Linear


class myModel_graph_sch_cnn(nn.Module):
    def __init__(self, args, device, num_class=2, cutoff = 10.0,num_layers = 6,hidden_channels = 128,
                 num_filters = 128,num_gaussians = 50, g_out_channels = 64):
        super(myModel_graph_sch_cnn, self).__init__()
        self.args = args
        self.device = device
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.model1 = MetaSchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters, num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)
        self.model2 = MetaSchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters, num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)

        self.fc1 = MetaLinearPReLULayer(g_out_channels, g_out_channels * 2, use_bias=True)
        self.fc2 = MetaLinearLayer(g_out_channels * 2, g_out_channels, use_bias=True)
        self.fc3 = MetaLinearPReLULayer(g_out_channels, g_out_channels * 2, use_bias=True)
        self.fc4 = MetaLinearLayer(g_out_channels * 2, g_out_channels, use_bias=True)

        self.cnn = MetaCNN_g(in_channel=g_out_channels, out_channel=64)

        self.rel_dim = nn.Parameter(torch.Tensor(self.args.class_num, 64))
        nn.init.xavier_uniform_(self.rel_dim)


    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        s_head_list_task, s_tail_list_task, s_rel_lis_taskt = x
        s_rel_lis_taskt = s_rel_lis_taskt.to(device=self.device)
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        pos1 = s_head_list_task.pos.to(device=self.device)
        z1 = s_head_list_task.z.to(device=self.device)
        batch1 = s_head_list_task.batch.to(device=self.device)
        self.pred1 = self.model1(pos1, z1, batch1, param_dict["model1"])
        pos2 = s_tail_list_task.pos.to(device=self.device)
        z2 = s_tail_list_task.z.to(device=self.device)
        batch2 = s_tail_list_task.batch.to(device=self.device)
        self.pred2 = self.model2(pos2, z2, batch2, param_dict["model2"])

        self.pred1 = self.fc1(self.pred1, param_dict["fc1"])
        self.pred1 = self.fc2(self.pred1, param_dict["fc2"])


        self.pred2 = self.fc3(self.pred2, param_dict["fc3"])
        self.pred2 = self.fc4(self.pred2, param_dict["fc4"])
        self.pred1 = self.pred1.unsqueeze(1)
        self.pred2 = self.pred2.unsqueeze(1)
        self.pred = torch.cat((self.pred1, self.pred2), 1)
        out = self.cnn(self.pred, param_dict["cnn"])

        rel_data = F.embedding(s_rel_lis_taskt, param_dict["rel_dim"])
        out = (out * rel_data).sum(-1)
        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            # print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    # aa = param.grad
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            # print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class MetaCNN_g(nn.Module):
    def __init__(self, in_channel=64, out_channel=2):
        super(MetaCNN_g, self).__init__()
        fc1_hid_dim = in_channel * in_channel
        self.conv1 = MetaConv1dLayer(2, in_channel, kernel_size=3, padding=1)
        self.conv2 = MetaConv1dLayer(in_channel, in_channel*2, kernel_size=3, padding=1)
        self.conv31 = MetaConv1dLayer(in_channel*2, in_channel*2, kernel_size=3, padding=1)
        self.conv32 = MetaConv1dLayer(in_channel*2, in_channel*2, kernel_size=3, padding=1)
        self.conv4 = MetaConv1dLayer(in_channel*2, in_channel, kernel_size=3, padding=1)
        self.fc1 = MetaLinearLayer(fc1_hid_dim, in_channel)
        self.fc2 = MetaLinearLayer(in_channel, out_channel)

    def forward(self, x, params=None):
        param_dict = dict()

        if params is not None:
            #param_dict = parallel_extract_top_level_dict(current_dict=params)
            param_dict = extract_top_level_dict(current_dict=params)
        else:
            param_dict["conv1"] = None
            param_dict["conv2"] = None
            param_dict["conv31"] = None
            param_dict["conv32"] = None
            param_dict["conv4"] = None
            param_dict["fc1"] = None
            param_dict["fc2"] = None


        x = F.leaky_relu_(self.conv1(x, param_dict["conv1"]))  # batchsize *2 * 32 -> batchsize *64 * 32
        x = F.leaky_relu_(self.conv2(x, param_dict["conv2"]))  # batchsize *128 * 32
        res = x
        x = F.leaky_relu_(self.conv31(x, param_dict["conv31"]))  # batchsize *128 * 32
        x = F.leaky_relu_(self.conv32(x, param_dict["conv32"]))  # batchsize *128 * 32
        x = res + x  # batchsize *128 * 32
        x = F.leaky_relu_(self.conv4(x, param_dict["conv4"]))  # batchsize *256 * 32
        x = F.leaky_relu_(self.fc1(x.view(x.shape[0], -1), param_dict["fc1"]))  # batchsize * 64
        x = self.fc2(x, param_dict["fc2"])  # batchsize * out_channel 输出通道数代表预测的类别数量 根据任务的分类类别来确定  ddi2013里面是5 drugbank里面是2

        return x



class myModel_text_cnn(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_class=2, freeze_bert=False,
                 max_len=128):  # , freeze_bert=False   ,model_name):
        super(myModel_text_cnn, self).__init__()
        self.max_len = max_len
        self.bert = AutoModel.from_pretrained(model_name, cache_dir='../cache', output_hidden_states=True,
                                              return_dict=True)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size * 6, num_class, bias=False)
        )
        self.cnn = CNN()

    def forward(self, batch_data):
        outputs = self.bert(input_ids=batch_data.token_ids.view(-1, self.max_len),
                            token_type_ids=batch_data.token_type_ids.view(-1, self.max_len),
                            attention_mask=batch_data.attn_masks.view(-1, self.max_len))
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4, -5, -6]]),
                                  dim=-1).view(outputs.hidden_states[-1].shape[0], -1,
                                               outputs.hidden_states[-1].shape[1],
                                               outputs.hidden_states[-1].shape[-1])  # [bs, seq_len, hidden_dim*6]
        self.pred = self.cnn(hidden_states)
        return self.pred




class myModel_text_pos_cnn(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_class=2, freeze_bert=False, max_len=128,
                 emb_dim=64):  # , freeze_bert=False   ,model_name):
        super(myModel_text_pos_cnn, self).__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.bert = AutoModel.from_pretrained(model_name, cache_dir='../cache', output_hidden_states=True,
                                              return_dict=True)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size * 6, num_class, bias=False)
        )

        self.emb = nn.Embedding(self.max_len + 1, self.emb_dim)

        self.fc_emb = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.emb_dim * 2, 32 * 2, bias=False),
            nn.PReLU(),
            nn.Linear(32 * 2, num_class, bias=False)
        )
        self.cnn = CNN()

    def forward(self, batch_data):
        outputs = self.bert(input_ids=batch_data.token_ids.view(-1, self.max_len),
                            token_type_ids=batch_data.token_type_ids.view(-1, self.max_len),
                            attention_mask=batch_data.attn_masks.view(-1, self.max_len))
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4, -5, -6]]),
                                  dim=-1).view(outputs.hidden_states[-1].shape[0], -1,
                                               outputs.hidden_states[-1].shape[1],
                                               outputs.hidden_states[-1].shape[-1])  # [bs, seq_len, hidden_dim*6]
        logits = self.cnn(hidden_states)

        drug1_pos = batch_data.drug1_pos
        drug2_pos = batch_data.drug2_pos
        drug1_pos[drug1_pos == -1] = self.max_len
        drug2_pos[drug2_pos == -1] = drug1_pos[drug2_pos == -1]
        self.emb1 = self.emb(drug1_pos)
        self.emb2 = self.emb(drug2_pos)
        self.emb_cat = torch.cat((self.emb1, self.emb2), 1)
        self.emb_cat = self.fc_emb(self.emb_cat)

        self.pred = (logits + 0.1 * self.emb_cat) / 1.1
        return self.pred


class MetaStepLossNetwork(nn.Module):
    def __init__(self, input_dim, args, device):
        super(MetaStepLossNetwork, self).__init__()

        self.device = device
        self.args = args
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x

        self.linear1 = MetaLinearLayer(in_features=self.input_dim,
                                       num_filters=self.input_dim, use_bias=True)

        self.linear2 = MetaLinearLayer(in_features=self.input_dim,
                                       num_filters=1, use_bias=True)

        out = self.linear1(out)
        out = F.relu_(out)
        out = self.linear2(out)

    def forward(self, x, params=None):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """

        linear1_params = None
        linear2_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            linear1_params = params['linear1']
            linear2_params = params['linear2']

        out = x

        out = self.linear1(out, linear1_params)
        out = F.relu_(out)
        out = self.linear2(out, linear2_params)

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()

class MetaLossNetwork(nn.Module):
    def __init__(self, input_dim, args, device):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(MetaLossNetwork, self).__init__()

        self.device = device
        self.args = args
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)

        self.num_steps = args.number_of_training_steps_per_iter  # number of inner-loop steps

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_steps):
            self.layer_dict['step{}'.format(i)] = MetaStepLossNetwork(self.input_dim, args=self.args,
                                                                      device=self.device)

            out = self.layer_dict['step{}'.format(i)](x)

    def forward(self, x, num_step, params=None):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        out = self.layer_dict['step{}'.format(num_step)](out, param_dict['step{}'.format(num_step)])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


class StepLossAdapter(nn.Module):
    def __init__(self, input_dim, num_loss_net_layers, args, device):
        super(StepLossAdapter, self).__init__()

        self.device = device
        self.args = args
        output_dim = num_loss_net_layers * 2 * 2  # 2 for weight and bias, another 2 for multiplier and offset

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(input_dim, output_dim)

        self.multiplier_bias = nn.Parameter(torch.zeros(output_dim // 2))
        self.offset_bias = nn.Parameter(torch.zeros(output_dim // 2))

    def forward(self, task_state, num_step, loss_params):

        out = self.linear1(task_state)
        out = F.relu_(out)
        out = self.linear2(out)

        generated_multiplier, generated_offset = torch.chunk(out, chunks=2, dim=-1)

        i = 0
        updated_loss_weights = dict()
        for key, val in loss_params.items():
            if 'step{}'.format(num_step) in key:
                updated_loss_weights[key] = (1 + self.multiplier_bias[i] * generated_multiplier[i]) * val + \
                                            self.offset_bias[i] * generated_offset[i]
                i += 1

        return updated_loss_weights


class LossAdapter(nn.Module):
    def __init__(self, input_dim, num_loss_net_layers, args, device):
        super(LossAdapter, self).__init__()

        self.device = device
        self.args = args

        self.num_steps = args.number_of_training_steps_per_iter  # number of inner-loop steps

        self.loss_adapter = nn.ModuleList()
        for i in range(self.num_steps):
            self.loss_adapter.append(StepLossAdapter(input_dim, num_loss_net_layers, args=args, device=device))

    def forward(self, task_state, num_step, loss_params):
        return self.loss_adapter[num_step](task_state, num_step, loss_params)

class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha_t is not None and self.gamma == 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss
