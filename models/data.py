import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle, joblib
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from torch_geometric.data import Data, Batch
import pandas as pd


class FewShotLearningDatasetParallel(Dataset):
    def __init__(self, args):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.data_loaded_in_memory = False

        self.args = args
        self.indexes_of_folders_indicating_class = args.indexes_of_folders_indicating_class
        self.reverse_channels = args.reverse_channels
        self.labels_as_int = args.labels_as_int
        self.train_val_test_split = args.train_val_test_split
        self.current_set_name = "train"
        self.num_target_samples = args.num_target_samples
        self.reset_stored_filepaths = args.reset_stored_filepaths
        val_rng = np.random.RandomState(seed=args.val_seed)
        val_seed = val_rng.randint(1, 999999)
        train_rng = np.random.RandomState(seed=args.train_seed)
        train_seed = train_rng.randint(1, 999999)
        test_rng = np.random.RandomState(seed=args.val_seed)
        test_seed = test_rng.randint(1, 999999)
        test_rng2 = np.random.RandomState(seed=args.val_seed)
        test2_seed = test_rng2.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed
        args.test2_seed = test2_seed
        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test1': args.test_seed, 'test2': args.test2_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test1': args.test_seed, 'test2': args.test2_seed}
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set

        self.rng = np.random.RandomState(seed=self.seed['val'])

        self.pass_smiles = set()
        self.atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}
        self.NOTINDICT = 19

        self.datasets = self.load_dataset()
        self.indexes = {"train": 0, "val": 0, 'test1': 0, 'test2': 0}
        self.dataset_size_dict = {
            "train": {key: len(self.datasets['train'][key]) for key in list(self.datasets['train'].keys())},
            "val": {key: len(self.datasets['val'][key]) for key in list(self.datasets['val'].keys())},
            'test1': {key: len(self.datasets['test1'][key]) for key in list(self.datasets['test1'].keys())},
            'test2': {key: len(self.datasets['test2'][key]) for key in list(self.datasets['test2'].keys())}}
        # self.label_set = self.get_label_set()
        self.data_length = {name: np.sum([len(self.datasets[name][key])
                                          for key in self.datasets[name]]) for name in self.datasets.keys()}

        print("data", self.data_length)
        self.observed_seed_set = None

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """


        if os.path.exists(os.path.join(self.args.dataset_path, self.dataset_name, "drug_3Ddata.pkl")):
            print("Load data at", os.path.join(self.args.dataset_path, self.dataset_name, "drug_3Ddata.pkl"))
            self.h_to_t_dict, self.t_to_h_dict, self.id_set, \
            self.id_smiles_dict, self.smile_pos_dict, self.smile_z_dict, self.pass_smiles \
                = joblib.load(os.path.join(self.args.dataset_path, self.dataset_name, "drug_3Ddata.pkl"))
        else:
            print("Loading data into RAM")
            self.data_loaded_in_memory = True
            # if "drugbank" in self.dataset_name:
            self.h_to_t_dict, self.t_to_h_dict, self.id_set, \
            self.id_smiles_dict , self.smile_pos_dict, self.smile_z_dict = self.get_drugbank_alltrain()
            joblib.dump((self.h_to_t_dict, self.t_to_h_dict, self.id_set,
                         self.id_smiles_dict , self.smile_pos_dict, self.smile_z_dict, self.pass_smiles),
                            os.path.join(self.args.dataset_path, self.dataset_name, "drug_3Ddata.pkl"))

        print("self.pass_smiles", len(self.pass_smiles))
        dataset_splits = dict()

        train_data = joblib.load(os.path.join(self.args.dataset_path, self.dataset_name, "meta_train.pkl"))
        dataset_splits["train"] = self.do_filter(train_data)

        val_data = joblib.load(os.path.join(self.args.dataset_path, self.dataset_name, "meta_val.pkl"))
        dataset_splits["val"] = self.do_filter(val_data)

        test_data = joblib.load(os.path.join(self.args.dataset_path, self.dataset_name, "meta_test1.pkl"))
        dataset_splits["test1"] = self.do_filter(test_data)

        test_data = joblib.load(os.path.join(self.args.dataset_path, self.dataset_name, "meta_test2.pkl"))
        dataset_splits["test2"] = self.do_filter(test_data)
        return dataset_splits

    def do_filter(self, task_data):

        new_dict = dict()
        for k,v in task_data.items():
            for smi in self.pass_smiles:
                index = v.index[v["Drug1"] == smi].tolist()
                index.extend(v.index[v["Drug2"] == smi].tolist())
                index = list(set(index))
                v.drop(axis=0, index=index, inplace=True)
            if len(v) >= 10:
                new_dict[k] = v
        return new_dict

    def get_pos_z(self, smile1):
        # print(smile1)
        m1 = rdkit.Chem.MolFromSmiles(smile1)

        if m1 is None:
            self.pass_smiles.add(smile1)
            return None, None

        if m1.GetNumAtoms() == 1:
            self.pass_smiles.add(smile1)
            return None, None
        m1 = Chem.AddHs(m1)

        ignore_flag1 = 0
        ignore1 = False

        while AllChem.EmbedMolecule(m1) == -1:
            print('retry')
            ignore_flag1 = ignore_flag1 + 1
            if ignore_flag1 >= 10:
                ignore1 = True
                break
        if ignore1:
            self.pass_smiles.add(smile1)
            return None, None
        AllChem.MMFFOptimizeMolecule(m1)
        m1 = Chem.RemoveHs(m1)
        m1_con = m1.GetConformer(id=0)

        pos1 = []
        for j in range(m1.GetNumAtoms()):
            pos1.append(list(m1_con.GetAtomPosition(j)))
        np_pos1 = np.array(pos1)
        ten_pos1 = torch.Tensor(np_pos1)

        z1 = []
        for atom in m1.GetAtoms():
            if self.atomType.__contains__(atom.GetSymbol()):
                z = self.atomType[atom.GetSymbol()]
            else:
                z = self.NOTINDICT
            z1.append(z)

        z1 = np.array(z1)
        z1 = torch.tensor(z1)
        return ten_pos1, z1

    def get_drugbank_alltrain(self):
        file_name = "drugbank.csv" if "drugbank" in self.dataset_name else "twosides_ge_500.csv"
        df_all = pd.read_csv(os.path.join(self.args.dataset_path, self.dataset_name, file_name))
        h_to_t_dict = {}
        t_to_h_dict = {}
        id_set = set()
        id_smiles_dict = {}
        for i in range(df_all.shape[0]):
            head = df_all.loc[i, 'Drug1_ID']
            tail = df_all.loc[i, 'Drug2_ID']
            head_smile = df_all.loc[i, 'Drug1']
            tail_smile = df_all.loc[i, 'Drug2']
            id_smiles_dict[head] = head_smile
            id_smiles_dict[tail] = tail_smile
            id_set.add(head)
            id_set.add(tail)

            if h_to_t_dict.__contains__(head):
                h_to_t_dict[head].append(tail)
            else:
                h_to_t_dict[head] = []
                h_to_t_dict[head].append(tail)

            if t_to_h_dict.__contains__(tail):
                t_to_h_dict[tail].append(head)
            else:
                t_to_h_dict[tail] = []
                t_to_h_dict[tail].append(head)

        smile_pos_dict = {}
        smile_z_dict = {}
        for i, (id, smiles) in enumerate(id_smiles_dict.items()):
            print('Converting SMILES to 3Dgraph: {}/{}'.format(i + 1, len(id_set)))
            if smile_pos_dict.__contains__(smiles):
                continue
            else:
                ten_pos1, z1 = self.get_pos_z(smiles)
                if ten_pos1 == None:
                    self.pass_smiles.add(smiles)
                    smile_pos_dict[smiles] = None
                    smile_z_dict[smiles] = None
                else:
                    smile_pos_dict[smiles] = ten_pos1
                    smile_z_dict[smiles] = z1

        return h_to_t_dict, t_to_h_dict, id_set, id_smiles_dict, smile_pos_dict, smile_z_dict

    def get_drugbank_neg(self, drug1_ID, drug2_ID, rel):
        smile1 = self.id_smiles_dict[drug1_ID]
        smile2 = self.id_smiles_dict[drug2_ID]

        neg_flag = True
        while neg_flag:
            if random.random() > 0.5:  # 换尾
                tail_set = self.h_to_t_dict[drug1_ID]
                pes_tail = random.sample(self.id_set - set(tail_set), 1)
                smile2 = self.id_smiles_dict[pes_tail[0]]
            else:
                head_set = self.t_to_h_dict[drug2_ID]
                pes_head = random.sample(self.id_set - set(head_set), 1)
                smile1 = self.id_smiles_dict[pes_head[0]]

            if self.smile_pos_dict.__contains__(smile1):
                ten_pos1 = self.smile_pos_dict[smile1]
                z1 = self.smile_z_dict[smile1]

            else:
                ten_pos1, z1 = self.get_pos_z(smile1)
            if ten_pos1 == None or z1 == None:
                continue

            if self.smile_pos_dict.__contains__(smile2):
                ten_pos2 = self.smile_pos_dict[smile2]
                z2 = self.smile_z_dict[smile2]
            else:
                ten_pos2, z2 = self.get_pos_z(smile2)
            if ten_pos2 == None or z2 == None:
                continue

            data1 = Data(pos=ten_pos1, z=z1, y=torch.tensor(0), rel=rel)
            data2 = Data(pos=ten_pos2, z=z2, y=torch.tensor(0), rel=rel)
            neg_flag = False
        return data1, data2

    def get_batch(self, x_data):
        class_num, x_num, csv_len = x_data.shape

        batch_head_list = []
        batch_tail_list = []
        batch_label_list = []
        batch_rel_list = []
        for i in range(class_num):
            class_head_list = []
            class_tail_list = []
            class_label_list = []
            class_rel_list = []
            for s in range(x_num):
                drug1_ID = x_data[i][s][0]
                drug1_smiles = x_data[i][s][1]
                drug2_ID = x_data[i][s][2]
                drug2_smiles = x_data[i][s][3]
                rel = int(x_data[i][s][4]) + self.args.Y_is_zero
                if self.smile_pos_dict.__contains__(drug1_smiles):
                    ten_pos1 = self.smile_pos_dict[drug1_smiles]
                    z1 = self.smile_z_dict[drug1_smiles]
                else:
                    ten_pos1, z1 = self.get_pos_z(drug1_smiles)
                if ten_pos1 == None or z1 == None:
                    print(" drug1_smiles pass" , drug1_smiles)
                    continue

                if self.smile_pos_dict.__contains__(drug1_smiles):
                    ten_pos2, z2 = self.smile_pos_dict[drug2_smiles], self.smile_z_dict[drug2_smiles]
                else:
                    ten_pos2, z2 = self.get_pos_z(drug2_smiles)
                if ten_pos2 == None or z1 == None:
                    print(" drug2_smiles pass", drug2_smiles)
                    continue

                neg_data1, neg_data2 = self.get_drugbank_neg(drug1_ID, drug2_ID, rel)
                if neg_data1 == None or neg_data2 == None:
                    print(" neg_data pass", drug1_ID, drug2_ID)
                    continue

                data1 = Data(pos=ten_pos1, z=z1, y=torch.tensor(0), rel=rel)
                data2 = Data(pos=ten_pos2, z=z2, y=torch.tensor(0), rel=rel)

                class_head_list.append(data1)
                class_head_list.append(neg_data1)

                class_tail_list.append(data2)
                class_tail_list.append(neg_data2)

                class_rel_list.append(torch.LongTensor([rel]))
                class_rel_list.append(torch.LongTensor([rel]))

                class_label_list.append(torch.FloatTensor([1]))
                class_label_list.append(torch.FloatTensor([0]))

            if len(class_head_list) == 0 or len(class_tail_list) == 0:
                print("class_head_list or class_tail_list is null")
                continue
            class_head = Batch.from_data_list(class_head_list)
            class_tail = Batch.from_data_list(class_tail_list)
            class_rel = torch.cat(class_rel_list, dim=0)
            class_label = torch.cat(class_label_list, dim=0)

            batch_head_list.append(class_head)
            batch_tail_list.append(class_tail)
            batch_label_list.append(class_label)
            batch_rel_list.append(class_rel)
        return batch_head_list, batch_tail_list, batch_label_list, batch_rel_list
        # return class_head


    def collate_fn(self, batch):

        s_head_list = []
        s_tail_list = []
        s_label_list = []
        s_rel_list = []

        q_head_list = []
        q_tail_list = []
        q_label_list = []
        q_rel_list = []

        # s_head_list_np = np.ones((2, 5))
        for bs_i, (s_data, q_data) in enumerate(batch):

            s_batch_head_list, s_batch_tail_list, s_batch_label_list, s_batch_rel_list = self.get_batch(s_data)
            q_batch_head_list, q_batch_tail_list, q_batch_label_list, q_batch_rel_list = self.get_batch(q_data)
            # test = self.get_batch(s_data)

            s_head_list.append(s_batch_head_list)
            s_tail_list.append(s_batch_tail_list)
            s_label_list.append(s_batch_label_list)
            s_rel_list.append(s_batch_rel_list)

            q_head_list.append(q_batch_head_list)
            q_tail_list.append(q_batch_tail_list)
            q_label_list.append(q_batch_label_list)
            q_rel_list.append(q_batch_rel_list)

        # s_head = np.array(s_head_list)
        # s_tail = np.array(s_tail_list)
        # s_label = np.array(s_label_list)
        # s_rel = np.array(s_rel_list)
        #
        # q_head = np.array(q_head_list)
        # q_tail = np.array(q_tail_list)
        # q_label = np.array(q_label_list)
        # q_rel = np.array(q_rel_list)

        return s_head_list, s_tail_list, s_label_list, s_rel_list, \
                q_head_list, q_tail_list, q_label_list, q_rel_list




    def get_set(self, dataset_name, seed):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """
        #seed = seed % self.args.total_unique_tasks
        rng = np.random.RandomState(seed)
        selected_classes = rng.choice(list(self.dataset_size_dict[dataset_name].keys()),
                                      size=self.num_classes_per_set, replace=False)
        rng.shuffle(selected_classes)

        x_smiles = []

        for class_entry in selected_classes:
            choose_samples_list = rng.choice(self.dataset_size_dict[dataset_name][class_entry],
                                             size=self.num_samples_per_class + self.num_target_samples, replace=False)
            class_smiles = []
            for sample in choose_samples_list:
                choose_samples = self.datasets[dataset_name][class_entry].iloc[sample]
                class_smiles.append((choose_samples["Drug1_ID"], choose_samples["Drug1"],
                                      choose_samples["Drug2_ID"], choose_samples["Drug2"], choose_samples[self.args.Y_name]))

            x_smiles.append(class_smiles)
        x_data = np.array(x_smiles)
        support_set_data = x_data[:, :self.num_samples_per_class]
        target_set_data = x_data[:, self.num_samples_per_class:]

        return support_set_data, target_set_data

    def __len__(self):
        total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        support_set_data, target_set_data = \
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx)

        return support_set_data, target_set_data

    def reset_seed(self):
        self.seed = self.init_seed


class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = FewShotLearningDatasetParallel(args=args)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                          shuffle=False, drop_last=True, collate_fn=self.dataset.collate_fn)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_val_batches(self, total_batches=-1):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_test1_batches(self, total_batches=-1):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test1'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test1')
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_test2_batches(self, total_batches=-1):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test2'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test2')
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

