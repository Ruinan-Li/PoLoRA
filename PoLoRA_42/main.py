import shutil
from datetime import datetime
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils import *
from src.parse_args import args
from src.data_load.KnowledgeGraph import KnowledgeGraph
from src.model.LoraKGE import TransE as LoraKGE_Layers
from src.train import *
from src.test import *
from src.poincare.rsgd import RiemannianSGD
from src.replay.per_buffer import PERBuffer, build_training_facts_for_snapshot

class Instructor():
    """ The instructor of the model """
    def __init__(self, args) -> None:

        self.args = args

        """ 1. Prepare for path, logger and device """
        self.prepare()

        """ 2. Load data """
        self.kg = KnowledgeGraph(args)

        # PER experience buffer
        self.per_buffer = PERBuffer(self.args, self.kg) if getattr(self.args, "per_enable", False) else None

        """ 3. Create models and optimizer """
        self.model, self.optimizer = self.create_model()

        self.args.logger.info(self.args)

    def create_model(self):
        """ Create KGE model and optimizer """
        if self.args.model_name == "MuRP":
            # still instantiate LoraKGE model to provide containers, MuRP handled in trainer
            model = LoraKGE_Layers(self.args, self.kg)
        else:
            model = LoraKGE_Layers(self.args, self.kg)
        model.to(self.args.device)
        optimizer = self._build_optimizer(model, self._get_optimizer_lr())
        return model, optimizer

    def reset_model(self, model=False, optimizer=False, snapshot_id=None):
        """
        Reset model or optimizer
        :param model: If True: reset the model and optimizer
        :param optimizer: If True: reset the optimizer
        """
        if model:
            self.model, self.optimizer = self.create_model()
        if optimizer:
            lr = self._get_optimizer_lr(snapshot_id)
            self.optimizer = self._build_optimizer(self.model, lr)

    def _get_optimizer_lr(self, snapshot_id=None):
        if snapshot_id is None:
            snapshot_idx = int(getattr(self.args, "snapshot", 0))
        else:
            snapshot_idx = int(snapshot_id)
        if snapshot_idx == 0:
            return float(self.args.poincare_lr)
        return float(getattr(self.args, "lora_lr", self.args.poincare_lr))

    def _build_optimizer(self, model, lr):
        named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        params = [param for _, param in named_params]
        param_names = [name for name, _ in named_params]
        if not params:
            return None
        scheduler_cfg = None
        if getattr(self.args, "euclid_scheduler", False):
            scheduler_cfg = {
                "enable": True,
                "T0": int(getattr(self.args, "euclid_scheduler_T0", 50)),
                "Tmult": int(getattr(self.args, "euclid_scheduler_Tmult", 2)),
                "eta_min": float(getattr(self.args, "euclid_scheduler_eta_min", 0.0)),
            }
        return RiemannianSGD(params, lr=lr, param_names=param_names, scheduler_cfg=scheduler_cfg)

    def prepare(self):
        """ Set data path """
        os.makedirs(args.data_path, exist_ok=True)
        self.args.data_path = args.data_path + args.dataset + "/"

        """ Set save path """
        # Add timestamp and process ID to avoid conflicts during parallel runs
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        process_id = os.getpid()
        self.args.save_path = args.save_path + args.dataset + f"_{timestamp}_{process_id}"
        if os.path.exists(self.args.save_path):
            shutil.rmtree(self.args.save_path, True)
        os.makedirs(self.args.save_path, exist_ok=True)
        if self.args.note != '':
            self.args.save_path += self.args.note
        if os.path.exists(self.args.save_path):
            shutil.rmtree(self.args.save_path, True)
        os.makedirs(self.args.save_path, exist_ok=True)

        """ Set log path """
        # Add timestamp and process ID to avoid log conflicts during parallel runs
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        process_id = os.getpid()
        os.makedirs(args.log_path, exist_ok=True)
        self.args.log_path = args.log_path + f"{timestamp}_{process_id}/"
        os.makedirs(self.args.log_path, exist_ok=True)
        self.args.log_path = self.args.log_path + args.dataset
        if self.args.note != "":
            self.args.log_path += self.args.note

        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = f'{args.log_path}.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        """ Set device """
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device
        
        # Get actual physical GPU ID used (from CUDA_VISIBLE_DEVICES environment variable)
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if ',' in cuda_visible_devices:
            # If multiple GPUs are set, take the first one
            physical_gpu_id = int(cuda_visible_devices.split(',')[0])
        else:
            physical_gpu_id = int(cuda_visible_devices) if cuda_visible_devices else 0
        
        # Log actual physical GPU ID used
        self.args.logger.info(f"Actual physical GPU ID used: {physical_gpu_id} (PyTorch device ID: {self.args.device})")

    def next_snapshot_setting(self):
        """ Prepare for next snapshot """
        self.model.switch_snapshot()

    def run(self):
        """ Run the instructor of the model. The training process on all snapshots """
        report_results = PrettyTable()
        report_results.field_names = ['Snapshot', 'Time', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10']
        test_results = []
        training_times = []
        BWT = [] # h(n, i) - h(i, i)
        FWT = [] # h(i- 1, i)
        first_learning_res = []

        """ training process """
        for ss_id in range(int(self.args.snapshot_num)):
            best_checkpoint = os.path.join(
                self.args.save_path, f'{str(ss_id - 1)}model_best.tar'
            )

            self.args.snapshot = ss_id
            self.args.snapshot_test = ss_id
            self.args.snapshot_valid = ss_id


            """ preprocess before training on a snapshot """
            self.model.pre_snapshot()

            if ss_id > 0:
                self.args.test_FWT = True
                res_before = self.test()
                FWT.append(res_before['mrr'])
            self.args.test_FWT = False

            training_time = self.train()

            """ prepare result table """
            test_res = PrettyTable()
            test_res.field_names = [
                f'Snapshot:{str(ss_id)}',
                'MRR',
                'Hits@1',
                'Hits@3',
                'Hits@5',
                'Hits@10',
            ]

            best_checkpoint = os.path.join(
                self.args.save_path, f'{str(ss_id)}model_best.tar'
            )
            self.load_checkpoint(best_checkpoint)

            self.model.snapshot_post_processing()

            reses = []
            for test_ss_id in range(ss_id + 1):
                self.args.snapshot_test = test_ss_id
                res = self.test()
                if test_ss_id == ss_id:
                    first_learning_res.append(res['mrr'])
                test_res.add_row([
                    test_ss_id, res['mrr'], res['hits1'], res['hits3'], res['hits5'], res['hits10']
                ])
                reses.append(res)
            if ss_id == self.args.snapshot_num - 1:
                BWT.extend(
                    reses[iid]['mrr'] - first_learning_res[iid]
                    for iid in range(self.args.snapshot_num - 1)
                )
            self.args.logger.info(f"\n{test_res}")
            test_results.append(test_res)

            """ record report results """
            whole_mrr, whole_hits1, whole_hits3, whole_hits10 = self.get_report_results(reses)
            report_results.add_row([ss_id, training_time, whole_mrr, whole_hits1, whole_hits3, whole_hits10])
            training_times.append(training_time)

            if self.per_buffer is not None:
                facts_for_buffer = build_training_facts_for_snapshot(self.kg, ss_id, self.args.train_new)
                self.per_buffer.add_snapshot(ss_id, facts_for_buffer)

            if self.args.snapshot < int(self.args.snapshot_num) - 1:
                self.next_snapshot_setting()
                if self.per_buffer is not None:
                    self.per_buffer.recompute_all_losses(self.model)
                self.reset_model(optimizer=True, snapshot_id=ss_id + 1)
        self.args.logger.info(f'Final Result:\n{test_results}')
        self.args.logger.info(f'Report Result:\n{report_results}')
        self.args.logger.info(f'Sum_Training_Time:{sum(training_times)}')
        self.args.logger.info(f'Every_Training_Time:{training_times}')
        self.args.logger.info(
            f'Forward transfer: {sum(FWT) / len(FWT)} Backward transfer: {sum(BWT) / len(BWT)}'
        )

    def get_report_results(self, results):
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.snapshots[idx].test))
        whole_mrr = sum(
            mrr * num_test[i] for i, mrr in enumerate(mrrs)
            ) / sum(num_test)
        whole_hits1 = sum(
            hits1 * num_test[i] for i, hits1 in enumerate(hits1s)
        ) / sum(num_test)
        whole_hits3 = sum(
            hits3 * num_test[i] for i, hits3 in enumerate(hits3s)
        ) / sum(num_test)
        whole_hits10 = sum(
            hits10 * num_test[i] for i, hits10 in enumerate(hits10s)
        ) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def train(self):
        """ Training process, return training time """
        start_time = time.time()
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer, per_buffer=self.per_buffer)
        backup_poincare_eval = getattr(self.args, "use_poincare_eval", False)
        self.args.use_poincare_eval = False

        """ Training iteration """
        # Select different epoch numbers based on snapshot
        if int(self.args.snapshot) == 0:
            epoch_num = int(getattr(self.args, "epoch_num_snapshot0", 30))
        else:
            epoch_num = int(getattr(self.args, "epoch_num_snapshot1plus", 200))
        
        
        if hasattr(self.args, "epoch_num") and int(self.args.epoch_num) != 100:
            epoch_num = int(self.args.epoch_num)
        
        for epoch in range(epoch_num):
            self.args.epoch = epoch
            """ training """
            loss, valid_res = trainer.run_epoch()
            """ early stop """
            if self.args.debug:
                if epoch > 0:
                    break
            if valid_res[self.args.valid_metrics] > self.best_valid:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = 0
                if self.args.snapshot == 0:
                    self.save_model(is_best=True, lora=False)
                else:
                    self.save_model(is_best=True, lora=True)
            else:
                self.stop_epoch += 1
                if self.args.snapshot == 0:
                    self.save_model(lora=False)
                else:
                    self.save_model(lora=True)
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info(
                        f'Early Stopping! Snapshot:{self.args.snapshot} Epoch: {epoch} Best Results: {round(self.best_valid * 100, 3)}'
                    )
                    break
            """ logging """
            if epoch % 1 == 0:
                self.args.logger.info(
                    f"Snapshot:{self.args.snapshot}\tEpoch:{epoch}\tLoss:{round(loss, 3)}\tMRR:{round(valid_res['mrr'] * 100, 3)}\tHits@10:{round(valid_res['hits10'] * 100, 3)}\tBest:{round(self.best_valid * 100, 3)}"
                )
        end_time = time.time()
        self.args.use_poincare_eval = backup_poincare_eval
        return end_time - start_time

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        return tester.test()

    def save_model(self, is_best=False, lora=False):
        if self.args.snapshot == 0:
            poincare_trainer = getattr(self.model, "poincare_trainer", None)
            if poincare_trainer is None:
                return
            checkpoint_dict = {
                'state_dict': poincare_trainer.model.state_dict(),
                'entities': poincare_trainer.entities,
                'relations': poincare_trainer.relations,
            }
            out_tar = os.path.join(
                self.args.save_path,
                f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
            )
            torch.save(checkpoint_dict, out_tar)
            if is_best:
                best_path = os.path.join(
                    self.args.save_path, f'{str(self.args.snapshot)}model_best.tar'
                )
                shutil.copyfile(out_tar, best_path)
            return
        if lora == False:
            checkpoint_dict = {'state_dict': self.model.state_dict()}
            checkpoint_dict['epoch_id'] = self.args.epoch
            out_tar = os.path.join(
                self.args.save_path,
                f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
            )
            torch.save(checkpoint_dict, out_tar)
            if is_best:
                best_path = os.path.join(
                    self.args.save_path, f'{str(self.args.snapshot)}model_best.tar'
                )
                shutil.copyfile(out_tar, best_path)
        else:
            with torch.no_grad():
                ent_emb, rel_emb = self.model.get_full_poincare_embeddings()
                # Update model's poincare_ent_embeddings to complete embeddings after training
                self.model.poincare_ent_embeddings = ent_emb.detach().clone()
                self.model.poincare_rel_embeddings = rel_emb.detach().clone()
            
            checkpoint_dict = {
                'state_dict': self.model.state_dict(),  
                'poincare_ent_embeddings': self.model.poincare_ent_embeddings,  
                'poincare_rel_embeddings': self.model.poincare_rel_embeddings,
                'lora_state': self.model.get_lora_state() if hasattr(self.model, "get_lora_state") else None,
                'epoch_id': self.args.epoch,
            }
            out_tar = os.path.join(
                self.args.save_path,
                f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
            )
            torch.save(checkpoint_dict, out_tar)
            if is_best:
                best_path = os.path.join(
                    self.args.save_path, f'{str(self.args.snapshot)}model_best.tar'
                )
                shutil.copyfile(out_tar, best_path)


    def load_checkpoint(self, input_file):
        if self.args.snapshot == 0:
            if os.path.isfile(input_file):
                logging.info(f"=> loading checkpoint '{input_file}'")
                checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
                poincare_trainer = getattr(self.model, "poincare_trainer", None)
                if poincare_trainer is not None:
                    poincare_trainer.model.load_state_dict(checkpoint['state_dict'])
                else:
                    logging.info("=> poincare trainer not initialized, skip loading.")
            else:
                logging.info(f'=> no checking found at \'{input_file}\'')
        else:
            if os.path.isfile(input_file):
                logging.info(f"=> loading checkpoint \'{input_file}\'")
                checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
                # Load model parameters (LoRA and MuRP scorer)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                # Load Riemannian embeddings (if exists)
                if 'poincare_ent_embeddings' in checkpoint:
                    self.model.poincare_ent_embeddings = checkpoint['poincare_ent_embeddings'].to(self.args.device)
                if 'poincare_rel_embeddings' in checkpoint:
                    self.model.poincare_rel_embeddings = checkpoint['poincare_rel_embeddings'].to(self.args.device)
                if hasattr(self.model, "load_lora_state"):
                    self.model.load_lora_state(checkpoint.get('lora_state'))
                elif hasattr(self.model, "ensure_lora_from_embeddings"):
                    self.model.ensure_lora_from_embeddings()
            else:
                logging.info(f'=> no checking found at \'{input_file}\'')


""" Main function """
if __name__ == "__main__":
    set_seeds(args.random_seed)
    ins = Instructor(args)
    ins.run()
