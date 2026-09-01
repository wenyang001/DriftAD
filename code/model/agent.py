from header import *

class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.image_decoder.named_parameters():
            param.requires_grad = True

        for name, param in self.model.adapter.named_parameters():
            param.requires_grad = True
        for name, param in self.model.aGNN.named_parameters():
            param.requires_grad = True
        for name, param in self.model.mmci.named_parameters():
            param.requires_grad = True

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        # 覆盖scheduler参数为实际计算的值
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))

        # 初始化DeepSpeed（使用配置文件中的scheduler，已被上面覆盖）
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )


    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step, total_epochs=50):
        """
        path: 保存路径
        current_step: 当前的 epoch 数 (根据您的描述)
        total_epochs: 总 epoch 数
        """
        # 筛选只保存可训练参数
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                # print(k) # 可选：打印参数名
                checkpoint[k] = v.to(torch.device("cpu"))

        # 保存 iter 计数器
        checkpoint['iter'] = self.ds_engine.module.iter
        
        # =======================================================
        # 修改后的保存逻辑
        # =======================================================
        # 1. 如果当前 epoch 还没到最后 5 个阶段 -> 覆盖保存为 pytorch_model.pt
        if current_step < 40:
            save_name = 'pytorch_model.pt'

        # 2. 最后 10 个 epoch (40-49) -> 每个 epoch 独立保存
        else:
            save_name = f'pytorch_model_{current_step}.pt'
            
        save_full_path = os.path.join(path, save_name)
        
        torch.save(checkpoint, save_full_path)
        
        # 只在主进程打印日志
        if self.args['local_rank'] == 0:
            print(f"Model saved to {save_full_path}")