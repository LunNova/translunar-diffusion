import pytorch_lightning as pl

from typing import Any

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from torch.utils.data import random_split, DataLoader, Dataset, Subset

from functools import partial


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=1,
        num_workers=None,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        use_worker_init_fn=False,
    ):
        super().__init__()
        self.datasets = None
        self.dataset_configs = {}
        self.use_worker_init_fn = use_worker_init_fn
        num_workers = num_workers if num_workers is not None else batch_size
        default_dataloader_params = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'shuffle': True,
            'drop_last': True,
            'persistent_workers': num_workers > 0,
        }
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = partial(
                self._build_dataloader, mode='train',
                dataloader_params=dict(default_dataloader_params, **train.get('dataloader_params', {}))
            )
        default_dataloader_params['shuffle'] = False
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = partial(
                self._build_dataloader,
                mode='validation',
                dataloader_params=dict(default_dataloader_params, **validation.get('dataloader_params', {}))
            )
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = partial(
                self._build_dataloader,
                mode='test',
                dataloader_params=dict(default_dataloader_params, **test.get('dataloader_params', {}))
            )
        if predict is not None:
            self.dataset_configs['predict'] = predict
            self.predict_dataloader = partial(
                self._build_dataloader,
                mode='predict',
                dataloader_params=dict(default_dataloader_params, **predict.get('dataloader_params', {}))
            )
        self.wrap = wrap

    def _setup_datasets(self):
        if self.datasets:
            print(
                f'Skipping dataloader setup call, already have {len(self.datasets.items())}'
            )
            return
        self.datasets = {
            k: instantiate_from_config(self.dataset_configs[k])
            for k in self.dataset_configs
        }

    def prepare_data(self):
        self._setup_datasets()
        for k, v in self.datasets.items():
            if hasattr(v, 'on_prepare_data'):
                v.on_prepare_data()
        # Do nothing, no need to download anything
        pass

    def setup(self, stage=None):
        self._setup_datasets()
        for k, v in self.datasets.items():
            if hasattr(v, 'on_setup'):
                v.on_setup()
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        for k, v in self.datasets.items():
            print(f'dataset\t{k}\t{v}\tlen={len(v)}')

    def _build_dataloader(self, mode: str, dataloader_params: dict):
        is_iterable_dataset = isinstance(
            self.datasets[mode], Txt2ImgIterableBaseDataset
        )
        init_fn = (
            worker_init_fn
            if is_iterable_dataset or self.use_worker_init_fn
            else None
        )
        if init_fn:
            print("Using worker_init_fn ", init_fn)
        print(f'Initing {mode} DataLoader, params={dataloader_params}')
        return DataLoader(
            self.datasets[mode],
            **dataloader_params,
            worker_init_fn=init_fn
        )


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[
            worker_id * split_size : (worker_id + 1) * split_size
        ]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    dataset: Any

    def __init__(self, dataset: Any):
        self.data = dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]
