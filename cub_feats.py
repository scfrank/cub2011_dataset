import os
from itertools import product
import numpy as np
# import pandas as pd
import polars as pl
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


"""
Pytorch dataset for CUB with descriptions from Scott Reed.

Downloading CUB images is not allowed here because descriptions need to be set up as well.
in /data/no_oov_descriptions.json

Assume  ./data/no_oov_descriptions
    and ./data/CUB_200_2011 
(can be softlinks)

Excuse to learn polars.
"""

class CubFeats(Dataset):
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False,
                 feats_only=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.feats_only = feats_only

        if download:
            self._download()
            raise Exception("You still need to download the Scott Reed descriptions and the fix by Franz Reiger!")

        if not self._check_integrity():  # this sets up data (in _load_metadata)
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        print(f"Created CUBFeats dataset N{len(self.data)} feats only? {self.feats_only}")

    def _load_metadata(self):
        images = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), separator=' ',
                                      has_header=False, new_columns=['img_id', 'filepath'])
        image_class_labels = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                      has_header=False, separator=' ', new_columns=['img_id', 'target'])
        train_test_split = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                      has_header=False, separator=' ', new_columns=['img_id', 'is_training_img'])
        attribute_names = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes.txt'),
                                      has_header=False, separator=' ', new_columns=['attr_id', 'att_name'])
        attributes = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes/image_attribute_labels.txt'),
                                 separator=' ', has_header=False,
                                 new_columns=['img_id', 'attr_id', 'is_present', 'certainty_id', 'time'],
                                 truncate_ragged_lines=True)  # why is this necesary?

        data = images.join(image_class_labels, on='img_id')
        self.data = data.join(train_test_split, on='img_id')

        # creates a (img_id, attr-1, attr-2, etc) table of boolean values.
        attributes = attributes.select(
            'img_id',
            pl.col('attr_id').map_elements(lambda x: f"attr-{x}", return_dtype=str),
            pl.col('is_present').cast(pl.Boolean)
        ).pivot(on='attr_id', values='is_present', index='img_id') #, columns='attr_id')

        # combines attributes columns into a single list column
        attributes = attributes.select(
            'img_id',
            pl.concat_list([f'attr-{x}' for x in range(1,313)]).alias('attributes')  # fix magic number (of CUB attributes)
        )

        self.data = self.data.join(attributes, on='img_id')

        if self.train:
            self.data = self.data.filter(pl.col("is_training_img") == 1)
        else:
            self.data = self.data.filter(pl.col("is_training_img") == 0)

        # hacky ways of getting everything out of polars for iterating
        self.items = self.data.get_column('img_id').to_list()  # for indexing

        ad = attributes.to_dict()
        self.attr_dict = dict(zip(ad['img_id'], ad['attributes']))

        dd = self.data.to_dict()
        self.target_dict = dict(zip(dd['img_id'], dd['target']))
        self.files_dict = dict(zip(dd['img_id'], dd['filepath']))


    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            raise e

        for row in self.data.iter_rows(named=True):  # this is discouraged
            filepath = os.path.join(self.root, self.base_folder, row['filepath'])
            if not os.path.isfile(filepath):
                print("Missing!", filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem_pl__(self, idx):
        """this hangs on next() in a pytorch dataloader."""
        img_id = self.items[idx]  # dict or array

        sample = self.data.filter(pl.col('img_id') == img_id).row(0, named=True)
        #print(img_id, 'sampled', sample)

        attributes = np.array(sample['attributes'])
        target = sample['target'] - 1  # Targets start at 1 by default, so shift to 0 # (inspection/debugging zombie)

        if self.feats_only:
            return attributes, target
        else:
            path = os.path.join(self.root, self.base_folder, sample['filepath'])
            img = self.loader(path)  # this is where all the time gets spent, everything else is negligible

            if self.transform is not None:
                img = self.transform(img)

            return (img, attributes), target

    def __getitem__(self, idx):
        """polars-free version for *feats_only*""" # TODO FIX BUG DEBUG
        img_id = self.items[idx]  # dict or array
        attr = np.array(self.attr_dict[img_id])
        target = self.target_dict[img_id] - 1 # Targets start at 1 by default, so shift to 0 # (inspection/debugging zombie)
        return (attr, target)


if __name__ == "__main__":
    cc=CubFeats('./data', download=False, feats_only=True)
    print(cc[0])

    def loop_test():
        for i in range(100):
            cc[i]
    import timeit
    print(timeit.timeit('loop_test()', globals=globals(), number=100))
