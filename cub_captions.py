import os
from itertools import product
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

class CubCaps(Dataset):
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    descriptions_file = 'no_oov_descriptions.json'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()
            raise Exception("You still need to download the Scott Reed descriptions and the fix by Franz Reiger!")

        if not self._check_integrity():  # this sets up data (in _load_metadata)
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), separator=' ',
                             new_columns=['img_id', 'filepath'])
        image_class_labels = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         separator=' ', new_columns=['img_id', 'target'])
        train_test_split = pl.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       separator=' ', new_columns=['img_id', 'is_training_img'])

        data = images.join(image_class_labels, on='img_id')
        self.data = data.join(train_test_split, on='img_id')

        self.captions = pl.read_json(os.path.join(self.root, self.descriptions_file))

        # number of captions per image (assume all same)
        num_caps = self.captions.select(pl.col("1").list.len()).item()
        # self.data = data.merge(captions_long, on='img_id')  # this multiplies the images, makes data unnecessarily huge

        if self.train:
            self.data = self.data.filter(pl.col("is_training_img") == 1)
        else:
            self.data = self.data.filter(pl.col("is_training_img") == 0)

        img_ids = self.data.get_column('img_id').to_list()
        self.items = list(product(img_ids, range(num_caps)))  # for __len__ and __getitem__

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            raise e
            #return False

        # This is slow
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
        return len(self.items)

    def __getitem__(self, idx):
        img_id, descr_id = self.items[idx]  # dict or array

        sample = self.data.filter(pl.col('img_id') == img_id).row(0, named=True)
        #print('sampled', sample)

        path = os.path.join(self.root, self.base_folder, sample['filepath'])
        target = sample['target'] - 1  # Targets start at 1 by default, so shift to 0 # (inspection/debugging zombie)
        img = self.loader(path)  # this is where all the time gets spent, everything else is negligible

        description = self.captions.get_column(str(img_id)).list.get(descr_id).item()

        if self.transform is not None:
            img = self.transform(img)

        return (img, description), target

if __name__ == "__main__":
    cc=CubCaps('./data', download=False)
    print(cc[0])

    def loop_test():
        for i in range(100):
            cc[i]
    import timeit
    print(timeit.timeit('loop_test()', globals=globals(), number=100))
