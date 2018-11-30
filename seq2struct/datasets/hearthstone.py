import attr

from seq2struct.utils import registry
from seq2struct.utils import indexed_file


@registry.register('dataset', 'hearthstone')
class HearthstoneDataset(torch.utils.data.Dataset):

    def __init__(self, config):
        self.filename = config['filename']
        self.reader = indexed_file.IndexedFileReader(self.filename)
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        item = json.loads(self.reader[idx].decode('utf-8'))
        # TODO: process item
        return item
