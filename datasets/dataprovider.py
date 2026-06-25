class DataProvider:
    def __init__(self, override_dimensions=(-1, -1)):
        self.override_dimensions = override_dimensions
        self.data_names = []

    def __len__(self):
        return len(self.data_names)

    def _construct_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        if 0 <= index < len(self.data_names):
            return self._construct_data(index)
        raise IndexError("Data index out of range")
