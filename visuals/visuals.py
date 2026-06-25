class Visuals:
    def __init__(self, image_size=512):
        self.image_size = image_size

    def visualize(self, data):
        return self._visualize(data)

    def _visualize(self, data):
        raise NotImplementedError
