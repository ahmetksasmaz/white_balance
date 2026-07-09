class DataProvider:
    def __init__(self, override_dimensions=(-1, -1)):
        self.override_dimensions = override_dimensions
        self.data_names = []
        # Per-image Data cache, keyed by index. Populated on demand by
        # get_cached() and evicted by mark_algorithm_done() once every
        # algorithm expected to run on that image (in this run) has done so.
        self._data_cache = {}
        self._completed_algorithms = {}

    def __len__(self):
        return len(self.data_names)

    def get_image_name(self, index):
        """Lightweight accessor for the image name that avoids decoding pixel data.

        Default assumes data_names[index] already is the image name; providers
        that derive the name differently (e.g. from a file path) override this.
        """
        return self.data_names[index]

    def _construct_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        if 0 <= index < len(self.data_names):
            return self._construct_data(index)
        raise IndexError("Data index out of range")

    def get_cached(self, index, camera=None, resize_factor=None):
        """Load the Data object for `index` on first access and reuse it for
        every subsequent algorithm evaluated against the same image, instead
        of re-decoding it from disk per algorithm. camera/resize are applied
        exactly once, at load time, so repeat callers don't re-resize an
        already-resized cached image."""
        if index not in self._data_cache:
            data = self._construct_data(index)
            if camera is not None:
                data.set_camera(camera)
            data.resize(resize_factor)
            self._data_cache[index] = data
            self._completed_algorithms[index] = set()
        return self._data_cache[index]

    def mark_algorithm_done(self, index, algorithm_key, expected_algorithm_keys):
        """Record that `algorithm_key` has finished evaluating the image at
        `index`. Once every key in `expected_algorithm_keys` (the full set of
        algorithms this run will evaluate for that image) has been recorded,
        the cached Data is evicted to free its pixel data."""
        if index not in self._data_cache:
            return
        completed = self._completed_algorithms.setdefault(index, set())
        completed.add(algorithm_key)
        if completed.issuperset(expected_algorithm_keys):
            self._data_cache.pop(index, None)
            self._completed_algorithms.pop(index, None)
