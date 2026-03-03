from .nus8_dataprovider import NUS8DataProvider


class NUS8ExtendedDataProvider(NUS8DataProvider):
    """NUS8 Extended: includes all 9 cameras (standard 8 + NikonD40)."""
    EXCLUDED_CAMERAS = []
