from gdino import GroundingDINOAPIWrapper


class GroundingDINOAPI(GroundingDINOAPIWrapper):
    api_key = "your_api_key"

    def __init__(self):
        super().__init__(self.api_key)
