import base64
import json


class ImageSetMap(dict):

    def __repr__(self):
        return f"ImageSet object. {len(self)} channels: {self}"

    def compress(self):
        return base64.b64encode(json.dumps(self).encode('utf-8'))

    def decompress(self, encoded):
        json_string = base64.b64decode(encoded).decode('utf-8')
        channels_map = json.loads(json_string)
        for chan, value in channels_map.items():
            self[chan] = value

