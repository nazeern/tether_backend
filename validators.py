from dataclasses import dataclass
import numpy as np

from transformers import Loader


@dataclass
class ImageValidator():

    extension: str = '.jpeg'
    xmax: int = 3000
    ymax: int = 5000

    def validate(
        self,
        filestorage,
    ) -> tuple[np.ndarray | None, str | None]:
        
        if not filestorage:
            return None, "Image file not provided in form data"

        if filestorage.content_type != 'image/jpeg':
            return None, "Image must be in JPEG format"

        filestorage.save("upload.jpeg")

        ld = Loader()
        arr = ld.transform("upload.jpeg")

        y, x = arr.shape[:2]

        if x > self.xmax or y > self.ymax:
            return None, "Image exceeds maximum dimensions"

        return arr, None