from pathlib import Path

from utils.image_utils import (
    load_image,
    normalize_image,
    save_temp_png,
)


def test_image_pipeline():

    image = load_image(
        Path("data/incoming/invoices/000.jpg")
    )

    image = normalize_image(image)

    output = save_temp_png(image)

    assert output.exists()

    output.unlink()