import os
from PIL import Image, ImageDraw
from pathlib import Path
import shutil

def create_rounded_background(img, padding=20, radius=20):
    img = img.convert("RGBA")
    w, h = img.size

    # New size with padding
    new_w = w + 2 * padding
    new_h = h + 2 * padding

    # Create transparent canvas
    result = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))

    # Create white rounded background
    bg = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    mask = Image.new("L", (new_w, new_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (new_w, new_h)], radius=radius, fill=255)

    rounded_bg = Image.composite(bg, result, mask)

    # Paste original image on top
    rounded_bg.paste(img, (padding, padding), img)

    return rounded_bg

def process_images_in_folder(folder="."):
    folder = Path(folder)
    backup_folder = folder / "backup"
    backup_folder.mkdir(exist_ok=True)

    png_files = list(folder.glob("*.png"))

    for img_path in png_files:
        print(f"Processing: {img_path.name}")

        # Backup original
        shutil.copy2(img_path, backup_folder / img_path.name)

        # Process image
        with Image.open(img_path) as img:
            rounded_img = create_rounded_background(img, padding=20, radius=20)
            rounded_img.save(img_path)

    print(f"\nDone. Originals backed up to: {backup_folder.resolve()}")

# Run the script
if __name__ == "__main__":
    process_images_in_folder(".")
