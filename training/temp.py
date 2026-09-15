from detection import *

template_dir = Path('template_samples')
paths = [Path(template_dir) / name for name in os.listdir(template_dir)]
templates = [load_image(path) for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

os.makedirs('rgb_templates', exist_ok=True)
for path, template in zip(paths, templates):
    print(f"Template: {path.name}, shape: {template.shape}")
    cv2.imwrite(f'rgb_templates/{path.name}', template)