import hydra
from hm3d_data_collector.utils.io_utils import get_abs_path
import os
from tqdm import tqdm
from imageio.v2 import imread, imwrite

@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    rgb_dir = cfg.get("rgb_dir", None)
    if rgb_dir is None:
        raise ValueError("rgb_dir is not provided in the config file.")

    list_img = os.listdir(rgb_dir)
    for img_fn in tqdm(list_img, desc="vis images"):
        img = imread(os.path.join(rgb_dir, img_fn))
        fn = os.path.join(cfg.log_dir, "vis.jpg")
        imwrite(fn, img)
        print(f"Visualized image: {fn}")
        input("Press Enter to continue...")
        

if __name__ == "__main__":
    main()
