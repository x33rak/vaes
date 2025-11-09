from pathlib import Path
import argparse
import math
import numpy as np
import cv2

import torch
from model_builder import SkipVAE
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim
import lpips


# device agnostic code setup
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--test_data_path", type=str, default="./datasets/agan/test_b/",
                    help="str path to test data folder")
parser.add_argument("--weights_path", type=str, default="./weights/attentive_vae_last.pth")
parser.add_argument("--save_path", type=str, default="./output/test_b/",
                    help="str path to save data folder")
opt = parser.parse_args()

def format_input(x:np.ndarray)->torch.Tensor:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(size=(480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    x = transform(x)
    x = x.unsqueeze(dim=0)
    x = x.to(device)
    return x

def format_output(x:torch.Tensor):
    x = (x + 0.5) / 0.5 * 255
    x = x.to(torch.uint8)
    x = x.squeeze().permute(1,2,0).detach().cpu().numpy()
    return x


def PSNR(model_clean, gt_clean) -> float:
    # Higher PSNR means better results
    MSE = np.square(model_clean-gt_clean).mean()

    if MSE == 0:
        return 100.0

    RMSE = math.sqrt(MSE)

    MAX = 255.0
    return 20 * math.log10(MAX/RMSE)


def CD(model_clean, gt_clean) -> float:
    # Lower value is better
    B1, G1, R1 = cv2.split(model_clean)
    B2, G2, R2 = cv2.split(gt_clean)

    # Compute the colour difference
    delta_R = np.abs(R1 - R2)
    delta_G = np.abs(G1 - G2)
    delta_B = np.abs(B1 - B2)

    # Compute the CD
    cd = np.mean(np.abs(delta_R + delta_G + delta_B) / 3)

    return cd


def LPIPS(model_clean, gt_clean, lpips_model) -> float:
    # Lower value is better

    # Compute LPIPS distance
    with torch.inference_mode():
        distance = lpips_model(model_clean, gt_clean.to(device))

    return float(distance)


def main():
    Path(opt.save_path).mkdir(parents=True, exist_ok=True)

    model = SkipVAE()  # instantiate model
    model.load_state_dict(torch.load(opt.weights_path, map_location=device))  # load weights
    model = model.to(device)  # model to traget device
    model.eval()  # put model in evaluation mode

    # Initialise LPIPS model: AlexNet
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model = lpips_model.eval()

    data_list = sorted(list(Path(opt.test_data_path).glob("data/*.jpg")))
    gt_list = sorted(list(Path(opt.test_data_path).glob("gt/*.jpg")))

    psnr_list, ssim_list, cd_list, distance_list = [], [], [], []
    for i in range(len(data_list)):
        data, gt = cv2.imread(data_list[i], cv2.IMREAD_UNCHANGED), cv2.imread(gt_list[i], cv2.IMREAD_UNCHANGED)
        data, gt = cv2.cvtColor(data, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        with torch.inference_mode():
            data, gt = format_input(data), format_input(gt)
            out, _, _, mask = model(data)
            out_np, gt_np = format_output(out), format_output(gt)
            psnr_val = PSNR(out_np, gt_np)  # Calculate PSNR
            ssim_val = ssim(out_np, gt_np, data_range=255, channel_axis=-1)  # Calculate SSIM
            cd_val = CD(out_np, gt_np)  # Calculate SSIM
            distance = LPIPS(out, gt, lpips_model)  # calculate LPIPS distance

            name = data_list[i].name
            # save model output as image files
            cv2.imwrite(opt.save_path + name, out_np)

            print(f"Image: {name} | PSNR: {psnr_val:.3f} | SSIM: {ssim_val:.3f} "
                  f"| CD: {cd_val:.3f} | Distance: {distance:.3f}")

            # append each val to list
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            cd_list.append(cd_val)
            distance_list.append(distance)

    print(f"Average PSNR: {sum(psnr_list)/len(psnr_list):.3f} "
          f"| Average SSIM: {sum(ssim_list)/len(ssim_list):.3f} "
          f"| Average CD: {sum(cd_list)/len(cd_list):.3f} "
          f"| Average Distance: {sum(distance_list)/len(distance_list):.3f}")

if __name__ == "__main__":
    main()
