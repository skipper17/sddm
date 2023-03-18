from tool.eval_score import fid_l2_psnr_ssim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    task = 'male2female'

    if task == 'cat2dog':
        translate_path = 'myoutput/cat2dog/50_100_25_2_fullegsde_noisy_bestfid'
        source_path = '/home/data/afhq/val/cat'
        gt_path = '/home/data/afhq/val/dog'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)

    if task == 'wild2dog':
        translate_path = 'myoutput/wild2dog/50_100_25_2_fullegsde_noisy_bestfid_mix'
        source_path = '/home/data/afhq/val/wild'
        gt_path = '/home/data/afhq/val/dog'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)

    if task == 'male2female':
        translate_path = 'myoutput/male2female/52_100_25_2_fullegsde_noisy_bestfid'
        source_path = '/home/data/celeba_hq/val/male'
        gt_path = '/home/sunsk/data/celeba_hq/train/fid_celebahq_female.npz' #'/home/data/celeba_hq/val/female'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)





