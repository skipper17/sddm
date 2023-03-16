from tool.eval_score import fid_l2_psnr_ssim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    task = 'cat2dog'

    if task == 'cat2dog':
        translate_path = 'myoutput/cat2dog/65_100_25_2_egsde_nonoise'
        source_path = '/home/data/afhq/val/cat'
        gt_path = '/home/data/afhq/val/dog'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)

    if task == 'wild2dog':
        translate_path = 'myoutput/[thesubpath]'
        source_path = 'data/afhq/val/wild'
        gt_path = 'data/afhq/val/dog'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)

    if task == 'male2female':
        translate_path = '/home/sunsk/Projects/EGSDE/runs/male2female/2'
        source_path = '/home/data/celeba_hq/val/male'
        gt_path = '/home/data/celeba_hq/val/female'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)





