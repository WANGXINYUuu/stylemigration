from mmgen.apis import init_model, sample_img2img_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# 对单张图片使用训练完的模型进行风格迁移
# 参数说明: 
# input_path —— 输入文件地址
# model —— 装载的模型
# target_domain —— 图像风格域 'Fortnite' or 'PUBG'
# figsize —— pltshow的图像大小
# save_paht —— 结果保存路径
# is_pltshow —— 是否展示到屏幕上
def picture_migration(input_path, model, target_domain, figsize, save_path, is_pltshow):
    input_img = cv2.imread(input_path)
    generated_picture = sample_img2img_model(model, input_path, target_domain = target_domain)
    img_size = generated_picture.shape[2]

    RGB = np.zeros((img_size, img_size, 3))
    RGB[:,:,0] = generated_picture[0][2]
    RGB[:,:,1] = generated_picture[0][1]
    RGB[:,:,2] = generated_picture[0][0]
    RGB = cv2.resize(RGB, dsize = (input_img.shape[1], input_img.shape[0]))
    RGB = 255 * (RGB - RGB.min()) / (RGB.max() - RGB.min())
    RGB = RGB.astype('uint8')

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB))
    if(is_pltshow):
        plt.figure(figsize = (figsize, figsize))
        plt.subplot(1, 2, 1)
        plt.title('input')
        input_RGB = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(input_RGB)

        plt.subplot(1, 2, 2)
        plt.title(target_domain)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(RGB)
        plt.show()
        # plt.waitforbuttonpress(0)
    else:
        return
# 对文件夹中的所有图片进行风格迁移
# 参数说明: 
# input_path —— 输入图片所在的文件夹地址
# model —— 装载的模型
# target_domain —— 图像风格域 'Fortnite' or 'PUBG'
# figsize —— pltshow的图像大小
# save_path —— 输出结果所在的文件夹地址
def pictures_migration(input_path, model, target_domain, figsize, save_path):
    for each in os.listdir(input_path):
        path = input_path + '/' + each
        result_path = save_path + '/' + target_domain + '_' + each
        try:
            picture_migration(path,model,target_domain, figsize, result_path, False)
        except:
            pass

# 对视频使用训练完的模型进行风格迁移
# 参数说明：
# model —— 装载的模型
# img —— 视频中的单帧画面
# target_domain —— 图像风格域'Fortnite' or 'PUBG'
def process_frame(model, img, target_domain):   # 处理单帧
    cv2.imwrite('outputs/F3_temp.jpg', img)
    generated_picture = sample_img2img_model(model, 'outputs/F3_temp.jpg', target_domain = target_domain)
    img_size = generated_picture.shape[2]
    
    RGB = np.zeros((img_size, img_size, 3))
    RGB[:, :, 0] = generated_picture[0][2]
    RGB[:, :, 1] = generated_picture[0][1]
    RGB[:, :, 2] = generated_picture[0][0]
    RGB = cv2.resize(RGB, dsize = (img.shape[1], img.shape[0]))
    RGB = 255 * (RGB - RGB.min()) / (RGB.max() - RGB.min())
    RGB = RGB.astype('uint8')
    RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
    return RGB
# 参数说明：
# model —— 装载的模型
# input_path —— 视频输入路径
# output_path —— 视频输出保存路径
def video_migration(model, target_domain, input_path, output_path = 'data/output.mp4'):
    print('视频开始处理', input_path)
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while(cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))
    with tqdm(total = frame_count-1) as pbar:
        try:
            while(cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break
                try:
                    output = process_frame(model, frame, target_domain)
                except Exception as e:
                    print(e)
                    pass
                if success == True:
                    out.write(output)
                    # 实时展示处理前的视频和处理后的视频
                    imgStack = np.hstack((frame, cv2.cvtColor(output, cv2.COLOR_RGB2BGR)))
                    cv2.imshow('imgStack', imgStack)
                    k = cv2.waitKey(20)
                    if k & 0xff == ord('q'):
                        break
                    pbar.update(1)
        except:
            print('中途中断')
            pass
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)

if __name__ == '__main__':
    in_fortnite_path = 'data/test/Fortnite'
    out_fortnite2pubg_path = 'outputs/f2p'
    in_pubg_path = 'data/test/PUBG'
    out_pubg2fortnite_path = 'outputs/p2f'
    # os.chdir('MMGeneration_Tutorials/mmgeneration')
    # 指定配置文件
    config_file = 'configs/cyclegan/cyclegan_lsgan_resnet_in_facades_b1x1_80k_Fortnite2PUBG.py'
    # 指定训练完的模型文件
    checkpoint_file = 'work_dirs/experiments/cyclegan_Fortnite2PUBG/ckpt/cyclegan_Fortnite2PUBG/latest.pth'
    # 加载模型
    model = init_model(config_file, checkpoint_file, device = 'cuda:0')
    # 对单张图片进行风格迁移
    # picture_migration('data/test/Fortnite/fortnite3.jpg', model, target_domain='PUBG', figsize=20, save_path='outputs/fortnite2pubg.jpg', is_pltshow = True)
    
    # 对文件夹中所有图片进行风格迁移
    # pictures_migration(in_pubg_path, model, 'Fortnite', 20, out_pubg2fortnite_path)
    
    # 对视频进行风格迁移
    video_migration(model, target_domain = 'PUBG', input_path = 'data/Fortnite2_640x360.mp4', output_path='outputs/f2p_640x360.mp4')