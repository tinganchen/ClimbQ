import pandas as pd
import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, default = '/data_dir/path/cifar10', help = 'Data directory')
parser.add_argument('--output_csv', type = str, default = 'cifar10.csv', help = 'Output csv')
parser.add_argument('--gamma', type = int, default = 10, help = 'Ratio of long tail')

args = parser.parse_args()

def main(args):
    classes = os.listdir(os.path.join(args.data_dir, 'train'))
    imgs = glob.glob(os.path.join(args.data_dir, 'train', '*/*.*'))
    imgs = ['/'.join(img.split('/')[-3:]) for img in imgs]
    
    # count number of images per class
    num_imgs_per_class = []
    clss_imgs_list = []
    COUNT = 0
    for clss in classes:
        img_list = []
        for img in imgs:
            if clss == img.split('/')[1]:
                COUNT += 1
                img_list.append(img)
        clss_imgs_list.append(img_list)
        num_imgs_per_class.append(COUNT)
        COUNT = 0
    
    # sort by number of images
    clss_order = np.argsort(num_imgs_per_class)[::-1]
    classes = np.array(classes)[clss_order]
    num_imgs_per_class = np.array(num_imgs_per_class)[clss_order]
    clss_imgs_list = np.array(clss_imgs_list)[clss_order]
    
    # count long-tail image number per class
    n_classes = len(classes)
    EXP_RATIO = 1 / args.gamma ** (1/(n_classes-1))
    
    num_major = len(clss_imgs_list[0])
    
    lt_num_imgs_per_class = [int(num_major)] 
    
    
    for i in range(1, len(classes)):
        lt_num_imgs = min(int(lt_num_imgs_per_class[i-1] * EXP_RATIO), 
                          num_imgs_per_class[i])
        lt_num_imgs_per_class.append(lt_num_imgs)
                                                    
    # sample images per class
    lt_imgs = []
    lt_labels = []
    for i in range(len(classes)):
        np.random.seed(i)
        num_sampled = lt_num_imgs_per_class[i]
        sampled = np.random.choice(range(len(clss_imgs_list[i])), 
                                   num_sampled, replace = False)
        lt_imgs.extend(np.array(clss_imgs_list[i])[sampled])
        lt_labels.extend([i]*num_sampled)
    
    n = 4
    train_csv = args.output_csv[:-n] + f'_train{args.gamma}' + args.output_csv[-n:]
    
    train_df = pd.DataFrame(columns = ['image', 'label'])
    train_df['image'] = list(lt_imgs)
    train_df['label'] = lt_labels
    
    train_df.to_csv(f'{train_csv}', index = False)
    
    
    # testing csv
    test_csv = args.output_csv[:-n] + f'_test' + args.output_csv[-n:]
    
    if not os.path.isfile(f'{test_csv}'):
        imgs = glob.glob(os.path.join(args.data_dir, 'val', '*/*.*'))
        img_classes = [img.split('/')[-2] for img in imgs]
        labels = [list(classes).index(img_class) for img_class in img_classes]
        
        test_df = pd.DataFrame(columns = ['image', 'label'])
        imgs = ['/'.join(img.split('/')[-3:]) for img in imgs]
        test_df['image'] = list(imgs)
        test_df['label'] = labels
        
        test_df.to_csv(f'{test_csv}', index = False)


if __name__ == '__main__':
    main(args)