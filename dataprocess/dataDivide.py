import os
import shutil
import argparse
import gzip

def extract_images_and_labels(input_folder, output_image_folder, output_mask_folder):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    image_files = [file for file in os.listdir(input_folder) if file.endswith('_image.nii.gz')]
    label_files = [file for file in os.listdir(input_folder) if file.endswith('_label.nii.gz')]

    for image_file in image_files:
        image_name = image_file.replace('_image.nii.gz', '')
        corresponding_label = [label for label in label_files if image_name in label]

        # Extract image file
        with gzip.open(os.path.join(input_folder, image_file), 'rb') as f_in:
            with open(os.path.join(output_image_folder, image_file.replace('_image.nii.gz', '.nii')), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if len(corresponding_label) != 0:
            corresponding_label = corresponding_label[0]
            # Extract corresponding label file
            with gzip.open(os.path.join(input_folder, corresponding_label), 'rb') as f_in:
                with open(os.path.join(output_mask_folder, corresponding_label.replace('_label.nii.gz', '.nii')), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract images and labels from input folder')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing image and label files')
    parser.add_argument('--output_image_folder', type=str, default=None, help='Path to the output folder for images')
    parser.add_argument('--output_mask_folder', type=str, default=None, help='Path to the output folder for masks')
    args = parser.parse_args()

    if args.output_image_folder is None:
        args.output_image_folder = os.path.join(args.input_folder, 'Image')
    if args.output_mask_folder is None:
        args.output_mask_folder = os.path.join(args.input_folder, 'Mask')

    extract_images_and_labels(args.input_folder, args.output_image_folder, args.output_mask_folder)

# python script.py your_input_folder --output_image_folder Image --output_mask_folder Mask
