import cv2
import numpy as np
import os

def create_dataset_from_csv(
        dataset_path, 
        output_path, 
        dataset_type="train"):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    label_file_path = os.path.join(output_path, dataset_type) + ".txt"
    with open(dataset_path, "r") as file, open(label_file_path, "w") as train_file:
        for idx, line in enumerate(file):
            line = line.strip().split(",")
            label = line[0]
            image_name = f"{idx}.png"
            image = np.array(line[1:], dtype=np.uint8).reshape(28, 28)
            cv2.imwrite(os.path.join(output_path, image_name), image)
            train_file.write(f"{image_name} {label}\n")

            print(f"Processing train with: {idx} images", end="\r")
        
        print(f"Created {dataset_type} dataset with {idx+1} images in {output_path}")

if __name__ == "__main__":
    train_dataset = "mnist_train.csv"
    test_dataset = "mnist_test.csv"
    create_dataset_from_csv(
        train_dataset, 
        "train", 
        "train")
    create_dataset_from_csv(
        test_dataset, 
        "test", 
        "test")

