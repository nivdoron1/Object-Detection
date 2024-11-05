# Project 2 - Object Detection

# Object Detection System

## Overview

**Authors:** Niv Doron

To present inference results from a pre-trained MobileNet V3 network, we need to consider its application in a typical object detection or classification task. Since MobileNet V3 can be used both for image classification and as a backbone for more complex object detection frameworks (like SSD or YOLO), the inference results will vary based on the specific application and dataset used. For the purpose of this discussion, we'll focus on a generic scenario of image classification on the ImageNet dataset, a common benchmark for such models.

### **Inference Results on ImageNet**

The ImageNet dataset is a large-scale dataset used for image classification, containing over 14 million images and 1000 classes. It's a standard benchmark for evaluating the performance of deep learning models in computer vision.

### Accuracy Metrics

MobileNet V3 models are typically evaluated using two primary metrics:

- **Top-1 Accuracy**: This is the percentage of times the model's highest-confidence prediction (the top prediction) matches the ground truth label.
- **Top-5 Accuracy**: This is the percentage of times the correct label is among the model's top 5 predictions.

For MobileNet V3, the results generally showcase its efficiency and accuracy. Although specific numbers might vary slightly depending on the exact version and configuration (e.g., Large or Small, input resolution, etc.), MobileNet V3 Large models typically achieve top-1 accuracy in the range of 75.0% to 75.7% on the ImageNet validation set, with top-5 accuracy reaching above 92%. The Small versions, designed for even lower computational budgets, show slightly lower accuracy but still perform remarkably well given their constraints.

### Efficiency Metrics

- **Latency**: MobileNet V3 is designed with a focus on reducing latency, making it suitable for real-time applications on mobile devices. Latency can vary based on the device and the specific model variant, but the architecture's optimizations ensure it remains competitive.
- **Model Size and Parameters**: The model's efficiency is also evident in its size, with MobileNet V3 models having significantly fewer parameters compared to other architectures with similar accuracy levels. This reduction in model size and computational complexity is crucial for deployment on mobile and edge devices.

### **Demonstrating Capabilities**

To demonstrate the capabilities of a pre-trained MobileNet V3 model, one can run inference on a set of diverse images from the ImageNet validation set or real-world images. The inference process typically involves:

- Preprocessing the input images to match the input size and format expected by the model (e.g., resizing, normalization).
- Feeding the preprocessed images through the model to obtain predictions.
- Analyzing the predictions to extract the top-1 and top-5 predicted classes along with their confidence scores.

These steps can be easily implemented using machine learning frameworks like TensorFlow or PyTorch, which provide pre-trained MobileNet V3 models.

MobileNet V3 is a compact and efficient architecture that is particularly well-suited for mobile and edge devices due to its balance between latency and accuracy. This analysis will explore the architecture's innovative aspects, including the introduction of new blocks or components not covered in standard course lectures, and discuss its inference results based on original research papers and additional resources.

### MobileNet V3: Overview

MobileNet V3 is a continuation of the MobileNet series, designed by Google researchers to provide highly efficient neural networks for mobile vision applications. It incorporates lessons from MobileNetV1 and V2, and introduces new concepts such as lightweight attention mechanisms, which further enhance its efficiency and performance. The architecture is optimized through a combination of hardware-aware network design and AutoML, specifically through the use of the NetAdapt algorithm and the novel architecture search that focuses on finding an optimal balance between latency and accuracy.

### Key Innovations in MobileNet V3

1. **Lightweight Attention Mechanisms**: MobileNet V3 introduces the use of squeeze-and-excitation (SE) blocks, a form of lightweight attention mechanism that allows the network to recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels. This mechanism improves the representational power of the network without significant computational overhead.
2. **Hard-Swish Activation**: Another innovation is the introduction of the hard-swish activation function, which is a computationally efficient version of the swish function.  This function provides a non-linear activation that helps in capturing complex patterns in the data while being more efficient for computation on mobile devices.
3. **Architecture Search for Component Blocks**: The design of MobileNet V3 involved a mix of automated search techniques and expert knowledge. The network architecture was optimized for mobile devices by using a combination of AutoML and manual tuning, focusing on reducing latency without compromising accuracy. This resulted in a fine-grained design of the depthwise separable convolutions and the introduction of minimalistic bottlenecks.
4. **Optimized Last Stage**: The architecture tailors the last stage of the network to significantly reduce computation while ensuring feature richness for the task at hand. This includes modifications to the expansion layers and the addition of efficient pooling layers.

### Performance and Inference Results

MobileNet V3 demonstrates superior performance in terms of accuracy and efficiency on various benchmarks, including ImageNet for classification tasks. Its design choices, focusing on reducing computational cost and memory usage, make it highly suitable for real-time object detection tasks on mobile devices. The network achieves this with minimal loss in accuracy, showcasing the effectiveness of its lightweight attention mechanisms and optimized layers.

### Sum Up

The MobileNet V3 architecture represents a significant advancement in the development of efficient and powerful neural networks for mobile vision applications. Its innovative components, such as the squeeze-and-excitation blocks and hard-swish activation, contribute to its high performance on classification and object detection tasks. By leveraging AutoML and expert knowledge for its design, MobileNet V3 strikes an optimal balance between latency and accuracy, making it an ideal backbone for object detection models intended for use in resource-constrained environments.

# Project summary:

This code is a comprehensive example of training a Faster R-CNN object detection model using a custom dataset, specifically formatted in the Pascal VOC format.

### Setup and Imports

- **Installation**: Installs required Python packages `matplotlib` for plotting and `roboflow` for accessing the RoboFlow dataset API.
- **Imports**: Loads necessary libraries and modules for handling datasets, neural networks, image transformations, and file operations.

### Data Preparation

- **Dataset Download**: Uses the `Roboflow` API to download a specific version of the "pascal-voc-2012" dataset.
- **Drive Mounting**: Mounts Google Drive for accessing and saving files directly in Google Colab.
- **PascalVOCDataset Class**: Defines a custom `Dataset` class for loading and processing the Pascal VOC dataset. It reads images and their corresponding annotations (bounding boxes and class labels), applies optional transformations, and formats the data for the object detection model.

The **`PascalVOCDataset`** class is a custom implementation designed to load the Pascal VOC dataset, which is a popular dataset for object detection containing images annotated with bounding boxes and class labels for objects. This dataset class extends PyTorch's **`Dataset`** class, making it compatible with PyTorch's data loading utilities such as **`DataLoader`**. The implementation focuses on reading images and their corresponding annotations, applying transformations if specified, and formatting the data into a structure suitable for training an object detection model.

Here's a simplified explanation of the **`PascalVOCDataset`** class along with a code example:

### **Key Components of PascalVOCDataset:**

1. **Initialization (`__init__` method)**: The constructor takes three parameters: the root directory of the dataset (**`root_dir`**), the dataset split (**`split`**) which could be 'train' or 'valid' (validation), and an optional **`transform`** argument for applying data augmentation or preprocessing transformations.
2. **Length (`__len__` method)**: This method returns the number of items in the dataset.
3. **Get Item (`__getitem__` method)**: This crucial method retrieves an image and its annotations (targets) by index (**`idx`**). It performs the following steps:
    - Loads the image using the **`PIL.Image`** module.
    - Reads the corresponding annotation file (in XML format) to extract bounding boxes (**`boxes`**), labels (**`labels`**), area of the boxes (**`area`**), and a binary flag indicating if the instance is crowded (**`iscrowd`**).
    - Converts the annotations into PyTorch tensors and constructs a dictionary (**`target`**) containing these tensors.
    - Applies the specified transformations to the image (if any).
4. **Class to Index Mapping (`class_to_idx` method)**: A helper method that maps class names to integer labels. This is necessary because the model expects numerical labels.

### Model Preparation

- **Transformations**: Specifies image transformations for training and validation datasets, including conversion to tensor format and random horizontal flipping.
- **Dataloader Preparation**: Sets up `DataLoader` instances for batching, shuffling, and parallel data loading for the training and validation datasets.
- **Model Definition**: Describes the process of creating a Faster R-CNN model with a MobileNet backbone. This involves setting up the backbone network, anchor generator, and region of interest (RoI) pooler, and combining them into the final model.
- **Hyperparameters**: Defines various hyperparameters for training, such as the learning rate, batch size, and scheduler settings.

**visulaziation dataset with bbox**

![הורדה (9).png](_(9).png)

![הורדה (8).png](_(8).png)

**THE MODEL**

This code snippet demonstrates how to customize a pre-trained Faster R-CNN model with a MobileNet V3 large backbone and a 320-pixel FPN (Feature Pyramid Network) for a specific object detection task that requires detecting a custom number of classes. Let's break down the key components and steps involved in this process:

### Importing Required Modules

First, the necessary classes and functions are imported from `torchvision`:

- `fasterrcnn_mobilenet_v3_large_320_fpn`: This function loads a Faster R-CNN model pre-trained on the COCO dataset with a MobileNet V3 Large backbone and a 320-pixel feature pyramid network for multi-scale feature extraction.
- `FastRCNNPredictor`: A class to create a new predictor for the classification of objects, which will replace the existing predictor in the pre-trained model to adapt to the number of classes in the new dataset.
- `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights`: A class that provides access to pre-trained weights for the specified architecture, ensuring the model is initialized with weights that are optimized for object detection tasks.

### Defining the Model Customization Function

A function named `get_model` is defined to customize the pre-trained model for a specific number of object classes (`num_classes`):

### Loading the Pre-trained Model

```python
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

```

- The model is initialized with pre-trained weights (`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`), which helps in leveraging transfer learning for improving performance on the new object detection task.

### Customizing the Predictor

```python
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

```

- The number of input features (`in_features`) for the classifier in the region of interest (RoI) head is obtained from the pre-trained model's box predictor.
- A new `FastRCNNPredictor` is created with the same number of input features (`in_features`) but with the output size set to `num_classes`, the number of classes for the new detection task.

### Training Loop

- **Training Functionality**: Implements functions for training and validation phases, including loss calculation, backpropagation, model evaluation using Intersection over Union (IoU) as a metric, and utilities for saving model checkpoints.
- **Training Execution**: Executes the training process for a specified number of epochs, logging the training loss and validation accuracy to TensorBoard, and saving model checkpoints periodically.

explain the functions:

The **`train_one_epoch`** function encapsulates the training logic for one epoch of the training loop for an object detection model using PyTorch.

### **Parameters:**

- **`epoch_index`**: The current epoch number. It's used for logging purposes.
- **`train_dataloader`**: A PyTorch DataLoader instance that provides batches of images and their corresponding targets (annotations) for training.
- **`tb_writer`**: A TensorBoard writer object used for logging training metrics, allowing you to visualize them in TensorBoard.

### **Function Operation:**

```python
#load the data for the 5 backbones sammples
for param in model.backbone.parameters():
    param.requires_grad = False
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
```

### Initialization of Running Losses:

Variables to accumulate losses over batches are initialized to zero. This includes the total loss and specific losses like classifier loss, bounding box regression loss, objectness loss, and RPN (Region Proposal Network) bounding box regression loss.

### Batch Processing Loop:

The function iterates over each batch of images and their targets provided by **`train_dataloader`**:

1. **Filter Valid Samples**: Not all samples might be valid (e.g., some might not have any bounding boxes). This step filters out such samples to ensure the model is trained only on valid data.
2. **Device Assignment**: The images and targets are moved to the appropriate device (e.g., GPU), which is essential for computation.
3. **Zero Gradients**: Before the model can backpropagate, the gradients are zeroed out to prevent accumulation from previous iterations.
4. **Model Forward Pass and Loss Calculation**:
    - The model makes predictions for the batch of images.
    - Losses are calculated based on the difference between the predictions and the ground truth targets. These losses include the classifier loss, box regression loss, objectness loss, and RPN box regression loss.
5. **Backward Propagation and Optimization Step**:
    - The gradients of the losses are computed.
    - Gradient clipping is applied to avoid exploding gradients.
    - The optimizer updates the model parameters based on the gradients.
6. **Update Running Losses**: The running losses are updated with the losses from the current batch. These accumulations are used to calculate average losses over a fixed number of batches.
7. **Logging**: Every 50 batches, the average losses for the last 50 batches are logged, and the TensorBoard writer logs these metrics for visualization. This helps in monitoring the training progress.

### Return Value:

- After completing the iteration over all batches in the DataLoader, the function returns the average loss for the last batch as an indication of the training loss for the epoch.

```python
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch and calculate the losses
        loss_dict = model(images, targets)

        # Sum up all losses
        losses = sum(loss for loss in loss_dict.values())

        # Compute the backward gradients and adjust learning weights
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
```

### **valid data loader accuracy**

The Intersection over Union (IoU) is a metric used to evaluate the accuracy of an object detector on a dataset, measuring the overlap between predicted and ground truth bounding boxes. It's calculated as the area of overlap between the two boxes divided by the area of their union, with a value ranging from 0 (no overlap) to 1 (perfect overlap).

Here's the code snippet for calculating IoU:

```python
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    union_area = ((box1[2]-box1[0]) * (box1[3]-box1[1])) + ((box2[2]-box2[0]) * (box2[3]-box2[1])) - intersection_area
    return intersection_area / union_area if union_area != 0 else 0

```

This function calculates IoU by finding the intersection area and the union area of two bounding boxes (`box1`, `box2`) and returns their ratio as the IoU score.

results of loss function after freezing:

![הורדה (5).png](_(5).png)

The training loss graph displays a consistent decrease over five epochs, indicating that the model is learning effectively with no signs of overfitting or convergence issues at this early stage of training.

The training loop fine-tunes a pre-trained model for additional epochs, selectively freezing layers, training, and validating the model's performance. Here's an overview in six lines with key code snippets:

1. **Selective Layer Freezing**: Based on the layer's name, decide whether to freeze it (`requires_grad = False`) to prevent updates during backpropagation, focusing fine-tuning on specific layers of the model.
    
    ```python
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = False if name.split(".")[0] in ["0", "1", "2", "3"] else True
    
    ```
    
2. **Optimizer Setup**: Define the optimizer, specifying parameters to update, learning rate, momentum, and weight decay.
    
    ```python
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    ```
    
3. **Learning Rate Scheduler**: Adjust the learning rate at specified intervals to fine-tune training.
    
    ```python
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
    
    ```
    
4. **Training Loop**: Iterate over epochs, setting the model to training mode, training for one epoch, then evaluating model performance.
    
    ```python
    for epoch in range(init_epochs, num_epochs + init_epochs):
        avg_loss = train_one_epoch(epoch, train_dataloader, writer)
    
    ```
    
5. **Validation and Logging**: After training, validate the model, log accuracy and loss, and save the checkpoint.
    
    ```python
    accuracy = validate(val_dataloader, device)
    save_checkpoint(model, optimizer, epoch, avg_loss, accuracy, "checkpoint.pth")
    
    ```
    
6. **Learning Rate Adjustment**: Update the learning rate based on the scheduler after each epoch to optimize training.
    
    ```python
    lr_scheduler.step()
    
    ```
    

This loop efficiently fine-tunes a neural network by focusing on updating parameters in specific layers, using an optimizer to minimize loss, validating model performance, logging key metrics, saving checkpoints for future use or analysis, and dynamically adjusting the learning rate to improve training outcomes.

### accuracy and loss function

![הורדה (11).png](_(11).png)

The accuracy chart shows a positive trend over 10 epochs, with model performance improving significantly, particularly between epochs 1 to 5, and then achieving a more stable, yet still increasing accuracy from epochs 5 to 10.

![הורדה (10).png](_(10).png)

Over the course of 10 epochs following the freezing of the backbone, the training loss exhibits an unusual spike at epoch 5 but otherwise trends downward overall, indicating a learning process with some variability but general improvement in the model's performance.

### Visualization and Inference

- **Visualization**: Contains functions for visualizing the dataset images with annotated bounding boxes and for plotting training/validation loss and accuracy over epochs.

- **Prediction on New Data**: Demonstrates how to load a trained model checkpoint and use it to make predictions on new images or videos, including drawing predicted bounding boxes on images.

The `predict` function runs a forward pass on a preprocessed image using a trained model to detect objects, returning a list of classified objects with their bounding boxes and scores, filtered by a confidence threshold.

```python
# Process the model's prediction to filter out objects with a confidence score above a threshold
classified_objects = [{"class": classes[label], "box": box.tolist(), "score": score.item()}
                      for label, box, score in zip(labels, boxes, scores) if score > threshold]

```

### 

The `save_video_with_predicted_bboxes` function processes a video frame-by-frame to detect objects using a trained model, draws bounding boxes around detected objects, and saves the annotated video to a specified path.

```python
# This snippet draws the predicted bounding box and class label on each frame
cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
cv2.putText(frame, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

```

# SUM UP

This project involves fine-tuning a pre-trained object detection model for a specific dataset. Key components include:

- **Selective Layer Freezing**: Certain layers of the neural network are frozen during fine-tuning to retain learned features and focus training on the upper layers.
- **Training and Validation Loops**: The model is trained over several epochs, with performance metrics such as loss and accuracy recorded to monitor progress. Validation is performed to ensure the model generalizes well to unseen data.
- **Intersection over Union (IoU)**: A metric to quantify the accuracy of predicted bounding boxes against ground truth, guiding the model's precision in localization tasks.
- **Prediction on New Data**: The model's capability is showcased through functions that predict and annotate objects within images and videos.
- **Utility Functions**: The codebase includes functions for batching efficiency and evaluation metrics, supporting effective training and insightful model performance assessment.

The outcome is a robust model tailored to accurately detect and classify objects within a given domain, evidenced by performance metrics tracked across training epochs.

**URL TO WATCH THE OUTPUT VIDEO**

[https://www.youtube.com/watch?v=lcUBx3zpFm0](https://www.youtube.com/watch?v=lcUBx3zpFm0)