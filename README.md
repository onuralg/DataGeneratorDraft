# DataGeneratorDraft
Keras DataGenerator draft used for different tasks


- If your RAM is big enough for your problem, this kind of code block might solve your data read problem: 
    - X, y = np.load('some_training_set_with_labels.npy')  # Load entire dataset

- If your RAM is not big enough and you are doing simple classification (binary or multi-label classification), you can use ImageDataGenerator of Keras library directly. 
    - https://keras.io/preprocessing/image/
    - “.flow_from_directory(directory)” might help for reading images and their labels from directory and subdirectories.Subdirectories are used as labels.

- If your RAM is not big enough and you are doing a different task rather than simple classification, for instance, if you need to give multiple images to your network or retrieve multiple labels from your network such as tasks like one-shot learning, multi-task learning, segmentation etc., you must override your own DataGenerator specific to your own problem.
    - This is a very good tutorial on DataGenerators: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    - By using the blog post above, I wrote my own DataGenerator class for multi-task learning problem 
        - You need to give four text files for both training and testing (train-data.txt, train-label.txt, test-data.txt, test-label.txt)
        - train-data.txt and test-data.txt contain file paths of images
        - train-label.txt and test-label.txt contain one-hot encoding vectors of image labels
        





