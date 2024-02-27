# BadNets Implementation on MNIST Dataset

This project marked my initial foray into machine learning research, focusing on BadNets—a specific form of backdoor attack. A backdoor attack involves subtly modifying AI models during their training phase to cause them to behave incorrectly under certain conditions, without affecting their performance on normal inputs. My journey into this project began after immersing myself in numerous tutorials on PyTorch and deep learning, followed by a comprehensive study of the BadNets research paper. This endeavor is a component of a broader initiative to develop a comprehensive library of backdoor and adversarial attacks.

In the context of BadNets, the attack mechanism involves embedding specific pixels or pixel patterns into images within the dataset, serving as triggers. To implement this attack, I adjusted the tensor objects within the data loader and modified the associated labels. The goal was to train the model to misclassify images containing these triggers while accurately identifying those without. Remarkably, the project achieved both clean data accuracy and attack success rates exceeding 95%.

Throughout this project, Jupyter Notebook proved to be an invaluable tool, especially for defining functions that allowed for real-time monitoring of the modifications and recognition processes of image tensors. Additionally, I leveraged existing GitHub repositories that feature deep learning models built with PyTorch, which served as practical references. Insights from the original research paper were instrumental in diagnosing and addressing issues that prevented achieving the desired accuracy levels. For more intensive testing phases at higher epochs, I utilized GPUs provided by my research department, facilitating more efficient computation.

<p align="center">
Trigger - An image of 8 mislabelled as 9 with the trigger being white pixel pattern <br/>
<img src="https://i.imgur.com/62TgaWL.png" height="80%" width="80%" alt="Trigger"/>
