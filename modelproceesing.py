import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

UPLOAD_FOLDER="./Train"


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})




# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1,3, (1,1))
        )
        self.cnn2 = InceptionResnetV1(pretrained='vggface2').eval()

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512)
            # # nn.ReLU(inplace=True),

            # nn.Linear(256, 128),
            # # nn.ReLU(inplace=True),

            # # nn.Linear(1024,256)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiaritynn.ReLU(inplace=True),

        # nn.Linear(8192, 4096),
        output = self.cnn1(x)
        output = self.cnn2(output)
        # output = model.features(output)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
# net = torch.jit.load('./models/cpu_my_model.pt')

device = torch.device('cpu')
net = SiameseNetwork()
net.load_state_dict(torch.load('./models/siamese_model_pytorch_transfer_inceptionnet_augumented_100x100.pt', map_location = device))


# net = torch.jit.load('./models/cpu_my_model.pt')


test_folder_path = './Checkeye'
validation_folder_path = './Addeye'





def get_test_image(test_folder_path):
  folder_dataset_test = datasets.ImageFolder(root = test_folder_path)
  x0 = Image.open(folder_dataset_test.imgs[0][0])
  x0 = x0.convert('L')
  x0 = transformation(x0)
  x0 = torch.unsqueeze(x0,0)  # 3D tensor to 4D tensor
  return x0


#get the testing image





# create a list of images to comapre with test image
def get_validation_image(path):
    folder_dataset_validation = datasets.ImageFolder(root=path)
    class_to_idx = folder_dataset_validation.class_to_idx
    id = -1
    image_list = []
    for _, tuple0 in enumerate(folder_dataset_validation.imgs):
        img_path = tuple0[0]
        label_idx = tuple0[1]
        y0 = Image.open(img_path)
        y0 = y0.convert('L')
        y0 = transformation(y0)
        y0 = torch.unsqueeze(y0, 0)
        image_list.append([y0, label_idx,img_path])
        id = label_idx

    return class_to_idx, image_list



# print(class_to_idx)

def check_image_similarity():
    class_to_idx, image_list = get_validation_image(validation_folder_path)
    x0 = get_test_image(test_folder_path)
    prediction_values  = []
    for _, tuple0 in enumerate(image_list):
        y1 = tuple0[0]
        label_idx = tuple0[1]
        concatenated = torch.cat((x0, y1), 0)
        output1, output2 = net(x0, y1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        # imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
        prediction_values.append([float("{:.5f}".format(euclidean_distance.item())) ,label_idx,tuple0[2]])

    prediction_values.sort()
    print(prediction_values)
    return prediction_values[0],class_to_idx


# pd=check_image_similarity()
# print(pd)
# prid=check_image_similarity()
# print(prid)

def sol(ch):
    (value, index, _), class_to_idx = check_image_similarity()
    print(class_to_idx)
    idx_to_class = dict((v, k) for k, v in class_to_idx.items())
    if value < 1.2:
        percent = ((1.2 - value) / 1.2 * 100)
        img_class = idx_to_class[index]
        folder_path = os.path.join(UPLOAD_FOLDER, img_class)
        img_name = os.listdir(folder_path)[0]
        imag = 'Train/' + str(img_class) + "/" + str(img_name)
        return imag
    else:
       print("image is not in dataset")

       return



