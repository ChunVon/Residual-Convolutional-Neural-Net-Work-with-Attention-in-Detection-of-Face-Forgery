# headers
from email.errors import ObsoleteHeaderDefect
from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torch.autograd
import numpy
import os


torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# constant
DROP_OUT_RATE = 0.5
LEARN_RATE = 0.02
EPCO = 50
BATCH_SIZE = 5


# model
class ResNet_Attention(nn.Module):
    def __init__(self):
        super(ResNet_Attention, self).__init__()
        # activaion functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # batch normalization function
        self.batchnormalize_block_1 = nn.BatchNorm2d(32)
        self.batchnormalize_block_2 = nn.BatchNorm2d(96)
        self.batchnormalize_block_3 = nn.BatchNorm2d(256)
        self.batchnormalize_block_4 = nn.BatchNorm2d(768)

        # attention function
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)

        # pooling function
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        # 1*1 convolutional kernel to change the channels and sizes of the residuel part
        self.convol_1_1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )  # output: 112*112*32
        self.convol_1_1_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )  # output: 56*56*96
        self.convol_1_1_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )  # output: 28*28*256
        self.convol_1_1_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=768,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )  # output: 14*14*768

        # block-1-convolution
        self.convolution1_1 = nn.Sequential(
            nn.Conv2d(  # input: 224*224*3
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.convolution1_2 = nn.Sequential(
            nn.Conv2d(  # input: 224*224*16
                in_channels=16,
                out_channels=24,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 224*224*24
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.convolution1_3 = nn.Sequential(
            nn.Conv2d(  # input: 224*224*24
                in_channels=24,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 224*224*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
            # output: 224*224*32
        )

        # block-1-attention
        self.channel_attetion_mlp1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 224*224*6
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=6,
                out_channels=3,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 224*224*3
            nn.Sigmoid()
        )

        self.spatial_attention1_1 = nn.Sequential(
            nn.Conv2d(  # input: 112*112*2
                in_channels=2,
                out_channels=1,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 112*112*1
            nn.Sigmoid()
        )

        # block-2-convolution
        self.convolution2_1 = nn.Sequential(
            nn.Conv2d(  # input: 112*112*32
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
            ),  # output: 112*112*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.convolution2_2 = nn.Sequential(
            nn.Conv2d(  # input: 112*112*64
                in_channels=64,
                out_channels=80,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 112*112*80
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )

        self.convolution2_3 = nn.Sequential(
            nn.Conv2d(  # input: 112*112*80
                in_channels=80,
                out_channels=96,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 112*112*96
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
            # output: 112*112*96
        )

        # block-2-attention
        self.channel_attetion_mlp2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 112*112*16
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 112*112*32
            nn.Sigmoid()
        )

        self.spatial_attention2_1 = nn.Sequential(
            nn.Conv2d(  # input: 112*112*2
                in_channels=2,
                out_channels=1,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 112*112*1
            nn.Sigmoid()
        )

        # block-3-convolution
        self.convolution3_1 = nn.Sequential(
            nn.Conv2d(  # input: 56*56*96
                in_channels=96,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
            ),  # output: 224*224*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.convolution3_2 = nn.Sequential(
            nn.Conv2d(  # input: 56*56*7128
                in_channels=128,
                out_channels=192,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 56*56*192
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.convolution3_3 = nn.Sequential(
            nn.Conv2d(  # input: 56*56*192
                in_channels=192,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
            # output: 56*56*256
        )

        # block-3-attention
        self.channel_attetion_mlp3_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 56*56*48
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 56*56*96
            nn.Sigmoid()
        )

        self.spatial_attention3_1 = nn.Sequential(
            nn.Conv2d(  # input: 56*56*2
                in_channels=2,
                out_channels=1,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 56*56*1
            nn.Sigmoid()
        )

        # block-4-convolution
        self.convolution4_1 = nn.Sequential(
            nn.Conv2d(  # input: 28*28*256
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
            ),  # output: 224*224*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.convolution4_2 = nn.Sequential(
            nn.Conv2d(  # input: 28*28*512
                in_channels=512,
                out_channels=640,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 28*28*640
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True)
        )

        self.convolution4_3 = nn.Sequential(
            nn.Conv2d(  # input: 28*28*640
                in_channels=640,
                out_channels=768,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 28*28*768
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
            # output: 14*14*768
        )

        # block-4-attention
        self.channel_attetion_mlp4_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 28*28*128
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)
            ),  # output: 28*28*256
            nn.Sigmoid()
        )

        self.spatial_attention4_1 = nn.Sequential(
            nn.Conv2d(  # input: 28*28*2
                in_channels=2,
                out_channels=1,
                kernel_size=(5, 5),
                stride=1,
                padding=(2, 2)
            ),  # output: 28*28*1
            nn.Sigmoid()
        )

        # full connected layer as classification
        self.full_connect = nn.Sequential(
            nn.Linear(
                in_features=14*14*768,
                out_features=2352,  # 28*28*768 / 64
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=2352,
                out_features=147,  # 9408 / 64
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=147,
                out_features=2,  # two categories: true and fake
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # block-1
        x = self.convolution1_1(x)
        x = self.convolution1_2(x)
        x = self.convolution1_3(x)

        # share aguments of full-connected layer
        channel_avg = self.channel_attetion_mlp1_1(self.channel_avg(identity))
        channel_max = self.channel_attetion_mlp1_1(self.channel_max(identity))
        channel_out = self.sigmoid(channel_avg + channel_max)

        identity = identity * channel_out

        saptial_avg = torch.mean(identity, dim=1, keepdim=True)
        spatial_max, _ = torch.max(identity, dim=1, keepdim=True)
        spatial_out = torch.cat([saptial_avg, spatial_max], dim=1)
        spatial_out = self.spatial_attention1_1(spatial_out)

        identity = identity * spatial_out

        identity = self.convol_1_1_1(identity)
        x = self.maxpool(x)  # x: 112*112*32

        x = x + identity
        x = self.relu(x)
        x = self.batchnormalize_block_1(x)
        identity = x

        # block-2
        x = self.convolution2_1(x)
        x = self.convolution2_2(x)
        x = self.convolution2_3(x)

        # share aguments of full-connected layer
        channel_avg = self.channel_attetion_mlp2_1(self.channel_avg(identity))
        channel_max = self.channel_attetion_mlp2_1(self.channel_max(identity))
        channel_out = self.sigmoid(channel_avg + channel_max)

        identity = identity * channel_out

        saptial_avg = torch.mean(identity, dim=1, keepdim=True)
        spatial_max, _ = torch.max(identity, dim=1, keepdim=True)
        spatial_out = torch.cat([saptial_avg, spatial_max], dim=1)
        spatial_out = self.spatial_attention2_1(spatial_out)

        identity = identity * spatial_out

        identity = self.convol_1_1_2(identity)
        x = self.maxpool(x)

        x = x + identity
        x = self.relu(x)
        x = self.batchnormalize_block_2(x)
        identity = x

        # block-3
        x = self.convolution3_1(x)
        x = self.convolution3_2(x)
        x = self.convolution3_3(x)

        # share aguments of full-connected layer
        channel_avg = self.channel_attetion_mlp3_1(self.channel_avg(identity))
        channel_max = self.channel_attetion_mlp3_1(self.channel_max(identity))
        channel_out = self.sigmoid(channel_avg + channel_max)

        identity = identity * channel_out

        saptial_avg = torch.mean(identity, dim=1, keepdim=True)
        spatial_max, _ = torch.max(identity, dim=1, keepdim=True)
        spatial_out = torch.cat([saptial_avg, spatial_max], dim=1)
        spatial_out = self.spatial_attention3_1(spatial_out)

        identity = identity * spatial_out

        identity = self.convol_1_1_3(identity)
        x = self.maxpool(x)

        x = x + identity
        x = self.relu(x)
        x = self.batchnormalize_block_3(x)
        identity = x

        # block-4
        x = self.convolution4_1(x)
        x = self.convolution4_2(x)
        x = self.convolution4_3(x)

        # share aguments of full-connected layer
        channel_avg = self.channel_attetion_mlp4_1(self.channel_avg(identity))
        channel_max = self.channel_attetion_mlp4_1(self.channel_max(identity))
        channel_out = self.sigmoid(channel_avg + channel_max)

        identity = identity * channel_out

        saptial_avg = torch.mean(identity, dim=1, keepdim=True)
        spatial_max, _ = torch.max(identity, dim=1, keepdim=True)
        spatial_out = torch.cat([saptial_avg, spatial_max], dim=1)
        spatial_out = self.spatial_attention4_1(spatial_out)

        identity = identity * spatial_out

        identity = self.convol_1_1_4(identity)
        x = self.maxpool(x)

        x = x + identity
        x = self.relu(x)
        # x = self.batchnormalize_block_4(x)
        # identity = x

        x = x.view(x.size(0), -1)  # convert to vector in column
        out = self.full_connect(x)

        return out
# end


# construct dataset:  "deepfake_in_the_wild"
train_data = torchvision.datasets.ImageFolder(
    "/home/dataset/train_data/",  # root
    transform=torchvision.transforms.Compose([  # data argumentation
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomVerticalFlip(0.5),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
)
test_data = torchvision.datasets.ImageFolder(
    "/home/dataset/test_data/",  # root
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
)
print(" Sucessfully construct train dataset.\n The classes and the labels are: ",
      train_data.class_to_idx)

# load data
print("begin load data")
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True)
print("train data finished loading")
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True)
print("test data finished loading")
# end

device = torch.device("cuda")
my_model = ResNet_Attention()
my_model = my_model.to(torch.float32).to(device)
# my_model.load_state_dict(torch.load("16000my_model.pth"))


# optimize and loss function
optim_func = torch.optim.Adam(my_model.parameters(), lr=LEARN_RATE)
loss_func = nn.CrossEntropyLoss()


# train
print("begin to train")
my_model = my_model.train()
for epco in range(EPCO):
    print("the ", epco, "epco")
    loss_batch_avg = 0
    for batch_idx, (data_image, data_label) in enumerate(train_loader):
        # save training data in files
        print("the ", batch_idx, "batch")
        data_image = data_image.to(torch.float32).to(device)
        data_label = data_label.to(device)
        train_results = my_model(data_image)
        optim_func.zero_grad()  # clear gradients
        train_loss = loss_func(train_results, data_label)
        train_loss.backward()
        optim_func.step()

        loss_batch_avg += train_loss.item()

        print("loss: ", train_loss.item())

        loss_file_detail = open("loss_in_train_detail.txt", "a")
        loss_file_detail.write(
            str(epco)+" "+str(batch_idx)+" "+str(train_loss.item())+"\n"
        )
        loss_file_detail.close()
        if (batch_idx % 1000 == 0) and (batch_idx != 0):
            torch.save(my_model.state_dict(), str(batch_idx)+"my_model.pth")

    loss_batch_avg = loss_batch_avg / (batch_idx + 1)
    loss_file_avg = open("loss_in_train_avg.txt", "a")
    loss_file_avg.write(
        str(epco)+" "+str(loss_batch_avg)+"\n"
    )
    loss_file_avg.close()
    torch.save(my_model.state_dict(), "my_model.pth")

# test
print("begin to test")

my_model.eval()
test_accuracy = 0
test_accuracy_avg = 0
correct = torch.zeros(1).squeeze()
total = torch.zeros(1).squeeze()

for batch_idx, (data_image, data_label) in enumerate(test_loader):
    print("the ", batch_idx, "batch")
    data_image = data_image.to(torch.float32).to(device)
    data_label = data_label.to(device)
    test_result = my_model(data_image)
    test_loss = loss_func(test_result, data_label)
    prediction = torch.argmax(test_result, 1)
    correct += (prediction == data_label).sum().float()
    total += len(data_label)
    test_accuracy = (correct/total).cpu().detach().data.numpy()
    test_accuracy_avg += test_accuracy

    test_loss_file = open("test_loss.txt", "a")
    test_loss_file.write(
        str(batch_idx)+" "+str(test_loss.item())+"\n"
    )
    test_loss_file.close()
    test_accuracy_file = open("test_acuracy.txt", "a")
    test_accuracy_file.write(
        str(batch_idx)+" "+str(test_accuracy)+"\n"
    )
    test_accuracy_file.close()

    test_accuracy = 0

test_accuracy_avg = test_accuracy_avg / (batch_idx + 1)
test_accuracy_file = open("test_acuracy.txt", "a")
test_accuracy_file.write(
    "avg: " + str(test_accuracy_avg)+"\n"
)
test_accuracy_file.close()
