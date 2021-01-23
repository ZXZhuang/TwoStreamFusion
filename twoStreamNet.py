from dataloader import *
from fusion import FusionNet
import matplotlib.pyplot as plt
import numpy as np


minibatch_train = 10
minibatch_test = 10
epoch_num = 1000
save_net = 1
BATCH_SIZE = 256
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.01
num_classes = 4
classes = ('brush_hair', 'cartwheel', 'catch', 'chew')

def reload():
    pth = './save/SpatialTemporalNet.pth'
    checkpoint = torch.load(pth)
    net_S = SpatialNet().cuda()
    net_T = TemporalNet().cuda()
    net_S.load_state_dict(checkpoint['net_S_state_dict'])
    net_T.load_state_dict(checkpoint['net_T_state_dict'])
    torch.save({'net_S_state_dict': net_S.state_dict(),
                'net_T_state_dict': net_T.state_dict()}, pth, _use_new_zipfile_serialization=False)

def imshow(img):
    img = img/2 + 0.5 #unnormalize [-1,1] -> [0,1]
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))
    plt.show()

def train(net, pth):
    print('Train start')
    loader = dataloader(minibatch_train, train=True)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(1, epoch_num):
        for i, data in enumerate(loader):
            # get the inputs
            (img, of, target) = data
            img, target, of = img.cuda(), target.cuda(), of.cuda()
            # print(img)
            # print(of)
            # print(target)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(img, of)
            #print(output_S, target_S)
            #print(output_T, target_T)
            loss = loss_function(output, target.to(DEVICE, dtype=torch.int64))
            loss.backward()
            optimizer.step()

            print(loss)

            # get the accuracies
            '''
            result_S = [int(torch.argmax(output_S, 1)[j]) == int(target_S[j]) for j in range(len(target_S))]
            result_T = [int(torch.argmax(output_T, 1)[j]) == int(target_T[j]) for j in range(len(target_T))]
            accuracy_S = round((sum(result_S) / len(result_S)) * 100)
            accuracy_T = round((sum(result_T) / len(result_T)) * 100)
            '''

            # print statistics
            '''
            print('Spatial : [{:03}, {:03}] loss:{:.2f}, accuracy:{:03}%'
                  .format(epoch, i, loss_S.item(), accuracy_S), end='    ', flush=True)
            print('Temporal: [{:03}, {:03}] loss:{:.2f}, accuracy:{:03}%'
                  .format(epoch, i, loss_T.item(), accuracy_T), end='    ', flush=True)
            '''

            if i == len(loader):
                print()
            else:
                print(end='\n')

        if (epoch % save_net) == 0:
            torch.save({'net_state_dict': net.state_dict()}, pth, _use_new_zipfile_serialization=False)
            print('Save Finished')
    print('Train Finished')

def test(net_S, net_T):
    print('Test Start')
    correct_S = 0
    total_S = 0
    correct_T = 0
    total_T = 0
    loader_S, loader_T = dataloader_S(minibatch_test, train=False), dataloader_T(minibatch_test, train=False)
    for i, (data_S, data_T) in enumerate(zip(loader_S, loader_T), 1):
        (img_S, target_S), (img_T, target_T) = data_S, data_T
        img_S, target_S = img_S.cuda(), target_S.cuda()
        img_T, target_T = img_T.cuda(), target_T.cuda()

        with torch.no_grad():
            output_S, output_T = net_S(img_S), net_T(img_T)
            _, predicted_S = torch.max(output_S, 1)
            _, predicted_T = torch.max(output_T, 1)

            result_S = [int(torch.argmax(output_S, 1)[j]) == int(target_S[j]) for j in range(len(target_S))]
            result_T = [int(torch.argmax(output_T, 1)[j]) == int(target_T[j]) for j in range(len(target_T))]
            correct_S += sum(result_S)
            correct_T += sum(result_T)
            total_S += len(result_S)
            total_T += len(result_T)
            accuracy_S = round((sum(result_S) / len(result_S)) * 100)
            accuracy_T = round((sum(result_T) / len(result_T)) * 100)

            print('Spatial : {:03}, accuracy:{:03}%'.format(i, accuracy_S), flush=True)
            print('Predicted   : ', ''.join('%s ' % classes[predicted_S[j]] for j in range(minibatch_test)))
            print('GroundTruth : ', ''.join('%s ' % classes[target_S[j]] for j in range(minibatch_test)))
            #imshow(torchvision.utils.make_grid(img_S.cpu(), nrow=5))

            print('Temporal: {:03}, accuracy:{:03}%'.format(i, accuracy_T), flush=True)
            print('Predicted   : ', ''.join('%s ' % classes[predicted_T[j]] for j in range(minibatch_test)))
            print('GroundTruth : ', ''.join('%s ' % classes[target_T[j]] for j in range(minibatch_test)))
            #imshow(torchvision.utils.make_grid(img_T.cpu(), nrow=5))
    print('Test Finish')
    accuracy_S = round(correct_S / total_S *100)
    accuracy_T = round(correct_T / total_T * 100)
    print("Spatial's accuracy: {:03}".format(accuracy_S))#67
    print("Temporal's accuracy: {:03}".format(accuracy_T))#58

def main(mode, pretrained):
    pth = './save/SpatialTemporalNet.pth'
    net = FusionNet().cuda()
    if pretrained:
        checkpoint = torch.load(pth)
        net.load_state_dict(checkpoint['net_state_dict'])
        print('Load Finished')
    if mode =='train':
        train(net, pth)
    elif mode == 'test':
        test(net)

if __name__ == '__main__':
    main('train', False)
    #main('test', True)