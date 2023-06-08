import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0，3'
import time
import random
import numpy as np
import setproctitle
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.HDT_Dataset import HDT

from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
import pandas as pd
from models.HDT_Net import HDT_Net



parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--root', default='/home/cyx/Datasets/BraData/', type=str)

parser.add_argument('--valid_dir', default='Val', type=str)

parser.add_argument('--valid_file', default='IDH_test.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='My_Net_plus', type=str) #TransBraTS

parser.add_argument('--test_date', default='2023-04-29', type=str)

parser.add_argument('--test_file', default='model_epoch_19.pth', type=str) #model_epoch_last   model_epoch_74.pth model_epoch_best.pth

parser.add_argument('--use_TTA', default=True, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='HDT_Net_In_Out', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1', type=str)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()



def tailor_and_concat(x, model):
    temp = []
    idh_temp=[]
    # 图像裁剪为八块
    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    for i in range(len(temp)):
        idh_out = model(temp[i])
        # grade_out = model['grade'](encoder_outs[3], encoder_outs[4])
        print("idh_out:",idh_out)
        idh_temp.append(idh_out)

    idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    print("idh_out mean:",idh_out)
    return idh_out#,grade_out






def main(model):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    IDH_model = model
    IDH_model = torch.nn.DataParallel(IDH_model).cuda()
    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment + args.model_name + args.test_date, args.test_file)
    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        IDH_model.load_state_dict(checkpoint['idh_state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.model_name+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = HDT(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("valid_loader",valid_loader)

    start_time = time.time()

    with torch.no_grad():

        IDH_model.eval()
        idh_prob = []
        idh_class = []
        idh_truth = []
        idh_error_case = []
        ids = []
        names = valid_set.names

        for i,data in enumerate(valid_loader):
            print('-------------------------------------------------------------------')
            msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))

            print("data[0]:", data[0].shape, 'data[1]', data[1])
            data = [t.cuda(non_blocking=True) for t in data]

            x, idh = data[:2]
            x = x[..., :155]
            # 测试时增强
            TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8 = tailor_and_concat(x, IDH_model), \
                                                                     tailor_and_concat(x.flip(dims=(2,)), IDH_model), \
                                                                     tailor_and_concat(x.flip(dims=(3,)),IDH_model),\
                                                                     tailor_and_concat(x.flip(dims=(4,)), IDH_model), \
                                                                     tailor_and_concat(x.flip(dims=(2, 3)),IDH_model), \
                                                                     tailor_and_concat(x.flip(dims=(2, 4)), IDH_model), \
                                                                     tailor_and_concat(x.flip(dims=(3, 4)),IDH_model), \
                                                                     tailor_and_concat(x.flip(dims=(2, 3, 4)), IDH_model)
            idh_probs = []
            # 将所作八个增强后的结果计算平均值
            for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
                idh_probs.append(F.softmax(pred, 1))
            idh_pred = torch.mean(torch.stack(idh_probs), dim=0)
            print("idh_pred:", idh_pred)
            idh_prob.append(idh_pred[0][1].item())

            idh_pred_class = torch.argmax(idh_pred, dim=1)

            idh_class.append(idh_pred_class.item())
            print('id:', names[i], 'IDH_truth:', idh.item(), 'IDH_pred:', idh_pred_class.item())

            ids.append(names[i])
            idh_truth.append(idh.item())
            if not (idh_pred_class.item() == idh.item()):
                idh_error_case.append({'id': names[i], 'truth:': idh.item(), 'pred': idh_pred_class.item()})

            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)

            print(msg)

        print("--------------------------------IDH evaluation report---------------------------------------")

        data = pd.DataFrame({"ID": ids, "pred": idh_prob, "pred_class": idh_class, "idh_truth": idh_truth})
        data.to_csv("revised/"+args.model_name+"BraTS_pred_Focus_loss_dual.csv")
        confusion = confusion_matrix(idh_truth, idh_class)
        print(confusion)
        labels = [0, 1]
        target_names = ["wild", "Mutant"]

        print(classification_report(idh_truth, idh_class, labels=labels, target_names=target_names))
        print("AUC:", roc_auc_score(idh_truth, idh_prob))
        print("Acc:", accuracy_score(idh_truth, idh_class))
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        print("Precision: " + str(precision))
        print("-------------------------- error cases----------------------------------------")
        for case in idh_error_case:
            print(case)



    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))





if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    model = HDT_Net()
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main(model)


