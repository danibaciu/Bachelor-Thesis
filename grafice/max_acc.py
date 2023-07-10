model_number = 6

output_lr4e = """Train loss:  0.37527649213333386
Dev loss:  0.2967184106019711
Acc train:  0.8261663286004057
Acc dev:  0.8682133838383839
Train loss:  0.17176801772737826
Dev loss:  0.25464111120637617
Acc train:  0.9342461122379987
Acc dev:  0.8988320707070707
Train loss:  0.07006917755660677
Dev loss:  0.3643805261858918
Acc train:  0.974949290060852
Acc dev:  0.8926767676767676
Train loss:  0.044107401284036805
Dev loss:  0.36129160003849503
Acc train:  0.9847870182555781
Acc dev:  0.8934659090909091
Train loss:  0.032246328200610054
Dev loss:  0.4017393691512295
Acc train:  0.9887762001352265
Acc dev:  0.8947285353535354"""

output_lr2e = """Train loss:  0.027246586257478576
Dev loss:  0.5057387295834053
Acc train:  0.9918864097363083
Acc dev:  0.8888888888888888
Train loss:  0.009247332973251548
Dev loss:  0.5547429469632775
Acc train:  0.9968221771467207
Acc dev:  0.889520202020202
Train loss:  0.00575949800905545
Dev loss:  0.6028922771392072
Acc train:  0.9980392156862745
Acc dev:  0.8997790404040404
Train loss:  0.007151446371631087
Dev loss:  0.512072066085018
Acc train:  0.9977687626774848
Acc dev:  0.8882575757575758
Train loss:  0.00601453330537707
Dev loss:  0.5474239962150941
Acc train:  0.9980392156862745
Acc dev:  0.8975694444444444"""

output_test_lr_4e = """Test loss:  0.2829320089114671
Acc test:  0.8816204287515763
Test loss:  0.23925730184974833
Acc test:  0.9046343001261034
Test loss:  0.3366202077730067
Acc test:  0.9036885245901639
Test loss:  0.37161404757535826
Acc test:  0.9014817150063051
Test loss:  0.4131156715415108
Acc test:  0.9033732660781841"""

output_test_lr_2e = """Test loss:  0.41277075349060754
Acc test:  0.9066834804539723
Test loss:  0.5032654620335778
Acc test:  0.9040037831021438
Test loss:  0.4928675162784074
Acc test:  0.9036885245901639
Test loss:  0.3677506497410306
Acc test:  0.9068411097099621
Test loss:  0.5564702905611735
Acc test:  0.9027427490542245"""


def from_output_vector_test(output):
    output = output.split('\n')

    test_loss = []
    test_acc = []

    for line in output:
        label, acc = line.split(':')
        if 'Test loss' in label:
            test_loss.append(float(acc[2:]))
        else:
            test_acc.append(float(acc[2:]))
    return test_acc, test_loss


def from_output_vector_dev(output):
    output = output.split('\n')

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for line in output:
        label, acc = line.split(':')
        if 'Train loss' in label:
            train_loss.append(float(acc[2:]))
        elif 'Dev loss' in label:
            val_loss.append(float(acc[2:]))
        elif 'Acc train' in label:
            train_acc.append(float(acc[2:]))
        else:
            val_acc.append(float(acc[2:]))
    return train_acc, val_acc, train_loss, val_loss


lr4e = from_output_vector_dev(output_lr4e)
lr2e = from_output_vector_dev(output_lr2e)

test_lr4e = from_output_vector_test(output_test_lr_4e)
test_lr2e = from_output_vector_test(output_test_lr_2e)

x, y, z, t = round(max(max(lr4e[1]),max(lr2e[1])), 3), round(max(max(lr4e[3]),max(lr2e[3])), 3), round(max(max(test_lr4e[0]),max(test_lr2e[0])),3), round(max(max(test_lr4e[1]),max(test_lr2e[1])),3)

print(f"max acc dev = {round(max(max(lr4e[1]),max(lr2e[1])), 3)}")
print(f"max loss dev = {round(max(max(lr4e[3]),max(lr2e[3])), 3)}")
print(f"max acc test = {round(max(max(test_lr4e[0]),max(test_lr2e[0])),3)}")
print(f"max loss test = {round(max(max(test_lr4e[1]),max(test_lr2e[1])),3)}")

print(f"Model {model_number}& {x}& {y}& {z}& {t}\\\\")

