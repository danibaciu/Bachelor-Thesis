model_number = "4-B"

output_lr4e = """Train loss:  0.30916251242492526
Test loss:  0.33433252865982116
Acc train:  0.8634591825370308
Acc test:  0.8540353089533418
Train loss:  0.14373947491409889
Test loss:  0.19787237432846172
Acc train:  0.9440360841964585
Acc test:  0.9170870113493065
Train loss:  0.07046712241216396
Test loss:  0.2608093983334625
Acc train:  0.9748023165163159
Acc test:  0.9101513240857503
Train loss:  0.04013944299024333
Test loss:  0.2839130084369631
Acc train:  0.9864405835839181
Acc test:  0.9169293820933165
Train loss:  0.027583752683035135
Test loss:  0.33859114463753465
Acc train:  0.990004454839069
Acc test:  0.9137767969735183"""

output_lr2e = """Train loss:  0.021077525984328563
Test loss:  0.3583049141070934
Acc train:  0.9959628020937744
Acc test:  0.9222887767969735
Train loss:  0.0068767548696366445
Test loss:  0.3910068512224971
Acc train:  0.9980510079073394
Acc test:  0.9166141235813366
Train loss:  0.007634239264907778
Test loss:  0.4769680074323013
Acc train:  0.9976612094888072
Acc test:  0.9188209331651954
Train loss:  0.006811665524607049
Test loss:  0.4679224571828767
Acc train:  0.9977725804655306
Acc test:  0.9164564943253468
Train loss:  0.00705979848443432
Test loss:  0.4425054644964999
Acc train:  0.9977447377213499
Acc test:  0.9174022698612863"""


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
        elif 'Test loss' in label:
            val_loss.append(float(acc[2:]))
        elif 'Acc train' in label:
            train_acc.append(float(acc[2:]))
        else:
            val_acc.append(float(acc[2:]))
    return train_acc, val_acc, train_loss, val_loss


def from_output_to_acc(lr):
    plm = []

    for i in [0, 1]:
        max_acc = 0
        for a in range(len(lr[i])):
            if lr[i][a] > lr[i][max_acc]:
                max_acc = a
        plm.append(lr[i][max_acc])
        plm.append(lr[i+2][max_acc])

    return plm


lr4e = from_output_vector_dev(output_lr4e)
lr2e = from_output_vector_dev(output_lr2e)

lr4e = from_output_to_acc(lr4e)
lr2e = from_output_to_acc(lr2e)

print(f" max 4e = {lr4e}")
print(f" max 2e = {lr2e}")


print("\n")
s = f"Model {model_number}& "

for i in range(4):
    if i %2 == 0:
        s += f" {round(max(lr4e[i],lr2e[i]) *100, 3)}\% "
    else:
        s += f" {round(max(lr4e[i], lr2e[i]), 3)} "
    if i < 3:
        s += " & "
s+="\\\\"

print(s)