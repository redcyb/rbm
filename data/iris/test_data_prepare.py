from random import shuffle

fit_dict = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": -1
}

with open("iris.data", "r") as f:
    data = f.read()
    data = data.split("\n")

shuffle(data)

type_1 = []
type_0 = []
type_m1 = []

var0 = []
var1 = []
var2 = []
var3 = []


for d in data:
    try:
        dd = d.split(",")
        dd[-1] = str(fit_dict[dd[-1]])

        if dd[-1] == "1":
            type_1.append(dd)

        if dd[-1] == "0":
            type_0.append(dd)

        if dd[-1] == "-1":
            type_m1.append(dd)

        var0.append(float(dd[0]))
        var1.append(float(dd[1]))
        var2.append(float(dd[2]))
        var3.append(float(dd[3]))

    except Exception as e:
        print(e)


range_v0 = min(var0), max(var0)
range_v1 = min(var1), max(var1)
range_v2 = min(var2), max(var2)
range_v3 = min(var3), max(var3)

print("var0:", range_v0)
print("var1:", range_v1)
print("var2:", range_v2)
print("var3:", range_v3)

print(len(data))

range_items = []
other_items = []

for d in type_0 + type_1 + type_m1:
    if (
            (float(d[0]) == range_v0[0] or float(d[0]) == range_v0[1]) or
            (float(d[1]) == range_v1[0] or float(d[1]) == range_v1[1]) or
            (float(d[2]) == range_v2[0] or float(d[2]) == range_v2[1]) or
            (float(d[3]) == range_v3[0] or float(d[3]) == range_v3[1])
    ):
        range_items.append(d)
    else:
        other_items.append(d)

print(range_items)

shuffle(other_items)

train_set = range_items + other_items[:-30]
test_set = other_items[-30:]

with open("irisTrain1.dat", "w") as f:
    # shuffle(train_set)
    for d in train_set:
        f.write("\t".join(d) + "\n")

with open("irisTest1.dat", "w") as f:
    # shuffle(test_set)
    for d in test_set:
        f.write("\t".join(d) + "\n")

# with open("trainingSetOutCheck.dat", "w") as f:
#     ddd = type_1[-5:] + type_0[-5:] + type_m1[-5:]
#     shuffle(ddd)
#     for d in ddd:
#         f.write("\t".join(d) + "\n")


# with open("trainingSetOutTrain.dat", "w") as f:
#     ddd = type_1[:-10] + type_0[:-10] + type_m1[:-10]
#     shuffle(ddd)
#     for d in ddd:
#         f.write("\t".join(d) + "\n")
#
# with open("trainingSetOutTest.dat", "w") as f:
#     ddd = type_1[-10:-5] + type_0[-10:-5] + type_m1[-10:-5]
#     shuffle(ddd)
#     for d in ddd:
#         f.write("\t".join(d) + "\n")
#
# with open("trainingSetOutCheck.dat", "w") as f:
#     ddd = type_1[-5:] + type_0[-5:] + type_m1[-5:]
#     shuffle(ddd)
#     for d in ddd:
#         f.write("\t".join(d) + "\n")
