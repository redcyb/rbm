from random import shuffle

fit_dict = {
    "Iris-setosa": 1,
    "Iris-versicolor": 0,
    "Iris-virginica": -1
}

with open("iris.data", "r") as f:
    data = f.read()
    data = data.split("\n")

# shuffle(data)

type_1 = []
type_0 = []
type_m1 = []

var0 = []
var1 = []


for d in data:
    try:
        dd = d.split(",")
        dd = dd[2:]
        dd[-1] = str(fit_dict[dd[-1]])

        if dd[-1] == "1":
            type_1.append(dd)

        if dd[-1] == "0":
            type_0.append(dd)

        if dd[-1] == "-1":
            type_m1.append(dd)

        var0.append(float(dd[0]))
        var1.append(float(dd[1]))

    except Exception as e:
        print(e)


range_v0 = min(var0), max(var0)
range_v1 = min(var1), max(var1)

print("var0:", range_v0)
print("var1:", range_v1)

print(len(data))

range_items = []
other_items = []

for d in type_0 + type_1 + type_m1:
    if (
            (float(d[0]) == range_v0[0] or float(d[0]) == range_v0[1]) or
            (float(d[1]) == range_v1[0] or float(d[1]) == range_v1[1])
    ):
        range_items.append(d)
    else:
        other_items.append(d)

print(range_items)

shuffle(other_items)

train_set = range_items + other_items[:-30]
test_set = other_items[-30:]

with open("irisTrain2.dat", "w") as f:
    shuffle(train_set)
    for d in train_set:
        f.write("\t".join(d) + "\n")

with open("irisTest2.dat", "w") as f:
    shuffle(test_set)
    for d in test_set:
        f.write("\t".join(d) + "\n")
