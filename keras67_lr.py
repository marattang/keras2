weight = 1.0
input = 0.8
goal_prediction = 0.8   #이놈 빼고 다 튜닝된다.
lr = 100                #0.001 0.1 / 1 / 0.0001 / 100
epochs = 50

for iteration in range(epochs): # range => epochs
    prediction = input * weight
    error = (prediction - goal_prediction) **2

    print('Error : ' + str(error) + "\tPrediction : " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) ** 2

    if(down_error < up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr