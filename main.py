import backPropagation

if __name__ == '__main__':
    trainSet = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    flag = 'y'
    while flag == 'y':
        rate = float(input('Please input learning rate : '))
        error = float(input('Please input expected error : '))

        bp = backPropagation.backPropagation(trainSet, rate, error)

        i = 0
        firstErr = finalErr = 0.0
        numBatch = 0

        init_inWeight = bp.get_inWeight()
        print('the initial input weight---------------')
        print(init_inWeight)
        init_hideWeight = bp.get_hideWeight()
        print('the initial hide weight---------------')
        print(init_hideWeight)

        while True:
            i += 1
            bp.train()
            if i == 1:
                firstErr = bp.error
            if bp.error < error:
                finalErr = bp.error
                numBatch = i
                break
            if i >= 100000:
                numBatch = 100000
                finalErr = bp.error
                break

        print('first-batch error-----------------------')
        print(firstErr)
        print('final input node weight-----------------')
        print(bp.get_inWeight())
        print('final hide node weight-----------------')
        print(bp.get_hideWeight())
        print('final-batch error-----------------------')
        print(finalErr)
        print('total number of batches-----------------')
        print(numBatch)

        flag = input('Continue...?')
