def min(a, b):
    return a if a < b else b

def max(a, b):
    return a if a > b else b

def median():
    a1 = [int(i) for i in input().split()]
    a2 = [int(i) for i in input().split()]

    l1 = len(a1)
    l2 = len(a2)
    l = l1 + l2

    # nếu a1 nhiều phần tử  hơn thì hoán đổi hai mảng
    if (l1 > l2):
        temp = a1
        a1 = a2
        a2 = temp

        l1 = len(a1)
        l2 = len(a2)

    first = 0
    last = l1 - 1
    median = -1

    while (first <= last):
        mid = (first + last) // 2
        # chọn vị trí ở mảng 2 sao cho 
        # tổng số phần từ của hai nửa đầu mảng = (l1 + l2)//2
        index2 = (l1 + l2) //2 - mid - 2

        if (mid == 0 and a1[mid] > a2[index2 + 1]):
            index2 = (l-1) // 2 
            if (l % 2 ==0):
                if (index2 == l2 - 1):
                    median = ( a2[index2] + a1[0] ) / 2
                else:
                    median = ( a2[index2] + a2[index2 + 1] ) / 2 
            else:
                median = a2[index2]
            break

        elif (mid == l1 - 1):
            if (l % 2 == 0):
                median = (max (a1[mid], a2[index2]) + a2[index2 + 1]) / 2
            else:
                median = a2[index2]
            break

        elif ( (a1[mid] <= a2[index2 + 1]) and (a2[index2] <= a1[mid + 1]) ):
            if (l % 2 == 0):
                median = ( max (a1[mid], a2[index2]) +
                     min (a1[mid + 1], a2[index2 + 1]) ) / 2
            else:
                median = min (a1[mid + 1], a2[index2 + 1])
            break 

        elif ( a1[mid] > a2[index2 + 1]):
            last = mid - 1

        elif ( a2[index2] > a1[mid + 1]):
            first = mid + 1

    print('median = {}'.format(median))
    

if __name__ == '__main__':
    median()
    
    


