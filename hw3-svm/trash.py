from collections import Counter

def bag_of_words(list):

    cnt = Counter()
    for word in list:
        cnt[word] += 1

    return cnt

a = ['the', 'bankrupt', 'man', 'dances', 'man', 'the']
b = bag_of_words(a)

print(b)

temp_w = w.copy()
        increment(temp_w, s-1, temp_w)
        loss_real = loss(x,y,l,temp_w)
        if abs(temp_loss - loss_real) < 10**-2:
            flag = False
        temp_loss = loss_real