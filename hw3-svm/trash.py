from collections import Counter

def bag_of_words(list):

    cnt = Counter()
    for word in list:
        cnt[word] += 1

    return cnt

a = ['the', 'bankrupt', 'man', 'dances', 'man', 'the']
b = bag_of_words(a)

print(b)