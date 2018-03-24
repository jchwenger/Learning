# Tutorial playlist here: https://www.youtube.com/watch?v=2sQp7jJJmeg&list=PLcTXcpndN-Sl9eYrKM6jtcOTgC52EJnqH&index=7

from nltk.corpus import wordnet

sarr1 = wordnet.synsets('cake')
sarr2 = wordnet.synsets('loaf')
sarr3 = wordnet.synsets('bread')

print(sarr1)
print(sarr2)
print(sarr3)

cake = sarr1[0]
loafb = sarr2[0]
loaf = sarr2[1]
bread = sarr3[0]

print(cake.wup_similarity(loaf))
print(cake.wup_similarity(loafb))
print(loaf.wup_similarity(loafb))
print(bread.wup_similarity(loaf))
print(bread.wup_similarity(loafb))

print(loaf.hypernyms()[0])

ref = loaf.hypernyms()[0]

print(loaf.shortest_path_distance(ref))
print(bread.shortest_path_distance(ref))
print(loafb.shortest_path_distance(ref))
print(cake.shortest_path_distance(ref))

