new_accu = 0.9324524172324531
div = new_accu/a[0]
print("New values: ")

for i in range(len(a)):
    cute = div*(a[i]/b[i])
    c.append(cute)
c[0] = new_accu
print(list(c))