import matplotlib.pyplot as plt
import numpy as np

m = np.arange(1,11)
c = 299792458
E = m*c**2

fig,axes = plt.subplots()



# ax.set_title('E=mc^2')
# ax.set_xlabel('Mass in Grams')
# ax.set_ylabel('Energy in Joules')

# ax.set_xlim(m.min(),10)
# ax.grid(axis='y',which='minor')


# ax.semilogy(m,E,color='red',lw='5')

# res = E


labels = ['1 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']

july16_2007 =[4.75,4.98,5.08,5.01,4.89,4.89,4.95,4.99,5.05,5.21,5.14]
july16_2020 = [0.12,0.11,0.13,0.14,0.16,0.17,0.28,0.46,0.62,1.09,1.31]


ax1 = axes


ax1.plot(labels,july16_2007)
ax1.set_title('july 16th Yield Curves')

ax2 = ax1.twinx()

ax1.set_ylabel('2007')
ax2.set_ylabel('2020')

ax2.plot(labels,july16_2020)


#ax1.legend(loc=[1.1,0.5])


plt.tight_layout()
plt.show()