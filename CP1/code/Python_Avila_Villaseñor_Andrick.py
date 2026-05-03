import numpy as np
import matplotlib.pyplot as plt

def dibujar_circulo(r, xc, yc):
    theta = np.linspace(0, 2*np.pi, 400)
    x = r * np.cos(theta) + xc
    y = r * np.sin(theta) + yc
    plt.plot(x, y)

plt.figure(figsize=(8,8))


num_circulos = 40000
for k in range(num_circulos):
    xc = (3/2) * (np.cos(141*np.pi*k/40000))**9 * \
     (1 - (1/2)*np.sin(np.pi*k/40000)) * \
     \
     (1 - (1/4)*((np.cos(2*np.pi*k/40000))**30) * \
     (1 + (np.cos(32*np.pi*k/40000))**20)) * \
     \
     (1 - (1/2)*((np.sin(2*np.pi*k/40000))**30) * \
     ((np.sin(6*np.pi*k/40000))**10) * \
     ((1/2 + (1/2)*(np.sin(18*np.pi*k/40000))**20)) \
     )

    yc = np.cos(2*np.pi*k/40000) * \
     (np.cos(141*np.pi*k/40000))**2 * \
     (1 + (1/4) * (np.cos(np.pi*k/40000) * \
                   np.cos(3*np.pi*k/40000) * \
                   np.cos(21*np.pi*k/40000))**24)

    radio = (1/40) * (
    (np.cos(141*np.pi*k/40000))**14 + (np.sin(141*np.pi*k/40000))**6
) * (
    1 - (
        np.cos(np.pi*k/40000) * np.cos(3*np.pi*k/40000) * np.cos(12*np.pi*k/40000)
    )**16
) + (1/100)

    dibujar_circulo(radio, xc, yc)

plt.gca().set_aspect("equal", "box")
plt.title("Yenageh de Mariposa")
plt.savefig("assets/Butterfly.png", dpi=300)
plt.show()