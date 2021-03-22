from GraphVisualizations import *
from UtilityFunctions import *
colors = ["#313695", "#a50026", "#fee838", "#004529"]

if __name__ == "__main__":
    # This script runs two compartmental models, one with included beta calculation over time, one without

    # With beta
    rn.seed(42)
    days = 100
    res_100 = calculate_new_states_beta(days=days)
    t = res_100[0]
    R_0_over_time = res_100[-1]
    fig = plt.figure(figsize=(18, 10))
    plt.subplot(1, 2, 1)
    plt.plot(range(days), t, c=colors[1])
    plt.title("T over time")
    plt.subplot(1, 2, 2)
    plt.plot(range(days), R_0_over_time, c=colors[2])
    plt.title("R_0 over time")
    plt.show()
    plot_seir(res_100[1], res_100[2], res_100[3], res_100[4])

    # No beta
    days = 100
    res_100 = calculate_new_states(days=days)
    plot_seir(res_100[0], res_100[1], res_100[2], res_100[3])
