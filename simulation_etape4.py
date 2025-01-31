import numpy
import matplotlib.pyplot as plt

def sird_forecast(beta, gamma, mu, step, duration):
    time = [0]
    susceptible_to_be_infected = [0.99]
    infected = [0.01]
    recovered = [0]
    deceased = [0]

    nb_points = int(duration / step)

    for _ in range(1, nb_points):
        new_value_time = time[-1] + step
        
        new_susceptible_to_be_infected = - beta * susceptible_to_be_infected[-1] * infected[-1] * step + susceptible_to_be_infected[-1]
        new_infected = (beta * susceptible_to_be_infected[-1] * infected[-1] - gamma * infected[-1] - mu * infected[-1]) * step + infected[-1]
        new_recovered = gamma * infected[-1] * step + recovered[-1]
        new_deceased = mu * infected[-1] * step + deceased[-1]

        time.append(new_value_time)
        susceptible_to_be_infected.append(new_susceptible_to_be_infected)
        infected.append(new_infected)
        recovered.append(new_recovered)
        deceased.append(new_deceased)
    
    time = time[::int(1 / step)]
    susceptible_to_be_infected = numpy.array(susceptible_to_be_infected[::int(1 / step)])
    infected = numpy.array(infected[::int(1 / step)])
    recovered = numpy.array(recovered[::int(1 / step)])
    deceased = numpy.array(deceased[::int(1 / step)])
    
    return time, susceptible_to_be_infected, infected, recovered, deceased

def plot_data(time, susceptible_to_be_infected, infected, recovered, deceased):
    plt.figure(figsize=(15, 6))
    plt.plot(time, susceptible_to_be_infected, "--b", label='Prediction : Susceptible to be infected')
    plt.plot(time, infected, "--y", label='Prediction : Infected')
    plt.plot(time, recovered, "--g", label='Prediction : Recovered')
    plt.plot(time, deceased, "--r", label='Prediction : Deceased')
    plt.xlabel('Time (Days)', weight='bold')
    plt.ylabel('Population (%)', weight='bold')
    plt.title('SIRD Model, Taux de transmission = 35%', weight='bold', fontsize=18)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    beta = 0.5
    gamma = 0.15
    mu = 0.015
    step = 0.01
    duration = 90
    time, susceptible_to_be_infected, infected, recovered, deceased = sird_forecast(beta, gamma, mu, step, duration)
    plot_data(time,susceptible_to_be_infected, infected, recovered, deceased)