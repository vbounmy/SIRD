import numpy
import pandas
import matplotlib.pyplot as plt
from tqdm.contrib.itertools import product

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

def mse(model_prediction, ground_data):
    mse = numpy.square(numpy.subtract(model_prediction, ground_data)).mean()
    return mse

def plot_data(time, susceptible_to_be_infected, infected, recovered, deceased, ground_truth):
    plt.figure(figsize=(15, 6))
    plt.plot(time, susceptible_to_be_infected, "-b", label='Prediction : Susceptible to be infected')
    plt.plot(time, infected, "-y", label='Prediction : Infected')
    plt.plot(time, recovered, "-g", label='Prediction : Recovered')
    plt.plot(time, deceased, "-r", label='Prediction : Deceased')
    plt.plot(time, ground_truth["Susceptibles"], "--b", label='Ground truth : Susceptible to be infected')
    plt.plot(time, ground_truth["Infectés"], "--y", label='Ground truth : Infected')
    plt.plot(time, ground_truth["Rétablis"], "--g", label='Ground truth : Recovered')
    plt.plot(time, ground_truth["Décès"], "--r", label='Ground truth : Deceased')
    plt.xlabel('Time (Days)', weight='bold')
    plt.ylabel('Population (%)', weight='bold')
    plt.title('Modèle SIRD', weight='bold', fontsize=18)
    plt.legend()
    plt.show()

def grid_search(step, duration, ground_truth):
    betas = numpy.linspace(0.25, 0.5, 3)
    gammas = numpy.linspace(0.08, 0.15, 3)
    mus = numpy.linspace(0.005, 0.015, 3)

    best_beta, best_gamma, best_mu = None, None, None
    best_mse, best_mse_susceptible_to_be_infected, best_mse_infected, best_mse_recovered, best_mse_deceased = float("inf"), float("inf"), float("inf"), float("inf"), float("inf")

    for beta, gamma, mu in product(betas, gammas, mus):
        time, susceptible_to_be_infected, infected, recovered, deceased = sird_forecast(beta, gamma, mu, step, duration)
        mse_susceptible_to_be_infected = mse(susceptible_to_be_infected, ground_truth["Susceptibles"].values)
        mse_infected = mse(infected, ground_truth["Infectés"].values)
        mse_recovered = mse(recovered, ground_truth["Rétablis"].values)
        mse_deceased = mse(deceased, ground_truth["Décès"].values)
        actual_mse = mse_susceptible_to_be_infected + mse_infected + mse_recovered + mse_deceased

        if actual_mse < best_mse:
            best_mse, best_mse_susceptible_to_be_infected, best_mse_infected, best_mse_recovered, best_mse_deceased = actual_mse, mse_susceptible_to_be_infected, mse_infected, mse_recovered, mse_deceased
            best_beta, best_gamma, best_mu = beta, gamma, mu
    
    print(f"best global MSE = {best_mse}")
    print(f"best 'Susceptible to be infected' MSE = {best_mse_susceptible_to_be_infected}")
    print(f"best 'Infected' MSE = {best_mse_infected}")
    print(f"best 'Recovered' MSE = {best_mse_recovered}")
    print(f"best 'Deceased' MSE = {best_mse_deceased}")
    print(f"best beta = {best_beta}, best gamma = {best_gamma}, best_mu = {best_mu}")
    time, susceptible_to_be_infected, infected, recovered, deceased = sird_forecast(best_beta, best_gamma, best_mu, step, duration)
    plot_data(time, susceptible_to_be_infected, infected, recovered, deceased, ground_truth)

if __name__ == "__main__":
    step = 0.01
    duration = 90
    ground_truth = pandas.read_csv("sird_dataset.csv")
    grid_search(step, duration, ground_truth)
