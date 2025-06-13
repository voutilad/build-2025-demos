import numpy as np
import matplotlib.pyplot as plt

# Constants (after applying 80:20 split)
o3_token_cost_per_1M = 4.8  # USD (revised to June '25 pricing)
gpt41_token_cost_per_1M = 4.4  # USD
ft_gpt41_token_cost_per_1M = 4.4  # USD
ft_gpt41_hosting_cost_per_hour = 1.7  # USD

# Per token cost
o3_cost_per_token = o3_token_cost_per_1M / 1_000_000
gpt41_cost_per_token = gpt41_token_cost_per_1M / 1_000_000
ft_gpt41_cost_per_token = ft_gpt41_token_cost_per_1M / 1_000_000

def plot_cost_comparison():
    """
    Plots the cost comparison between o3, gpt-4.1, and fine-tuned gpt-4.1 models.
    """
    # Range of TPM values
    tpm_values = np.linspace(100, 1_000_000, 200)  # 100 to 1 Million TPM

    # Calculate hourly costs
    o3_costs = o3_cost_per_token * tpm_values * 60
    gpt41_costs = gpt41_cost_per_token * tpm_values * 60
    ft_gpt41_costs = ft_gpt41_cost_per_token * tpm_values * 60 + ft_gpt41_hosting_cost_per_hour

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(tpm_values, o3_costs, label="o3 (Base Model)", color="blue")
    plt.plot(tpm_values, gpt41_costs, label="gpt-4.1 (Base Model)", color="green")
    plt.plot(tpm_values, ft_gpt41_costs, label="ft-gpt-4.1 (Fine-Tuned Model)", color="orange")
    plt.xlabel("Tokens per Minute (TPM)")
    plt.ylabel("Total Cost per Hour (USD)")
    plt.title("Cost Comparison: o3 vs gpt-4.1 vs ft-gpt-4.1 (June 2025 Pricing, 80:20 Input:Output)")
    plt.legend()
    plt.xscale("log")
    plt.grid(True)
    plt.show()

def plot_cost_break_even():
    """
    Plots the cost comparison with a zoomed-in view and highlights the break-even point.
    """
    # Range of TPM values
    tpm_values = np.linspace(50_000, 100_000, 400)

    # Cost calculations
    o3_costs = o3_cost_per_token * tpm_values * 60
    gpt41_costs = gpt41_cost_per_token * tpm_values * 60
    ft_gpt41_costs = ft_gpt41_cost_per_token * tpm_values * 60 + ft_gpt41_hosting_cost_per_hour

    # Break-even TPM calculation
    break_even_tpm = ft_gpt41_hosting_cost_per_hour / ((o3_cost_per_token - ft_gpt41_cost_per_token) * 60)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(tpm_values, o3_costs, label="o3 (Base Model)", color="blue")
    plt.plot(tpm_values, gpt41_costs, label="gpt-4.1 (Base Model)", color="green")
    plt.plot(tpm_values, ft_gpt41_costs, label="ft-gpt-4.1 (Fine-Tuned Model)", color="orange")

    # Add break-even line
    plt.axvline(x=break_even_tpm, color="red", linestyle="--", label=f"Break-even (~{int(break_even_tpm)} TPM)")

    plt.xlim(50_000, 100_000)
    plt.xlabel("Tokens per Minute (TPM)")
    plt.ylabel("Total Cost per Hour (USD)")
    plt.title("Cost Comparison and Break-Even: o3 vs gpt-4.1 vs ft-gpt-4.1 (Zoomed)")
    plt.legend()
    plt.grid(True)
    plt.show()