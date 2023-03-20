using CSV
using Plots
using DataFrames
using StatsBase

# data = CSV.read("src/experiments/pomcpow_c=1.0_n=8.csv", DataFrame)
# grouped = combine(groupby(data, :size), Not(:size) .=> mean)
# grouped[!, "reward_online_mean"] - grouped[!, "reward_default_mean"]

# Group data together for different values of c

c_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
reward_diffs_8 = []
for c in c_vals
    path = "src/experiments/pomcpow_c=" * string(c) * "_n=8.csv"
    data = CSV.read(path, DataFrame)
    grouped = combine(groupby(data, :size), Not(:size) .=> mean)
    diff = grouped[!, "reward_online_mean"] - grouped[!, "reward_default_mean"]
    append!(reward_diffs_8, diff)
end

# Plot all hyperparameter tuning

param_tuning_first = plot(c_vals, reward_diffs_4, color=:green, xscale=:log10, ma=0.7, label="")
scatter!(c_vals, reward_diffs_4, ma=0.7, color=:green, label="\$n = 4\$")
plot!(c_vals, reward_diffs_6, ma=0.7, color=:blue, label="")
scatter!(c_vals, reward_diffs_6, ma=0.7, label="\$n = 6\$", color=:blue)
plot!(c_vals, reward_diffs_8, ma=0.7, color=:red, label="")
scatter!(c_vals, reward_diffs_8, ma=0.7, label="\$n = 8\$", color=:red)
title!("UCB1 Constant Hyperparameter Tuning for \$n = 4,6,8\$")
xlabel!("UCB1 Parameter \$c\$")
ylabel!("Average Advantage \$δ\$")

savefig(param_tuning_first, "src/figures/mcts_hyperparam_tuning_small.png")

param_tuning_10 = plot(c_vals, reward_diffs_10, ma=0.7, xscale=:log10, color=:orange, label="")
scatter!(c_vals, reward_diffs_10, ma=0.7, color=:orange, label="\$n=10\$")
title!("UCB1 Constant Hyperparameter Tuning for \$n = 10\$")
xlabel!("UCB1 Parameter \$c\$")
ylabel!("Average Advantage \$δ\$")

savefig(param_tuning_10, "src/figures/mcts_hyperparam_tuning_n=10.png")

# data = CSV.read("src/experiments/pomcpow_c=0.2_n=8.csv", DataFrame)
# grouped = combine(groupby(data, :size), Not(:size) .=> mean)
# grouped[!, "reward_online_mean"] - grouped[!, "reward_default_mean"]

data = CSV.read("src/experiments/pomcpow_c=0.2_n=8.csv", DataFrame)
grouped = combine(groupby(data, :size), Not(:size) .=> mean)
advantage = data[!, "reward_online"] - data[!, "reward_default"]
hist_8 = histogram(advantage, bins=range(-50, 250, 12), ma=0.2, color=:lightcoral, label="")
title!("POMCPOW Advantage Over Default Policy for \$n=8\$")
xlabel!("Advantage \$δ\$")
ylabel!("Count")

savefig(hist_8, "src/figures/advantage_resids_n=8.png")