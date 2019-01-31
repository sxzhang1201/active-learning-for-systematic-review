from added_value.active_learning import active_learning, plot_added_value
from added_value.correlation import correlation
import recipe


#Perform active learning over all 20 reviews
indices_full_dataset = list(range(1, 21))
# Train a model and evaluate its performance by WSS
for index in indices_full_dataset:
  active_learning(index, query_strategy=recipe.QUERY_STRATEGY, importance=recipe.SAMPLE_WEIGHT)

# Plot trend graph
plot_added_value()

# Perform pearson coefficient analysis
correlation()