import matplotlib.pyplot as plt

# Sample x and y lists
breakout_aug_scores = [1.969, 28.53, 68.19, 164.3, 189.5, 207.8]
breakout_mlr_scores = [1.75, 0.09375, 0.0, 57.25, 23.53, 97.75]
breakout_no_aug_scores = [1.656, 0.0, 0.0, 0.0, 0.0, 0.0]
breakout_temporal_mlr_scores = [1.719, 0.125, 0.0, 0.0, 13.94, 96.25]
breakout_new_mask_mlr_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
breakout_data = [breakout_no_aug_scores, breakout_aug_scores, breakout_mlr_scores, breakout_temporal_mlr_scores, breakout_new_mask_mlr_scores]

mspacman_no_aug_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mspacman_aug_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mspacman_mlr_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mspacman_temporal_mlr_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

x_values = [0, 10000, 20000, 30000, 40000, 50000]

# Plotting the data
for y_values in breakout_data:
    plt.plot(x_values, y_values, 'o-')  # 'o-' represents circles connected with lines

plt.xlabel('Training Steps')
plt.ylabel('Mean Score')
plt.title('Breakout Scores for Different Representation Learning Methods')
plt.legend(['No Augmentation', 'Augmentation', 'MLR', 'Temporal MLR', 'Improved MLR'])  # Add a legend for the x lists
plt.grid(True)
plt.savefig('images/models/breakout_scores.png')
plt.show()

