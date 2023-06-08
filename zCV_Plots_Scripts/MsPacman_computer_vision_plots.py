import matplotlib.pyplot as plt

mspacman_aug_scores = [963, 537, 361, 551, 701, 936]
mspacman_mlr_scores = [308.4, 476.3, 584.1, 495.1, 505.6, 659.4]
mspacman_no_aug_scores = [417.8, 60, 60, 60, 60, 60]
mspacman_temporal_mlr_scores = [332.8, 573.8, 445, 419.4, 434.4, 683.1]
mspacman_data = [mspacman_no_aug_scores, mspacman_aug_scores, mspacman_mlr_scores, mspacman_temporal_mlr_scores]

x_values = [0, 10000, 20000, 30000, 40000, 50000]

for y_values in mspacman_data:
    plt.plot(x_values, y_values, 'o-')

plt.xlabel('Training Steps')
plt.ylabel('Mean Score')
plt.title('MsPacman Scores for Different Representation Learning Methods')
plt.legend(['No Augmentation', 'Augmentation', 'MLR', 'Temporal MLR'])
plt.grid(True)
plt.savefig('images/models/mspacman_scores.png')
plt.show()

