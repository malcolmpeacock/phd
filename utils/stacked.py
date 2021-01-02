import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    'cat': [3, 5, 4, 3, 3, 23, 4],
    'dog': [3, 23, 4, 3, 5, 4, 3],
    'mouse': [3, 23, 4, 3, 5, 4, 3]
})

# Save the chart that's drawn
ax = df.plot(stacked=True, kind='barh', figsize=(10, 5))

# .patches is everything inside of the chart, lines and
# rectangles and circles and stuff. In this case we only
# have rectangles!
for rect in ax.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()
    
    # The padding is NOT pixels, it's whatever your axis is.
    # So this means pad by half an animal
    padding = 0.5

    # The width of the bar is also not pixels, it's the
    # number of animals. So we can use it as the label!
    label_text = width
    
    # ax.text(x, y, text)
    label_x = x + width - padding
    label_y = y + height / 2
    ax.text(label_x, label_y, label_text, ha='right', va='center')

plt.show()
