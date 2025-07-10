import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Define the intervals and corresponding percentages for three models
intervals = ['[0,3]', '[4,6]', '[7,9]', '[10,12]']
cir3 =                  [04.00, 76.00, 19.50, 00.50]
cir3_no_perspective =   [09.00, 69.00, 17.00, 05.00]
cir3_no_curmudgeon =    [17.50, 57.00, 14.00, 11.50]
cir3_agent_only =       [06.50, 71.00, 16.50, 06.00] # [-2.5, 5.0, 3.0, -5.5] from CIR3
cir3_vendi_only =       [09.50, 60.50, 11.00, 19.00] # [-5.5, 15.5, 8.5, -18.5] from CIR3
# approve: p = 0.3
cir3_random_rejection = [84.00, 10.00, 04.00, 02.00] # [-80.0, 66.0, 15.5, -1.5]



# Combine all percentages into a single list
all_percentages = cir3 + cir3_no_perspective + cir3_no_curmudgeon + cir3_agent_only + cir3_vendi_only + cir3_random_rejection

# Calculate the maximum percentage value
max_percentage = max(all_percentages)

# Set the positions and width for the bars
x = range(len(intervals))
width = 0.13  # Width of the bars (reduced to prevent overlap)

# Define hatch patterns for each model
hatches = ['xx', '||', '..', '//', '\\\\', '--']



# colors = sns.color_palette("Greys", n_colors=3)
# colors = ['#add8e6', '#fffdd0', '#f5f5dc']
# Define the colors for light gray, light blue, and beige
colors = ['#d3d3d3', '#add8e6', '#f5f5dc', '#ffa500', '#008000', '#c3aed6']  # Light gray, light blue, beige, orange, green, light purple

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))  

# List of all data groups and their labels
data_groups = [
    (cir3, 'CIR3'),
    (cir3_no_perspective, 'CIR3 w/o perspectives'),
    (cir3_no_curmudgeon, 'CIR3 w/o Curmudgeon'),
    (cir3_agent_only, 'CIR3 Agent Only'),
    (cir3_vendi_only, 'CIR3 Vendi Only'),
    (cir3_random_rejection, 'CIR3 Random Rejection')
]

num_groups = len(data_groups)
bar_containers = []

for i, (data, label) in enumerate(data_groups):
    offset = (i - (num_groups - 1) / 2) * width
    bar = ax.bar(
        [p + offset for p in x],
        data,
        width=width,
        label=label,
        hatch=hatches[i],
        edgecolor='black',
        color=colors[i]
    )
    bar_containers.append(bar)

# Set the x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(intervals)

# Set the y-axis to display percentage
# ax.yaxis.set_major_formatter(PercentFormatter())

# Set the upper limit of the y-axis slightly above the maximum percentage value
ax.set_ylim(0, max_percentage * 1.1)

# Set the labels and title
# ax.set_xlabel('Cycle Ranges')
ax.set_ylabel('Processed Documents (%)')
# ax.set_title('Context distribution across cycle ranges (%)')

# Add a legend
ax.legend()

# Display the plot
plt.show()